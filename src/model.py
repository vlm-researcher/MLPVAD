from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from clip import clip
from utils.layers import GraphConvolution, DistanceAdj

class LayerNorm(nn.LayerNorm):

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor, padding_mask: torch.Tensor):
        padding_mask = padding_mask.to(dtype=bool, device=x.device) if padding_mask is not None else None
        self.attn_mask = self.attn_mask.to(device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, key_padding_mask=padding_mask, attn_mask=self.attn_mask)[0]

    def forward(self, x):
        x, padding_mask = x
        x = x + self.attention(self.ln_1(x), padding_mask)
        x = x + self.mlp(self.ln_2(x))
        return (x, padding_mask)


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)
    
class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, ff_dim):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=dim, num_heads=8, batch_first=True)
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ff_dim),
            QuickGELU(),
            nn.Linear(ff_dim, dim)
        )

    def forward(self, visual_input, text_features):
        # visual_input: (B, 256, D)
        # text_features: (B, 14, D)

        Q = visual_input # (B, 256, D)
        K = text_features # (B, 14, D)
        V = text_features # (B, 14, D)

        attn_output, _ = self.mha(Q, K, V)  # (256, B, D)
    
        # Add & Norm 1
        x = self.ln1(visual_input + attn_output)

        # Feed-forward
        ffn_output = self.ffn(x)

        # Add & Norm 2
        out = self.ln2(x + ffn_output)
        return out

class EntityCrossAttention(nn.Module):
    def __init__(self, dim, num_heads=4, noun_dict=None):
        super().__init__()
        self.query_proj = nn.Linear(dim, dim)
        self.key_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.num_heads = num_heads
        self.dim = dim

        self.class_noun_weights = nn.ParameterDict()
        for cls, nouns in noun_dict.items():  # noun_dict: dict[str, list[str]]
            n = len(nouns)
            self.class_noun_weights[cls] = nn.Parameter(torch.ones(n), requires_grad=True)

    def forward(self, visual_feat, noun_feats_list, text_labels):
        """
        visual_feat: (B, T, D)
        noun_feats_list: dict of length 14, each [N_i, D]
        Returns: (B, T, D)
        """
        B, T, D = visual_feat.shape
        if B!=len(text_labels):
            text_labels= [text_labels[0]]*B
        
        modulated = []
        for i in range(B):
            V_i = visual_feat[i]  # (T, D)
            class_label= text_labels[i]
            E = noun_feats_list[class_label]  # (N_i, D)
            W = self.class_noun_weights[class_label]  # (N_i,)

            Q = self.query_proj(V_i)  # (T, D)
            K = self.key_proj(E)    # (N_i, D)
            V = self.value_proj(E) # (N_i, D)

            # Compute attention (N_i queries over T frames)
            attn = torch.softmax(Q @ K.T / (D ** 0.5), dim=-1)  # (T, N_i)
            weighted_attn = attn * W.unsqueeze(0)  # (T, N_i) * (1, N_i)
            weighted_attn = weighted_attn / (weighted_attn.sum(dim=-1, keepdim=True) + 1e-6)
            out = weighted_attn @ V  # (T, D)
            modulated.append(out)

        return torch.stack(modulated, dim=0)  # (B, T, D)


class CLIPVAD(nn.Module):
    def __init__(self,
                 num_class: int,
                 embed_dim: int,
                 visual_snippets: int,
                 visual_dim: int,
                 visual_head: int,
                 visual_layers: int,
                 attn_window: int,
                 prompt_prefix: int,
                 prompt_postfix: int,
                 args,
                 class_nouns,
                 device):
        super().__init__()

        self.num_class = num_class
        self.visual_snippets = visual_snippets
        self.visual_dim = visual_dim
        self.embed_dim = embed_dim
        self.attn_window = attn_window
        self.prompt_prefix = prompt_prefix
        self.prompt_postfix = prompt_postfix
        self.args=args
        self.class_nouns= class_nouns
        self.device = device

        self.temporal = Transformer(
            width=visual_dim,
            layers=visual_layers,
            heads=visual_head,
            attn_mask=self.build_attention_mask(self.attn_window)
        )

        width = int(visual_dim / 2)
        self.gc1 = GraphConvolution(visual_dim, width, residual=True)
        self.gc2 = GraphConvolution(width, width, residual=True)
        self.gc3 = GraphConvolution(visual_dim, width, residual=True)
        self.gc4 = GraphConvolution(width, width, residual=True)
        self.disAdj = DistanceAdj()
        self.linear = nn.Linear(visual_dim, visual_dim)

        if self.args.feat_encoder=='vit14':
            self.down_proj = nn.Linear(1408, embed_dim)
        self.gelu = QuickGELU()

        self.mlp1 = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(visual_dim, visual_dim * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(visual_dim * 4, visual_dim))
        ]))
        self.mlp2 = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(visual_dim, visual_dim * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(visual_dim * 4, visual_dim))
        ]))
        
        self.classifier = nn.Linear(visual_dim, 1)
        
        if self.args.fusion=='cross_attn':
            # self.cross_attn= CrossAttentionBlock(self.embed_dim, self.embed_dim)
            self.entity_modulator= EntityCrossAttention(dim=self.embed_dim, noun_dict=self.class_nouns)
            
        self.clipmodel, _ = clip.load("ViT-B/16", device)
        for clip_param in self.clipmodel.parameters():
            clip_param.requires_grad = False

        self.frame_position_embeddings = nn.Embedding(visual_snippets, self.visual_dim) #(256, 512)
        self.text_prompt_embeddings = nn.Embedding(77, self.embed_dim)

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.text_prompt_embeddings.weight, std=0.01)
        nn.init.normal_(self.frame_position_embeddings.weight, std=0.01)

    def build_attention_mask(self, attn_window):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.visual_snippets, self.visual_snippets)
        mask.fill_(float('-inf'))
        for i in range(int(self.visual_snippets / attn_window)):
            if (i + 1) * attn_window < self.visual_snippets:
                mask[i * attn_window: (i + 1) * attn_window, i * attn_window: (i + 1) * attn_window] = 0
            else:
                mask[i * attn_window: self.visual_snippets, i * attn_window: self.visual_snippets] = 0

        return mask

    def adj4(self, x, seq_len):
        soft = nn.Softmax(1)
        x2 = x.matmul(x.permute(0, 2, 1)) # B*T*T
        x_norm = torch.norm(x, p=2, dim=2, keepdim=True)  # B*T*1
        x_norm_x = x_norm.matmul(x_norm.permute(0, 2, 1))
        x2 = x2/(x_norm_x+1e-20)
        output = torch.zeros_like(x2)
        if seq_len is None:
            for i in range(x.shape[0]):
                tmp = x2[i]
                adj2 = tmp
                adj2 = F.threshold(adj2, 0.7, 0)
                adj2 = soft(adj2)
                output[i] = adj2
        else:
            for i in range(len(seq_len)):
                tmp = x2[i, :seq_len[i], :seq_len[i]]
                adj2 = tmp
                adj2 = F.threshold(adj2, 0.7, 0)
                adj2 = soft(adj2)
                output[i, :seq_len[i], :seq_len[i]] = adj2

        return output

    def encode_video(self, images, padding_mask, lengths):
        images = images.to(torch.float)
        position_ids = torch.arange(self.visual_snippets, device=self.device)
        position_ids = position_ids.unsqueeze(0).expand(images.shape[0], -1)
        frame_position_embeddings = self.frame_position_embeddings(position_ids)
        frame_position_embeddings = frame_position_embeddings.permute(1, 0, 2)
        images = images.permute(1, 0, 2) + frame_position_embeddings
        x, _ = self.temporal((images, None))
        x = x.permute(1, 0, 2)

        adj = self.adj4(x, lengths)
        disadj = self.disAdj(x.shape[0], x.shape[1])
        x1_h = self.gelu(self.gc1(x, adj))
        x2_h = self.gelu(self.gc3(x, disadj))

        x1 = self.gelu(self.gc2(x1_h, adj))
        x2 = self.gelu(self.gc4(x2_h, disadj))

        x = torch.cat((x1, x2), 2)
        x = self.linear(x)
        return x

    def build_class_noun_features(self, class_nouns):
        """
        class_nouns: dict[class_name] = list of nouns (str)
        clip_model: CLIP model from OpenAI or similar
        Returns: dict[class_name] = Tensor of shape (N, D)
        """
        noun_text_feats = {}
        for cls, nouns in class_nouns.items():
            # Create fixed prompts
            prompts = [f"a photo of a {noun}" for noun in nouns]  # Or: "a scene with {noun}"
            word_tokens = clip.tokenize(prompts).to(self.device)                         # (N, 77)
            word_embedding = self.clipmodel.encode_token(word_tokens)                   # (N, 77, D)

            # Use token IDs directly
            text_embeddings = word_embedding  # No learnable prompt tokens involved
            text_tokens = word_tokens
            with torch.no_grad():
                text_feat = self.clipmodel.encode_text(text_embeddings, text_tokens)    # (N, D)

            noun_text_feats[cls] = text_feat  # (N, D)
        return noun_text_feats

    def encode_nounprompt(self, class_nouns: dict) -> dict:
        """
        Converts class-wise noun lists into text features using prompt-tuned embeddings.

        Returns:
            class_noun_text_features: dict[class_name] = Tensor of shape (num_nouns, D)
        """
        noun_text_feats = {}
        for cls, noun_list in class_nouns.items():
            prompts = [f"a {noun}" for noun in noun_list]  # simple consistent format
            N = len(prompts)
            word_tokens = clip.tokenize(prompts).to(self.device)               # (N, 77)
            word_embedding = self.clipmodel.encode_token(word_tokens)         # (N, 77, D)
            text_embeddings = self.text_prompt_embeddings(torch.arange(77).to(self.device)).unsqueeze(0).repeat(N, 1, 1)                                     # (N, 77, D)     
            text_tokens = torch.zeros(N, 77).to(self.device)

            for i in range(N):
                # Find actual text token length (exclude padding)
                ind = torch.argmax(word_tokens[i], -1)
                text_embeddings[i, 0] = word_embedding[i, 0]  # CLS token
                text_embeddings[i, self.prompt_prefix + 1 : self.prompt_prefix + ind] = word_embedding[i, 1 : ind]
                text_embeddings[i, self.prompt_prefix + ind + self.prompt_postfix] = word_embedding[i, ind]
                text_tokens[i, self.prompt_prefix + ind + self.prompt_postfix] = word_tokens[i, ind]

            text_feat = self.clipmodel.encode_text(text_embeddings, text_tokens)  # (N, D)
            noun_text_feats[cls] = text_feat  # class_name → (num_nouns, D)

        return noun_text_feats

    def encode_textprompt(self, text):
        word_tokens = clip.tokenize(text).to(self.device)
        word_embedding = self.clipmodel.encode_token(word_tokens)
        text_embeddings = self.text_prompt_embeddings(torch.arange(77).to(self.device)).unsqueeze(0).repeat([len(text), 1, 1])
        text_tokens = torch.zeros(len(text), 77).to(self.device)

        for i in range(len(text)):
            ind = torch.argmax(word_tokens[i], -1)
            text_embeddings[i, 0] = word_embedding[i, 0]
            text_embeddings[i, self.prompt_prefix + 1: self.prompt_prefix + ind] = word_embedding[i, 1: ind]
            text_embeddings[i, self.prompt_prefix + ind + self.prompt_postfix] = word_embedding[i, ind]
            text_tokens[i, self.prompt_prefix + ind + self.prompt_postfix] = word_tokens[i, ind]
            #[CLS] [prompt prefix tokens] [actual words] [prompt postfix tokens] [EOS]

        text_features = self.clipmodel.encode_text(text_embeddings, text_tokens)
        return text_features
        
    def select_topk_bottomk(self, visual_features, classification_scores, lengths, top_k=8):
        """
        visual_features: (B, T, D)
        classification_scores: (B, T, 1)
        lengths: list of valid lengths per video
        top_k: number of frames to select
        """
        B, T, D = visual_features.shape
        half_B = B // 2
        norm_features = []
        abnorm_features = []

        for i in range(B):
            valid_len = lengths[i]
            scores = classification_scores[i, :, 0]           # (T,)
            feats = visual_features[i, :, :]                  # (T, D)

            if i < half_B:
                _, indices = torch.topk(scores, k=top_k, largest=True)
                norm_features.append(feats[indices])
            else:
                _, indices = torch.topk(scores, k=top_k, largest=True)
                abnorm_features.append(feats[indices])
        
        # Stack into (N, D)
        norm_features = torch.cat(norm_features, dim=0)     # [(B/2) * top_k, D]
        abnorm_features = torch.cat(abnorm_features, dim=0) # [(B/2) * top_k, D]
        norm_features = norm_features.view(half_B, top_k, -1)
        abnorm_features = abnorm_features.view(half_B, top_k, -1)
        return norm_features, abnorm_features

    def forward(self, visual, padding_mask, text, class_nouns, text_labels, lengths, test):
        if self.args.feat_encoder=='vit14':
            visual = self.down_proj(visual)

        visual_features = self.encode_video(visual, padding_mask, lengths) #(B, T, D)
        
        noutn_features = self.build_class_noun_features(class_nouns)
        entity_modulated_feat= self.entity_modulator(visual_features, noutn_features, text_labels)
        entity_modulated_vis = visual_features + entity_modulated_feat  # residual connection

        classificaion_scores = self.classifier(entity_modulated_vis)

        logits_attn = classificaion_scores.permute(0, 2, 1)
        visual_attn = logits_attn @ visual_features
        visual_attn = visual_attn / visual_attn.norm(dim=-1, keepdim=True)

        text_features_ori = self.encode_textprompt(text)
        text_features = text_features_ori
        text_features = text_features.unsqueeze(0)
        text_features = text_features.expand(visual.shape[0], text_features.shape[1], text_features.shape[2])

        visual_attn = visual_attn.expand(visual_attn.shape[0], text_features_ori.shape[0], visual_attn.shape[2])
        text_features = text_features + visual_attn
        text_features = text_features + self.mlp1(text_features) #(B, C, D)

        visual_features_norm = visual_features / visual_features.norm(dim=-1, keepdim=True)
        text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)
        text_features_norm = text_features_norm.permute(0, 2, 1)
        sim_matrix = visual_features_norm @ text_features_norm.type(visual_features_norm.dtype) / 0.07 #(B, T, C)
        
        if self.args.contrast and not test:
            norm_feat, abn_feat= self.select_topk_bottomk(visual_features, classificaion_scores, lengths)
            return text_features_ori, classificaion_scores, sim_matrix, norm_feat, abn_feat
        
        return text_features_ori, classificaion_scores, sim_matrix, None, None
    