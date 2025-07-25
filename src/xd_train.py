import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
import random

from model import CLIPVAD
from xd_test import test
from utils.dataset import XDDataset
from utils.tools import get_prompt_text, get_batch_label
import xd_option
import os
import json

def CLASM(logits, labels, lengths, device):
    instance_logits = torch.zeros(0).to(device)
    labels = labels / torch.sum(labels, dim=1, keepdim=True)
    labels = labels.to(device)

    for i in range(logits.shape[0]):
        tmp, _ = torch.topk(logits[i, 0:lengths[i]], k=int(lengths[i] / 16 + 1), largest=True, dim=0)
        instance_logits = torch.cat([instance_logits, torch.mean(tmp, 0, keepdim=True)], dim=0)

    milloss = -torch.mean(torch.sum(labels * F.log_softmax(instance_logits, dim=1), dim=1), dim=0)
    return milloss

def CLAS2(logits, labels, lengths, device):
    instance_logits = torch.zeros(0).to(device)
    labels = 1 - labels[:, 0].reshape(labels.shape[0])
    labels = labels.to(device)
    logits = torch.sigmoid(logits).reshape(logits.shape[0], logits.shape[1])

    for i in range(logits.shape[0]):
        tmp, _ = torch.topk(logits[i, 0:lengths[i]], k=int(lengths[i] / 16 + 1), largest=True)
        tmp = torch.mean(tmp).view(1)
        instance_logits = torch.cat((instance_logits, tmp))

    clsloss = F.binary_cross_entropy(instance_logits, labels)
    return clsloss

def randomized_contrastive_loss(abn_feat, norm_feat, pos_keys=2, neg_keys=4):
    """
    abn_feat: (B, T, D) - features from anomalous videos
    norm_feat: (B, T, D) - features from normal videos
    """
    B, T, D = abn_feat.shape
    device = abn_feat.device

    anchors = []
    positives = []
    negatives = []

    for b in range(B):
        # ----- Random anchor -----
        all_indices = torch.randperm(T)
        anchor_idx = all_indices[0]
        anchor = abn_feat[b, anchor_idx]  # (D,)

        # ----- Random positives (excluding anchor_idx) -----
        remaining_indices = all_indices[1:]
        pos_idxs = remaining_indices[:pos_keys]
        pos = abn_feat[b, pos_idxs]  # (pos_keys, D)

        # ----- Random negatives from normal -----
        neg_indices = torch.randperm(T)[:neg_keys]
        neg = norm_feat[b, neg_indices]  # (neg_keys, D)

        anchors.append(anchor)
        positives.append(pos)
        negatives.append(neg)

    anchor = torch.stack(anchors, dim=0)                     # (B, D)
    positives = torch.stack(positives, dim=0)                # (B, pos_keys, D)
    negatives = torch.stack(negatives, dim=0)                # (B, neg_keys, D)

    # Normalize
    anchor = F.normalize(anchor, dim=-1)
    positives = F.normalize(positives, dim=-1)
    negatives = F.normalize(negatives, dim=-1)

    # Similarity
    sim_pos = torch.einsum("bd,bkd->bk", anchor, positives) / 0.07  # (B, pos_keys)
    sim_neg = torch.einsum("bd,bkd->bk", anchor, negatives) / 0.07  # (B, neg_keys)

    logits = torch.cat([sim_pos, sim_neg], dim=1)  # (B, pos_keys + neg_keys)
    labels = torch.zeros(B, dtype=torch.long, device=device)  # anchor matches pos[0]

    return F.cross_entropy(logits, labels)


def compute_contrastive_loss(abn_feat, norm_feat, pos_keys=1, neg_keys=4):
    # topk_feats, bottomk_feats: [B, K, D]
    B, T, D= abn_feat.shape
    
    anchor = abn_feat[:, 0, :]  # positive anchor [B, D]
    positives = abn_feat[:, 1:pos_keys+1, :]  # [B, negkeys, D]
    negatives = norm_feat[:, :neg_keys, :]  # [B, poskeys, D]

    # normalize
    anchor = F.normalize(anchor, dim=-1)
    positives = F.normalize(positives, dim=-1)
    negatives = F.normalize(negatives, dim=-1)

    # compute similarities
    sim_pos = torch.einsum("bd,bkd->bk", anchor, positives) / 0.07  # [B, 2]
    sim_neg = torch.einsum("bd,bkd->bk", anchor, negatives) / 0.07  # [B, 4]

    logits = torch.cat([sim_pos, sim_neg], dim=1)  # [B, 6]
    labels = torch.zeros(B, dtype=torch.long, device=logits.device)  # anchor matched with pos[0]

    return F.cross_entropy(logits, labels)

def train(model, train_loader, test_loader, args, label_map, class_nouns, device):
    save_path=args.checkpoint_path+'My run_3/'
    os.makedirs(save_path, exist_ok=True)
    print("Save path: ",save_path)
    print("Saving configuration...")
    
    with open(os.path.join(save_path, "run_config.json") , "w") as f:
        json.dump({
            "args": vars(args),
        }, f, indent=4)
        print("run config saved!")

    model.to(device)

    gt = np.load(args.gt_path)
    gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
    gtlabels = np.load(args.gt_label_path, allow_pickle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = MultiStepLR(optimizer, args.scheduler_milestones, args.scheduler_rate)
    prompt_text = get_prompt_text(label_map)
    ap_best = 0
    epoch = 0

    if args.use_checkpoint == True:
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        ap_best = checkpoint['ap']
        print("checkpoint info:")
        print("epoch:", epoch+1, " ap:", ap_best)

    for e in range(args.max_epoch):
        model.train()
        loss_total1 = 0
        loss_total2 = 0
        for i, item in enumerate(train_loader):
            step = 0
            visual_feat, text_labels, feat_lengths = item
            visual_feat = visual_feat.to(device)
            feat_lengths = feat_lengths.to(device)
            text_labels_ori= text_labels
            text_labels = get_batch_label(text_labels_ori, prompt_text, label_map).to(device)

            text_features, classificaion_scores, sim_matrix, n_feat, a_feat = model(visual_feat, None, prompt_text, class_nouns, text_labels_ori, feat_lengths, False) 

            loss1 = CLAS2(classificaion_scores, text_labels, feat_lengths, device) 
            loss_total1 += loss1.item()

            loss2 = CLASM(sim_matrix, text_labels, feat_lengths, device)
            loss_total2 += loss2.item()

            loss3 = torch.zeros(1).to(device)
            text_feature_normal = text_features[0] / text_features[0].norm(dim=-1, keepdim=True)
            for j in range(1, text_features.shape[0]):
                text_feature_abr = text_features[j] / text_features[j].norm(dim=-1, keepdim=True)
                loss3 += torch.abs(text_feature_normal @ text_feature_abr)
            loss3 = loss3 / 6
            loss4= randomized_contrastive_loss(a_feat, n_feat) if args.contrast else torch.zeros(1).to(device)
            loss = loss1 + loss2 + loss3 + loss4 if args.contrast else loss1 + loss2 + loss3 * 1e-4

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += i * train_loader.batch_size
            if step % 4800 == 0 and step != 0:
                if args.contrast:
                    print('epoch: ', e+1, '| step: ', step, '| loss1: ', loss_total1 / (i+1), '| loss2: ', loss_total2 / (i+1), '| loss3: ',
                           loss3.item(), '| loss4: ', loss4.item())
                else:
                    print('epoch: ', e+1, '| step: ', step, '| loss1: ', loss_total1 / (i+1), '| loss2: ', loss_total2 / (i+1), '| loss3: ',
                           loss3.item()) 
                
        scheduler.step()
        AUC, AP, mAP = test(model, test_loader, args, prompt_text, class_nouns, gt, gtsegments, gtlabels, save_path, ap_best, device)

        if AP > ap_best:
            ap_best = AP 
            checkpoint = {
                'epoch': e,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'ap': ap_best}
            torch.save(checkpoint, os.path.join(save_path, "checkpoint.pth"))

        checkpoint = torch.load(os.path.join(save_path, "checkpoint.pth"))
        model.load_state_dict(checkpoint['model_state_dict'])

    checkpoint = torch.load(os.path.join(save_path, "checkpoint.pth"))
    torch.save(checkpoint['model_state_dict'], os.path.join(save_path, "model_ucf.pth"))

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    #torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = xd_option.parser.parse_args()
    setup_seed(args.seed)

    label_map = dict({'A': 'normal', 'B1': 'fighting', 'B2': 'shooting', 'B4': 'riot', 'B5': 'abuse', 'B6': 'car accident', 'G': 'explosion'})

    class_nouns = {   
        "A": ["person", "street", "car", "store", "road", "bicycle", "building", "traffic light"],

        "B5": ["person", "child", "victim", "hand", "injury", "bruise", "shout", "push", "slap"],
        
        "B4": ["police", "barricade", "protester", "person", "crowd", "fire", "smoke", "shield", "banner"],
        
        "G": ["smoke", "fire", "blast", "cloud", "car", "building", "debris", "flash"],
        
        "B1": ["person", "fist", "kick", "crowd", "face", "punch", "injury", "body"],
        
        "B6": ["car", "truck", "collision", "crash", "road", "glass", "fire", "bicycle"],
        
        "B2": ["gun", "bullet", "weapon", "person", "blood", "hand", "target", "officer"],

    }

    train_dataset = XDDataset(args, args.train_list, False, label_map, device)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    test_dataset = XDDataset(args, args.test_list, True, label_map, device)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = CLIPVAD(args.classes_num, args.embed_dim, args.snippets, args.visual_width, args.visual_head,
                     args.visual_layers, args.attn_window, args.prompt_prefix, args.prompt_postfix, args, class_nouns, device)

    train(model, train_loader, test_loader, args, label_map, class_nouns, device)