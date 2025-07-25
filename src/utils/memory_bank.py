import torch

class MemoryBankManager:
    def __init__(self, max_memory_size=100, top_k=5, selection_strategy="top_bottom"):
        self.memory_bank = dict()  # {video_id: tensor of (M, D)}
        self.max_memory_size = max_memory_size
        self.top_k = top_k
        self.selection_strategy = selection_strategy

    def add(self, video_id, features, scores, label):
        features = features.detach()   # Always store features without gradients
        scores = scores.detach()

        # Select features based on label
        if self.selection_strategy == "top_bottom":
            selected_feats = self.select_top_bottom(features, scores, label)
        else:
            raise NotImplementedError(f"Selection strategy {self.selection_strategy} not implemented.")

        # Initialize or Append
        if video_id not in self.memory_bank:
            self.memory_bank[video_id] = selected_feats
        else:
            self.memory_bank[video_id] = torch.cat([self.memory_bank[video_id], selected_feats], dim=0)

        # Prune if needed
        if self.memory_bank[video_id].size(0) > self.max_memory_size:
            self.memory_bank[video_id] = self.prune(self.memory_bank[video_id])

    def select_top_bottom(self, features, scores, label):
        if label == 1:
            _, indices = torch.topk(scores, k=min(self.top_k, scores.size(0)), largest=True)
        else:
            _, indices = torch.topk(scores, k=min(self.top_k, scores.size(0)), largest=False)

        selected_feats = features[indices]  # (K, D)
        return selected_feats

    def fuse(self, video_id, current_features):
        if video_id in self.memory_bank and self.memory_bank[video_id] is not None:
            memory_feats = self.memory_bank[video_id]
            fused = torch.cat([memory_feats, current_features], dim=0)
        else:
            fused = current_features
        return fused

    def reset(self, video_id):
        if video_id in self.memory_bank:
            del self.memory_bank[video_id]

    def detach_all(self):

        for vid in self.memory_bank:
            self.memory_bank[vid] = self.memory_bank[vid].detach()
