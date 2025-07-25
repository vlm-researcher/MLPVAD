import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
import random
import json

from model import CLIPVAD
from ucf_test import test
from utils.dataset import UCFDataset
from utils.tools import get_prompt_text, get_batch_label
import ucf_option
import os

def CLASM(logits, labels, lengths, device):
    instance_logits = torch.zeros(0).to(device)
    labels = labels / torch.sum(labels, dim=1, keepdim=True)
    labels = labels.to(device)

    for i in range(logits.shape[0]):
        tmp, _ = torch.topk(logits[i, 0:lengths[i]], k=int(lengths[i] / 16 + 1), largest=True, dim=0)
        instance_logits = torch.cat([instance_logits, torch.mean(tmp, 0, keepdim=True)], dim=0)
        
    # Step 1: Apply log-softmax over class dimension (dim=1)
    log_probs = F.log_softmax(instance_logits, dim=1)  # shape: [B, C]
    # Step 2: Multiply with normalized label (assumes soft labels or one-hot)
    weighted_log_probs = labels * log_probs  # shape: [B, C]
    # Step 3: Sum over class dimension → gives loss per instance
    instance_losses = torch.sum(weighted_log_probs, dim=1)  # shape: [B]
    # Step 4: Take mean over batch and negate (cross-entropy loss)
    milloss = -torch.mean(instance_losses)  # scalar
    return milloss

def CLAS2(logits, labels, lengths, device):
    instance_logits = torch.zeros(0).to(device)
    labels = 1 - labels[:, 0].reshape(labels.shape[0])
    labels = labels.to(device) # Labels are 0 and 1. First B/2 are 0 Last B/2 are 1
    logits = torch.sigmoid(logits).reshape(logits.shape[0], logits.shape[1])

    for i in range(logits.shape[0]):
        tmp, _ = torch.topk(logits[i, 0:lengths[i]], k=int(lengths[i] / 16 + 1), largest=True)
        tmp = torch.mean(tmp).view(1)
        instance_logits = torch.cat([instance_logits, tmp], dim=0)

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
    
def train(model, normal_loader, anomaly_loader, testloader, args, label_map, class_nouns, device):
    save_path=args.checkpoint_path+'final_run_text_grounded/'
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

    model.train()
    for e in range(args.max_epoch):
        loss_total1 = 0
        loss_total2 = 0
        normal_iter = iter(normal_loader)
        anomaly_iter = iter(anomaly_loader)
        for i in range(min(len(normal_loader), len(anomaly_loader))):
            step = 0
            normal_features, normal_label, normal_lengths = next(normal_iter)
            anomaly_features, anomaly_label, anomaly_lengths = next(anomaly_iter)
            visual_features = torch.cat([normal_features, anomaly_features], dim=0).to(device)
            text_labels_ori = list(normal_label) + list(anomaly_label)
            feat_lengths = torch.cat([normal_lengths, anomaly_lengths], dim=0).to(device)
            text_labels = get_batch_label(text_labels_ori, prompt_text, label_map).to(device)

            text_features, classificaion_scores, sim_matrix, n_feat, a_feat  = model(visual_features, None, prompt_text, class_nouns, text_labels_ori, feat_lengths, False) 
            #loss1
            loss1 = CLAS2(classificaion_scores, text_labels, feat_lengths, device) 
            loss_total1 += loss1.item()
            #loss2
            loss2 = CLASM(sim_matrix, text_labels, feat_lengths, device)
            loss_total2 += loss2.item()
            #loss3
            loss3 = torch.zeros(1).to(device)
            text_feature_normal = text_features[0] / text_features[0].norm(dim=-1, keepdim=True)
            for j in range(1, text_features.shape[0]):
                text_feature_abr = text_features[j] / text_features[j].norm(dim=-1, keepdim=True)
                loss3 += torch.abs(text_feature_normal @ text_feature_abr)
            loss3 = loss3 / 13 * 1e-1

            #compute contrastive loss
            loss4= randomized_contrastive_loss(a_feat, n_feat) if args.contrast else torch.zeros(1).to(device)
            loss = loss1 + loss2 + loss3 + loss4 if args.contrast else loss1 + loss2 + loss3

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += i * normal_loader.batch_size * 2
            if step % 1280 == 0 and step != 0:
                if args.contrast:
                    print('epoch: ', e+1, '| step: ', step, '| loss1: ', loss_total1 / (i+1), '| loss2: ', loss_total2 / (i+1), '| loss3: ',
                           loss3.item(), '| loss4: ', loss4.item())
                else:
                    print('epoch: ', e+1, '| step: ', step, '| loss1: ', loss_total1 / (i+1), '| loss2: ', loss_total2 / (i+1), '| loss3: ',
                           loss3.item())                 
            
        AUC, AP = test(model, testloader, args, prompt_text, class_nouns, gt, gtsegments, gtlabels, save_path, ap_best, device)
        AP = AUC

        if AP > ap_best:
            ap_best = AP 
            checkpoint = {
                'epoch': e,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'ap': ap_best}
            torch.save(checkpoint, os.path.join(save_path, "checkpoint.pth"))
                
        scheduler.step()
        
        # torch.save(model.state_dict(), f"{save_path}/epoch_{e}.pth")
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
    args = ucf_option.parser.parse_args()
    setup_seed(args.seed)

    label_map = dict({'Normal': 'normal', 'Abuse': 'abuse', 'Arrest': 'arrest', 'Arson': 'arson', 'Assault': 'assault', 
                      'Burglary': 'burglary', 'Explosion': 'explosion', 'Fighting': 'fighting', 'RoadAccidents': 'roadAccidents', 
                      'Robbery': 'robbery', 'Shooting': 'shooting', 'Shoplifting': 'shoplifting', 'Stealing': 'stealing', 'Vandalism': 'vandalism'})
    class_nouns = {   
        "Normal": ["person", "street", "car", "store", "road", "bicycle", "building", "traffic light"],

        "Abuse": ["person", "child", "victim", "hand", "injury", "bruise", "shout", "push", "slap"],
        
        "Arrest": ["police", "handcuffs", "officer", "person", "uniform", "vehicle", "badge"],
        
        "Arson": ["fire", "flame", "smoke", "building", "gasoline", "lighter", "window", "burning object"],
        
        "Assault": ["person", "fist", "face", "weapon", "crowd", "injury", "blood", "punch"],
        
        "Burglary": ["thief", "mask", "window", "door", "jewelry", "cash", "safe", "bag"],
        
        "Explosion": ["smoke", "fire", "blast", "cloud", "car", "building", "debris", "flash"],
        
        "Fighting": ["person", "fist", "kick", "crowd", "face", "punch", "injury", "body"],
        
        "RoadAccidents": ["car", "truck", "collision", "crash", "road", "glass", "fire", "bicycle"],
        
        "Robbery": ["gun", "cash", "mask", "store", "weapon", "bag", "person", "money"],
        
        "Shooting": ["gun", "bullet", "weapon", "person", "blood", "hand", "target", "officer"],
        
        "Shoplifting": ["bag", "store", "item", "product", "pocket", "shelf", "customer", "camera"],
        
        "Stealing": ["wallet", "person", "pocket", "bag", "cash", "hand", "item", "store"],
        
        "Vandalism": ["spray", "paint", "wall", "glass", "building", "graffiti", "tool", "window"]
    }
    normal_dataset = UCFDataset(args, args.train_list, False, label_map, device, True)
    normal_loader = DataLoader(normal_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    anomaly_dataset = UCFDataset(args, args.train_list, False, label_map, device, False)
    anomaly_loader = DataLoader(anomaly_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    test_dataset = UCFDataset(args, args.test_list, True, label_map, device)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = CLIPVAD(args.classes_num, args.embed_dim, args.snippets, args.visual_width, args.visual_head,
                     args.visual_layers, args.attn_window, args.prompt_prefix, args.prompt_postfix, args, class_nouns, device)

    train(model, normal_loader, anomaly_loader, test_loader, args, label_map, class_nouns, device)