import numpy as np
import torch
import torch.utils.data as data
import pandas as pd
import utils.tools as tools
from utils.tools import ClipProcessor
from lavis.models import load_model_and_preprocess

class UCFDataset(data.Dataset):
    def __init__(self, args, data_path, test_mode: bool, label_map: dict, device, normal: bool = False):
        self.args=args
        self.df = pd.read_csv(data_path)
        self.snippets = self.args.snippets
        self.test_mode = test_mode
        self.label_map = label_map
        
        if normal==False and test_mode==False and self.args.frame_selection=='text_grounded':
            model, vis_processor, text_processor= load_model_and_preprocess("blip2_image_text_matching", "pretrain", device=device, is_eval=True)
            self.processor = ClipProcessor(model, vis_processor, text_processor, normal=normal, device=device)
        else:
            self.processor = ClipProcessor(normal=normal, device=device)

        if normal == True and test_mode == False:
            self.df = self.df.loc[self.df['label'] == 'Normal']
            self.df = self.df.reset_index()
        elif test_mode == False:
            self.df = self.df.loc[self.df['label'] != 'Normal']
            self.df = self.df.reset_index()
        
    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        clip_feature = np.load(self.df.loc[index]['path'])
        filename= self.df.loc[index]['path'].split('/')[-1]
        clip_label = self.df.loc[index]['label']
        clip_feature= clip_feature.squeeze()

        if self.test_mode == False:
            clip_feature, clip_length = self.processor.process_feat(clip_feature, self.snippets, self.label_map[clip_label], 
                                                                    self.args.frame_selection, filename)
        else:
            clip_feature, clip_length = self.processor.process_split(clip_feature, self.snippets)

        clip_feature = torch.tensor(clip_feature)
        return clip_feature, clip_label, clip_length

class XDDataset(data.Dataset):
    def __init__(self, args: int, data_path: str, test_mode: bool, label_map: dict, device):
        self.args=args
        self.df = pd.read_csv(data_path)
        self.snippets = self.args.snippets
        self.test_mode = test_mode
        self.label_map = label_map
        self.processor = ClipProcessor(normal=True, device=device)

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        clip_feature = np.load(self.df.loc[index]['path'])
        clip_label = self.df.loc[index]['label'].split('-')[0]

        if self.test_mode == False:
            clip_feature, clip_length = self.processor.process_feat(clip_feature, self.snippets, None,
                                                                     self.args.frame_selection, None)
        else:
            clip_feature, clip_length = self.processor.process_split(clip_feature, self.snippets)

        clip_feature = torch.tensor(clip_feature)
        return clip_feature, clip_label, clip_length
    
class OnlineUCFDataset(data.Dataset):
    def __init__(self, snippets: int, file_path: str, label_map: dict, test_mode: bool = False, normal: bool = False):
        self.snippets = snippets
        self.test_mode = test_mode
        self.label_map = label_map
        self.normal = normal
        self.df = pd.read_csv(file_path)

        # Filter based on label
        if normal and not test_mode:
            self.df = self.df.loc[self.df['label'] == 'Normal'].reset_index(drop=True)
        elif not test_mode:
            self.df = self.df.loc[self.df['label'] != 'Normal'].reset_index(drop=True)

        # Precompute valid chunks
        self.entries = []
        for idx in range(len(self.df)):
            length = self.df.loc[idx]['video_len']
            video_id = idx

            for start_idx in range(0, length - self.snippets + 1):
                self.entries.append({
                    'path': self.df.loc[idx]['path'],
                    'label': self.df.loc[idx]['label'],
                    'video_id': video_id,
                    'start_idx': start_idx,
                    'video_length': length
                })

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        entry = self.entries[index]

        clip_feature = np.load(entry['path'])  # Full feature [T_total, D]
        start = entry['start_idx']
        end = start + self.snippets
        snippet_feature = clip_feature[start:end]  # Slice [T, D]

        # Convert to tensor
        snippet_feature = torch.tensor(snippet_feature, dtype=torch.float)

        # Get label mapped to int
        label = self.label_map[entry['label']]

        return {
            'video': snippet_feature,                # (T, D)
            'label': label,                           # int (0=Normal, 1=Abnormal)
            'video_id': entry['video_id'],            # video name or ID
            'frame_idx': entry['start_idx'],          # starting frame index
            'video_length': entry['video_length']     # total frames in the video
        }