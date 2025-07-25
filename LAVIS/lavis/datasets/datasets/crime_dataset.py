from torch.utils.data import DataLoader, Dataset
import torch
from PIL import Image
import random
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.io import read_video
import numpy as np
import torch.nn.functional as F
import os

class VideoTextDataProcess(Dataset):
    def __init__(self, root_path, video_paths, interval=15, sampling="uniform", num_frames=8, image_size=224):
        self.root_path = root_path
        self.video_paths = video_paths
        self.sampling = sampling
        self.interval = interval
        self.num_frames = num_frames

        self.caption_dict = {
            'abuse': 'a person is being abused or mistreated in a violent situation',
            'arrest': 'law enforcement officers are arresting a person',
            'arson': 'someone is intentionally setting fire to property',
            'assault': 'an individual is attacking or physically assaulting another person',
            'burglary': 'a break-in is occurring where someone is stealing from a property',
            'explosion': 'an explosion or blast is happening, causing destruction',
            'fighting': 'a fight is taking place between two or more people',
            'roadaccidents': 'a traffic accident is occurring on the road involving vehicles',
            'robbery': 'a person is committing a robbery, stealing forcefully from another person',
            'shooting': 'a person is firing a gun or there is a gunfight',
            'shoplifting': 'someone is stealthily stealing items from a store',
            'stealing': 'a person is stealing property or goods unlawfully',
            'testing_normal_videos_anomaly': 'a normal daily life activity is happening with no anomaly',
            'training_normal_videos_anomaly': 'a normal activity is being recorded with no unusual event',
            'vandalism': 'a person is damaging or defacing public or private property'
        }

        # Define basic transforms (similar to LAVIS default processor)
        self.vis_processor = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def sample_frames(self, video_path):
        """Samples frames from a video using uniform, stepwise, or random sampling."""
        full_path = os.path.join(self.root_path, video_path.strip())
        try:
            video, _, _ = read_video(full_path, output_format="TCHW")  # (C, T, H, W)
        except Exception as e:
            print(f"Failed to read video: {full_path}, error: {e}")
            return None, None

        video = video.permute(1, 0, 2, 3)  # (T, C, H, W)
        total_frames = video.shape[0]

        if total_frames == 0:
            return None, None

        if self.sampling == "uniform":
            indices = torch.linspace(0, total_frames - 1, steps=self.num_frames).long()
        elif self.sampling == "stepwise":
            indices = torch.arange(0, total_frames, step=self.interval)[:self.num_frames]
        elif self.sampling == "random":
            if total_frames < self.num_frames:
                indices = torch.arange(0, total_frames)
            else:
                indices = torch.tensor(sorted(random.sample(range(total_frames), self.num_frames)))
        else:
            raise ValueError(f"Unknown sampling type: {self.sampling}")

        # If not enough frames, pad by repeating last frame
        if len(indices) < self.num_frames:
            pad_len = self.num_frames - len(indices)
            indices = torch.cat([indices, indices[-1:].repeat(pad_len)])

        sampled_frames = video[indices]  # (num_frames, C, H, W)
        return sampled_frames, full_path

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, index):
        video_path = self.video_paths[index]
        frames, full_path = self.sample_frames(video_path)

        if frames is None:
            # Skip broken videos
            return None

        # Apply visual processor frame-by-frame
        processed_frames = torch.stack([self.vis_processor(frame.permute(1, 2, 0).numpy()) for frame in frames])

        # Get folder name (category) from video path, e.g., 'abuse/video1.mp4' → 'abuse'
        category = os.path.dirname(video_path).split(os.sep)[-1].lower()

        # Get caption from dictionary (default fallback)
        caption = self.caption_dict.get(category, f"an event of type {category}")

        return {
            "image": processed_frames,      # shape: (num_frames, C, H, W)
            "text_input": caption,
            "video_name": os.path.basename(video_path)
        }

