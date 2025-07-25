import torch
import numpy as np
import os
from torchvision.io import read_video
from torchvision import transforms
from PIL import Image
from lavis.models import load_model_and_preprocess
import torch.nn.functional as F
import random
from collections import defaultdict

class VideoTextDataProcess():
    def __init__(self, root_path, cap_model, vis_processors_cap, device, 
                 video_paths, num_frames=3000, interval=60, sampling="uniform"):
        self.root_path = root_path
        self.video_paths = video_paths
        self.num_frames = num_frames
        self.sampling = sampling
        self.interval = interval
        self.device = device

        self.cap_model = cap_model
        self.vis_processors_cap = vis_processors_cap
        self.saved_dict=defaultdict(int)

    def sample_frames(self, video_path):
        """Samples frames from a video."""
        full_path = os.path.join(self.root_path, video_path.strip())
        video, _, _ = read_video(full_path, output_format="TCHW")
        total_frames = video.shape[0]

        if total_frames == 0:
            return None

        if self.sampling == "uniform":
            indices = torch.linspace(0, total_frames - 1, steps=self.num_frames).long()
        elif self.sampling == "stepwise":
            indices = torch.arange(0, total_frames, step=self.interval)[:self.num_frames]
        else:  # Random sampling
            indices = torch.tensor(sorted(random.sample(range(total_frames), self.num_frames)))

        return video[indices]  # (num_frames, C, H, W)

    def preprocess_videos(self, caption_path=""):
        """Process videos and save features."""
        for video_path in self.video_paths:
            captions = []

            file_name = os.path.splitext(os.path.basename(video_path))[0]
            class_name= video_path.split('/')[0]
            save_cap_path = os.path.join(caption_path, file_name + ".txt")

            # Skip if already processed
            if os.path.exists(save_cap_path) or self.saved_dict[class_name]>10:
                print(f"Skipping {file_name}, already processed.")
                continue

            frames = self.sample_frames(video_path)
            if frames is None:
                print(f"Skipped {video_path} (no frames).")
                continue

            print(f"Processing {file_name}...")

            for frame in frames:
                img_pil = transforms.ToPILImage()(frame).convert("RGB")

                # (1) Generate caption from captioning model
                processed_img_cap = self.vis_processors_cap["eval"](img_pil).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    generated_caption = self.cap_model.generate({"image": processed_img_cap})[0]
                    captions.append(generated_caption)
                # print(f"Generated caption: {generated_caption}")

            # Save as .txt
            with open(save_cap_path, 'w', encoding='utf-8') as f:
                for caption in captions:
                    f.write(caption + '\n')
            print(f"✅ Saved: {save_cap_path}")
            self.saved_dict[class_name]+=1

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # # (2) Load captioning model (BLIP Caption)
    caption_model, vis_processors_cap, _ = load_model_and_preprocess(
        name="blip_caption",
        model_type="large_coco",
        is_eval=True,
        device=device,
    )

    root_path = "/home/username/Anomaly_Datasets/UCFCrimeDataset/Anomaly-Videos/"
    data_path = '/home/username/Anomaly_Datasets/UCFCrimeDataset/Anomaly_Train.txt'
    caption_save_path = "/home/username/Anomaly_Datasets/UCFCrimeDataset/lavis_saved_caption/"

    os.makedirs(caption_save_path, exist_ok=True)

    def read_data(file_path):
        with open(file_path, 'r') as f:
            return [line.strip() for line in f.readlines()]

    train_video_paths = read_data(data_path)

    print(f"Total videos to process: {len(train_video_paths)}")

    processor = VideoTextDataProcess(
        root_path,
        caption_model,
        vis_processors_cap,
        device,
        train_video_paths,
        interval=60,
        sampling="stepwise"
    )

    processor.preprocess_videos(caption_save_path)
