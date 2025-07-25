import torch
import numpy as np
import os
from torchvision.io import read_video
from torchvision import transforms
from PIL import Image
from lavis.models import load_model_and_preprocess
import torch.nn.functional as F
import random

class VideoTextDataProcess():
    def __init__(self, root_path, feat_model, vis_processors_feat, device, video_paths, num_frames=3000, interval=16, sampling="uniform"):
        self.root_path = root_path
        self.video_paths = video_paths
        self.num_frames = num_frames
        self.sampling = sampling
        self.interval = interval
        self.device = device

        self.feat_model = feat_model
        self.vis_processors_feat = vis_processors_feat

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
            if total_frames>self.num_frames:
                indices = torch.linspace(0, total_frames - 1, steps=self.num_frames).long()
            else:
                indices = torch.arange(0, total_frames, step=self.interval)
        else:  # Random sampling
            indices = torch.tensor(sorted(random.sample(range(total_frames), self.num_frames)))

        # Pad if needed
        # if len(indices) < self.num_frames:
        #     pad_len = self.num_frames - len(indices)
        #     indices = torch.cat([indices, indices[-1:].repeat(pad_len)])

        return video[indices]  # (num_frames, C, H, W)

    def preprocess_videos(self, dest_path_img="", batch_size=320):
        """Process videos and save features in batches."""
        for video_path in self.video_paths:
            file_name = os.path.splitext(os.path.basename(video_path))[0]
            save_img_path = os.path.join(dest_path_img, file_name + ".npy")

            # Skip if already processed
            if os.path.exists(save_img_path):
                load_data= np.load(save_img_path)
                print("Processed data: ",load_data.shape)
                if load_data.shape[0]<3000:
                    print(f"Skipping {file_name}, already processed.")
                    continue

            frames = self.sample_frames(video_path)
            if frames is None:
                print(f"Skipped {video_path} (no frames).")
                continue

            print(f"Processing {file_name}...")

            imgs_pil = [transforms.ToPILImage()(frame).convert("RGB") for frame in frames]
            total_frames = len(imgs_pil)
            all_features = []

            # Process in mini-batches
            for start_idx in range(0, total_frames, batch_size):
                end_idx = min(start_idx + batch_size, total_frames)
                batch_imgs = imgs_pil[start_idx:end_idx]

                processed_imgs_feat = torch.stack(
                    [self.vis_processors_feat["eval"](img) for img in batch_imgs]
                ).to(self.device)

                with torch.no_grad():
                    raw_feats = self.feat_model.visual_encoder(processed_imgs_feat)
                    image_embeds = self.feat_model.ln_vision(raw_feats)
                    image_embeds = image_embeds[:, 0, :].unsqueeze(1)
                    print(f"Batch {start_idx}-{end_idx}: Image feature: {image_embeds.shape}")

                    all_features.append(image_embeds.cpu())  # Move to CPU to free GPU memory

            # Concatenate all batches
            all_features = torch.cat(all_features, dim=0)  # (num_frames, 257, 1408)

            # Save as .npy
            np.save(save_img_path, all_features.numpy())
            print(f"✅ Saved: {save_img_path}, Shape:{all_features.shape}")


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # (1) Load feature extractor model (BLIP-2)
    feat_model, vis_processors_feat, _ = load_model_and_preprocess(
        name="blip2_feature_extractor",
        model_type="pretrain",
        is_eval=True,
        device=device,
    )

    feat_model.visual_encoder = feat_model.visual_encoder.float()
    root_path = "/home/username/Anomaly_Datasets/UCFCrimeDataset/Anomaly-Videos/"
    data_path = '/home/username/Anomaly_Datasets/UCFCrimeDataset/Anomaly_Test.txt'
    img_save_path = "/home/username/Anomaly_Datasets/UCFCrimeDataset/lavis_saved_img_features/"

    os.makedirs(img_save_path, exist_ok=True)

    def read_data(file_path):
        with open(file_path, 'r') as f:
            return [line.strip() for line in f.readlines()]

    train_video_paths = read_data(data_path)
    print(f"Total videos to process: {len(train_video_paths)}")

    processor = VideoTextDataProcess(
        root_path,
        feat_model,
        vis_processors_feat,
        device,
        train_video_paths,
        interval=16,
        sampling="stepwise"
    )

    processor.preprocess_videos(img_save_path)
