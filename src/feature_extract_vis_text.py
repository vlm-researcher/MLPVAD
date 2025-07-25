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
    def __init__(self, root_path, feat_model, cap_model, vis_processors_feat, vis_processors_cap, device, video_paths, num_frames=3000, interval=16, sampling="uniform"):
        self.root_path = root_path
        self.video_paths = video_paths
        self.num_frames = num_frames
        self.sampling = sampling
        self.interval = interval
        self.device = device

        self.feat_model = feat_model
        self.cap_model = cap_model
        self.vis_processors_feat = vis_processors_feat
        self.vis_processors_cap = vis_processors_cap

    def sample_frames(self, video_path):
        """Samples frames from a video."""
        full_path = os.path.join(self.root_path, video_path.strip())
        video, _, _ = read_video(full_path, output_format="TCHW")
        print(video.shape)
        total_frames = video.shape[0]

        if total_frames == 0:
            return None

        if self.sampling == "uniform":
            indices = torch.linspace(0, total_frames - 1, steps=self.num_frames).long()
        elif self.sampling == "stepwise":
            indices = torch.arange(0, total_frames, step=self.interval)[:self.num_frames]
        else:  # Random sampling
            indices = torch.tensor(sorted(random.sample(range(total_frames), self.num_frames)))

        # Pad if needed
        # if len(indices) < self.num_frames:
        #     pad_len = self.num_frames - len(indices)
        #     indices = torch.cat([indices, indices[-1:].repeat(pad_len)])

        return video[indices]  # (num_frames, C, H, W)

    def preprocess_videos(self, dest_path_img="", dest_path_text=""):
        """Process videos and save features."""
        for video_path in self.video_paths:
            img_features = []
            text_features = []

            file_name = os.path.splitext(os.path.basename(video_path))[0]
            save_img_path = os.path.join(dest_path_img, file_name + ".npy")
            save_text_path = os.path.join(dest_path_text, file_name + ".npy")

            # Skip if already processed
            if os.path.exists(save_img_path):
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
                # processed_img_cap = self.vis_processors_cap["eval"](img_pil).unsqueeze(0).to(self.device)
                # with torch.no_grad():
                #     generated_caption = self.cap_model.generate({"image": processed_img_cap})[0]
                # print(f"Generated caption: {generated_caption}")

                # (2) Extract visual features using feature extractor
                processed_img_feat = self.vis_processors_feat["eval"](img_pil).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    raw_feats = self.feat_model.visual_encoder(processed_img_feat)
                    image_embeds = self.feat_model.ln_vision(raw_feats)

                # (3) Extract text features from generated caption
                # text_input = self.feat_model.tokenizer(
                #     generated_caption,
                #     padding="max_length",
                #     truncation=True,
                #     max_length=50,
                #     return_tensors="pt"
                # ).to(self.device)

                # with torch.no_grad():
                #     text_output = self.feat_model.Qformer.bert(
                #         input_ids=text_input.input_ids,
                #         attention_mask=text_input.attention_mask,
                #         return_dict=True,
                #     )
                #     text_feat = torch.nn.functional.normalize(
                #         self.feat_model.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1
                #     ).squeeze(0).cpu()  # (256,)

                print("Image feature: ",image_embeds.shape)
                # Append features
                img_features.append(image_embeds)
                # text_features.append(text_feat)

            # Save as .npy
            np.save(save_img_path, np.stack(img_features))     # (num_frames, 257, 1408)
            # np.save(save_text_path, np.stack(text_features))   # (num_frames, 256)
            print(f"✅ Saved: {save_img_path}")

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

    # # (2) Load captioning model (BLIP Caption)
    # caption_model, vis_processors_cap, _ = load_model_and_preprocess(
    #     name="blip_caption",
    #     model_type="large_coco",
    #     is_eval=True,
    #     device=device,
    # )

    root_path = "/home/username/Anomaly_Datasets/UCFCrimeDataset/Anomaly-Videos/"
    data_path = '/home/username/Anomaly_Datasets/UCFCrimeDataset/Anomaly_Train.txt'
    img_save_path = "/home/username/Anomaly_Datasets/UCFCrimeDataset/lavis_saved_img_features/"
    text_save_path = "/home/username/Anomaly_Datasets/UCFCrimeDataset/lavis_saved_text_features/"

    os.makedirs(img_save_path, exist_ok=True)
    os.makedirs(text_save_path, exist_ok=True)

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
        interval=20,
        sampling="stepwise"
    )

    processor.preprocess_videos(img_save_path, text_save_path)
