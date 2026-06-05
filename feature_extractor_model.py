import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import numpy as np
from tqdm import tqdm
from PIL import Image
import cv2
import timm

# =================== CONFIG ====================
DATASET_FOLDER = r"woman_abuse_dataset"
FEATURE_SAVE_FOLDER = r"/home/shivji/Brajesh_OCIT/Brajesh_main/features"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLIP_LEN = 8
IMG_SIZE = 224

# =================== MODELS ====================

class I3DExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.video.r3d_18(pretrained=True)
        self.model.fc = nn.Identity()

    def forward(self, x):  # [B, 3, T, H, W]
        return self.model(x)  # [B, 512]

class RealTimeSformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model('vit_base_patch16_224', pretrained=True)
        self.model.eval()

    def forward(self, x):  # [B, 3, T, H, W]
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W).to(DEVICE)
        with torch.no_grad():
            x = self.model.forward_features(x)  # [B*T, Tokens, D]
            x = x.mean(dim=1)  # [B*T, D]
            x = x.view(B, T, -1)  # [B, T, 768]
        return x

class FusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.i3d = I3DExtractor().to(DEVICE)
        self.timesformer = RealTimeSformer().to(DEVICE)
        self.fuse = nn.Linear(512 + 768, 64).to(DEVICE)

    def forward(self, x):  # [B, 3, T, H, W]
        x = x.to(DEVICE)
        i3d_feat = self.i3d(x)  # [B, 512]
        tsf_feat = self.timesformer(x)  # [B, T, 768]
        B, T, _ = tsf_feat.shape
        i3d_feat_expanded = i3d_feat.unsqueeze(1).expand(B, T, -1)  # [B, T, 512]
        fused = torch.cat([i3d_feat_expanded, tsf_feat], dim=2).to(DEVICE)  # [B, T, 1280]
        return self.fuse(fused)  # [B, T, 64]

# ================ UTILS ===================

transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize([0.43216, 0.394666, 0.37645],
                [0.22803, 0.22145, 0.216989])
])

def extract_frames(video_path, num_frames=CLIP_LEN):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(1, total // num_frames)
    for i in range(num_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * interval)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        frames.append(frame)
    cap.release()
    return frames

def load_clip_tensor(frames):
    tensors = [transform(f) for f in frames]
    clip = torch.stack(tensors, dim=1)  # [3, T, H, W]
    return clip.unsqueeze(0).to(DEVICE)  # [1, 3, T, H, W]

# ================ FEATURE EXTRACTION ===================

@torch.no_grad()
def extract_features_from_dataset(dataset_folder, output_folder):
    model = FusionModel().to(DEVICE)
    model.eval()

    # Save fusion model for later testing
    os.makedirs(output_folder, exist_ok=True)
    fusion_model_path = os.path.join(output_folder, "fusion_model.pth")
    torch.save(model.state_dict(), fusion_model_path)
    print(f"✅ Fusion model saved at: {fusion_model_path}")

    # Loop through all class folders
    for class_name in os.listdir(dataset_folder):
        class_path = os.path.join(dataset_folder, class_name)
        if not os.path.isdir(class_path):
            continue

        save_class_folder = os.path.join(output_folder, class_name)
        os.makedirs(save_class_folder, exist_ok=True)

        print(f"\n🔹 Processing class: {class_name}")
        for video_file in tqdm(os.listdir(class_path)):
            if not video_file.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                continue

            video_path = os.path.join(class_path, video_file)
            frames = extract_frames(video_path)
            if len(frames) < CLIP_LEN:
                print(f"⚠️ Skipping {video_file}: Not enough frames")
                continue

            clip_tensor = load_clip_tensor(frames)
            features = model(clip_tensor).squeeze(0).cpu().numpy()  # [T, 64]

            save_path = os.path.join(save_class_folder, os.path.splitext(video_file)[0] + ".npy")
            np.save(save_path, features)

    print(f"\n✅ All features saved in: {output_folder}")

# ==================== RUN ====================
if __name__ == "__main__":
    extract_features_from_dataset(DATASET_FOLDER, FEATURE_SAVE_FOLDER)
