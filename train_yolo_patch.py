import os
import sys
import time
import torch
import torch.optim as optim
import numpy as np
from PIL import Image
from torchvision import transforms
import torchvision.utils as tvutils

# Configuration
CFG = {
    "yolo_cfg":     "yolov2/yolo.cfg", # YOLOv2 configuration
    "yolo_weights": "yolov2/yolov2.weights", # YOLOv2 weights
    "img_dir":      "data/test_images", # Directory containing images of people
    "patch_size":   300, # Patch size in pixels
    "num_epochs":   100, # Number of epochs
    "batch_size":   1, # Batch size (1 - 4GB VRAM)
    "lr":           0.03, # Learning rate
    "tv_weight":    2.5, # Patch smoothness weight
    "img_size":     416, # YOLOv2 input image size
    "output_dir":   "output_patches", # Directory to save patches
    "save_every":   10, # Save patch every N epochs
    "num_cls":      80, # Number of COCO classes
}

# Setup
os.makedirs(CFG["output_dir"], exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# Loading YOLOv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from yolov2.darknet_v2 import Darknet
from yolov2.loss import objectness_darknet

def load_yolov2(cfg_path, weights_path, device):
    print(f"Loading YOLOv2 from {weights_path}...")
    model = Darknet(cfg_path)
    model.load_weights(weights_path)
    model = model.to(device)
    model.eval()
    # Freeze weights because we are training the patch, not the model
    for param in model.parameters():
        param.requires_grad = False
    return model

# Loading images
from torch.utils.data import Dataset, DataLoader

class SimpleImageDataset(Dataset):
    def __init__(self, img_dir, img_size):
        self.img_size = img_size
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])
        extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        self.images = [
            os.path.join(img_dir, f)
            for f in os.listdir(img_dir)
            if f.lower().endswith(extensions)
        ]
        if len(self.images) == 0:
            raise ValueError(f"No images found in directory: {img_dir}\n"
                             f"Please add some JPG/PNG images with people to this folder.")
        print(f"Found {len(self.images)} images in {img_dir}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')
        return self.transform(img)

# Loss function
class TotalVariationLoss(torch.nn.Module):
    def forward(self, patch):
        tv_h = (patch[:, 1:, :] - patch[:, :-1, :]).pow(2).sum()
        tv_w = (patch[:, :, 1:] - patch[:, :, :-1]).pow(2).sum()
        return (tv_h + tv_w) / patch.numel()

# Applying patch to image
def apply_patch_center(img_batch, patch, scale=0.25):
    b, c, h, w = img_batch.shape
    patch_h = int(h * scale)
    patch_w = int(w * scale)

    # Scaling the patch
    patch_resized = torch.nn.functional.interpolate(
        patch.unsqueeze(0), size=(patch_h, patch_w), mode='bilinear', align_corners=False
    ).squeeze(0)

    # Image center
    start_h = (h - patch_h) // 2
    start_w = (w - patch_w) // 2

    img_patched = img_batch.clone()
    img_patched[:, :, start_h:start_h+patch_h, start_w:start_w+patch_w] = patch_resized

    return img_patched

# Training
def train():
    if not os.path.exists(CFG["img_dir"]):
        os.makedirs(CFG["img_dir"], exist_ok=True)
        return

    # Load model
    model = load_yolov2(CFG["yolo_cfg"], CFG["yolo_weights"], device)

    # Load dataset
    dataset = SimpleImageDataset(CFG["img_dir"], CFG["img_size"])
    dataloader = DataLoader(dataset, batch_size=CFG["batch_size"], shuffle=True, num_workers=0)

    # Initialize patch - random pixels (dimensions: 3 x patch_size x patch_size)
    patch = torch.rand(3, CFG["patch_size"], CFG["patch_size"], device=device, requires_grad=True)
    optimizer = optim.Adam([patch], lr=CFG["lr"])
    tv_loss_fn = TotalVariationLoss()

    print(f"\nStarting training: {CFG['num_epochs']} epochs, patch {CFG['patch_size']}x{CFG['patch_size']}")
    print("\n")

    for epoch in range(CFG["num_epochs"]):
        epoch_det_loss = 0.0
        epoch_tv_loss = 0.0
        epoch_total_loss = 0.0
        n_batches = 0

        for img_batch in dataloader:
            img_batch = img_batch.to(device)

            # Clamp patch to [0, 1]
            patch_clamped = patch.clamp(0, 1)

            # Apply patch to image
            img_patched = apply_patch_center(img_batch, patch_clamped, scale=0.25)

            # Forward pass through YOLOv2
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                output = model(img_patched)
                # Objectness score - target is to minimize this
                det_loss = objectness_darknet(output, num_class=CFG["num_cls"]).mean()
                # Patch smoothness
                tv_loss = tv_loss_fn(patch_clamped)
                # Total loss
                total_loss = det_loss + CFG["tv_weight"] * tv_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_det_loss += det_loss.item()
            epoch_tv_loss += tv_loss.item()
            epoch_total_loss += total_loss.item()
            n_batches += 1

        avg_det = epoch_det_loss / n_batches
        avg_tv = epoch_tv_loss / n_batches
        avg_total = epoch_total_loss / n_batches

        print(f"Epoch [{epoch+1:3d}/{CFG['num_epochs']}] "
              f"total={avg_total:.4f}  det={avg_det:.4f}  tv={avg_tv:.4f}")

        if (epoch + 1) % CFG["save_every"] == 0 or epoch == 0:
            patch_path = os.path.join(CFG["output_dir"], f"patch_epoch_{epoch+1:04d}.png")
            tvutils.save_image(patch.clamp(0, 1), patch_path)
            print(f"Saved patch: {patch_path}")

    final_path = os.path.join(CFG["output_dir"], "patch_final.png")
    tvutils.save_image(patch.clamp(0, 1), final_path)
    print(f"\nFinal patch saved: {final_path}")

if __name__ == "__main__":
    train()