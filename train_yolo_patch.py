import os
import sys
import time
import random

import torch
import torch.optim as optim
import torch.nn.functional as F

from PIL import Image
from torchvision import transforms
import torchvision.utils as tvutils
from torch.utils.data import Dataset, DataLoader

#config for local tests
CFG = {
    "yolo_cfg":     "yolov2/yolo.cfg",
    "yolo_weights": "yolov2/yolov2.weights",
    "img_dir":      "data/test_images",
    "patch_size":   400,
    "img_size":     416,
    #for now for testing
    "num_epochs":   200,
    "batch_size":   4,
    "lr":           0.01,
    "tv_weight":    4.0,
    "output_dir":   "output_patches",
    "save_every":   10,
    "num_cls":      80,
    #only subset
    "max_images":   200,
}

torch.backends.cudnn.benchmark = True

if torch.cuda.is_available():
    torch.set_float32_matmul_precision('high')

random.seed(0)
torch.manual_seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"\nDevice: {device}")

if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("WARNING: CUDA not available -> training will be slower")

os.makedirs(CFG["output_dir"], exist_ok=True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from yolov2.darknet_v2 import Darknet
from yolov2.loss import objectness_darknet

def load_yolov2(cfg_path, weights_path, device):
    print(f"\nLoading YOLOv2 from {weights_path}")

    model = Darknet(cfg_path)
    model.load_weights(weights_path)
    model = model.to(device)
    model.eval()
    # freeze model
    for param in model.parameters():
        param.requires_grad = False
    return model

class FastImageDataset(Dataset):
    def __init__(self, img_dir, img_size, max_images=None):
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])
        extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        all_images = [
            os.path.join(img_dir, f)
            for f in os.listdir(img_dir)
            if f.lower().endswith(extensions)
        ]
        if len(all_images) == 0:
            raise ValueError(f"No images found in {img_dir}")
        random.shuffle(all_images)

        if max_images is not None:
            all_images = all_images[:max_images]
        self.images = all_images
        print(f"Using {len(self.images)} images")

        self.cached = []
        print("Caching images into RAM")
        for path in self.images:
            try:
                img = Image.open(path).convert("RGB")
                tensor = self.transform(img)
                self.cached.append(tensor)
            except Exception as e:
                print(f"Failed loading {path}: {e}")
        print("Caching complete")

    def __len__(self):
        return len(self.cached)

    def __getitem__(self, idx):
        return self.cached[idx]

class TotalVariationLoss(torch.nn.Module):
    def forward(self, patch):
        tv_h = (patch[:, 1:, :] - patch[:, :-1, :]).pow(2).mean()
        tv_w = (patch[:, :, 1:] - patch[:, :, :-1]).pow(2).mean()
        return tv_h + tv_w

#applying patch
def apply_patch_random(img_batch, patch):
    b, c, h, w = img_batch.shape

    scale = random.uniform(0.20, 0.30)

    patch_h = int(h * scale)
    patch_w = int(w * scale)
    patch_resized = F.interpolate(
        patch.unsqueeze(0),
        size=(patch_h, patch_w),
        mode='bilinear',
        align_corners=False
    ).squeeze(0)

    max_h = h - patch_h
    max_w = w - patch_w
    start_h = random.randint(0, max(0, max_h))
    start_w = random.randint(0, max(0, max_w))

    patched = img_batch.clone()
    patched[:, :, start_h:start_h + patch_h,
                  start_w:start_w + patch_w] = patch_resized
    return patched


def train():
    model = load_yolov2(
        CFG["yolo_cfg"],
        CFG["yolo_weights"],
        device
    )
    dataset = FastImageDataset(
        CFG["img_dir"],
        CFG["img_size"],
        CFG["max_images"]
    )
    dataloader = DataLoader(
        dataset,
        batch_size=CFG["batch_size"],
        shuffle=True,
        num_workers=4 if os.name != "nt" else 0,
        pin_memory=(device.type == "cuda"),
        persistent_workers=False
    )
    patch = torch.rand(
        3,
        CFG["patch_size"],
        CFG["patch_size"],
        device=device,
        requires_grad=True
    )
    optimizer = optim.Adam([patch], lr=CFG["lr"])
    tv_loss_fn = TotalVariationLoss()

    print("Start training \n")

    print(f"epochs      : {CFG['num_epochs']}")
    print(f"batch_size  : {CFG['batch_size']}")
    print(f"img_size    : {CFG['img_size']}")
    print(f"patch_size  : {CFG['patch_size']}")
    print("\n\n")

    start_time = time.time()
    for epoch in range(CFG["num_epochs"]):
        epoch_det = 0.0
        epoch_tv = 0.0
        epoch_total = 0.0
        n_batches = 0
        for batch_idx, img_batch in enumerate(dataloader):
            img_batch = img_batch.to(device, non_blocking=True)
            # clamp patch
            patch_clamped = patch.clamp(0, 1)
            # apply patch
            patched_images = apply_patch_random(
                img_batch,
                patch_clamped
            )

            with torch.amp.autocast( device_type='cuda', enabled=(device.type == "cuda") ):
                output = model(patched_images)

                det_loss = objectness_darknet(
                    output,
                    num_class=CFG["num_cls"]
                ).mean()
                tv_loss = tv_loss_fn(patch_clamped)
                total_loss = det_loss + CFG["tv_weight"] * tv_loss

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            optimizer.step()

            epoch_det += det_loss.item()
            epoch_tv += tv_loss.item()
            epoch_total += total_loss.item()
            n_batches += 1

        avg_det = epoch_det / n_batches
        avg_tv = epoch_tv / n_batches
        avg_total = epoch_total / n_batches
        elapsed = time.time() - start_time
        print(
            f"Epoch [{epoch+1:03d}/{CFG['num_epochs']}] "
            f"total={avg_total:.4f} "
            f"det={avg_det:.4f} "
            f"tv={avg_tv:.4f} "
            f"time={elapsed/60:.1f}m"
        )
        if ( (epoch + 1) % CFG["save_every"] == 0 or epoch == 0 ):
            save_path = os.path.join( CFG["output_dir"], f"patch_epoch_{epoch+1:03d}.png" )

            tvutils.save_image(
                patch.clamp(0, 1),
                save_path
            )
            print(f"Saved: {save_path}")

    final_path = os.path.join( CFG["output_dir"], "patch_final.png" )
    tvutils.save_image( patch.clamp(0, 1), final_path )

    print(f"Final patch saved: {final_path}")
    
if __name__ == "__main__":
    train()