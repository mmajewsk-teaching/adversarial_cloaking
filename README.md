# Adversarial Cloaking — YOLOv2 Patch Attack

Recreation of  
[Making an Invisibility Cloak: Real World Adversarial Attacks on Object Detectors](https://arxiv.org/abs/1910.14667)  
Based on the original implementation: [zxwu/adv_cloak](https://github.com/zxwu/adv_cloak)

This project trains an adversarial patch that reduces YOLOv2 person detection accuracy.

---

# Setup

## 1. Clone repo

```bash
git clone <your-repo-url>
cd adversarial_cloaking
```

---

## 2. Create virtual environment

### Linux / macOS

```bash
python -m venv .venv
source .venv/bin/activate
```

### Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
```

---

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

PyTorch with CUDA is recommended:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

Official install guide:  
https://pytorch.org/get-started/locally/

---

## 4. Download YOLOv2 weights

### Linux / macOS

```bash
bash download_weights.sh
```

### Windows (Git Bash)

```powershell
& "C:\Program Files\Git\bin\bash.exe" 
```
Inside Git Bash
```bash
bash download_weights.sh
```

Weights are downloaded from:  
https://pjreddie.com/darknet/yolov2/

---

# Repository layout

```text
adversarial_cloaking/
├── train_yolo_patch.py
├── download_weights.sh
├── requirements.txt
├── yolov2/
├── patch/
├── data/test_images/
└── output_patches/
```

---

# Training

## Prepare images

Put person images into:

```text
data/test_images/
```

Supported:
- .jpg
- .jpeg
- .png
- .bmp

---

## Run training

```bash
python train_yolo_patch.py
```

Main config is inside `CFG` in `train_yolo_patch.py`.

### Important parameters

| Parameter | Default |
|---|---|
| `patch_size` | 300 |
| `num_epochs` | 100 |
| `batch_size` | 1 |
| `lr` | 0.03 |
| `tv_weight` | 2.5 |
| `img_size` | 416 |

Generated patches are saved in:

```text
output_patches/
```

---

# How it works

Training loop:
1. Load person images
2. Apply patch to image
3. Run frozen YOLOv2
4. Minimize person objectness score
5. Update patch pixels via backpropagation

---

# Real-world testing

After training:

```text
output_patches/patch_final.png
```

Print the patch (~30×30 cm recommended), attach it to clothing, and test against YOLOv2 detection.

---

# References

- Wu et al. (2019) — https://arxiv.org/abs/1910.14667
- Thys et al. (2019) — https://arxiv.org/abs/1904.08653
- Original Repository — https://github.com/zxwu/adv_cloak
- Class Repository — https://github.com/mmajewsk-teaching/adversarial_cloaking
