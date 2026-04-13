# Nuclei Instance Segmentation — MobileSAM + LoRA

Parameter-efficient fine-tuning of [MobileSAM](https://github.com/ChaoningZhang/MobileSAM) with [LoRA](https://arxiv.org/abs/2106.09685) for nuclei instance segmentation on the [NuInsSeg](https://arxiv.org/abs/2308.01760) dataset.

---

## Overview

This project applies Low-Rank Adaptation (LoRA) to MobileSAM's TinyViT image encoder for nuclei instance segmentation in H&E-stained histological images. The mask decoder and prompt encoder are kept fully trainable while the image encoder is frozen except for injected LoRA adapters.

**Key design decisions:**
- **Encode once, decode many**: the image embedding is computed once per image and reused for every nucleus prompt, avoiding redundant encoder passes
- **All-nuclei supervision**: every nucleus instance in each image is supervised (centroid point prompt + random background negative), rather than sampling a single nucleus per step
- **LoRA on encoder only**: rank-16 LoRA adapters on `qkv` attention layers; mask decoder and prompt encoder are fully trainable
- **Lightweight checkpoints**: only trainable weights are saved (~4.2M parameters vs ~9.8M total)

## Results

Five-fold cross-validation on the full NuInsSeg dataset (665 images, 31 tissue types):

| Fold | Dice   | AJI    | PQ     | SQ     | RQ     |
|------|--------|--------|--------|--------|--------|
| 0    | 0.8164 | 0.4060 | 0.4315 | 0.7321 | 0.5814 |
| 1    | 0.8291 | 0.4534 | 0.4554 | 0.7268 | 0.6225 |
| 2    | 0.8053 | 0.4631 | 0.4806 | 0.7316 | 0.6526 |
| 3    | 0.7902 | 0.4532 | 0.4833 | 0.7305 | 0.6589 |
| 4    | 0.8039 | 0.4566 | 0.4831 | 0.7304 | 0.6580 |
| **Mean** | **0.8090** | **0.4465** | **0.4668** | **0.7303** | **0.6347** |
| Std  | 0.0131 | 0.0214 | 0.0218 | 0.0019 | 0.0305 |

**Trainable parameters:** 4,183,344 (42.6% of total)

## Project Structure

```
├── configs/
│   ├── debug.yaml              # Local debug config (3 epochs, 20 samples)
│   └── train_a100.yaml         # Full training config (10 epochs, all data)
├── data/
│   ├── dataset.py              # NuInsSeg dataset loader + fold splits
│   └── transforms.py           # Augmentation pipeline (albumentations)
├── models/
│   ├── lora.py                 # LoRA implementation (injection, freezing, param counting)
│   └── sam_lora.py             # MobileSAM + LoRA wrapper with encode/decode split
├── utils/
│   ├── logger.py               # Logging setup (console + file)
│   ├── losses.py               # BCE + Dice loss, IoU prediction loss
│   ├── metrics.py              # Dice, AJI, Panoptic Quality (PQ/SQ/RQ)
│   └── visualization.py        # 3-panel comparison + overlay figures
├── scripts/
│   └── verify_dataset.py       # Dataset integrity checker
├── train.py                    # Per-fold training entry point
├── evaluate.py                 # Per-fold evaluation (GT box prompting)
├── cross_validate.py           # 5-fold cross-validation orchestrator
└── requirements.txt            # Python dependencies
```

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/devvansh1997/Nuclei-Instance-Segmentation.git
cd Nuclei-Instance-Segmentation
```

### 2. Create environment and install dependencies

```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

pip install -r requirements.txt
pip install git+https://github.com/ChaoningZhang/MobileSAM.git
```

### 3. Download the MobileSAM checkpoint

Download `mobile_sam.pt` from the [MobileSAM repo](https://github.com/ChaoningZhang/MobileSAM) and place it in `weights/`:

```bash
mkdir -p weights
# Download mobile_sam.pt into weights/
```

### 4. Download the NuInsSeg dataset

Download from one of:
- [Kaggle](https://www.kaggle.com/datasets/ipateam/nuinsseg)
- [Zenodo](https://zenodo.org/records/10518968)
- [GitHub](https://github.com/masih4/NuInsSeg)

### 5. Update the config

Edit `configs/train_a100.yaml` and set `data.root` to your dataset path:

```yaml
data:
  root: /path/to/your/NuInsSeg
```

Also set `model.checkpoint` to point to your `mobile_sam.pt` file.

### 6. Verify dataset integrity (optional)

```bash
python scripts/verify_dataset.py --root /path/to/your/NuInsSeg
```

## Usage

### Full 5-fold cross-validation (train + eval)

```bash
python cross_validate.py --config configs/train_a100.yaml
```

### Train a single fold

```bash
python train.py --config configs/train_a100.yaml --fold 0
```

### Evaluate a single fold (requires trained checkpoint)

```bash
python evaluate.py --config configs/train_a100.yaml --fold 0
```

### Useful flags for cross_validate.py

```bash
# Resume — skip folds that already have a best_lora.pt checkpoint
python cross_validate.py --config configs/train_a100.yaml --resume

# Run specific folds only
python cross_validate.py --config configs/train_a100.yaml --folds 0,1,2

# Skip training (evaluate existing checkpoints only)
python cross_validate.py --config configs/train_a100.yaml --skip-train

# Skip evaluation (train only)
python cross_validate.py --config configs/train_a100.yaml --skip-eval
```

### Debug run (quick sanity check)

```bash
python cross_validate.py --config configs/debug.yaml
```

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Model | MobileSAM (TinyViT encoder) |
| LoRA rank | 16 |
| LoRA alpha | 32 (scale = 2.0) |
| LoRA dropout | 0.1 |
| LoRA targets | `qkv` (image encoder attention) |
| Optimizer | AdamW (lr=1e-4, wd=1e-4) |
| LR schedule | Cosine with 3-epoch warmup |
| Epochs | 10 |
| Batch size | 4 |
| Mixed precision | bfloat16 |
| Loss | BCE + Dice + 0.1 * IoU prediction |
| Evaluation | Ground-truth bounding box prompts |

## Hardware

The full 5-fold run was trained on an NVIDIA H100 PCIe (80 GB) and completed in approximately 10 hours.

## References

- [NuInsSeg](https://arxiv.org/abs/2308.01760) — Mahbod et al., 2024
- [Segment Anything (SAM)](https://arxiv.org/abs/2304.02643) — Kirillov et al., 2023
- [MobileSAM](https://arxiv.org/abs/2306.14289) — Zhang et al., 2023
- [LoRA](https://arxiv.org/abs/2106.09685) — Hu et al., 2021
