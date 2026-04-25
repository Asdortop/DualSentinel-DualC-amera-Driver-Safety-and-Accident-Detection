"""
DualSentinel — train.py
=======================
Fine-tunes YOLOv8n-cls (classification variant) on an eye open/closed dataset.

DATASET SETUP (do this BEFORE running train.py):
-------------------------------------------------
1. Download from Kaggle (requires kaggle CLI or manual download):

   Dataset 1 — Eyes Dataset:
   https://www.kaggle.com/datasets/charunisa/eyes-dataset

   Dataset 2 — MRL Eye Dataset (optional, for more data):
   https://www.kaggle.com/datasets/tauilabdelilah/mrl-eye-dataset

2. Organize images into this folder structure:

   DualSentinel/
   └── dataset/
       ├── train/
       │   ├── open/       ← images of open eyes
       │   └── closed/     ← images of closed eyes
       └── val/
           ├── open/
           └── closed/

   Tip: use ~90% of images for train/, ~10% for val/

3. Then run:  python train.py

OUTPUT:
   runs/train/eye_model/weights/best.pt  ← used by driver_stream.py
"""

import os
import sys
from pathlib import Path
from ultralytics import YOLO

# ── Config ────────────────────────────────────────────────────────────────────
DATASET_DIR  = Path("dataset")
EPOCHS       = 20
IMG_SIZE     = 64       # eye crops are small — 64px is sufficient
BATCH        = 64       # RTX 3050 6GB can handle 64 at imgsz=64
WORKERS      = 4        # data loader workers
DEVICE       = "cuda"   # force GPU (RTX 3050)
MODEL_NAME   = "eye_model"
PROJECT_DIR  = "runs/train"
BASE_WEIGHTS = "yolov8n-cls.pt"   # classification variant (auto-downloaded)

# ── Validation ────────────────────────────────────────────────────────────────
def validate_dataset():
    required = [
        DATASET_DIR / "train" / "open",
        DATASET_DIR / "train" / "closed",
        DATASET_DIR / "val"   / "open",
        DATASET_DIR / "val"   / "closed",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        print("\n[ERROR] Dataset folders not found:")
        for m in missing:
            print(f"  ✗  {m}")
        print("\nPlease follow the dataset setup instructions at the top of train.py")
        sys.exit(1)

    counts = {
        "train/open"  : len(list((DATASET_DIR / "train" / "open").glob("*.*"))),
        "train/closed": len(list((DATASET_DIR / "train" / "closed").glob("*.*"))),
        "val/open"    : len(list((DATASET_DIR / "val"   / "open").glob("*.*"))),
        "val/closed"  : len(list((DATASET_DIR / "val"   / "closed").glob("*.*"))),
    }
    print("\n[INFO] Dataset summary:")
    for folder, count in counts.items():
        status = "✓" if count > 0 else "✗ EMPTY"
        print(f"  {status}  {folder}: {count} images")

    if any(c == 0 for c in counts.values()):
        print("\n[ERROR] One or more folders are empty. Add images and retry.")
        sys.exit(1)

# ── Training ──────────────────────────────────────────────────────────────────
def train():
    validate_dataset()

    print(f"\n[INFO] Loading base weights: {BASE_WEIGHTS}")
    model = YOLO(BASE_WEIGHTS)   # downloads automatically if not cached

    print(f"[INFO] Starting fine-tuning — {EPOCHS} epochs, imgsz={IMG_SIZE}, batch={BATCH}, device={DEVICE}")
    results = model.train(
        data    = str(DATASET_DIR),   # path to folder containing train/ and val/
        epochs  = EPOCHS,
        imgsz   = IMG_SIZE,
        batch   = BATCH,
        device  = DEVICE,             # GPU: RTX 3050
        workers = WORKERS,
        name    = MODEL_NAME,
        project = PROJECT_DIR,
        patience= 10,                 # early stopping
        verbose = True,
    )

    best_pt = Path(PROJECT_DIR) / MODEL_NAME / "weights" / "best.pt"
    if best_pt.exists():
        print(f"\n[✓] Training complete!")
        print(f"    Best model saved to: {best_pt}")
        print(f"\n[→] You can now run: python fusion.py")
    else:
        print(f"\n[WARNING] best.pt not found at expected path: {best_pt}")

if __name__ == "__main__":
    train()
