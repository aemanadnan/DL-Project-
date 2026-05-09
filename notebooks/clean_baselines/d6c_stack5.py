"""
D6c — 2.5D U-Net with STACK_SIZE = 5

The simplest possible extension of D6: feed 5 adjacent slices instead of 3.
Tests whether more spatial context further helps tumor segmentation.

Single-variable change vs D6: STACK_SIZE 3 -> 5. Everything else identical.

Run:  python d6c_stack5.py
Time: ~25 min on M4 Mac (slightly slower than D6 due to 5-channel input)
"""
import os, json, time
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from lits_core import (get_device, load_lits_paths, make_split,
                       LiTS25DDataset, UNet2D, train_model)


# ============================================================
# CONFIG  (edit DATA_DIR for your machine)
# ============================================================
DATA_DIR    = '/Users/aemanadnan/Desktop/DL proj/archive-2'
OUT_DIR     = './out_d6c'
TAG         = 'D6c'
STACK_SIZE  = 5
IMG_SIZE    = 128
BATCH_SIZE  = 16
NUM_EPOCHS  = 20
LR          = 1e-4
SEED        = 42

os.makedirs(OUT_DIR, exist_ok=True)
device = get_device()
print(f'Device: {device}')


# ============================================================
# DATA
# ============================================================
volume_paths, seg_paths = load_lits_paths(DATA_DIR)
train_v, train_s, val_v, val_s, test_v, test_s = make_split(volume_paths, seg_paths,
                                                            seed=SEED)

train_dataset = LiTS25DDataset(train_v, train_s, img_size=IMG_SIZE,
                               stack_size=STACK_SIZE, augment=False)
val_dataset   = LiTS25DDataset(val_v, val_s, img_size=IMG_SIZE,
                               stack_size=STACK_SIZE, augment=False)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=0, pin_memory=False)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=0, pin_memory=False)


# ============================================================
# MODEL  (in_channels = 5)
# ============================================================
model = UNet2D(in_channels=STACK_SIZE, num_classes=3, base_features=64,
               use_attention=False).to(device)


# ============================================================
# TRAIN
# ============================================================
history, best_state, best_tumor = train_model(
    model, train_loader, val_loader, device,
    num_epochs=NUM_EPOCHS, lr=LR, tag=TAG,
    save_path=os.path.join(OUT_DIR, 'best_unet_d6c.pth'),
    seed=SEED,
)


# ============================================================
# REPORT
# ============================================================
best_liver = float(np.max(history['val_liver_dice']))
best_ep    = int(np.argmax(history['val_tumor_dice']) + 1)

result = {
    'tag': TAG,
    'stack_size': STACK_SIZE,
    'img_size': IMG_SIZE,
    'num_epochs': NUM_EPOCHS,
    'best_tumor_dice': best_tumor,
    'best_liver_dice': best_liver,
    'best_epoch': best_ep,
    'history': history,
}
with open(os.path.join(OUT_DIR, 'results.json'), 'w') as f:
    json.dump(result, f, indent=2)


D3_TUMOR, D6_TUMOR = 0.5011, 0.5314
print()
print('=' * 78)
print(f'D6c RESULTS — Stack Size {STACK_SIZE}')
print('=' * 78)
print(f'Best Tumor Dice: {best_tumor:.4f}  (epoch {best_ep})')
print(f'Best Liver Dice: {best_liver:.4f}')
print(f'  vs D3 baseline: {best_tumor - D3_TUMOR:+.4f}')
print(f'  vs D6 (stack=3): {best_tumor - D6_TUMOR:+.4f}')


# ============================================================
# CURVES
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ep = range(1, NUM_EPOCHS + 1)
axes[0].plot(ep, history['train_loss'], label='Train', color='#3498db', lw=2)
axes[0].plot(ep, history['val_loss'],   label='Val',   color='#e74c3c', lw=2)
axes[0].set(xlabel='Epoch', ylabel='Loss',
            title=f'{TAG}: Train vs Val Loss')
axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(ep, history['val_liver_dice'], label='Liver', color='#2ecc71', lw=2)
axes[1].plot(ep, history['val_tumor_dice'], label='Tumor', color='#e74c3c', lw=2)
axes[1].axhline(D3_TUMOR, ls='--', color='gray',  alpha=0.5, label='D3 Tumor')
axes[1].axhline(D6_TUMOR, ls='-.', color='red',   alpha=0.5, label='D6 Tumor')
axes[1].set(xlabel='Epoch', ylabel='Dice', ylim=(0, 1),
            title=f'{TAG}: Val Dice')
axes[1].legend(loc='lower right', fontsize=8); axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'training_curves_d6c.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f'Curves saved to {OUT_DIR}/training_curves_d6c.png')
