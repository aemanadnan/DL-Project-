"""
lits_core.py — Shared building blocks for D7 sprint experiments.

This module consolidates everything that's identical across experiments so that
each experiment notebook stays short and the differences are obvious.

Contains:
  - LiTS data loading helpers (paths, splits)
  - LiTS25DDataset (configurable: stack_size, augment, label noise)
  - UNet2D (configurable: in_channels, attention gates on/off)
  - Loss functions (Dice + CE)
  - train_model() — unified training loop with best-model checkpointing

Drop this file in the same directory as your notebooks. Import what you need.
"""

import os
import re
import glob
import time
import random
import copy

import numpy as np
import nibabel as nib
from scipy import ndimage

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# =====================================================================
# 1. DEVICE
# =====================================================================
def get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


# =====================================================================
# 2. PATHS + SPLIT  (identical to D3-D6)
# =====================================================================
def load_lits_paths(data_dir):
    """Recursively find volume*.nii* and segmentation*.nii* under data_dir."""
    all_nii = glob.glob(os.path.join(data_dir, '**', '*.nii*'), recursive=True)
    vol_files = sorted(p for p in all_nii if 'volume' in os.path.basename(p).lower())
    seg_files = sorted(p for p in all_nii if 'segmentation' in os.path.basename(p).lower())

    def case_id(path):
        m = re.search(r'(\d+)', os.path.basename(path))
        return int(m.group(1)) if m else None

    vol_map = {case_id(p): p for p in vol_files if case_id(p) is not None}
    seg_map = {case_id(p): p for p in seg_files if case_id(p) is not None}
    common = sorted(set(vol_map).intersection(seg_map))

    if not common:
        raise FileNotFoundError(f'No matched pairs under {data_dir}')

    vol_paths = [vol_map[i] for i in common]
    seg_paths = [seg_map[i] for i in common]
    print(f'Found {len(vol_paths)} matched volume-segmentation pairs')
    return vol_paths, seg_paths


def make_split(volume_paths, seg_paths, seed=42, train_frac=0.70, val_frac=0.15):
    """SAME seed=42 split as all prior experiments. Deterministic."""
    np.random.seed(seed)
    n = len(volume_paths)
    indices = np.random.permutation(n)
    train_idx = indices[:int(train_frac * n)]
    val_idx = indices[int(train_frac * n):int((train_frac + val_frac) * n)]
    test_idx = indices[int((train_frac + val_frac) * n):]

    def gather(idxs):
        return [volume_paths[i] for i in idxs], [seg_paths[i] for i in idxs]

    train_v, train_s = gather(train_idx)
    val_v, val_s = gather(val_idx)
    test_v, test_s = gather(test_idx)
    print(f'Split — Train: {len(train_v)} | Val: {len(val_v)} | Test: {len(test_v)}')
    return train_v, train_s, val_v, val_s, test_v, test_s


# =====================================================================
# 3. DATASET — supports stack_size, augment, label noise (for D7)
# =====================================================================
class LiTS25DDataset(Dataset):
    """
    Configurable 2.5D dataset. Drop-in replacement for D6/D6b dataset.

    Args:
      stack_size:       1 (pure 2D), 3 (D6), 5 (D6c), 7 ...
      augment:          False (clean) | True (D6b-style geometric+intensity)
      aug_strength:     'mild' | 'medium' | 'strong'  (only used if augment=True)
      noise_type:       None | 'missing' | 'jitter' | 'combined'  (D7)
      noise_rate:       0.0 .. 1.0 — fraction of slices receiving noise
      noise_seed:       int — for reproducibility of which slices got noisy

    The noise is applied AT INDEX TIME (stable per slice across epochs):
      - 'missing'  : randomly zero out tumor pixels in some slices
                     (simulates radiologist missed a tumor)
      - 'jitter'   : dilate/erode tumor boundaries by 1-3 px
                     (simulates sloppy contouring)
      - 'combined' : 50/50 mix of the two

    Validation datasets MUST use noise_type=None to keep evaluation honest.
    """

    def __init__(self, volume_paths, seg_paths,
                 img_size=128, hu_min=-100, hu_max=400,
                 stack_size=3, filter_background=True,
                 augment=False, aug_strength='medium',
                 noise_type=None, noise_rate=0.0, noise_seed=123):

        assert stack_size % 2 == 1, 'stack_size must be odd (1, 3, 5, 7, ...)'
        assert aug_strength in ('mild', 'medium', 'strong')
        assert noise_type in (None, 'missing', 'jitter', 'combined')

        self.img_size = img_size
        self.hu_min = hu_min
        self.hu_max = hu_max
        self.stack_size = stack_size
        self.half = stack_size // 2

        self.augment = augment
        self.aug_strength = aug_strength
        self._set_aug_params(aug_strength)

        self.noise_type = noise_type
        self.noise_rate = noise_rate

        self.slices = []
        self._vol_cache = {}
        self._seg_cache = {}

        # 1) Build the slice index
        for vol_path, seg_path in zip(volume_paths, seg_paths):
            seg = nib.load(seg_path).get_fdata().astype(np.uint8)
            n_slices = seg.shape[2]
            for s in range(n_slices):
                if filter_background and not np.any(seg[:, :, s] > 0):
                    continue
                self.slices.append((vol_path, seg_path, s, n_slices))

        # 2) Decide which slices get noise (deterministic, per-slice)
        rng = np.random.RandomState(noise_seed)
        self._noisy_mask = rng.rand(len(self.slices)) < noise_rate
        self._noisy_kinds = rng.choice(['missing', 'jitter'], size=len(self.slices))
        self._noisy_seeds = rng.randint(0, 1_000_000, size=len(self.slices))

        n_noisy = int(self._noisy_mask.sum())
        print(f'Dataset: {len(self.slices)} slices | stack={stack_size} | '
              f'augment={augment} ({aug_strength}) | noise={noise_type}@{noise_rate:.0%} '
              f'({n_noisy} slices)')

    # --- augmentation strength presets ---
    def _set_aug_params(self, strength):
        if strength == 'strong':   # original D6b
            self.aug = dict(hflip_p=0.5, rot_p=0.5, rot_deg=10.0,
                            bright_p=0.5, bright_mag=0.10,
                            contrast_p=0.5, contrast_mag=0.10)
        elif strength == 'medium':
            self.aug = dict(hflip_p=0.5, rot_p=0.4, rot_deg=7.0,
                            bright_p=0.3, bright_mag=0.07,
                            contrast_p=0.3, contrast_mag=0.07)
        else:  # mild
            self.aug = dict(hflip_p=0.5, rot_p=0.3, rot_deg=5.0,
                            bright_p=0.2, bright_mag=0.05,
                            contrast_p=0.2, contrast_mag=0.05)

    # --- volume/seg caching ---
    def _get_vol(self, path):
        if path not in self._vol_cache:
            self._vol_cache[path] = nib.load(path).get_fdata().astype(np.float32)
        return self._vol_cache[path]

    def _get_seg(self, path):
        if path not in self._seg_cache:
            self._seg_cache[path] = nib.load(path).get_fdata().astype(np.uint8)
        return self._seg_cache[path]

    def _process_ct(self, slice_2d):
        x = np.clip(slice_2d, self.hu_min, self.hu_max)
        x = (x - self.hu_min) / (self.hu_max - self.hu_min)
        h, w = x.shape
        return ndimage.zoom(x, (self.img_size / h, self.img_size / w),
                            order=1).astype(np.float32)

    def _process_seg(self, slice_2d):
        h, w = slice_2d.shape
        return ndimage.zoom(slice_2d, (self.img_size / h, self.img_size / w),
                            order=0)

    # --- LABEL NOISE (D7 core) ---
    def _corrupt_label(self, seg, kind, seed):
        """seg: (H, W) int64 with values {0, 1, 2}. Returns corrupted seg."""
        rng = np.random.RandomState(seed)
        seg = seg.copy()
        tumor_mask = (seg == 2)

        if not tumor_mask.any():
            return seg  # nothing to corrupt

        if kind == 'missing':
            # Reassign tumor pixels to liver (not background) so corruption is realistic:
            # a radiologist who misses a tumor still labels it as liver, not background.
            seg[tumor_mask] = 1
            return seg

        if kind == 'jitter':
            # Dilate or erode tumor boundary by 1-3 pixels
            iters = rng.randint(1, 4)
            if rng.rand() < 0.5:
                # dilate: liver pixels next to tumor become tumor
                dilated = ndimage.binary_dilation(tumor_mask, iterations=iters)
                grew_into = dilated & (seg == 1)
                seg[grew_into] = 2
            else:
                # erode: tumor edges become liver
                eroded = ndimage.binary_erosion(tumor_mask, iterations=iters)
                lost = tumor_mask & (~eroded)
                seg[lost] = 1
            return seg

        return seg

    # --- AUGMENTATION ---
    def _augment(self, ct_stack, seg):
        a = self.aug
        # horizontal flip
        if random.random() < a['hflip_p']:
            ct_stack = ct_stack[:, ::-1, :].copy()
            seg = seg[::-1, :].copy()
        # rotation
        if random.random() < a['rot_p']:
            ang = random.uniform(-a['rot_deg'], a['rot_deg'])
            ct_stack = np.stack([
                ndimage.rotate(ct_stack[c], ang, reshape=False, order=1, mode='reflect')
                for c in range(ct_stack.shape[0])
            ], axis=0).astype(np.float32)
            seg = ndimage.rotate(seg.astype(np.float32), ang, reshape=False,
                                 order=0, mode='constant', cval=0).astype(np.int64)
        # brightness
        if random.random() < a['bright_p']:
            shift = random.uniform(-a['bright_mag'], a['bright_mag'])
            ct_stack = np.clip(ct_stack + shift, 0.0, 1.0).astype(np.float32)
        # contrast
        if random.random() < a['contrast_p']:
            scale = 1.0 + random.uniform(-a['contrast_mag'], a['contrast_mag'])
            mean = ct_stack.mean()
            ct_stack = np.clip((ct_stack - mean) * scale + mean, 0.0, 1.0).astype(np.float32)
        return ct_stack, seg

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        vol_path, seg_path, s, n_slices = self.slices[idx]
        vol = self._get_vol(vol_path)
        seg = self._get_seg(seg_path)

        stack_idxs = [max(0, min(n_slices - 1, s + off))
                      for off in range(-self.half, self.half + 1)]
        ct_channels = [self._process_ct(vol[:, :, i]) for i in stack_idxs]
        ct_stack = np.stack(ct_channels, axis=0)

        seg_slice = self._process_seg(seg[:, :, s].astype(np.int64)).astype(np.int64)

        # Apply label noise BEFORE augmentation (so geometric augs treat it like real label)
        if self.noise_type is not None and self._noisy_mask[idx]:
            kind = self._noisy_kinds[idx] if self.noise_type == 'combined' else self.noise_type
            seg_slice = self._corrupt_label(seg_slice, kind, self._noisy_seeds[idx])

        if self.augment:
            ct_stack, seg_slice = self._augment(ct_stack, seg_slice)

        return torch.from_numpy(ct_stack), torch.from_numpy(seg_slice)


# =====================================================================
# 4. MODELS — UNet2D (with optional attention gates)
# =====================================================================
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class AttentionGate(nn.Module):
    """
    Attention gate from Oktay et al. 2018 (Attention U-Net).

    Given:
      g  : gating signal from the deeper decoder layer (smaller spatial size, more channels)
      x  : skip-connection features from the encoder (larger spatial size)
    Returns: x re-weighted by attention coefficients learned from (g, x).

    Helps the decoder focus on relevant spatial regions (e.g. tumors) and
    suppress irrelevant activations from skip connections.
    """
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, 1, bias=False),
            nn.BatchNorm2d(F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, 1, bias=False),
            nn.BatchNorm2d(F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class UNet2D(nn.Module):
    """
    Standard 2D U-Net (~31M params at base_features=64).

    Args:
      in_channels     : 1 (D3-D5), 3 (D6), 5/7 (D6c)
      num_classes     : 3 (background, liver, tumor)
      base_features   : 64
      use_attention   : if True, insert AttentionGate on each skip connection
                        before concatenation. Adds ~0.5M params.
    """

    def __init__(self, in_channels=3, num_classes=3, base_features=64,
                 use_attention=False):
        super().__init__()
        self.use_attention = use_attention
        f = base_features

        # encoder
        self.enc1 = ConvBlock(in_channels, f)
        self.enc2 = ConvBlock(f, f*2)
        self.enc3 = ConvBlock(f*2, f*4)
        self.enc4 = ConvBlock(f*4, f*8)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(f*8, f*16)

        # decoder
        self.up4 = nn.ConvTranspose2d(f*16, f*8, 2, stride=2)
        self.dec4 = ConvBlock(f*16, f*8)
        self.up3 = nn.ConvTranspose2d(f*8, f*4, 2, stride=2)
        self.dec3 = ConvBlock(f*8, f*4)
        self.up2 = nn.ConvTranspose2d(f*4, f*2, 2, stride=2)
        self.dec2 = ConvBlock(f*4, f*2)
        self.up1 = nn.ConvTranspose2d(f*2, f, 2, stride=2)
        self.dec1 = ConvBlock(f*2, f)

        self.out_conv = nn.Conv2d(f, num_classes, 1)

        if use_attention:
            self.att4 = AttentionGate(F_g=f*8, F_l=f*8, F_int=f*4)
            self.att3 = AttentionGate(F_g=f*4, F_l=f*4, F_int=f*2)
            self.att2 = AttentionGate(F_g=f*2, F_l=f*2, F_int=f)
            self.att1 = AttentionGate(F_g=f,   F_l=f,   F_int=f//2)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        u4 = self.up4(b)
        s4 = self.att4(u4, e4) if self.use_attention else e4
        d4 = self.dec4(torch.cat([u4, s4], dim=1))

        u3 = self.up3(d4)
        s3 = self.att3(u3, e3) if self.use_attention else e3
        d3 = self.dec3(torch.cat([u3, s3], dim=1))

        u2 = self.up2(d3)
        s2 = self.att2(u2, e2) if self.use_attention else e2
        d2 = self.dec2(torch.cat([u2, s2], dim=1))

        u1 = self.up1(d2)
        s1 = self.att1(u1, e1) if self.use_attention else e1
        d1 = self.dec1(torch.cat([u1, s1], dim=1))

        return self.out_conv(d1)


# =====================================================================
# 5. LOSSES + METRICS  (Dice + CE, identical to D3-D6)
# =====================================================================
def dice_loss(preds, targets, num_classes=3, smooth=1e-6):
    preds_soft = F.softmax(preds, dim=1)
    targets_oh = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    inter = (preds_soft * targets_oh).sum(dims)
    card = (preds_soft + targets_oh).sum(dims)
    dice_per_class = (2.0 * inter + smooth) / (card + smooth)
    return 1.0 - dice_per_class.mean()


_CE = nn.CrossEntropyLoss()


def combined_loss(preds, targets):
    return dice_loss(preds, targets) + _CE(preds, targets)


def compute_dice_per_class(preds, targets, num_classes=3, smooth=1e-6):
    pred_cls = preds.argmax(dim=1)
    out = []
    for c in range(num_classes):
        p = (pred_cls == c).float()
        t = (targets == c).float()
        inter = (p * t).sum()
        denom = p.sum() + t.sum()
        out.append(((2 * inter + smooth) / (denom + smooth)).item() if denom > 0 else 1.0)
    return out


# =====================================================================
# 6. UNIFIED TRAINING LOOP
# =====================================================================
def train_model(model, train_loader, val_loader, device,
                num_epochs=20, lr=1e-4, weight_decay=1e-5,
                tag='exp', save_path=None, verbose_batch_every=100,
                seed=42):
    """
    One training loop to rule them all. Returns (history_dict, best_state, best_tumor).

    history keys: train_loss, val_loss, val_liver_dice, val_tumor_dice, epoch_mins
    """
    torch.manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=4, factor=0.5)

    history = {k: [] for k in
               ['train_loss', 'val_loss', 'val_liver_dice', 'val_tumor_dice', 'epoch_mins']}
    best_tumor = 0.0
    best_state = None

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('=' * 78)
    print(f'TRAINING: {tag}')
    print(f'  Epochs={num_epochs} | LR={lr} | Device={device} | Params={n_params:,}')
    print(f'  Train batches: {len(train_loader)} | Val batches: {len(val_loader)}')
    print('=' * 78)

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        model.train()
        train_losses = []
        for bi, (ct, seg) in enumerate(train_loader):
            ct, seg = ct.to(device), seg.to(device)
            optimizer.zero_grad()
            preds = model(ct)
            loss = combined_loss(preds, seg)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            if (bi + 1) % verbose_batch_every == 0:
                print(f'  [{tag}] Ep {epoch:02d} batch {bi+1}/{len(train_loader)} '
                      f'loss={loss.item():.4f}')

        model.eval()
        val_losses, livers, tumors = [], [], []
        with torch.no_grad():
            for ct, seg in val_loader:
                ct, seg = ct.to(device), seg.to(device)
                preds = model(ct)
                val_losses.append(combined_loss(preds, seg).item())
                d = compute_dice_per_class(preds, seg)
                livers.append(d[1])
                tumors.append(d[2])

        m_train = float(np.mean(train_losses))
        m_val = float(np.mean(val_losses))
        m_liver = float(np.mean(livers))
        m_tumor = float(np.mean(tumors))
        em = (time.time() - t0) / 60

        history['train_loss'].append(m_train)
        history['val_loss'].append(m_val)
        history['val_liver_dice'].append(m_liver)
        history['val_tumor_dice'].append(m_tumor)
        history['epoch_mins'].append(em)

        scheduler.step(m_tumor)
        flag = ''
        if m_tumor > best_tumor:
            best_tumor = m_tumor
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            flag = ' *** BEST'

        print(f'[{tag}] Ep {epoch:02d}/{num_epochs} | {em:4.1f}min | '
              f'Train: {m_train:.4f} | Val: {m_val:.4f} | '
              f'Liver: {m_liver:.4f} | Tumor: {m_tumor:.4f}{flag}')

    if save_path is not None and best_state is not None:
        torch.save(best_state, save_path)
        print(f'Saved best model to {save_path}')

    return history, best_state, best_tumor
