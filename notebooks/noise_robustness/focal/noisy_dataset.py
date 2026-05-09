"""
D7: NoisyLiTSDataset — a thin wrapper around an existing PyTorch Dataset
that applies label noise on-the-fly.

Why a wrapper and not a modification of the base dataset:
    Your D3-D6b notebooks each define a Dataset (some return single slices,
    some return 2.5D stacks). Wrapping them in NoisyLiTSDataset lets every
    strategy share the EXACT same corruption code without touching their
    individual implementations. This is critical for fair comparison.

Reproducibility design:
    Noise is determined by (base_seed, sample_idx, noise_type, rate). It
    does NOT depend on epoch number or shuffling — sample 0 always gets
    the same noisy mask every epoch, every run, every strategy. This is
    exactly the standard "fixed noisy training set" setup from the
    label-noise literature.

Fairness across strategies:
    D3 at (missing, 20%) and D6 at (missing, 20%) see byte-for-byte
    identical noisy labels. This makes "robustness" measurable as
    a function of the strategy alone.
"""
import numpy as np
import torch
from torch.utils.data import Dataset

from noise_injection import inject_noise


class NoisyLiTSDataset(Dataset):
    """
    Drop-in wrapper for any Dataset that returns (image, mask) where
    mask is an integer-valued tensor or numpy array with classes
    {0=bg, 1=liver, 2=tumor}.

    Args:
        base_dataset: the underlying Dataset (your D3-D6b dataset).
        noise_type: one of 'none', 'missing', 'boundary', 'combined'.
        noise_rate: float in [0.0, 1.0].
        base_seed: int, used as a per-config seed. Different configs
                   should use different base_seeds; same config across
                   strategies must use the same base_seed.
        precompute: if True, generate every corrupted mask up-front and
                    cache in memory. Faster, and guarantees strategies
                    see literally identical bytes. Recommended unless
                    your dataset is huge.
        verbose: print a one-line corruption summary at __init__.
    """

    def __init__(
        self,
        base_dataset,
        noise_type='none',
        noise_rate=0.0,
        base_seed=2026,
        precompute=True,
        verbose=True,
    ):
        if noise_type not in {'none', 'missing', 'boundary', 'combined'}:
            raise ValueError(f"Unknown noise_type={noise_type!r}")
        if not (0.0 <= noise_rate <= 1.0):
            raise ValueError(f"noise_rate must be in [0,1], got {noise_rate}")

        self.base = base_dataset
        self.noise_type = noise_type
        self.noise_rate = noise_rate
        self.base_seed = base_seed
        self.precompute = precompute and (noise_type != 'none' and noise_rate > 0)

        self._cache = None
        if self.precompute:
            self._build_cache(verbose=verbose)
        elif verbose:
            print(f"[NoisyLiTSDataset] noise_type={noise_type} rate={noise_rate} "
                  f"base_seed={base_seed} (lazy mode, no cache)")

    # ---------------------------------------------------------------
    # Core API
    # ---------------------------------------------------------------
    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        image, mask = self.base[idx]

        # No-op fast path
        if self.noise_type == 'none' or self.noise_rate == 0.0:
            return image, mask

        if self._cache is not None:
            noisy_mask_np = self._cache[idx]
        else:
            noisy_mask_np = self._corrupt(mask, idx)

        # Match the original mask's container type (tensor vs ndarray)
        if isinstance(mask, torch.Tensor):
            noisy_mask = torch.from_numpy(noisy_mask_np).to(
                dtype=mask.dtype, device=mask.device
            )
        else:
            noisy_mask = noisy_mask_np

        return image, noisy_mask

    # ---------------------------------------------------------------
    # Internals
    # ---------------------------------------------------------------
    def _sample_seed(self, idx):
        # Spread idx through prime multiplication so consecutive samples
        # don't get correlated noise patterns.
        return (self.base_seed * 1_000_003 + idx * 7919) & 0xFFFF_FFFF

    def _corrupt(self, mask, idx):
        if isinstance(mask, torch.Tensor):
            mask_np = mask.detach().cpu().numpy()
        else:
            mask_np = np.asarray(mask)
        # If the dataset stores mask with shape (1, H, W), squeeze for
        # connected-component analysis, then unsqueeze back.
        squeeze_axis = None
        if mask_np.ndim == 3 and mask_np.shape[0] == 1:
            mask_np = mask_np[0]
            squeeze_axis = 0

        noisy = inject_noise(
            mask_np.astype(np.uint8),
            noise_type=self.noise_type,
            rate=self.noise_rate,
            seed=self._sample_seed(idx),
        )

        if squeeze_axis is not None:
            noisy = noisy[None, ...]
        return noisy

    def _build_cache(self, verbose=True):
        """Pre-generate all noisy masks. Called once at __init__."""
        n = len(self.base)
        cache = [None] * n

        # Track aggregate stats so we can print a summary
        total_clean_tumor = 0
        total_noisy_tumor = 0
        total_changed = 0

        for i in range(n):
            _, mask = self.base[i]
            if isinstance(mask, torch.Tensor):
                mask_np = mask.detach().cpu().numpy()
            else:
                mask_np = np.asarray(mask)

            squeeze_axis = None
            if mask_np.ndim == 3 and mask_np.shape[0] == 1:
                mask_np = mask_np[0]
                squeeze_axis = 0

            noisy = inject_noise(
                mask_np.astype(np.uint8),
                noise_type=self.noise_type,
                rate=self.noise_rate,
                seed=self._sample_seed(i),
            )

            if squeeze_axis is not None:
                noisy_to_store = noisy[None, ...]
            else:
                noisy_to_store = noisy

            cache[i] = noisy_to_store

            if verbose:
                total_clean_tumor += int((mask_np == 2).sum())
                total_noisy_tumor += int((noisy == 2).sum())
                total_changed += int((mask_np != noisy).sum())

        self._cache = cache

        if verbose:
            mb = sum(c.nbytes for c in cache) / 1e6
            print(f"[NoisyLiTSDataset] noise_type={self.noise_type} "
                  f"rate={self.noise_rate} base_seed={self.base_seed}")
            print(f"  cached {n} masks  ({mb:.1f} MB)")
            print(f"  tumor pixels: {total_clean_tumor:,} -> {total_noisy_tumor:,} "
                  f"(delta {total_noisy_tumor - total_clean_tumor:+,})")
            print(f"  pixels changed: {total_changed:,}")

    # ---------------------------------------------------------------
    # Convenience: print a config tag for logging / filenames
    # ---------------------------------------------------------------
    def config_tag(self):
        """e.g. 'missing_r20' or 'clean'."""
        if self.noise_type == 'none' or self.noise_rate == 0.0:
            return 'clean'
        return f'{self.noise_type}_r{int(round(self.noise_rate * 100)):02d}'
