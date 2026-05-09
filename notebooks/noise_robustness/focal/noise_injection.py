"""
D7: Label noise injection for LiTS training masks.

All corruptions operate on integer masks where:
  0 = background
  1 = liver
  2 = tumor

Design rules:
- Corruption is applied ONLY to tumor pixels (class 2).
- Liver and background pixels are never modified directly.
- Removed tumor pixels become liver (because tumors sit inside liver).
- Dilated tumor pixels can ONLY come from liver, never from background
  (a tumor cannot "leak" out of the liver — that would be unrealistic).
- All randomness uses a seeded numpy.random.Generator so the same
  (mask, noise_type, rate, seed) always yields the same corrupted mask.

This file is imported by D7 training notebooks; it has no side-effects on import.
"""
import numpy as np
from scipy import ndimage


# -------------------------------------------------------------------
# 1. Missing tumors — randomly delete whole tumor components
# -------------------------------------------------------------------
def inject_missing_tumors(mask, rate, seed, small_bias=True):
    """
    Randomly remove a `rate` fraction of tumor connected components.

    Clinical motivation: radiologists most often miss SMALL tumors that
    are easy to overlook on a single slice. So we bias removal toward
    smaller components when small_bias=True.

    Removed tumor pixels are converted to liver (class 1), since LiTS
    tumors sit inside the liver — erasing a tumor reveals liver tissue
    underneath, not background.
    """
    rng = np.random.default_rng(seed)
    out = mask.copy()

    tumor = (mask == 2)
    if not tumor.any():
        return out

    labeled, n = ndimage.label(tumor)
    if n == 0:
        return out

    n_to_remove = int(np.round(rate * n))
    if n_to_remove == 0:
        return out
    n_to_remove = min(n_to_remove, n)

    components = np.arange(1, n + 1)

    if small_bias:
        sizes = ndimage.sum(tumor, labeled, components).astype(np.float64)
        # Inverse-size weighting: small tumors more likely to be missed
        inv = 1.0 / (sizes + 1.0)
        probs = inv / inv.sum()
        chosen = rng.choice(components, size=n_to_remove, replace=False, p=probs)
    else:
        chosen = rng.choice(components, size=n_to_remove, replace=False)

    for cid in chosen:
        out[labeled == cid] = 1  # tumor pixels -> liver
    return out


# -------------------------------------------------------------------
# 2. Boundary jitter — erode/dilate tumor edges
# -------------------------------------------------------------------
def inject_boundary_jitter(mask, rate, seed, max_kernel=2):
    """
    Apply random morphological erosion OR dilation to a `rate` fraction
    of tumor components.

    Clinical motivation: different annotators trace tumor boundaries
    differently — one is generous, another is conservative. Both
    introduce uncertainty at the boundary.

    For each chosen component:
      - Pick erode or dilate (50/50)
      - Pick kernel size in [1, max_kernel]
      - Erode: removed tumor pixels become liver
      - Dilate: only liver-adjacent pixels become tumor (never background)
    """
    rng = np.random.default_rng(seed)
    out = mask.copy()

    tumor = (mask == 2)
    if not tumor.any():
        return out

    labeled, n = ndimage.label(tumor)
    if n == 0:
        return out

    n_to_jitter = int(np.round(rate * n))
    if n_to_jitter == 0:
        return out
    n_to_jitter = min(n_to_jitter, n)

    components = np.arange(1, n + 1)
    chosen = rng.choice(components, size=n_to_jitter, replace=False)

    liver = (mask == 1)

    for cid in chosen:
        comp = (labeled == cid)
        ksize = int(rng.integers(1, max_kernel + 1))
        op = rng.choice(['erode', 'dilate'])

        # Build a square structuring element of side (2*ksize + 1)
        struct = np.ones((2 * ksize + 1,) * mask.ndim, dtype=bool)

        if op == 'erode':
            new_comp = ndimage.binary_erosion(comp, structure=struct)
            removed = comp & ~new_comp
            out[removed] = 1  # eroded tumor pixels become liver
        else:  # dilate
            new_comp = ndimage.binary_dilation(comp, structure=struct)
            # Only allow expansion into liver, NOT into background.
            # Also exclude pixels that are already tumor in the working `out`
            # (avoids double-counting if components are adjacent).
            added = new_comp & ~comp & liver & (out != 2)
            out[added] = 2

    return out


# -------------------------------------------------------------------
# 3. Combined — missing + boundary jitter
# -------------------------------------------------------------------
def inject_combined(mask, rate, seed):
    """
    Apply both noise types at the same rate.

    Order matters: we drop tumors FIRST, then jitter what's left.
    This is the most clinically realistic — annotator misses some
    tumors entirely, and the ones they do annotate have imperfect
    boundaries.

    Different sub-seeds so the two operations don't share their RNG.
    """
    seed_missing = seed
    seed_boundary = (seed * 2654435761) & 0xFFFFFFFF  # decorrelated

    out = inject_missing_tumors(mask, rate, seed_missing)
    out = inject_boundary_jitter(out, rate, seed_boundary)
    return out


# -------------------------------------------------------------------
# 4. Public dispatch
# -------------------------------------------------------------------
def inject_noise(mask, noise_type, rate, seed):
    """
    Single entry point for D7 training code.

    Args:
        mask: integer numpy array, values in {0, 1, 2}
        noise_type: one of 'missing', 'boundary', 'combined', 'none'
        rate: float in [0.0, 1.0]
        seed: int, used for reproducibility

    Returns:
        Corrupted mask, same shape and dtype as input.
    """
    if noise_type == 'none' or rate == 0.0:
        return mask.copy()
    if noise_type == 'missing':
        return inject_missing_tumors(mask, rate, seed)
    if noise_type == 'boundary':
        return inject_boundary_jitter(mask, rate, seed)
    if noise_type == 'combined':
        return inject_combined(mask, rate, seed)
    raise ValueError(f"Unknown noise_type={noise_type!r}; "
                     f"expected one of 'missing','boundary','combined','none'")


# -------------------------------------------------------------------
# 5. Diagnostic helper — quantify how much was changed
# -------------------------------------------------------------------
def noise_report(clean, noisy):
    """
    Compute how much a corruption actually changed the mask.
    Use this to sanity-check that 20% noise actually changes ~20%
    of tumor structure.
    """
    clean_tumor = (clean == 2).sum()
    noisy_tumor = (noisy == 2).sum()
    pixels_changed = (clean != noisy).sum()
    total_pixels = clean.size

    n_clean_comps = ndimage.label(clean == 2)[1]
    n_noisy_comps = ndimage.label(noisy == 2)[1]

    return {
        'clean_tumor_pixels': int(clean_tumor),
        'noisy_tumor_pixels': int(noisy_tumor),
        'tumor_pixel_delta': int(noisy_tumor) - int(clean_tumor),
        'pixels_changed': int(pixels_changed),
        'pct_pixels_changed': 100.0 * pixels_changed / total_pixels,
        'clean_n_components': int(n_clean_comps),
        'noisy_n_components': int(n_noisy_comps),
        'components_lost': int(n_clean_comps) - int(n_noisy_comps),
    }
