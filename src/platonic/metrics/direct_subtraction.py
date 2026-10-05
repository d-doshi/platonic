"""Direct subtraction of adjacent-layer representations."""
from __future__ import annotations

import numpy as np


def direct_subtraction(previous, current, scale):
    """Return (current - previous) / scale as float32 representations.

    Inputs contain the same objects and coordinates at adjacent layers.
    scale is the positive target RMS for this language and layer, computed
    from the fit and validation sets. Use scale=1.0 for unnormalized
    differences. This fixes the affine map to identity with zero bias.
    """
    previous, current = np.asarray(previous), np.asarray(current)
    if previous.ndim != 2 or previous.shape != current.shape or not previous.size:
        raise ValueError("adjacent representations must have the same nonempty matrix shape")
    if not (np.isfinite(previous).all() and np.isfinite(current).all()):
        raise ValueError("representations must be finite")
    if not np.isscalar(scale) or not np.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be a positive finite scalar")
    return ((current.astype(np.float64) - previous.astype(np.float64)) / scale).astype(np.float32)
