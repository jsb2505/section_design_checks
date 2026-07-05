from __future__ import annotations

from math import copysign, cos, inf, pi, remainder, sin
from typing import Any

import numpy as np


def cot(theta: float, *, tol: float = 1e-12, singular: str = "raise") -> float:
    """
    Cotangent of theta (radians): cot(theta) = cos(theta) / sin(theta)

    Args:
        theta: angle in radians
        tol: tolerance used to detect near-singularities
        singular: what to do when sin(theta) ~ 0
            - "raise": raise ZeroDivisionError
            - "inf": return +/- inf with the correct sign

    Returns:
        cot(theta)

    Notes:
        - Uses sin/cos directly (usually better than 1/tan near pi/2).
        - Detects singularities by |sin(theta)|, not by comparing theta to constants.
    """
    # Optional: reduce angle to a small magnitude for numerical stability
    theta = remainder(theta, 2 * pi)

    s = sin(theta)
    if abs(s) <= tol:
        if singular == "inf":
            # cot ~ cos/sin, sign follows cos and sin
            return copysign(inf, cos(theta) * s if s != 0.0 else cos(theta))
        raise ZeroDivisionError(f"cot undefined for sin(theta)≈0 (theta={theta})")

    c = cos(theta)
    if abs(c) <= tol:
        return 0.0

    return c / s


def as_float(x: Any) -> float:
    """
    Convert numpy scalars cleanly to Python float.

    For complex-step differentiation, extracts real part.
    """
    if np.iscomplexobj(x):
        return float(np.real(x))
    return float(x)  # raises if not convertible (good)
