from math import tan
from typing import Any
import numpy as np


def cot(angle_rads: float) -> float:
    '''Simple helper to return the cotangent of an angle in radians'''
    return 1 / tan(angle_rads)


def as_float(x: Any) -> float:
    """
    Convert numpy scalars cleanly to Python float.

    For complex-step differentiation, extracts real part.
    """
    if np.iscomplexobj(x):
        return float(np.real(x))
    return float(x)  # raises if not convertible (good)