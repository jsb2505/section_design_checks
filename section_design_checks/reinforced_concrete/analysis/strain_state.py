"""
Strain state representation for section analysis.

Supports both 1D (horizontal NA, eps varies only with y) and 2D (skewed NA,
eps varies with both x and y) strain distributions.

The 1D case is fully described by ``eps_top`` and ``eps_bottom`` (legacy API).
The 2D case additionally stores plane coefficients (a, b, c) where
``eps(x, y) = a * x + b * y + c`` with x, y relative to the section centroid.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class StrainState:
    """
    Immutable description of the strain distribution across a section.

    For backwards-compatible use, ``eps_top`` and ``eps_bottom`` are always
    populated.  They represent the strain at the vertical centroidal axis
    (``x = cx``) at ``y = y_top`` and ``y = y_bottom``, respectively.

    For the full 2D strain plane (biaxial / skewed NA), use ``strain_at(x, y)``
    or the raw plane coefficients ``plane_a, plane_b, plane_c``.

    Attributes:
        eps_top: Strain at (cx, y_top).  Compression positive.
        eps_bottom: Strain at (cx, y_bottom).  Compression positive.
        plane_a: d(eps)/dx coefficient (0 for horizontal NA).
        plane_b: d(eps)/dy coefficient.
        plane_c: Strain at centroid origin.
        is_biaxial: True when the NA is skewed (plane_a != 0).
        na_angle_deg: Neutral axis angle from horizontal (degrees), None for 1D.
    """

    eps_top: float
    eps_bottom: float
    plane_a: float = 0.0
    plane_b: float = 0.0
    plane_c: float = 0.0
    is_biaxial: bool = False
    na_angle_deg: float | None = None

    def strain_at(self, x: float, y: float) -> float:
        """Return strain at section coordinate (x, y) relative to centroid."""
        return self.plane_a * x + self.plane_b * y + self.plane_c

    def strain_field(
        self,
        x: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """
        Vectorised strain field over arrays of fibre coordinates (relative to centroid).
        """
        return self.plane_a * x + self.plane_b * y + self.plane_c

    def is_tension_at(self, x: float, y: float) -> bool:
        """True if the strain at centroid-relative (x, y) is tensile (negative)."""
        return self.strain_at(x, y) < 0

    @property
    def compression_direction(self) -> tuple[float, float]:
        """
        Unit vector pointing from tension toward compression, perpendicular to NA.

        The strain gradient ``(plane_a, plane_b)`` points in the direction of
        increasing strain (toward compression).  This property normalises it.

        For 1D (``plane_a == 0``): returns ``(0, 1)`` when ``plane_b > 0``
        (top in compression) or ``(0, -1)`` when ``plane_b < 0``.

        Returns ``(0, 0)`` for a uniform strain field (no gradient).
        """
        grad_mag = math.hypot(self.plane_a, self.plane_b)
        if grad_mag < 1e-18:
            return (0.0, 0.0)
        return (self.plane_a / grad_mag, self.plane_b / grad_mag)

    def project_along_compression(self, x: float, y: float) -> float:
        """
        Scalar projection of centroid-relative point (x, y) onto
        :attr:`compression_direction`.

        Larger values correspond to the compression side of the section.
        """
        dx, dy = self.compression_direction
        return dx * x + dy * y

    def perpendicular_extent(
        self,
        fibre_x: npt.NDArray[np.float64],
        fibre_y: npt.NDArray[np.float64],
        cx: float,
        cy: float,
    ) -> tuple[float, float]:
        """
        Range of all fibres projected along :attr:`compression_direction`.

        Returns:
            ``(min_proj, max_proj)`` where *max_proj* corresponds to the extreme
            compression fibre and *min_proj* to the extreme tension fibre.
        """
        dx, dy = self.compression_direction
        if dx == 0.0 and dy == 0.0:
            return (0.0, 0.0)
        proj = dx * (fibre_x - cx) + dy * (fibre_y - cy)
        return (float(np.min(proj)), float(np.max(proj)))

    def get_na_angle_deg(self) -> float:
        """
        Neutral axis angle from horizontal (degrees).

        Uses :attr:`na_angle_deg` if set, otherwise computes from
        ``atan2(plane_a, plane_b)`` (the angle the strain gradient makes with
        the y-axis, which equals the NA rotation from horizontal).
        """
        if self.na_angle_deg is not None:
            return self.na_angle_deg
        return math.degrees(math.atan2(self.plane_a, self.plane_b))

    def to_end_strains(self) -> tuple[float, float]:
        """Return (eps_top, eps_bottom) for legacy API compatibility."""
        return (self.eps_top, self.eps_bottom)

    @classmethod
    def from_end_strains(
        cls,
        eps_top: float,
        eps_bottom: float,
        y_top: float,
        y_bottom: float,
    ) -> StrainState:
        """
        Construct a 1D (horizontal NA) StrainState from end strains.

        Args:
            eps_top: Strain at top of section (compression positive).
            eps_bottom: Strain at bottom of section (compression positive).
            y_top: y-coordinate of top fibre (mm, relative to centroid).
            y_bottom: y-coordinate of bottom fibre (mm, relative to centroid).
        """
        h = y_top - y_bottom
        if abs(h) < 1e-18:
            return cls(eps_top=eps_top, eps_bottom=eps_bottom, plane_c=eps_top)

        plane_b = (eps_top - eps_bottom) / h
        plane_c = eps_bottom - plane_b * y_bottom
        return cls(
            eps_top=eps_top,
            eps_bottom=eps_bottom,
            plane_a=0.0,
            plane_b=plane_b,
            plane_c=plane_c,
            is_biaxial=False,
        )

    @classmethod
    def from_plane(
        cls,
        plane_a: float,
        plane_b: float,
        plane_c: float,
        y_top: float,
        y_bottom: float,
        na_angle_deg: float | None = None,
    ) -> StrainState:
        """
        Construct a (possibly biaxial) StrainState from plane coefficients.

        The plane equation is ``eps(x, y) = a * x + b * y + c`` where x, y are
        relative to the section centroid.

        Args:
            plane_a: d(eps)/dx.
            plane_b: d(eps)/dy.
            plane_c: Strain at centroid.
            y_top: y-coordinate of top fibre (relative to centroid).
            y_bottom: y-coordinate of bottom fibre (relative to centroid).
            na_angle_deg: Optional NA angle from horizontal in degrees.
        """
        # Project onto vertical centroidal axis (x = 0 in centroid-relative coords)
        eps_top = plane_b * y_top + plane_c
        eps_bottom = plane_b * y_bottom + plane_c
        return cls(
            eps_top=eps_top,
            eps_bottom=eps_bottom,
            plane_a=plane_a,
            plane_b=plane_b,
            plane_c=plane_c,
            is_biaxial=abs(plane_a) > 1e-15,
            na_angle_deg=na_angle_deg,
        )
