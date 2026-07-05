"""Regression tests for the hole product-of-inertia (I_xy) winding bug.

Hole ring integrals scale with the ring's signed area, and Shapely preserves the
caller's interior-ring winding. The bug subtracted I_xy with the raw (winding-
dependent) sign while area/I_xx/I_yy used abs(), so a clockwise-wound void
corrupted I_xy. The fix normalises every hole term by sign(area).
"""

import pytest

from materials.core.geometry import Point2D
from materials.reinforced_concrete.geometry import RCSection


def _rect(x0, y0, x1, y1, ccw=True):
    pts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]  # counter-clockwise
    if not ccw:
        pts = list(reversed(pts))
    return tuple(Point2D(x=float(x), y=float(y)) for x, y in pts)


OUTER = _rect(0, 0, 600, 400)  # 600 x 400 mm, centroid (300, 200)


def _section(void):
    return RCSection(outline_coords=OUTER, voids_coords=(void,), section_name="hollow")


def _ref_Ixy(outer, void):
    """Independent composite reference for I_xy about the global centroid.

    Two axis-aligned rectangles (outer minus void); each has zero I_xy about its
    own centroid, so by the parallel-axis theorem I_xy = sum(A_i * dx_i * dy_i).
    """
    def props(r):
        xs = [p.x for p in r]
        ys = [p.y for p in r]
        x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
        return (x1 - x0) * (y1 - y0), 0.5 * (x0 + x1), 0.5 * (y0 + y1)

    a_o, cxo, cyo = props(outer)
    a_v, cxv, cyv = props(void)
    a = a_o - a_v
    cx = (a_o * cxo - a_v * cxv) / a
    cy = (a_o * cyo - a_v * cyv) / a
    return a_o * (cxo - cx) * (cyo - cy) - a_v * (cxv - cx) * (cyv - cy)


class TestHoleIxyWinding:
    def test_ixy_independent_of_void_winding(self):
        """Same physical section: I_xy must not depend on the void's winding."""
        void_ccw = _rect(400, 250, 500, 350, ccw=True)
        void_cw = _rect(400, 250, 500, 350, ccw=False)
        ixy_ccw = _section(void_ccw).get_second_moment_area()[2]
        ixy_cw = _section(void_cw).get_second_moment_area()[2]
        assert ixy_ccw == pytest.approx(ixy_cw, rel=1e-9)
        # Off-centre void => I_xy is genuinely non-zero, so the test is meaningful.
        assert abs(ixy_ccw) > 1.0e6

    def test_ixy_matches_composite_reference(self):
        """I_xy magnitude and sign match an independent rectangle decomposition."""
        void = _rect(400, 250, 500, 350, ccw=True)
        ixy = _section(void).get_second_moment_area()[2]
        assert ixy == pytest.approx(_ref_Ixy(OUTER, void), rel=1e-6)

    def test_symmetric_section_zero_ixy_regardless_of_winding(self):
        """A void centred on x=300 keeps the section symmetric about the vertical
        axis: I_xy ~ 0 and is_symmetric_about_vertical_axis True for either winding.
        Before the fix a clockwise void produced spurious I_xy and broke this.
        """
        for ccw in (True, False):
            void = _rect(250, 250, 350, 350, ccw=ccw)  # centred on x=300, offset up
            sec = _section(void)
            ixy = sec.get_second_moment_area()[2]
            assert ixy == pytest.approx(0.0, abs=1.0), f"ccw={ccw}: I_xy={ixy}"
            assert sec.is_symmetric_about_vertical_axis() is True
