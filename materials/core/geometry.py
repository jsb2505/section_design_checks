"""
Base geometric abstractions for cross-sections.
"""

from abc import ABC, abstractmethod
from typing import Tuple
from pydantic import BaseModel, Field, ConfigDict


class BaseGeometry(BaseModel, ABC):
    """
    Abstract base class for cross-section geometry.

    Provides common interface for geometric properties.
    """

    model_config = ConfigDict(
        validate_assignment=True,
        arbitrary_types_allowed=False,
        extra="forbid",
    )

    @abstractmethod
    def get_area(self) -> float:
        """
        Calculate cross-sectional area.

        Returns:
            Area in mm²
        """
        pass

    @abstractmethod
    def get_centroid(self) -> Tuple[float, float]:
        """
        Calculate centroid coordinates.

        Returns:
            Tuple of (x, y) coordinates in mm from origin
        """
        pass

    @abstractmethod
    def get_second_moment_area(self) -> Tuple[float, float, float]:
        """
        Calculate second moments of area about centroidal axes.

        Returns:
            Tuple of (I_xx, I_yy, I_xy) in mm⁴
        """
        pass

    @abstractmethod
    def get_bounding_box(self) -> Tuple[float, float, float, float]:
        """
        Get bounding box of the geometry.

        Returns:
            Tuple of (min_x, min_y, max_x, max_y) in mm
        """
        pass


class Point2D(BaseModel):
    """2D point in mm."""

    model_config = ConfigDict(frozen=True)

    x: float = Field(..., description="X coordinate (mm)")
    y: float = Field(..., description="Y coordinate (mm)")

    def __repr__(self) -> str:
        return f"Point2D(x={self.x:.2f}, y={self.y:.2f})"

    def __str__(self) -> str:
        return f"({self.x:.2f}, {self.y:.2f})"
