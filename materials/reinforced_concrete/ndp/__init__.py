"""
Nationally Determined Parameters (NDP) registry for Eurocode 2.

Provides a centralised registry for NDP values that vary by country National Annex.
Classes use ``get_ndp()`` for their default values so that changing the active
country code automatically updates all subsequent object defaults.

Usage::

    from materials.reinforced_concrete.ndp import set_ndp_context, get_ndp, CountryCode

    # Switch to UK National Annex
    set_ndp_context(country=CountryCode.EU_UK)

    # All new objects will now use UK NDP defaults
    concrete = ConcreteMaterial(grade="C30/37")  # alpha_cc defaults to 1.0
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Callable, Optional, Union

from materials.reinforced_concrete.ndp.ndp import (
    EN1992_1_1_2004,
    EN1992_2_2005,
    _NDP_METADATA,
)

# Type alias for NDP values (can be constants or callables)
NDPValue = Union[float, int, Callable[..., float]]


class EurocodeVersion(StrEnum):
    """Supported Eurocode code versions."""
    EN1992_1_1_2004 = "EN1992_1_1_2004"
    EN1992_2_2005 = "EN1992_2_2005"


class CountryCode(StrEnum):
    """Supported National Annex country codes."""
    EU = "EU"
    EU_UK = "EU_UK"
    EU_DE = "EU_DE"  # German


# All NDP data keyed by code version
# Structure: {code_version: {country_code: {param: value}}}
_NDP_DATA: dict[str, dict[str, dict[str, NDPValue]]] = {
    EurocodeVersion.EN1992_1_1_2004: EN1992_1_1_2004,
    EurocodeVersion.EN1992_2_2005: EN1992_2_2005,
}


class NDPRegistry:
    """
    Singleton registry holding the active Eurocode version and country code.

    All NDP lookups go through this registry so that a single
    ``set_context()`` call changes defaults for every subsequently
    created object.
    """

    _instance: Optional["NDPRegistry"] = None

    def __init__(self) -> None:
        self._code: EurocodeVersion = EurocodeVersion.EN1992_1_1_2004
        self._country: CountryCode = CountryCode.EU

    @classmethod
    def instance(cls) -> "NDPRegistry":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @property
    def code(self) -> EurocodeVersion:
        return self._code

    @property
    def country(self) -> CountryCode:
        return self._country

    def set_context(
        self,
        code: Optional[EurocodeVersion] = None,
        country: Optional[CountryCode] = None,
    ) -> None:
        """
        Set the active code version and/or country code.

        Args:
            code: Eurocode version (e.g. ``EurocodeVersion.EN1992_1_1_2004``)
            country: Country code (e.g. ``CountryCode.EU_UK``)
        """
        if code is not None:
            self._code = EurocodeVersion(code)
        if country is not None:
            self._country = CountryCode(country)
        # Validate the combination exists
        self._get_country_data()

    def get(self, param: str) -> NDPValue:
        """
        Look up a single NDP value from the active code + country.

        Args:
            param: Parameter key (e.g. ``"gamma_c"``, ``"alpha_cc"``)

        Returns:
            The NDP value (can be a constant float or a callable).

        Raises:
            KeyError: If the parameter is not found.
        """
        country_data = self._get_country_data()
        if param not in country_data:
            raise KeyError(
                f"NDP parameter '{param}' not found for "
                f"{self._code} / {self._country}. "
                f"Available: {sorted(country_data.keys())}"
            )
        return country_data[param]["value"]

    def get_info(self, param: str) -> dict[str, Any]:
        """
        Return the full NDP entry (value, description, ref) for a parameter.
        """
        country_data = self._get_country_data()
        return dict(country_data[param])

    def list_params(self) -> list[str]:
        """Return all available NDP parameter names for the active context."""
        return sorted(self._get_country_data().keys())

    def _get_country_data(self) -> dict[str, dict[str, Any]]:
        """
        Build merged NDP data: EU base + country overrides + metadata.

        Returns dict of {param: {"value": ..., "description": ..., "ref": ...}}
        """
        code_data = _NDP_DATA.get(self._code)
        if code_data is None:
            raise KeyError(
                f"Eurocode version '{self._code}' not found. "
                f"Available: {sorted(_NDP_DATA.keys())}"
            )

        # Get EU base values (must exist)
        eu_base = code_data.get(CountryCode.EU)
        if eu_base is None:
            raise KeyError(
                f"EU base data not found for {self._code}. "
                f"Available country codes: {sorted(code_data.keys())}"
            )

        # Get country-specific overrides (may be empty if requesting EU)
        country_overrides = code_data.get(self._country, {})

        # Merge: start with EU base, override with country-specific values
        merged: dict[str, dict[str, Any]] = {}
        for param, base_value in eu_base.items():
            # Use country override if it exists, otherwise use EU base
            value = country_overrides.get(param, base_value)

            # Combine value with metadata
            metadata = _NDP_METADATA.get(param, {})
            merged[param] = {
                "value": value,
                "description": metadata.get("description", ""),
                "ref": metadata.get("ref", ""),
            }

        return merged


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------

def get_ndp(param: str) -> NDPValue:
    """
    Look up raw NDP value from the active code + country context.

    Returns either a constant (float/int) or a callable (lambda function).
    For uniform interface, prefer using get_ndp_callable() instead.
    """
    return NDPRegistry.instance().get(param)


def get_ndp_callable(param: str) -> Callable[..., float]:
    """
    Get NDP as a callable, wrapping constants if needed.

    This provides a uniform interface where both constant and formula-based
    NDPs can be called the same way.

    Args:
        param: Parameter key (e.g. "nu_shear", "alpha_cw")

    Returns:
        A callable that computes the NDP value. For constants, returns a
        function that ignores arguments and returns the constant.

    Example:
        nu_func = get_ndp_callable("nu_shear")
        nu = nu_func(f_ck=35)  # works for both constants and formulas
    """
    value = get_ndp(param)

    if callable(value):
        return value

    # Wrap constant in a named function for better tracebacks
    const = float(value)

    def _const_fn(*_args: Any, **_kwargs: Any) -> float:
        return const

    return _const_fn


def get_ndp_info(param: str) -> dict[str, Any]:
    """Return the full NDP entry for a parameter."""
    return NDPRegistry.instance().get_info(param)


def set_ndp_context(
    code: Optional[EurocodeVersion] = None,
    country: Optional[CountryCode] = None,
) -> None:
    """Set the active Eurocode version and/or country code."""
    NDPRegistry.instance().set_context(code=code, country=country)


def get_ndp_context() -> tuple[EurocodeVersion, CountryCode]:
    """Return the current (code, country) context."""
    reg = NDPRegistry.instance()
    return reg.code, reg.country


__all__ = [
    "EurocodeVersion",
    "CountryCode",
    "NDPRegistry",
    "get_ndp",
    "get_ndp_callable",
    "get_ndp_info",
    "set_ndp_context",
    "get_ndp_context",
]
