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

from enum import StrEnum
from typing import Any, Optional

from materials.reinforced_concrete.ndp.ndp import EN1992_1_1_2004


class EurocodeVersion(StrEnum):
    """Supported Eurocode code versions."""
    EN1992_1_1_2004 = "EN1992_1_1_2004"


class CountryCode(StrEnum):
    """Supported National Annex country codes."""
    EU = "EU"
    EU_UK = "EU_UK"


# All NDP data keyed by code version
_NDP_DATA: dict[str, dict[str, dict[str, dict[str, Any]]]] = {
    EurocodeVersion.EN1992_1_1_2004: EN1992_1_1_2004,
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

    def get(self, param: str) -> float:
        """
        Look up a single NDP value from the active code + country.

        Args:
            param: Parameter key (e.g. ``"gamma_c"``, ``"alpha_cc"``)

        Returns:
            The NDP value as a float.

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
        return float(country_data[param]["value"])

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
        code_data = _NDP_DATA.get(self._code)
        if code_data is None:
            raise KeyError(
                f"Eurocode version '{self._code}' not found. "
                f"Available: {sorted(_NDP_DATA.keys())}"
            )
        country_data = code_data.get(self._country)
        if country_data is None:
            raise KeyError(
                f"Country code '{self._country}' not found for {self._code}. "
                f"Available: {sorted(code_data.keys())}"
            )
        return country_data


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------

def get_ndp(param: str) -> float:
    """Look up an NDP value from the active code + country context."""
    return NDPRegistry.instance().get(param)


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
    "get_ndp_info",
    "set_ndp_context",
    "get_ndp_context",
]
