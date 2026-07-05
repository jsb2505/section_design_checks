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

from contextlib import contextmanager
from enum import StrEnum
from typing import Any, Callable, Generator, Optional, Union

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
        self._country: str = CountryCode.EU  # str to allow custom annex names
        self._temp_overrides: dict[str, NDPValue] = {}  # For context manager
        self._custom_annexes: dict[str, dict[str, NDPValue]] = {}  # For runtime registration

    @classmethod
    def instance(cls) -> "NDPRegistry":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @property
    def code(self) -> EurocodeVersion:
        return self._code

    @property
    def country(self) -> str:
        """Current country code (may be a CountryCode or custom annex name)."""
        return self._country

    def set_context(
        self,
        code: Optional[EurocodeVersion] = None,
        country: Optional[str] = None,
    ) -> None:
        """
        Set the active code version and/or country code.

        Args:
            code: Eurocode version (e.g. ``EurocodeVersion.EN1992_1_1_2004``)
            country: Country code (e.g. ``CountryCode.EU_UK``) or custom annex name
        """
        if code is not None:
            self._code = EurocodeVersion(code)
        if country is not None:
            # Accept both CountryCode enum and string (for custom annexes)
            self._country = str(country)
        # Validate the combination exists
        self._get_country_data()

    def get(self, param: str) -> NDPValue:
        """
        Look up a single NDP value from the active code + country.

        Priority order:
        1. Temporary overrides (from ndp_override context manager)
        2. Country data (built-in or custom annex)

        Args:
            param: Parameter key (e.g. ``"gamma_c"``, ``"alpha_cc"``)

        Returns:
            The NDP value (can be a constant float or a callable).

        Raises:
            KeyError: If the parameter is not found.
        """
        # Check temporary overrides first (highest priority)
        if param in self._temp_overrides:
            return self._temp_overrides[param]

        # Fall back to country data
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

        Supports both built-in country codes (EU, EU_UK, EU_DE) and
        custom annexes registered via register_custom_annex().

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

        # Get country-specific overrides
        # Check custom annexes first, then built-in country codes
        if self._country in self._custom_annexes:
            country_overrides = self._custom_annexes[self._country]
        elif self._country in code_data:
            country_overrides = code_data[self._country]
        elif self._country == CountryCode.EU:
            country_overrides = {}
        else:
            raise KeyError(
                f"Country '{self._country}' not found. "
                f"Built-in: {sorted(code_data.keys())}. "
                f"Custom: {sorted(self._custom_annexes.keys())}"
            )

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
    country: Optional[str] = None,
) -> None:
    """Set the active Eurocode version and/or country code (or custom annex name)."""
    NDPRegistry.instance().set_context(code=code, country=country)


def get_ndp_context() -> tuple[EurocodeVersion, str]:
    """Return the current (code, country) context."""
    reg = NDPRegistry.instance()
    return reg.code, reg.country


# ---------------------------------------------------------------------------
# Runtime NDP customization
# ---------------------------------------------------------------------------

@contextmanager
def ndp_override(**overrides: NDPValue) -> Generator[None, None, None]:
    """
    Temporarily override NDP values within a scope.

    Any NDP parameters passed as keyword arguments will override
    the normal lookup for the duration of the context.

    Args:
        **overrides: NDP parameter names and their temporary values.
            Values can be constants (float/int) or callables.

    Example::

        # Normal lookup
        gamma_c = get_ndp("gamma_c")  # Returns 1.5

        # Temporary override
        with ndp_override(gamma_c=1.6, alpha_cc=0.9):
            gamma_c = get_ndp("gamma_c")  # Returns 1.6
            check = ShearCheck(...)  # Uses gamma_c=1.6

        # Back to normal
        gamma_c = get_ndp("gamma_c")  # Returns 1.5 again
    """
    registry = NDPRegistry.instance()
    old_overrides = registry._temp_overrides.copy()
    registry._temp_overrides.update(overrides)
    try:
        yield
    finally:
        registry._temp_overrides = old_overrides


def register_custom_annex(
    name: str,
    overrides: dict[str, NDPValue],
    *,
    validate: bool = True,
) -> None:
    """
    Register a custom national annex at runtime.

    Custom annexes inherit all parameters from the EU base, with
    the provided overrides taking precedence.

    Args:
        name: Annex identifier (e.g., "PROJECT_X", "CLIENT_STANDARDS")
        overrides: Dict of NDP parameter names to override values.
            Values can be constants (float/int) or callables.
        validate: If True (default), verify that override keys exist
            in the EU base parameters.

    Raises:
        ValueError: If validate=True and unknown parameter names are provided.

    Example::

        # Register a project-specific annex
        register_custom_annex("PROJECT_BRIDGE", {
            "gamma_c": 1.4,
            "alpha_cc": 0.95,
            "nu_shear": lambda f_ck: 0.65 * (1 - f_ck / 250),
        })

        # Switch to it
        set_ndp_context(country="PROJECT_BRIDGE")

        # All subsequent objects use these values
        concrete = ConcreteMaterial(grade="C30/37")  # Uses gamma_c=1.4
    """
    registry = NDPRegistry.instance()

    if validate:
        # Get EU base keys for the current code version
        code_data = _NDP_DATA.get(registry._code)
        if code_data is None:
            raise KeyError(f"Eurocode version '{registry._code}' not found.")
        eu_keys = set(code_data.get(CountryCode.EU, {}).keys())
        invalid = set(overrides.keys()) - eu_keys
        if invalid:
            raise ValueError(
                f"Unknown NDP parameters: {sorted(invalid)}. "
                f"Valid parameters: {sorted(eu_keys)}"
            )

    registry._custom_annexes[name] = overrides


def unregister_custom_annex(name: str) -> bool:
    """
    Remove a custom annex registration.

    Args:
        name: Annex identifier to remove.

    Returns:
        True if the annex was removed, False if it didn't exist.
    """
    registry = NDPRegistry.instance()
    if name in registry._custom_annexes:
        del registry._custom_annexes[name]
        return True
    return False


def list_custom_annexes() -> list[str]:
    """
    List all registered custom annex names.

    Returns:
        List of custom annex identifiers.
    """
    return list(NDPRegistry.instance()._custom_annexes.keys())


__all__ = [
    # Enums
    "EurocodeVersion",
    "CountryCode",
    # Registry
    "NDPRegistry",
    # Basic lookup
    "get_ndp",
    "get_ndp_callable",
    "get_ndp_info",
    # Context management
    "set_ndp_context",
    "get_ndp_context",
    # Runtime customization
    "ndp_override",
    "register_custom_annex",
    "unregister_custom_annex",
    "list_custom_annexes",
]
