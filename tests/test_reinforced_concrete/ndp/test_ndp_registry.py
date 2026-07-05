"""
Tests for NDP registry/context helpers in reinforced_concrete.ndp.__init__.
"""

from __future__ import annotations

import pytest

import materials.reinforced_concrete.ndp as ndp_mod
from materials.reinforced_concrete.ndp import (
    CountryCode,
    EurocodeVersion,
    NDPRegistry,
    get_ndp_context,
    get_ndp,
    get_ndp_callable,
    get_ndp_info,
    list_custom_annexes,
    ndp_override,
    register_custom_annex,
    set_ndp_context,
    unregister_custom_annex,
)


@pytest.fixture
def _restore_registry_state():
    reg = NDPRegistry.instance()
    old_code = reg._code
    old_country = reg._country
    old_temp = reg._temp_overrides.copy()
    old_custom = reg._custom_annexes.copy()
    try:
        yield reg
    finally:
        reg._code = old_code
        reg._country = old_country
        reg._temp_overrides = old_temp
        reg._custom_annexes = old_custom


class TestNDPRegistryCore:
    """Tests for TestNDPRegistryCore."""
    def test_context_accessors(self, _restore_registry_state):
        """Test context accessors."""
        reg = _restore_registry_state
        set_ndp_context(code=EurocodeVersion.EN1992_2_2005, country=CountryCode.EU_DE)
        assert reg.code == EurocodeVersion.EN1992_2_2005
        assert reg.country == "EU_DE"
        assert get_ndp_context() == (EurocodeVersion.EN1992_2_2005, "EU_DE")

    def test_get_missing_param_get_info_and_list_params(self, _restore_registry_state):
        """Test get missing param get info and list params."""
        reg = _restore_registry_state
        set_ndp_context(code=EurocodeVersion.EN1992_1_1_2004, country=CountryCode.EU)

        params = reg.list_params()
        assert params
        assert params == sorted(params)

        info = reg.get_info(params[0])
        assert set(info.keys()) >= {"value", "description", "ref"}
        assert get_ndp_info(params[0])["value"] == info["value"]

        with pytest.raises(KeyError, match="NDP parameter 'definitely_missing' not found"):
            reg.get("definitely_missing")

    def test_get_country_data_error_branches(self, _restore_registry_state, monkeypatch):
        """Test get country data error branches."""
        reg = _restore_registry_state

        # Unknown code version.
        reg._code = "NOT_A_CODE"
        with pytest.raises(KeyError, match="Eurocode version 'NOT_A_CODE' not found"):
            reg._get_country_data()

        # Missing EU base.
        reg._code = EurocodeVersion.EN1992_1_1_2004
        reg._country = CountryCode.EU
        monkeypatch.setitem(
            ndp_mod._NDP_DATA,
            EurocodeVersion.EN1992_1_1_2004,
            {CountryCode.EU_UK: {}},
        )
        with pytest.raises(KeyError, match="EU base data not found"):
            reg._get_country_data()

    def test_country_resolution_custom_builtin_and_invalid(self, _restore_registry_state):
        """Test country resolution custom builtin and invalid."""
        reg = _restore_registry_state
        set_ndp_context(code=EurocodeVersion.EN1992_1_1_2004, country=CountryCode.EU)

        # EU base branch.
        eu_data = reg._get_country_data()
        assert "gamma_c" in eu_data

        # Custom annex branch.
        reg._custom_annexes["MY_CUSTOM"] = {"gamma_c": 1.4}
        set_ndp_context(country="MY_CUSTOM")
        custom_data = reg._get_country_data()
        assert custom_data["gamma_c"]["value"] == pytest.approx(1.4, rel=1e-12)

        # Unknown country branch.
        with pytest.raises(KeyError, match="Country 'UNKNOWN_COUNTRY' not found"):
            set_ndp_context(country="UNKNOWN_COUNTRY")

    def test_eu_fallback_branch_when_contains_check_skips_eu_key(
        self,
        _restore_registry_state,
        monkeypatch,
    ):
        """Test eu fallback branch when contains check skips eu key."""
        reg = _restore_registry_state

        class _CodeDataNoEuContains(dict):
            def __contains__(self, key):
                if str(key) == "EU":
                    return False
                return super().__contains__(key)

        monkeypatch.setitem(
            ndp_mod._NDP_DATA,
            EurocodeVersion.EN1992_1_1_2004,
            _CodeDataNoEuContains(
                {
                    CountryCode.EU: {"gamma_c": 1.5},
                    CountryCode.EU_UK: {"gamma_c": 1.4},
                }
            ),
        )

        reg._code = EurocodeVersion.EN1992_1_1_2004
        reg._country = CountryCode.EU
        merged = reg._get_country_data()
        assert merged["gamma_c"]["value"] == pytest.approx(1.5, rel=1e-12)

    def test_ndp_callable_and_override_context(self, _restore_registry_state):
        """Test ndp callable and override context."""
        _ = _restore_registry_state
        set_ndp_context(code=EurocodeVersion.EN1992_1_1_2004, country=CountryCode.EU)

        gamma_c_fn = get_ndp_callable("gamma_c")
        assert callable(gamma_c_fn)
        assert gamma_c_fn() == pytest.approx(float(get_ndp("gamma_c")), rel=1e-12)

        custom_fn = lambda **kwargs: 2.5  # noqa: E731
        with ndp_override(gamma_c=custom_fn):
            out = get_ndp_callable("gamma_c")
            assert out is custom_fn


class TestCustomAnnexRegistration:
    """Tests for TestCustomAnnexRegistration."""
    def test_register_validate_and_unregister_paths(self, _restore_registry_state):
        """Test register validate and unregister paths."""
        reg = _restore_registry_state
        set_ndp_context(code=EurocodeVersion.EN1992_1_1_2004, country=CountryCode.EU)

        with pytest.raises(ValueError, match="Unknown NDP parameters"):
            register_custom_annex("BAD_ANNEX", {"not_a_real_param": 1.0}, validate=True)

        register_custom_annex("PROJ_ANNEX", {"gamma_c": 1.4}, validate=True)
        assert "PROJ_ANNEX" in list_custom_annexes()

        set_ndp_context(country="PROJ_ANNEX")
        assert get_ndp("gamma_c") == pytest.approx(1.4, rel=1e-12)

        assert unregister_custom_annex("PROJ_ANNEX") is True
        assert unregister_custom_annex("PROJ_ANNEX") is False

        # validate=False allows non-EU keys to be stored without immediate validation failure.
        register_custom_annex("NO_VALIDATE", {"custom_key": 123.0}, validate=False)
        assert "NO_VALIDATE" in reg._custom_annexes

    def test_register_raises_when_active_code_data_missing(self, _restore_registry_state):
        """Test register raises when active code data missing."""
        reg = _restore_registry_state
        reg._code = "NOT_A_CODE"
        with pytest.raises(KeyError, match="Eurocode version 'NOT_A_CODE' not found"):
            register_custom_annex("ANY", {"gamma_c": 1.5}, validate=True)
