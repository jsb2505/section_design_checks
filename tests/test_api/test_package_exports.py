"""Tests for `section_design_checks.api` package namespace and exports."""

from __future__ import annotations

import importlib


def test_api_package_exposes_models_namespace():
    """`section_design_checks.api` should expose the `models` namespace via package exports."""
    api = importlib.import_module("section_design_checks.api")
    assert hasattr(api, "models"), "section_design_checks.api is expected to expose a `models` attribute."
    assert "models" in getattr(api, "__all__", []), "__all__ should include `models`."


def test_api_models_namespace_imports_cleanly():
    """`section_design_checks.api.models` should be importable even when no concrete models exist yet."""
    models = importlib.import_module("section_design_checks.api.models")
    assert isinstance(models.__all__, list), "`section_design_checks.api.models.__all__` should be a list."
    assert models.__all__ == [], "Expected empty __all__ until API model types are added."
