"""Smoke tests for DivPrune config loading and backend construction."""

import json
import pytest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_config_load_divprune():
    """load_benchmark_config + resolve_backend_config on divprune config."""
    from skillreuse.configuration import load_benchmark_config, resolve_backend_config
    config = load_benchmark_config(config_path=str(REPO_ROOT / "configs" / "androidcontrol" / "divprune" / "default.json"))
    svc_cfg = resolve_backend_config(config)
    assert svc_cfg.kind == "divprune"
    assert svc_cfg.keep_ratio == 0.098


def test_config_load_baseline_transformers():
    """load_benchmark_config + resolve_backend_config on baseline transformers config."""
    from skillreuse.configuration import load_benchmark_config, resolve_backend_config
    config = load_benchmark_config(config_path=str(REPO_ROOT / "configs" / "androidcontrol" / "baseline" / "default.json"))
    svc_cfg = resolve_backend_config(config)
    assert svc_cfg.kind == "local_transformers"
    assert svc_cfg.keep_ratio is None


def test_default_config_unbroken():
    """Existing default config still loads without keep_ratio."""
    from skillreuse.configuration import load_benchmark_config, resolve_backend_config
    config = load_benchmark_config(config_path=str(REPO_ROOT / "configs" / "androidcontrol" / "default.json"))
    svc_cfg = resolve_backend_config(config)
    assert svc_cfg.keep_ratio is None


def test_keep_ratio_passthrough():
    """keep_ratio flows from JSON → ModelServiceConfig correctly."""
    from skillreuse.model.service_backend import ModelServiceConfig
    cfg = ModelServiceConfig(kind="divprune", keep_ratio=0.098)
    assert cfg.kind == "divprune"
    assert cfg.keep_ratio == 0.098


def test_model_service_config_divprune_kind():
    """DIVPRUNE_BACKEND_KINDS contains 'divprune'."""
    from skillreuse.model.service_backend import DIVPRUNE_BACKEND_KINDS
    assert "divprune" in DIVPRUNE_BACKEND_KINDS


def test_resolve_model_spec_divprune():
    """resolve_model_spec handles divprune config."""
    from skillreuse.configuration import load_benchmark_config, resolve_model_spec
    config = load_benchmark_config(config_path=str(REPO_ROOT / "configs" / "androidcontrol" / "divprune" / "default.json"))
    spec = resolve_model_spec(config, benchmark="AndroidControl")
    assert spec.backend == "divprune"
    assert spec.base_model_path is not None
