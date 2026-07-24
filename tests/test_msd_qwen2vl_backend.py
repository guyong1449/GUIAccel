import pytest

from guiaccel.model.msd_qwen2vl_backend import MSDQwen2VLConfig
from scripts.androidcontrol.eval_msd import _summarize_mode


def test_msd_config_accepts_model_identifiers() -> None:
    config = MSDQwen2VLConfig(
        base_model_path="Qwen/Qwen2-VL-7B-Instruct",
        draft_model_path="lucylyn/MSD-Qwen2VL-7B-Instruct",
    )
    config.validate()


def test_msd_config_rejects_missing_base_model() -> None:
    config = MSDQwen2VLConfig(
        base_model_path="",
        draft_model_path="lucylyn/MSD-Qwen2VL-7B-Instruct",
    )
    with pytest.raises(ValueError):
        config.validate()


def test_summarize_mode_reports_accuracy_latency_and_tokens() -> None:
    stats = {
        "steps": 3,
        "matched": 2,
        "parse_failures": 1,
        "latency_sum_ms": 30.0,
        "generated_tokens": 24,
    }
    episode_matches = {
        "episode-a:high_level": [True, True],
        "episode-b:high_level": [False],
        "episode-a:low_level": [True],
    }

    summary = _summarize_mode(stats, episode_matches, mode="high_level")

    assert summary["step_accuracy"] == pytest.approx(2 / 3)
    assert summary["episode_accuracy"] == 0.5
    assert summary["parse_failure_rate"] == pytest.approx(1 / 3)
    assert summary["mean_end_to_end_latency_ms"] == 10.0
    assert summary["generated_tokens"] == 24
