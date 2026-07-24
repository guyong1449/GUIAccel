"""Tests for repository checkpoint example reload in evaluation orchestrator."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from skillreuse.evaluation.orchestrator import (
    _repository_examples_from_checkpoint_build,
    build_android_examples,
)
from skillreuse.repository import RepositoryCheckpointBuildResult


def _android_repository_example(*, episode_id: str = "ep-1", step_index: int = 0):
    return SimpleNamespace(
        metadata={"instruction_mode": "high_level", "partition": "IDD"},
        record=SimpleNamespace(normalized_metadata={"test_subsplit": "IDD"}),
        dataset_step=SimpleNamespace(step_index=step_index, metadata={"test_subsplit": "IDD"}),
        group_id=episode_id,
        observation_id=f"{episode_id}:{step_index}",
    )


def _checkpoint_build_result(
    *,
    examples: tuple = tuple(),
    checkpoint_root: Path | None = None,
) -> RepositoryCheckpointBuildResult:
    return RepositoryCheckpointBuildResult(
        examples=examples,
        checkpoint_root=(checkpoint_root or Path("/tmp/eval_repository_checkpoint")).resolve(),
        resumed=bool(examples),
        manifest_status="completed",
        completed_examples=len(examples),
    )


def test_repository_examples_from_checkpoint_build_uses_in_memory_examples():
    example = _android_repository_example()
    build_result = _checkpoint_build_result(examples=(example,))

    with patch("skillreuse.evaluation.orchestrator.load_repository_checkpoint") as load_mock:
        loaded = _repository_examples_from_checkpoint_build(build_result)

    assert loaded == (example,)
    load_mock.assert_not_called()


def test_repository_examples_from_checkpoint_build_reloads_when_empty():
    reloaded = (_android_repository_example(episode_id="ep-2"),)
    build_result = RepositoryCheckpointBuildResult(
        examples=tuple(),
        checkpoint_root=Path("/tmp/eval_repository_checkpoint").resolve(),
        resumed=False,
        manifest_status="completed",
        completed_examples=1,
    )
    manifest = MagicMock()

    with patch(
        "skillreuse.evaluation.orchestrator.load_repository_checkpoint_manifest",
        return_value=manifest,
    ) as manifest_mock:
        with patch(
            "skillreuse.evaluation.orchestrator.load_repository_checkpoint",
            return_value=reloaded,
        ) as load_mock:
            loaded = _repository_examples_from_checkpoint_build(build_result)

    assert loaded == reloaded
    manifest_mock.assert_called_once_with(build_result.checkpoint_root)
    load_mock.assert_called_once_with(manifest)


def test_build_android_examples_reloads_empty_checkpoint_build():
    reloaded = (_android_repository_example(episode_id="ep-3"),)
    build_result = _checkpoint_build_result(examples=tuple())
    build_result = RepositoryCheckpointBuildResult(
        examples=build_result.examples,
        checkpoint_root=build_result.checkpoint_root,
        resumed=False,
        manifest_status="completed",
        completed_examples=1,
    )

    with patch(
        "skillreuse.evaluation.orchestrator.build_training_repository_checkpointed",
        return_value=build_result,
    ) as build_mock:
        with patch(
            "skillreuse.evaluation.orchestrator._repository_examples_from_checkpoint_build",
            return_value=reloaded,
        ) as reload_mock:
            examples = build_android_examples(checkpoint_root=build_result.checkpoint_root)

    build_mock.assert_called_once()
    reload_mock.assert_called_once_with(build_result)
    assert len(examples) == 1
    assert examples[0].benchmark == "AndroidControl"
    assert examples[0].episode_id == "ep-3"
