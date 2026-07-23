"""Guards for the runtime-visible feature boundary."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

_FORBIDDEN_EXACT_KEYS = frozenset(
    {
        "future_annotation",
        "future_annotations",
        "future_trajectory",
        "next_step",
        "next_step_sketch",
        "teacher",
        "teacher_annotation",
        "teacher_annotations",
        "teacher_only",
    }
)
_FORBIDDEN_PREFIXES = ("future_", "next_step_", "teacher_")


def find_runtime_visibility_violations(payload: Any, *, path: str = "") -> tuple[str, ...]:
    """Return nested paths that expose teacher-only or future-derived fields."""

    violations: list[str] = []
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            normalized_key = str(key)
            child_path = f"{path}.{normalized_key}" if path else normalized_key
            if _is_forbidden_runtime_key(normalized_key):
                violations.append(child_path)
            violations.extend(find_runtime_visibility_violations(value, path=child_path))
        return tuple(violations)
    if isinstance(payload, (list, tuple)):
        for index, value in enumerate(payload):
            child_path = f"{path}[{index}]" if path else f"[{index}]"
            violations.extend(find_runtime_visibility_violations(value, path=child_path))
    return tuple(violations)


def assert_runtime_problem_sketch_clean(sketch: Mapping[str, Any], *, context: str) -> None:
    """Fail if a reduced-problem sketch carries teacher-only fields."""

    _raise_if_violations(
        find_runtime_visibility_violations(sketch),
        context=context,
    )


def assert_runtime_feature_map_clean(features: Mapping[str, Any], *, context: str) -> None:
    """Fail if a routing/calibration feature map carries teacher-only fields."""

    _raise_if_violations(
        find_runtime_visibility_violations(features),
        context=context,
    )


def assert_runtime_skill_artifact_clean(skill: Any, *, context: str) -> None:
    """Fail if one calibrated/runtime skill artifact exposes forbidden fields."""

    runtime_skill = getattr(skill, "skill", skill)
    issues: list[str] = []
    issues.extend(
        f"prototype_sketch.{violation}"
        for violation in find_runtime_visibility_violations(
            getattr(runtime_skill, "prototype_sketch", {}) or {}
        )
    )
    issues.extend(
        f"discovery_stats.{violation}"
        for violation in find_runtime_visibility_violations(
            getattr(runtime_skill, "discovery_stats", {}) or {}
        )
    )
    for feature_name in tuple(getattr(getattr(skill, "score_model", None), "feature_names", ()) or ()):
        if _is_forbidden_runtime_key(str(feature_name)):
            issues.append(f"score_model.feature_names[{feature_name}]")
    for index, observation in enumerate(tuple(getattr(skill, "observations", ()) or ())):
        issues.extend(
            f"observations[{index}].features.{violation}"
            for violation in find_runtime_visibility_violations(
                getattr(observation, "features", {}) or {}
            )
        )
    _raise_if_violations(tuple(issues), context=context)


def assert_runtime_skill_artifacts_clean(skills: Iterable[Any], *, context: str) -> None:
    """Fail if any skill in a library exposes teacher-only runtime fields."""

    for index, skill in enumerate(skills):
        skill_id = str(getattr(skill, "skill_id", getattr(getattr(skill, "skill", None), "skill_id", index)))
        assert_runtime_skill_artifact_clean(skill, context=f"{context} skill={skill_id}")


def _is_forbidden_runtime_key(key: str) -> bool:
    normalized = str(key).strip()
    if normalized in _FORBIDDEN_EXACT_KEYS:
        return True
    return any(normalized.startswith(prefix) for prefix in _FORBIDDEN_PREFIXES)


def _raise_if_violations(violations: Iterable[str], *, context: str) -> None:
    materialized = tuple(str(item) for item in violations)
    if materialized:
        raise ValueError(
            f"{context} exposes teacher-only or future-derived runtime fields: {', '.join(materialized)}"
        )
