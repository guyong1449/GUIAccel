"""AndroidControl data loader."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

from guiaccel.android_runtime import filter_android_accessibility_nodes
from guiaccel.data.base import BaseDatasetLoader
from guiaccel.types import DatasetEpisode, DatasetStep, ScreenshotAsset
from guiaccel.utils.image import screenshot_from_bytes
from guiaccel.utils.text import normalize_text, unique_preserve_order
from guiaccel.utils.tfrecord import decode_tf_example_features, iter_tfrecord_records, parse_tf_example

ASCII_FRAGMENT_RE = re.compile(rb"[ -~]{3,}")
PERCENT_ENCODED_RE = re.compile(r"(?:%[0-9A-Fa-f]{2}){4,}")
URL_LIKE_RE = re.compile(
    r"https?://|www\.|googleadservices\.com|doubleclick\.net|utm_|gclid(?:=|%3d)|adurl(?:=|%3d)|client=ca-app-pub",
    re.IGNORECASE,
)
MAX_SCREEN_DESCRIPTION_FRAGMENTS = 64
MAX_SCREEN_DESCRIPTION_CHARS = 4096
MAX_FRAGMENT_CHARS = 160
MAX_METADATA_ACCESSIBILITY_FRAGMENTS = 32


class AndroidControlDataset(BaseDatasetLoader):
    """Access AndroidControl TFRecord shards and prepared image cache."""

    default_manifest_key = "androidcontrol_dataset_manifest"
    CORE_FEATURE_NAMES = frozenset(
        {
            "accessibility_trees",
            "actions",
            "episode_id",
            "goal",
            "screen_description",
            "screen_descriptions",
            "screenshot_heights",
            "screenshot_widths",
            "step_instructions",
        }
    )
    STEP_COUNT_FEATURE_NAMES = frozenset({"actions", "episode_id"})
    SCREENSHOT_FEATURE_NAMES = frozenset({"screenshots"})

    def __init__(self, manifest_path: str | Path | None = None) -> None:
        super().__init__(manifest_path)
        self.shard_paths = sorted(self.dataset_root.glob("android_control-*"))
        self.prepared_images_root = self.dataset_root / "prepared_images"
        self.splits = json.loads((self.dataset_root / "splits.json").read_text())
        self.test_subsplits = json.loads((self.dataset_root / "test_subsplits.json").read_text())

        self.split_lookup: dict[int, str] = {}
        for split_name, episode_ids in self.splits.items():
            for episode_id in episode_ids:
                self.split_lookup[int(episode_id)] = split_name

        self.subsplit_lookup = self._build_test_subsplit_lookup(self.test_subsplits)

    @staticmethod
    def _build_test_subsplit_lookup(
        test_subsplits: Mapping[str, Sequence[Any]],
    ) -> dict[int, tuple[str, ...]]:
        memberships: dict[int, list[str]] = {}
        for subsplit_name, episode_ids in test_subsplits.items():
            normalized_name = str(subsplit_name)
            for episode_id in episode_ids:
                coerced_episode_id = int(episode_id)
                episode_memberships = memberships.setdefault(coerced_episode_id, [])
                if normalized_name not in episode_memberships:
                    episode_memberships.append(normalized_name)
        return {
            episode_id: tuple(episode_memberships)
            for episode_id, episode_memberships in memberships.items()
        }

    @staticmethod
    def _extract_accessibility_strings(blob: bytes) -> list[str]:
        fragments = [
            fragment.decode("utf-8", errors="ignore").strip()
            for fragment in ASCII_FRAGMENT_RE.findall(blob)
        ]
        filtered = [
            fragment
            for fragment in fragments
            if len(fragment) >= 3 and any(character.isalpha() for character in fragment)
        ]
        return unique_preserve_order(filtered)

    @staticmethod
    def _load_accessibility_proto() -> tuple[Any | None, str | None]:
        try:
            from android_env.proto.a11y import android_accessibility_forest_pb2
        except Exception as exc:
            return None, f"{type(exc).__name__}: {exc}"
        return android_accessibility_forest_pb2, None

    @staticmethod
    def _accessibility_fallback_warning(status: str) -> str:
        if status == "proto_unavailable":
            return "AndroidControl accessibility proto support is unavailable; continuing with text-only fallback."
        if status == "parse_failed":
            return "AndroidControl accessibility parsing failed; continuing with text-only fallback."
        return ""

    @classmethod
    def _fallback_accessibility_metadata(
        cls,
        heuristic_fragments: Sequence[str],
        *,
        status: str,
        error_text: str | None,
    ) -> dict[str, Any]:
        warning = cls._accessibility_fallback_warning(status)
        return {
            "accessibility_text_fragments": tuple(heuristic_fragments[:MAX_METADATA_ACCESSIBILITY_FRAGMENTS]),
            "accessibility_nodes": tuple(),
            "accessibility_node_count": 0,
            "actionable_accessibility_node_count": 0,
            "accessibility_parse_status": status,
            "accessibility_parse_error": error_text,
            "accessibility_degraded": status != "parsed",
            "accessibility_fallback_used": status != "parsed",
            "accessibility_warning": warning or None,
        }

    @classmethod
    def _proto_rect_to_bbox(cls, rect: Any) -> tuple[int, int, int, int] | None:
        if rect is None:
            return None
        left = int(getattr(rect, "left", 0))
        top = int(getattr(rect, "top", 0))
        right = int(getattr(rect, "right", 0))
        bottom = int(getattr(rect, "bottom", 0))
        if right <= left or bottom <= top:
            return None
        return (left, top, right, bottom)

    @staticmethod
    def _coerce_bbox(value: Any) -> tuple[int, int, int, int] | None:
        if not isinstance(value, (list, tuple)) or len(value) != 4:
            return None
        if not all(isinstance(item, (int, float)) for item in value):
            return None
        left, top, right, bottom = (int(item) for item in value)
        if right <= left or bottom <= top:
            return None
        return (left, top, right, bottom)

    @classmethod
    def _lookup_accessibility_target(
        cls,
        accessibility_nodes: Sequence[Mapping[str, Any]],
        *,
        x: int,
        y: int,
    ) -> tuple[tuple[int, int, int, int], str | None] | None:
        candidates: list[tuple[int, int, tuple[int, int, int, int], str | None]] = []
        for node in accessibility_nodes:
            bbox = cls._coerce_bbox(node.get("bbox"))
            if bbox is None:
                continue
            if not (bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]):
                continue
            label = normalize_text(
                node.get("text")
                or node.get("content_description")
                or node.get("view_id_resource_name")
                or node.get("class_name")
            )
            area = max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])
            depth = int(node.get("depth") or 0)
            candidates.append((area, -depth, bbox, label))
        if not candidates:
            return None
        _, _, bbox, label = min(candidates)
        return bbox, label

    @classmethod
    def _resolve_action_target_metadata(
        cls,
        action_payload: Mapping[str, Any],
        accessibility_nodes: Sequence[Mapping[str, Any]],
    ) -> tuple[tuple[int, int, int, int] | None, str | None]:
        bbox = cls._coerce_bbox(action_payload.get("target_bbox") or action_payload.get("bbox"))
        if bbox is not None:
            center_x = int(round((bbox[0] + bbox[2]) / 2.0))
            center_y = int(round((bbox[1] + bbox[3]) / 2.0))
            resolved = cls._lookup_accessibility_target(accessibility_nodes, x=center_x, y=center_y)
            if resolved is not None:
                return resolved
            return bbox, None
        x = action_payload.get("x")
        y = action_payload.get("y")
        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
            resolved = cls._lookup_accessibility_target(
                accessibility_nodes,
                x=int(x),
                y=int(y),
            )
            if resolved is not None:
                return resolved
        return None, None

    @classmethod
    def _extract_accessibility_metadata(
        cls,
        blob: bytes,
        *,
        screenshot_width: int,
        screenshot_height: int,
    ) -> dict[str, Any]:
        proto, proto_error = cls._load_accessibility_proto()
        heuristic_fragments = cls._extract_accessibility_strings(blob)
        if proto is None:
            return cls._fallback_accessibility_metadata(
                heuristic_fragments,
                status="proto_unavailable",
                error_text=proto_error,
            )

        forest = proto.AndroidAccessibilityForest()
        try:
            forest.ParseFromString(blob)
        except Exception as exc:
            return cls._fallback_accessibility_metadata(
                heuristic_fragments,
                status="parse_failed",
                error_text=f"{type(exc).__name__}: {exc}",
            )

        raw_nodes: list[dict[str, Any]] = []
        fragments: list[str] = []
        actionable_count = 0
        for window in getattr(forest, "windows", ()):
            window_title = str(getattr(window, "title", "") or "").strip()
            if window_title:
                fragments.append(window_title)
            tree = getattr(window, "tree", None)
            if tree is None:
                continue
            for node in getattr(tree, "nodes", ()):
                bbox = cls._proto_rect_to_bbox(getattr(node, "bounds_in_screen", None))
                text = str(getattr(node, "text", "") or "").strip()
                content_description = str(getattr(node, "content_description", "") or "").strip()
                view_id_resource_name = str(getattr(node, "view_id_resource_name", "") or "").strip()
                class_name = str(getattr(node, "class_name", "") or "").strip()
                clickable = bool(getattr(node, "is_clickable", False))
                editable = bool(getattr(node, "is_editable", False))
                scrollable = bool(getattr(node, "is_scrollable", False))
                checkable = bool(getattr(node, "is_checkable", False))
                checked = bool(getattr(node, "is_checked", False))
                visible = bool(getattr(node, "is_visible_to_user", True))
                enabled = bool(getattr(node, "is_enabled", True))
                focusable = bool(getattr(node, "is_focusable", False))
                if clickable or editable or scrollable or checkable:
                    actionable_count += 1
                for fragment in (text, content_description, view_id_resource_name):
                    normalized = normalize_text(fragment)
                    if normalized:
                        fragments.append(normalized)
                raw_nodes.append(
                    {
                        "bbox": bbox,
                        "text": text or None,
                        "content_description": content_description or None,
                        "class_name": class_name or None,
                        "view_id_resource_name": view_id_resource_name or None,
                        "clickable": clickable,
                        "editable": editable,
                        "scrollable": scrollable,
                        "checkable": checkable,
                        "checked": checked,
                        "visible": visible,
                        "enabled": enabled,
                        "focusable": focusable,
                        "depth": int(getattr(node, "depth", 0) or 0),
                    }
                )

        filtered_nodes = filter_android_accessibility_nodes(
            raw_nodes,
            screen_size=(screenshot_width, screenshot_height),
        )
        cleaned_fragments = [
            fragment
            for fragment in unique_preserve_order(fragments + heuristic_fragments)
            if not cls._is_noisy_accessibility_fragment(fragment)
        ]
        return {
            "accessibility_text_fragments": tuple(cleaned_fragments[:MAX_METADATA_ACCESSIBILITY_FRAGMENTS]),
            "accessibility_nodes": filtered_nodes,
            "accessibility_node_count": len(raw_nodes),
            "actionable_accessibility_node_count": actionable_count,
            "accessibility_parse_status": "parsed",
            "accessibility_parse_error": None,
            "accessibility_degraded": False,
            "accessibility_fallback_used": False,
            "accessibility_warning": None,
        }

    @staticmethod
    def _is_noisy_accessibility_fragment(fragment: str) -> bool:
        candidate = " ".join(fragment.strip().split())
        if len(candidate) < 3 or not any(character.isalpha() for character in candidate):
            return True
        if URL_LIKE_RE.search(candidate):
            return True
        if len(candidate) > 64 and PERCENT_ENCODED_RE.search(candidate):
            return True
        if len(candidate) > 120 and candidate.count("=") >= 2 and candidate.count("&") >= 2:
            return True
        return False

    @classmethod
    def _sanitize_screen_description(cls, raw_description: str) -> str:
        fragments = [fragment.strip() for fragment in raw_description.split("|")]
        if len(fragments) <= 1:
            fragments = [raw_description]

        cleaned: list[str] = []
        total_chars = 0
        for fragment in fragments:
            normalized = " ".join(fragment.strip().split())
            if not normalized or cls._is_noisy_accessibility_fragment(normalized):
                continue
            if len(normalized) > MAX_FRAGMENT_CHARS:
                normalized = normalized[:MAX_FRAGMENT_CHARS].rstrip()
            if not normalized:
                continue
            projected_chars = total_chars + len(normalized) + (3 if cleaned else 0)
            if projected_chars > MAX_SCREEN_DESCRIPTION_CHARS:
                break
            cleaned.append(normalized)
            total_chars = projected_chars
            if len(cleaned) >= MAX_SCREEN_DESCRIPTION_FRAGMENTS:
                break
        return " | ".join(unique_preserve_order(cleaned))

    def _heuristic_screen_descriptions(
        self,
        decoded_example: dict[str, list[Any]],
        screenshot_count: int,
    ) -> list[str]:
        serialized: list[str] = []
        tree_values = decoded_example.get("accessibility_trees", [])
        for index in range(screenshot_count):
            tree_blob = tree_values[index] if index < len(tree_values) else b""
            blob_bytes = bytes(tree_blob) if isinstance(tree_blob, (bytes, bytearray)) else b""
            fragments = self._extract_accessibility_strings(blob_bytes)
            serialized.append(
                self._sanitize_screen_description(
                    " | ".join(fragments[:MAX_SCREEN_DESCRIPTION_FRAGMENTS])
                )
            )
        return serialized

    def _serialize_screen_descriptions(
        self,
        decoded_example: dict[str, list[Any]],
        screenshot_count: int,
    ) -> tuple[list[str], str]:
        raw_descriptions = decoded_example.get("screen_description") or decoded_example.get("screen_descriptions")
        heuristic_descriptions = self._heuristic_screen_descriptions(decoded_example, screenshot_count)
        if not raw_descriptions:
            return heuristic_descriptions, "heuristic_accessibility_serialization"

        sanitized_provided = [
            self._sanitize_screen_description(
                item.decode("utf-8") if isinstance(item, bytes) else str(item)
            )
            for item in raw_descriptions
        ]
        aligned: list[str] = []
        used_heuristic_backfill = False
        for index in range(screenshot_count):
            provided_description = sanitized_provided[index] if index < len(sanitized_provided) else ""
            if provided_description:
                aligned.append(provided_description)
                continue
            aligned.append(heuristic_descriptions[index] if index < len(heuristic_descriptions) else "")
            used_heuristic_backfill = True
        if len(aligned) < screenshot_count:
            aligned.extend("" for _ in range(screenshot_count - len(aligned)))
        return (
            aligned[:screenshot_count],
            "provided_with_heuristic_backfill" if used_heuristic_backfill else "provided",
        )

    def _build_episode_from_example(
        self,
        *,
        shard_path: Path,
        decoded_example: dict[str, list[Any]],
    ) -> DatasetEpisode:
        episode_id = int(decoded_example["episode_id"][0])
        goal = decoded_example["goal"][0].decode("utf-8")
        split = self.split_lookup.get(episode_id)
        test_subsplits = self.subsplit_lookup.get(episode_id, tuple())
        test_subsplit = test_subsplits[0] if test_subsplits else None
        screenshot_blobs = decoded_example.get("screenshots", [])
        screenshot_widths = [int(value) for value in decoded_example.get("screenshot_widths", [])]
        screenshot_heights = [int(value) for value in decoded_example.get("screenshot_heights", [])]
        screenshot_count = self._expected_screenshot_count(decoded_example)
        actions = [json.loads(item.decode("utf-8")) for item in decoded_example.get("actions", [])]
        step_instructions = [
            item.decode("utf-8") if isinstance(item, bytes) else str(item)
            for item in decoded_example.get("step_instructions", [])
        ]
        screen_descriptions, screen_description_source = self._serialize_screen_descriptions(
            decoded_example,
            screenshot_count,
        )

        history_action_types: list[str] = []
        high_level_step_index = -1
        previous_instruction: str | None = None
        steps: list[DatasetStep] = []
        screenshot_cache: dict[int, Any] = {}
        accessibility_status_counts: dict[str, int] = {}
        degraded_accessibility_steps = 0

        def load_screenshot(step_index: int) -> Any:
            cached = screenshot_cache.get(step_index)
            if cached is not None:
                return cached
            prepared_image_path = (
                self.prepared_images_root
                / shard_path.name
                / str(episode_id)
                / f"{step_index:04d}.png"
            )
            screenshot = (
                ScreenshotAsset(
                    path=prepared_image_path,
                    width=screenshot_widths[step_index],
                    height=screenshot_heights[step_index],
                    source="path",
                )
                if prepared_image_path.exists()
                else screenshot_from_bytes(
                    screenshot_blobs[step_index],
                    width=screenshot_widths[step_index],
                    height=screenshot_heights[step_index],
                )
            )
            screenshot_cache[step_index] = screenshot
            return screenshot

        for step_index, action_payload in enumerate(actions):
            step_instruction = step_instructions[step_index] if step_index < len(step_instructions) else None
            if step_instruction != previous_instruction:
                high_level_step_index += 1
                previous_instruction = step_instruction

            screenshot = load_screenshot(step_index)
            next_screenshot = None
            next_metadata = None
            if step_index + 1 < screenshot_count:
                next_screenshot = load_screenshot(step_index + 1)
                next_metadata = {
                    "step_instruction": (
                        step_instructions[step_index + 1]
                        if step_index + 1 < len(step_instructions)
                        else None
                    ),
                    "screen_description": screen_descriptions[step_index + 1],
                }

            screen_description = screen_descriptions[step_index] if step_index < len(screen_descriptions) else ""
            accessibility_trees = decoded_example.get("accessibility_trees", [])
            tree_blob = accessibility_trees[step_index] if step_index < len(accessibility_trees) else b""
            blob_bytes = bytes(tree_blob) if isinstance(tree_blob, (bytes, bytearray)) else b""
            accessibility_metadata = self._extract_accessibility_metadata(
                blob_bytes,
                screenshot_width=screenshot_widths[step_index],
                screenshot_height=screenshot_heights[step_index],
            )
            action_target_bbox, action_target_label = self._resolve_action_target_metadata(
                action_payload,
                accessibility_metadata.get("accessibility_nodes") or (),
            )
            input_text_reference_bbox = None
            input_text_reference_label = None
            if (
                str(action_payload.get("action_type") or "").lower() == "input_text"
                and step_index > 0
                and str(actions[step_index - 1].get("action_type") or "").lower() == "click"
                and steps
            ):
                input_text_reference_bbox = steps[-1].metadata.get("action_target_bbox")
                input_text_reference_label = steps[-1].metadata.get("action_target_label")
            parse_status = str(accessibility_metadata.get("accessibility_parse_status") or "unknown")
            accessibility_status_counts[parse_status] = accessibility_status_counts.get(parse_status, 0) + 1
            if accessibility_metadata.get("accessibility_degraded"):
                degraded_accessibility_steps += 1
            metadata = {
                "benchmark": "AndroidControl",
                "representation": "processed-step",
                "episode_id": episode_id,
                "split": split,
                "test_subsplit": test_subsplit,
                "test_subsplits": test_subsplits,
                "step_index": step_index,
                "step_instruction": step_instruction,
                "high_level_step_index": high_level_step_index,
                "history_action_types": tuple(history_action_types),
                "screen_description": screen_description,
                "screen_description_source": screen_description_source,
                "screenshot_width": screenshot_widths[step_index],
                "screenshot_height": screenshot_heights[step_index],
                "action_target_bbox": action_target_bbox,
                "action_target_label": action_target_label,
                "input_text_reference_bbox": input_text_reference_bbox,
                "input_text_reference_label": input_text_reference_label,
                "processed_step": {
                    "goal": goal,
                    "step_instruction": step_instruction,
                    "screen_description": screen_description,
                    "history_action_types": tuple(history_action_types),
                    "test_subsplits": test_subsplits,
                },
                **accessibility_metadata,
            }
            steps.append(
                DatasetStep(
                    dataset="AndroidControl",
                    episode_id=str(episode_id),
                    step_index=step_index,
                    goal=goal,
                    screenshot=screenshot,
                    raw_action=action_payload,
                    metadata=metadata,
                    next_screenshot=next_screenshot,
                    next_metadata=next_metadata,
                )
            )
            history_action_types.append(action_payload.get("action_type", "unknown"))

        return DatasetEpisode(
            dataset="AndroidControl",
            episode_id=str(episode_id),
            goal=goal,
            steps=tuple(steps),
            split=split,
            app=None,
            domain=None,
            metadata={
                "shard_path": str(shard_path),
                "test_subsplit": test_subsplit,
                "test_subsplits": test_subsplits,
                "screen_description_source": screen_description_source,
                "accessibility_status_counts": dict(accessibility_status_counts),
                "accessibility_degraded_step_count": int(degraded_accessibility_steps),
            },
        )

    @staticmethod
    def _expected_screenshot_count(decoded_example: dict[str, list[Any]]) -> int:
        actions_count = len(decoded_example.get("actions", []))
        return max(
            len(decoded_example.get("screenshots", [])),
            len(decoded_example.get("screenshot_widths", [])),
            len(decoded_example.get("screenshot_heights", [])),
            len(decoded_example.get("screen_description", [])),
            len(decoded_example.get("screen_descriptions", [])),
            len(decoded_example.get("accessibility_trees", [])),
            actions_count + 1 if actions_count else 0,
        )

    def _prepared_episode_dir(self, shard_path: Path, episode_id: int) -> Path:
        return self.prepared_images_root / shard_path.name / str(episode_id)

    def _has_complete_prepared_images(
        self,
        *,
        shard_path: Path,
        episode_id: int,
        screenshot_count: int,
    ) -> bool:
        if screenshot_count <= 0:
            return False
        episode_dir = self._prepared_episode_dir(shard_path, episode_id)
        if not episode_dir.exists():
            return False
        return all((episode_dir / f"{step_index:04d}.png").exists() for step_index in range(screenshot_count))

    def _decode_core_example(self, serialized_example: bytes) -> dict[str, list[Any]]:
        return decode_tf_example_features(
            parse_tf_example(serialized_example, feature_names=self.CORE_FEATURE_NAMES)
        )

    def _decode_screenshots(self, serialized_example: bytes) -> list[Any]:
        return decode_tf_example_features(
            parse_tf_example(serialized_example, feature_names=self.SCREENSHOT_FEATURE_NAMES)
        ).get("screenshots", [])

    def iter_episodes(
        self,
        *,
        split: str | Sequence[str] | None = None,
        limit: int | None = None,
        episode_filter: Callable[[int], bool] | None = None,
        shard_paths: Sequence[str | Path] | None = None,
    ) -> Iterator[DatasetEpisode]:
        requested_splits = (
            {str(item) for item in split}
            if isinstance(split, (list, tuple, set, frozenset))
            else ({str(split)} if split is not None else None)
        )
        selected_shards = (
            tuple(Path(path).resolve() for path in shard_paths)
            if shard_paths is not None
            else tuple(self.shard_paths)
        )
        emitted = 0
        import time as _t
        for shard_idx, shard_path in enumerate(selected_shards):
            print(f"[{_t.strftime('%H:%M:%S')}] [shard {shard_idx+1}/{len(selected_shards)}] opening {shard_path.name} ...", flush=True)
            shard_emitted = 0
            for serialized_example in iter_tfrecord_records(shard_path):
                header_example = decode_tf_example_features(
                    parse_tf_example(serialized_example, feature_names=frozenset({"episode_id"}))
                )
                if "episode_id" not in header_example:
                    continue
                episode_id = int(header_example["episode_id"][0])
                episode_split = self.split_lookup.get(episode_id)
                if requested_splits is not None and episode_split not in requested_splits:
                    continue
                if episode_filter is not None and not episode_filter(episode_id):
                    continue

                decoded_example = self._decode_core_example(serialized_example)
                episode_id = int(decoded_example["episode_id"][0])
                screenshot_count = self._expected_screenshot_count(decoded_example)
                if not self._has_complete_prepared_images(
                    shard_path=shard_path,
                    episode_id=episode_id,
                    screenshot_count=screenshot_count,
                ):
                    decoded_example["screenshots"] = self._decode_screenshots(serialized_example)
                yield self._build_episode_from_example(shard_path=shard_path, decoded_example=decoded_example)
                emitted += 1
                shard_emitted += 1
                if shard_emitted % 50 == 0:
                    print(f"[{_t.strftime('%H:%M:%S')}] [shard {shard_idx+1}] {shard_emitted} episodes yielded (total={emitted})", flush=True)
                if limit is not None and emitted >= limit:
                    return

    def iter_episode_lengths(
        self,
        *,
        split: str | Sequence[str] | None = None,
        limit: int | None = None,
        episode_filter: Callable[[int], bool] | None = None,
        shard_paths: Sequence[str | Path] | None = None,
    ) -> Iterator[tuple[int, int]]:
        requested_splits = (
            {str(item) for item in split}
            if isinstance(split, (list, tuple, set, frozenset))
            else ({str(split)} if split is not None else None)
        )
        selected_shards = (
            tuple(Path(path).resolve() for path in shard_paths)
            if shard_paths is not None
            else tuple(self.shard_paths)
        )
        emitted = 0
        for shard_path in selected_shards:
            for serialized_example in iter_tfrecord_records(shard_path):
                decoded = decode_tf_example_features(
                    parse_tf_example(serialized_example, feature_names=self.STEP_COUNT_FEATURE_NAMES)
                )
                if "episode_id" not in decoded:
                    continue
                episode_id = int(decoded["episode_id"][0])
                episode_split = self.split_lookup.get(episode_id)
                if requested_splits is not None and episode_split not in requested_splits:
                    continue
                if episode_filter is not None and not episode_filter(episode_id):
                    continue
                yield (episode_id, len(decoded.get("actions", [])))
                emitted += 1
                if limit is not None and emitted >= limit:
                    return

    def resolve_shard_paths_for_episode_ids(
        self,
        episode_ids: Sequence[int | str],
        *,
        split: str | Sequence[str] | None = None,
        shard_paths: Sequence[str | Path] | None = None,
    ) -> tuple[Path, ...]:
        requested_episode_ids = {int(episode_id) for episode_id in episode_ids}
        if not requested_episode_ids:
            return tuple()

        requested_splits = (
            {str(item) for item in split}
            if isinstance(split, (list, tuple, set, frozenset))
            else ({str(split)} if split is not None else None)
        )
        selected_shards = (
            tuple(Path(path).resolve() for path in shard_paths)
            if shard_paths is not None
            else tuple(self.shard_paths)
        )
        matched_shards: list[Path] = []
        remaining_episode_ids = set(requested_episode_ids)
        for shard_path in selected_shards:
            shard_matched = False
            for serialized_example in iter_tfrecord_records(shard_path):
                header_example = decode_tf_example_features(
                    parse_tf_example(serialized_example, feature_names=frozenset({"episode_id"}))
                )
                if "episode_id" not in header_example:
                    continue
                episode_id = int(header_example["episode_id"][0])
                if episode_id not in remaining_episode_ids:
                    continue
                episode_split = self.split_lookup.get(episode_id)
                if requested_splits is not None and episode_split not in requested_splits:
                    continue
                shard_matched = True
                remaining_episode_ids.discard(episode_id)
                if not remaining_episode_ids:
                    matched_shards.append(shard_path)
                    return tuple(matched_shards)
            if shard_matched:
                matched_shards.append(shard_path)

        if remaining_episode_ids:
            missing = ", ".join(str(episode_id) for episode_id in sorted(remaining_episode_ids))
            raise KeyError(f"Unable to resolve AndroidControl shards for episode_ids: {missing}")
        return tuple(matched_shards)

    def resolve_shard_paths_for_episode_id_groups(
        self,
        episode_id_groups: Sequence[Sequence[int | str]],
        *,
        split: str | Sequence[str] | None = None,
        shard_paths: Sequence[str | Path] | None = None,
    ) -> tuple[tuple[Path, ...], ...]:
        normalized_groups = tuple(
            tuple(int(episode_id) for episode_id in episode_ids)
            for episode_ids in episode_id_groups
        )
        if not normalized_groups:
            return tuple()

        requested_splits = (
            {str(item) for item in split}
            if isinstance(split, (list, tuple, set, frozenset))
            else ({str(split)} if split is not None else None)
        )
        selected_shards = (
            tuple(Path(path).resolve() for path in shard_paths)
            if shard_paths is not None
            else tuple(self.shard_paths)
        )

        group_remaining = [set(group) for group in normalized_groups]
        episode_to_group_indices: dict[int, list[int]] = {}
        unresolved_group_count = 0
        for group_index, remaining_episode_ids in enumerate(group_remaining):
            if remaining_episode_ids:
                unresolved_group_count += 1
            for episode_id in remaining_episode_ids:
                episode_to_group_indices.setdefault(int(episode_id), []).append(int(group_index))

        matched_shards_by_group: list[list[Path]] = [[] for _ in normalized_groups]
        if not episode_to_group_indices:
            return tuple(tuple(paths) for paths in matched_shards_by_group)

        for shard_path in selected_shards:
            matched_groups_in_shard: set[int] = set()
            for serialized_example in iter_tfrecord_records(shard_path):
                header_example = decode_tf_example_features(
                    parse_tf_example(serialized_example, feature_names=frozenset({"episode_id"}))
                )
                if "episode_id" not in header_example:
                    continue
                episode_id = int(header_example["episode_id"][0])
                group_indices = episode_to_group_indices.get(episode_id)
                if not group_indices:
                    continue
                episode_split = self.split_lookup.get(episode_id)
                if requested_splits is not None and episode_split not in requested_splits:
                    continue
                for group_index in group_indices:
                    if episode_id not in group_remaining[group_index]:
                        continue
                    matched_groups_in_shard.add(int(group_index))
                    group_remaining[group_index].discard(episode_id)
                    if not group_remaining[group_index]:
                        unresolved_group_count -= 1
                if unresolved_group_count == 0:
                    break
            for group_index in sorted(matched_groups_in_shard):
                matched_shards_by_group[group_index].append(shard_path)
            if unresolved_group_count == 0:
                return tuple(tuple(paths) for paths in matched_shards_by_group)

        missing_groups = [
            ", ".join(str(episode_id) for episode_id in sorted(remaining_episode_ids))
            for remaining_episode_ids in group_remaining
            if remaining_episode_ids
        ]
        if missing_groups:
            raise KeyError(
                "Unable to resolve AndroidControl shards for episode groups: "
                + " | ".join(missing_groups)
            )
        return tuple(tuple(paths) for paths in matched_shards_by_group)

    def get_episode(self, episode_id: int | str) -> DatasetEpisode:
        target_episode_id = int(episode_id)
        for shard_path in self.shard_paths:
            for serialized_example in iter_tfrecord_records(shard_path):
                header_example = decode_tf_example_features(
                    parse_tf_example(serialized_example, feature_names=frozenset({"episode_id"}))
                )
                if "episode_id" not in header_example:
                    continue
                current_episode_id = int(header_example["episode_id"][0])
                if current_episode_id != target_episode_id:
                    continue
                decoded_example = self._decode_core_example(serialized_example)
                screenshot_count = self._expected_screenshot_count(decoded_example)
                if not self._has_complete_prepared_images(
                    shard_path=shard_path,
                    episode_id=current_episode_id,
                    screenshot_count=screenshot_count,
                ):
                    decoded_example["screenshots"] = self._decode_screenshots(serialized_example)
                return self._build_episode_from_example(
                    shard_path=shard_path,
                    decoded_example=decoded_example,
                )
        raise KeyError(f"Unknown AndroidControl episode_id: {episode_id}")
