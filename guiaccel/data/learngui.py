"""LearnGUI data loader."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from guiaccel.data.base import BaseDatasetLoader
from guiaccel.types import DatasetEpisode, DatasetStep
from guiaccel.utils.image import screenshot_from_path

EPISODE_ID_RE = re.compile(r"([0-9a-f]{32})")


@dataclass
class LearnGUITask:
    """Few-shot query/support task descriptor."""

    app: str
    split: str
    k_shot: int
    query_episode_id: str
    support_episode_ids: tuple[str, ...]
    metadata: dict[str, Any]


def _extract_episode_id(name: str) -> str | None:
    match = EPISODE_ID_RE.search(name)
    return match.group(1) if match else None


class LearnGUIDataset(BaseDatasetLoader):
    """Access the prepared LearnGUI offline assets."""

    default_manifest_key = "learngui_dataset_manifest"

    def __init__(self, manifest_path: str | Path | None = None) -> None:
        super().__init__(manifest_path)
        self.offline_root = self.dataset_root / "offline"
        self.screenshots_dir = self.offline_root / "screenshots"
        self.element_annotations_dir = self.offline_root / "element_anno" / "element_anno"
        self.instruction_annotations_dir = self.offline_root / "instruction_anno" / "instruction_anno"
        self.low_level_path = self.offline_root / "low_level_instructions.json"
        self.task_split_path = self.offline_root / "task_split.json"

        self.low_level_episodes: dict[str, dict[str, Any]] = json.loads(self.low_level_path.read_text())
        self.task_split: list[dict[str, Any]] = json.loads(self.task_split_path.read_text())
        self._instruction_annotation_index = {
            episode_id: path
            for path in sorted(self.instruction_annotations_dir.glob("*.json"))
            if (episode_id := _extract_episode_id(path.name))
        }
        self._instruction_annotation_cache: dict[str, dict[str, Any]] = {}
        self._screen_annotation_cache: dict[str, dict[str, Any]] = {}

    def _load_instruction_annotation(self, episode_id: str) -> dict[str, Any] | None:
        if episode_id in self._instruction_annotation_cache:
            return self._instruction_annotation_cache[episode_id]
        path = self._instruction_annotation_index.get(episode_id)
        if path is None:
            return None
        payload = json.loads(path.read_text())
        self._instruction_annotation_cache[episode_id] = payload
        return payload

    def _load_screen_annotation(self, image_name: str) -> dict[str, Any] | None:
        if image_name in self._screen_annotation_cache:
            return self._screen_annotation_cache[image_name]
        path = self.element_annotations_dir / f"{Path(image_name).stem}.json"
        if not path.exists():
            return None
        payload = json.loads(path.read_text())
        self._screen_annotation_cache[image_name] = payload
        return payload

    def get_episode(self, episode_id: str) -> DatasetEpisode:
        if episode_id not in self.low_level_episodes:
            raise KeyError(f"Unknown LearnGUI episode_id: {episode_id}")
        low_level_episode = self.low_level_episodes[episode_id]
        annotation_episode = self._load_instruction_annotation(episode_id) or {}
        annotation_steps = {
            int(step_payload["step_id"]): step_payload
            for step_payload in annotation_episode.get("steps", [])
        }

        low_steps = low_level_episode.get("steps", [])
        built_steps: list[DatasetStep] = []
        for index, step_payload in enumerate(low_steps):
            image_name = step_payload["image_path"]
            screenshot = screenshot_from_path(self.screenshots_dir / image_name)
            next_screenshot = None
            next_metadata = None
            if index + 1 < len(low_steps):
                next_low_step = low_steps[index + 1]
                next_screenshot = screenshot_from_path(self.screenshots_dir / next_low_step["image_path"])
                next_annotation = annotation_steps.get(int(next_low_step["step_id"]), {})
                next_metadata = {
                    "step_id": next_low_step["step_id"],
                    "action_description": next_low_step.get("action_description"),
                    "annotation_action": next_annotation.get("action"),
                    "package_name": next_annotation.get("package_name"),
                    "screen_annotation": self._load_screen_annotation(next_low_step["image_path"]),
                }

            annotation_step = annotation_steps.get(int(step_payload["step_id"]), {})
            metadata = {
                "benchmark": "LearnGUI",
                "representation": "screenshot-centric",
                "app": low_level_episode.get("app"),
                "domain": low_level_episode.get("domain"),
                "split": low_level_episode.get("split"),
                "step_id": step_payload["step_id"],
                "action_description": step_payload.get("action_description"),
                "image_path": image_name,
                "raw_low_level_action": step_payload["action"],
                "annotation_action": annotation_step.get("action"),
                "touch_coord": annotation_step.get("touch_coord"),
                "lift_coord": annotation_step.get("lift_coord"),
                "device_dim": annotation_step.get("device_dim"),
                "package_name": annotation_step.get("package_name"),
                "type_text": annotation_step.get("type_text"),
                "need_human_check": annotation_step.get("need_human_check"),
                "interest_region": annotation_step.get("interest_region"),
                "screen_annotation": self._load_screen_annotation(image_name),
            }
            raw_action = {
                "low_level_action": step_payload["action"],
                "annotation_action": annotation_step.get("action"),
                "touch_coord": annotation_step.get("touch_coord"),
                "lift_coord": annotation_step.get("lift_coord"),
                "type_text": annotation_step.get("type_text"),
            }
            built_steps.append(
                DatasetStep(
                    dataset="LearnGUI",
                    episode_id=episode_id,
                    step_index=index,
                    goal=low_level_episode["instruction"],
                    screenshot=screenshot,
                    raw_action=raw_action,
                    metadata=metadata,
                    next_screenshot=next_screenshot,
                    next_metadata=next_metadata,
                )
            )

        return DatasetEpisode(
            dataset="LearnGUI",
            episode_id=episode_id,
            goal=low_level_episode["instruction"],
            steps=tuple(built_steps),
            split=low_level_episode.get("split"),
            app=low_level_episode.get("app"),
            domain=low_level_episode.get("domain"),
            metadata={
                "instruction_annotation_path": str(self._instruction_annotation_index.get(episode_id, "")),
                "sample_count": len(built_steps),
            },
        )

    def iter_episodes(
        self,
        *,
        split: str | None = None,
        app: str | None = None,
        limit: int | None = None,
    ) -> Iterator[DatasetEpisode]:
        emitted = 0
        for episode_id in sorted(self.low_level_episodes):
            payload = self.low_level_episodes[episode_id]
            if split is not None and payload.get("split") != split:
                continue
            if app is not None and payload.get("app") != app:
                continue
            yield self.get_episode(episode_id)
            emitted += 1
            if limit is not None and emitted >= limit:
                return

    def iter_tasks(
        self,
        *,
        split: str | None = None,
        k_shot: int | None = None,
        limit: int | None = None,
    ) -> Iterator[LearnGUITask]:
        emitted = 0
        for task_payload in self.task_split:
            if split is not None and task_payload.get("split") != split:
                continue
            if k_shot is not None and int(task_payload.get("k_shot", -1)) != k_shot:
                continue
            yield LearnGUITask(
                app=task_payload["app"],
                split=task_payload["split"],
                k_shot=int(task_payload["k_shot"]),
                query_episode_id=task_payload["query"]["episode_id"],
                support_episode_ids=tuple(item["episode"]["episode_id"] for item in task_payload["support"]),
                metadata=task_payload,
            )
            emitted += 1
            if limit is not None and emitted >= limit:
                return
