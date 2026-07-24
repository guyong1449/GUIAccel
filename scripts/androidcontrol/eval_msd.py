"""Evaluate MSD-Qwen2-VL on AndroidControl TFRecords."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from guiaccel.data.android_control import AndroidControlDataset
from guiaccel.evaluation.android_eval import score_android_prediction
from guiaccel.model.msd_qwen2vl_backend import MSDQwen2VLBackend, MSDQwen2VLConfig
from guiaccel.routing.common import TokenUsage
from guiaccel.routing.fallback import FullModelRequest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--instruction-mode", choices=("high_level", "low_level", "both"), default="both")
    parser.add_argument("--episode-limit", type=int)
    parser.add_argument("--step-limit", type=int)
    parser.add_argument("--baseline", action="store_true", help="Use MSD naivegenerate instead of msdgenerate.")
    parser.add_argument("--dry-run", action="store_true", help="Load data and print planned requests without loading a model.")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    payload = json.loads(config_path.read_text())
    model_payload = dict(payload["model"])
    msd_config = MSDQwen2VLConfig(
        base_model_path=str(model_payload["base_model_path"]),
        draft_model_path=str(model_payload["draft_model_path"]),
        use_msd=not args.baseline,
        max_new_tokens=int(model_payload.get("max_new_tokens", 256)),
        max_pixels=int(model_payload.get("max_pixels", 1_000_000)),
        min_pixels=int(model_payload.get("min_pixels", 3_136)),
        total_token=int(model_payload.get("total_token", 59)),
        depth=int(model_payload.get("depth", 5)),
        top_k=int(model_payload.get("top_k", 10)),
        threshold=float(model_payload.get("threshold", 1.0)),
    )
    msd_config.validate()

    dataset = AndroidControlDataset(payload["dataset_manifest"])
    modes = ("high_level", "low_level") if args.instruction_mode == "both" else (args.instruction_mode,)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    results_path = output_dir / "results.jsonl"
    backend = None if args.dry_run else MSDQwen2VLBackend(msd_config, eager_load=True)

    total = 0
    matched = 0
    parse_failures = 0
    episode_matches: dict[str, list[bool]] = {}
    latency_sum_ms = 0.0
    generated_tokens = 0
    accepted_tokens = 0.0
    verification_rounds = 0.0
    peak_gpu_memory_bytes = 0
    mode_stats = {
        mode: {
            "steps": 0,
            "matched": 0,
            "parse_failures": 0,
            "latency_sum_ms": 0.0,
            "generated_tokens": 0,
        }
        for mode in modes
    }

    with results_path.open("w", encoding="utf-8") as sink:
        stop = False
        for episode in dataset.iter_episodes(split=args.split, limit=args.episode_limit):
            for step in episode.steps:
                for mode in modes:
                    if args.step_limit is not None and total >= args.step_limit:
                        stop = True
                        break
                    request = _build_request(step, instruction_mode=mode, max_new_tokens=msd_config.max_new_tokens)
                    if backend is None:
                        total += 1
                        response_payload = {
                            "episode_id": episode.episode_id,
                            "step_index": step.step_index,
                            "instruction_mode": mode,
                            "prompt_text": request.prompt_text,
                            "dry_run": True,
                        }
                    else:
                        response = backend.generate(request)
                        score = score_android_prediction(response.action, step)
                        is_match = bool(score["step_match"])
                        total += 1
                        matched += int(is_match)
                        parse_failures += int(response.action is None)
                        episode_matches.setdefault(f"{episode.episode_id}:{mode}", []).append(is_match)
                        latency_sum_ms += response.latency_ms
                        generated_tokens += response.token_usage.generated_tokens
                        timing = dict(response.timing or {})
                        accepted_tokens += float(timing.get("accepted_draft_tokens") or 0.0)
                        verification_rounds += float(timing.get("verification_rounds") or 0.0)
                        peak_gpu_memory_bytes = max(
                            peak_gpu_memory_bytes,
                            int(timing.get("peak_gpu_memory_bytes") or 0),
                        )
                        mode_stat = mode_stats[mode]
                        mode_stat["steps"] += 1
                        mode_stat["matched"] += int(is_match)
                        mode_stat["parse_failures"] += int(response.action is None)
                        mode_stat["latency_sum_ms"] += response.latency_ms
                        mode_stat["generated_tokens"] += response.token_usage.generated_tokens
                        response_payload = {
                            "episode_id": episode.episode_id,
                            "step_index": step.step_index,
                            "instruction_mode": mode,
                            "ground_truth": step.raw_action,
                            "prediction": asdict(response.action) if response.action is not None else None,
                            "step_match": is_match,
                            "latency_ms": response.latency_ms,
                            "token_usage": asdict(response.token_usage),
                            "timing": timing,
                            "raw_output": response.raw_output,
                        }
                    sink.write(json.dumps(response_payload, ensure_ascii=False) + "\n")
                    sink.flush()
                if stop:
                    break
            if stop:
                break

    summary: dict[str, Any] = {
        "benchmark": "AndroidControl",
        "method": "naive-Qwen2-VL" if args.baseline else "MSD-Qwen2-VL",
        "config": str(config_path),
        "split": args.split,
        "instruction_mode": args.instruction_mode,
        "dry_run": args.dry_run,
        "steps": total,
        "step_accuracy": matched / total if total else 0.0,
        "episode_accuracy": (
            sum(all(values) for values in episode_matches.values()) / len(episode_matches)
            if episode_matches
            else 0.0
        ),
        "parse_failure_rate": parse_failures / total if total else 0.0,
        "end_to_end_latency_ms": latency_sum_ms,
        "mean_end_to_end_latency_ms": latency_sum_ms / total if total else 0.0,
        "peak_gpu_memory_bytes": peak_gpu_memory_bytes,
        "generated_tokens": generated_tokens,
        "verification_rounds": verification_rounds,
        "accepted_draft_tokens": accepted_tokens,
        "accepted_tokens_per_round": (
            accepted_tokens / verification_rounds if verification_rounds else 0.0
        ),
        "by_instruction_mode": {
            mode: _summarize_mode(stats, episode_matches, mode=mode)
            for mode, stats in mode_stats.items()
        },
        "results_path": str(results_path),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _summarize_mode(
    stats: dict[str, float | int],
    episode_matches: dict[str, list[bool]],
    *,
    mode: str,
) -> dict[str, float | int]:
    steps = int(stats["steps"])
    episodes = [
        matches
        for key, matches in episode_matches.items()
        if key.endswith(f":{mode}")
    ]
    return {
        "steps": steps,
        "step_accuracy": float(stats["matched"]) / steps if steps else 0.0,
        "episode_accuracy": (
            sum(all(matches) for matches in episodes) / len(episodes)
            if episodes
            else 0.0
        ),
        "parse_failure_rate": (
            float(stats["parse_failures"]) / steps if steps else 0.0
        ),
        "end_to_end_latency_ms": float(stats["latency_sum_ms"]),
        "mean_end_to_end_latency_ms": (
            float(stats["latency_sum_ms"]) / steps if steps else 0.0
        ),
        "generated_tokens": int(stats["generated_tokens"]),
    }


def _build_request(step: Any, *, instruction_mode: str, max_new_tokens: int) -> FullModelRequest:
    instruction = (
        step.goal
        if instruction_mode == "high_level"
        else str(step.metadata.get("step_instruction") or step.goal)
    )
    history_types = tuple(step.metadata.get("history_action_types") or ())
    history_text = ", ".join(str(item) for item in history_types) if history_types else "none"
    prompt_text = (
        f"Task: {instruction}\n"
        f"Actions already performed: {history_text}\n"
        f"Current step: {step.step_index}"
    )
    return FullModelRequest(
        observation_id=f"{step.episode_id}:{step.step_index}:{instruction_mode}",
        reason="androidcontrol_msd_eval",
        benchmark="AndroidControl",
        screenshot=step.screenshot,
        prompt_text=prompt_text,
        history_length=len(history_types),
        support_context={},
        model_spec=None,
        temperature=0.0,
        top_p=1.0,
        max_new_tokens=max_new_tokens,
        repetition_penalty=1.0,
        image_max_pixels=1_000_000,
        estimated_token_usage=TokenUsage(),
    )


if __name__ == "__main__":
    main()
