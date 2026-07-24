#!/usr/bin/env python3
"""Fill ViT sdpa DivPrune columns in visionzip_four_way_eval_comparison.md from eval JSON."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from extract_eval_comparison_metrics import (
    _float1,
    _int_fmt,
    _pct,
    _pct_pp,
    _rel_pct,
    delta_pp,
    delta_rel,
    extract_metrics,
    load_eval,
)

DEFAULT_DOC = Path("docs/plans/visionzip_four_way_eval_comparison.md")
SDP_SECTION = "# ViT sdpa, LLM sdpa"


def _null_cell(rate: float | None, count: int | None) -> str:
    if rate is None:
        return "—"
    pct = _pct(rate)
    if count is not None:
        return f"{pct} ({int(count)})"
    return pct


def _token_delta(baseline: float | int | None, value: float | int | None) -> str:
    if baseline is None or value is None:
        return "—"
    base = float(baseline)
    if base == 0:
        return "—"
    return _rel_pct((float(value) - base) / base)


def build_sdpa_section(baseline: dict, divprune: dict) -> str:
    b_steps = max(1, int(baseline["result_count"]))
    d_steps = max(1, int(divprune["result_count"]))

    b_vis_total = baseline["visual_tokens_total"]
    d_vis_total = divprune["visual_tokens_total"]
    b_tok_total = baseline["total_tokens"]
    d_tok_total = divprune["total_tokens"]

    vis_ret = d_vis_total / b_vis_total if b_vis_total else None
    tok_ret = d_tok_total / b_tok_total if b_tok_total else None

    subset_rows = []
    for name, label in (
        ("in_distribution", "in_distribution"),
        ("app_unseen", "app_unseen"),
        ("category_unseen", "category_unseen"),
        ("task_unseen", "task_unseen"),
        ("overall", "overall"),
    ):
        b_sub = baseline["subsets"][name]
        d_sub = divprune["subsets"][name]
        subset_rows.append(
            f"| {label} | {_pct(b_sub['hl'])} | {_pct(b_sub['ll'])} | "
            f"{_pct(d_sub['hl'])} | {_pct(d_sub['ll'])} | "
            f"{_pct_pp(delta_pp(b_sub['hl'], d_sub['hl']))} | {_pct_pp(delta_pp(b_sub['ll'], d_sub['ll']))} |"
        )

    return f"""{SDP_SECTION}

| | Baseline (20803) | DivPrune | Δ DivPrune |
| --- | ---: | ---: | ---: |
| Job | 20803 | sdpa full | — |
| Backend | local_transformers | divprune | — |
| ViT attn | sdpa | sdpa | — |
| LLM attn | sdpa | sdpa | — |
| Split / steps | test / {b_steps:,} | test / {d_steps:,} | — |

### Accuracy (overall)

| Metric | Baseline (20803) | DivPrune | Δ DivPrune |
| --- | ---: | ---: | ---: |
| HL step accuracy | {_pct(baseline['hl_step_accuracy'])} | {_pct(divprune['hl_step_accuracy'])} | {_pct_pp(delta_pp(baseline['hl_step_accuracy'], divprune['hl_step_accuracy']))} |
| LL step accuracy | {_pct(baseline['ll_step_accuracy'])} | {_pct(divprune['ll_step_accuracy'])} | {_pct_pp(delta_pp(baseline['ll_step_accuracy'], divprune['ll_step_accuracy']))} |
| HL episode accuracy | {_pct(baseline['hl_episode_accuracy'])} | {_pct(divprune['hl_episode_accuracy'])} | {_pct_pp(delta_pp(baseline['hl_episode_accuracy'], divprune['hl_episode_accuracy']))} |
| Null-action rate | {_null_cell(baseline['null_action_rate'], baseline['null_action_count'])} | {_null_cell(divprune['null_action_rate'], divprune['null_action_count'])} | {_pct_pp(delta_pp(baseline['null_action_rate'], divprune['null_action_rate']))} |

### Accuracy (per subset)

| Subset | HL step (20803) | LL step (20803) | HL step (DivPrune) | LL step (DivPrune) | Δ HL DivPrune | Δ LL DivPrune |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(subset_rows)}

### Tokens & compression

| Metric | Baseline (20803) | DivPrune | Δ DivPrune |
| --- | ---: | ---: | ---: |
| Visual tokens (total) | {_int_fmt(b_vis_total)} | {_int_fmt(d_vis_total)} | {_rel_pct(-(1 - vis_ret) if vis_ret is not None else None)} |
| Visual tokens / step (mean) | {_float1(baseline['visual_tokens_per_step'])} | {_float1(divprune['visual_tokens_per_step'])} | {_token_delta(baseline['visual_tokens_per_step'], divprune['visual_tokens_per_step'])} |
| Visual token compression | 0.0% | {_pct(1 - vis_ret if vis_ret is not None else None)} | — |
| Total tokens | {_int_fmt(b_tok_total)} | {_int_fmt(d_tok_total)} | {_rel_pct(-(1 - tok_ret) if tok_ret is not None else None)} |
| Total tokens / step (mean) | {_float1(baseline['total_tokens_per_step'])} | {_float1(divprune['total_tokens_per_step'])} | {_token_delta(baseline['total_tokens_per_step'], divprune['total_tokens_per_step'])} |
| Total token compression | 0.0% | {_pct(1 - tok_ret if tok_ret is not None else None)} | — |
| Input text tokens | {_int_fmt(baseline['input_text_tokens'])} | {_int_fmt(divprune['input_text_tokens'])} | {_token_delta(baseline['input_text_tokens'], divprune['input_text_tokens'])} |
| Output tokens | {_int_fmt(baseline['output_tokens'])} | {_int_fmt(divprune['output_tokens'])} | {_token_delta(baseline['output_tokens'], divprune['output_tokens'])} |
"""


def patch_comparison_doc(
    doc_path: Path,
    *,
    baseline_json: Path,
    divprune_json: Path,
) -> tuple[dict, dict]:
    baseline = extract_metrics(load_eval(baseline_json))
    divprune = extract_metrics(load_eval(divprune_json))
    new_section = build_sdpa_section(baseline, divprune)

    text = doc_path.read_text(encoding="utf-8")
    pattern = re.compile(
        rf"{re.escape(SDP_SECTION)}.*?(?=\n---\n\n# Eval JSON sources)",
        re.DOTALL,
    )
    if not pattern.search(text):
        raise SystemExit(f"Could not find sdpa section in {doc_path}")
    text = pattern.sub(new_section.rstrip() + "\n", text)

    divprune_path = str(divprune_json.resolve())
    text = re.sub(
        r"\| DivPrune ViT sdpa \|[^\n]+\n",
        f"| DivPrune ViT sdpa | — | `{divprune_path}` |\n",
        text,
    )
    doc_path.write_text(text, encoding="utf-8")
    print(f"Updated {doc_path} from {divprune_json}")
    return baseline, divprune


def embed_comparison_in_json(
    divprune_json: Path,
    *,
    baseline: dict,
    divprune: dict,
    baseline_json: Path,
) -> None:
    report = load_eval(divprune_json)
    report["four_way_comparison"] = {
        "doc": "docs/plans/visionzip_four_way_eval_comparison.md",
        "attn_config": "vit_sdpa_llm_sdpa",
        "baseline_eval_json": str(baseline_json.resolve()),
        "divprune_eval_json": str(divprune_json.resolve()),
        "baseline_job": 20803,
        "result_count": divprune["result_count"],
        "accuracy": {
            "hl_step_accuracy": divprune["hl_step_accuracy"],
            "ll_step_accuracy": divprune["ll_step_accuracy"],
            "hl_episode_accuracy": divprune["hl_episode_accuracy"],
            "null_action_rate": divprune["null_action_rate"],
            "null_action_count": divprune["null_action_count"],
            "delta_hl_step_pp": delta_pp(baseline["hl_step_accuracy"], divprune["hl_step_accuracy"]),
            "delta_ll_step_pp": delta_pp(baseline["ll_step_accuracy"], divprune["ll_step_accuracy"]),
        },
        "tokens": {
            "visual_tokens_total": divprune["visual_tokens_total"],
            "visual_tokens_per_step": divprune["visual_tokens_per_step"],
            "total_tokens": divprune["total_tokens"],
            "total_tokens_per_step": divprune["total_tokens_per_step"],
            "input_text_tokens": divprune["input_text_tokens"],
            "output_tokens": divprune["output_tokens"],
            "visual_token_retention_vs_baseline": (
                divprune["visual_tokens_total"] / baseline["visual_tokens_total"]
                if baseline["visual_tokens_total"]
                else None
            ),
        },
        "subsets": divprune["subsets"],
    }
    divprune_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Embedded four_way_comparison into {divprune_json}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--divprune-json",
        type=Path,
        required=True,
        help="DivPrune ViT sdpa androidcontrol_test_v0_evaluation.json",
    )
    parser.add_argument(
        "--baseline-json",
        type=Path,
        default=Path("/dkucc/home/rw335/SkillReuse/outputs/baseline_transformers/androidcontrol_test_v0_evaluation.json"),
        help="Baseline ViT sdpa eval JSON (job 20803)",
    )
    parser.add_argument(
        "--doc",
        type=Path,
        default=DEFAULT_DOC,
        help="Comparison markdown to update",
    )
    parser.add_argument(
        "--json-only",
        action="store_true",
        help="Only embed four_way_comparison in JSON, skip markdown update",
    )
    parser.add_argument(
        "--skip-json",
        action="store_true",
        help="Only update markdown, do not modify divprune eval JSON",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[3]
    doc_path = args.doc if args.doc.is_absolute() else repo_root / args.doc

    if not args.json_only:
        baseline, divprune = patch_comparison_doc(
            doc_path,
            baseline_json=args.baseline_json,
            divprune_json=args.divprune_json,
        )
    else:
        baseline = extract_metrics(load_eval(args.baseline_json))
        divprune = extract_metrics(load_eval(args.divprune_json))

    if not args.skip_json:
        embed_comparison_in_json(
            args.divprune_json,
            baseline=baseline,
            divprune=divprune,
            baseline_json=args.baseline_json,
        )


if __name__ == "__main__":
    main()
