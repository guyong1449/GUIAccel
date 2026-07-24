#!/usr/bin/env python3
"""Experiment: Coordinate regression head for GUI agent decode acceleration.

Three phases:
    extract  — Run Qwen3-VL on AndroidControl, collect (hidden_state, gt_coord) pairs
    train    — Fit a 2-layer MLP on the extracted data
    eval     — Compare regression-head coordinates against autoregressive decode

Usage:
    # Phase 1 — extract hidden states (needs 1 GPU, ~30 GB VRAM)
    python experiments/regression_head.py --mode extract \
        --output-dir outputs/regression_head \
        --split train --episode-limit 200

    # Phase 2 — train regression MLP (CPU is fine)
    python experiments/regression_head.py --mode train \
        --output-dir outputs/regression_head

    # Phase 3 — evaluate
    python experiments/regression_head.py --mode eval \
        --output-dir outputs/regression_head \
        --split test --episode-limit 100
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ── Phase 1: Extract ─────────────────────────────────────────────────────────

def _extract_worker(
    rank: int,
    num_gpus: int,
    episodes_for_worker: list,
    model_path: str,
    dtype: str,
    extraction_layer: int,
    max_new_tokens: int,
    output_dir: Path,
    split: str,
    thinking_mode: str,
    extract_point: str,
) -> None:
    """Worker function for multi-GPU extraction. Each worker loads its own model."""
    import torch
    from guiaccel.model.hidden_state_extractor import (
        COORD_ACTION_TYPES,
        extract_hidden_state,
        load_model_for_extraction,
        save_extracted_samples,
    )

    device = rank
    worker_output = output_dir / f"worker_{rank}"
    worker_output.mkdir(parents=True, exist_ok=True)

    print(
        f"[GPU {rank}] Loading model on cuda:{device} ... "
        f"(thinking_mode={thinking_mode}, extract_point={extract_point})",
        flush=True,
    )
    model, processor, _ = load_model_for_extraction(
        model_path, device=device, dtype=dtype,
    )
    print(f"[GPU {rank}] Model loaded. Processing {len(episodes_for_worker)} episodes.")

    samples = []
    step_count = 0
    skip_count = 0

    for ep_idx, episode in enumerate(episodes_for_worker):
        for step in episode.steps:
            action_type = str(step.raw_action.get("action_type", "")).lower()
            if action_type not in COORD_ACTION_TYPES:
                skip_count += 1
                continue

            step_count += 1
            t0 = time.perf_counter()
            sample = extract_hidden_state(
                model, processor, step,
                layer=extraction_layer,
                max_new_tokens=max_new_tokens,
                thinking_mode=thinking_mode,
                extract_point=extract_point,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            if sample is None:
                print(f"  [GPU {rank}][SKIP] ep={episode.episode_id} step={step.step_index}")
                continue

            samples.append(sample)
            multi_note = ""
            if sample.extract_point == "multi":
                multi_note = (
                    f" lens=te{sample.prefix_token_len_thinking_end}/"
                    f"act{sample.prefix_token_len_action}/"
                    f"cb{sample.prefix_token_len_coord_bracket}"
                )
            print(
                f"  [GPU {rank}][{len(samples):4d}] ep={sample.episode_id} "
                f"step={sample.step_index} type={sample.action_type} "
                f"gt=({sample.gt_coord_999[0]},{sample.gt_coord_999[1]}) "
                f"gen_tokens={sample.generated_tokens} "
                f"h_dim={sample.hidden_state.shape[0]}{multi_note} {elapsed_ms:.0f}ms"
            )

            if len(samples) % 200 == 0:
                ckpt_path = worker_output / f"checkpoint_{len(samples)}.pt"
                save_extracted_samples(samples, ckpt_path)

        if (ep_idx + 1) % 10 == 0:
            print(f"  [GPU {rank}] Progress: {ep_idx + 1}/{len(episodes_for_worker)} episodes, "
                  f"{len(samples)} samples extracted")

    if samples:
        final_path = worker_output / f"{split}_hidden_states.pt"
        save_extracted_samples(samples, final_path)

    summary = {
        "rank": rank,
        "episodes_processed": len(episodes_for_worker),
        "coord_steps": step_count,
        "extracted_samples": len(samples),
        "skipped_non_coord": skip_count,
        "thinking_mode": thinking_mode,
        "extract_point": extract_point,
    }
    (worker_output / "worker_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[GPU {rank}] Done: {len(samples)} samples from {len(episodes_for_worker)} episodes")


def run_extract(args: argparse.Namespace) -> None:
    """Run Qwen3-VL on AndroidControl and save (hidden_state, gt_coord) pairs."""
    import torch
    from guiaccel.data import AndroidControlDataset
    from guiaccel.model.hidden_state_extractor import (
        COORD_ACTION_TYPES,
        extract_hidden_state,
        load_model_for_extraction,
        load_extracted_samples,
        save_extracted_samples,
        worker_meta_list,
    )

    output_dir = Path(args.output_dir) / "extracted"
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = os.environ.get(
        "GUIACCEL_BASE_MODEL_PATH",
        str(PROJECT_ROOT / "models" / "Qwen3-VL-8B-Instruct"),
    )
    thinking_mode = getattr(args, "thinking_mode", "template")
    extract_point = getattr(args, "extract_point", "action")

    num_gpus = args.num_gpus
    if num_gpus > 1:
        print(f"Multi-GPU extraction: {num_gpus} GPUs")
        print(f"thinking_mode={thinking_mode} extract_point={extract_point}")
        print(f"Loading AndroidControl ({args.split} split) episode list ...")
        dataset = AndroidControlDataset()
        all_episodes = list(dataset.iter_episodes(split=args.split, limit=args.episode_limit))
        print(f"Total episodes: {len(all_episodes)}")

        # Round-robin partition episodes across GPUs
        per_gpu = [[] for _ in range(num_gpus)]
        for i, ep in enumerate(all_episodes):
            per_gpu[i % num_gpus].append(ep)
        for rank in range(num_gpus):
            print(f"  GPU {rank}: {len(per_gpu[rank])} episodes")

        import torch.multiprocessing as mp
        mp.set_start_method("spawn", force=True)
        processes = []
        for rank in range(num_gpus):
            p = mp.Process(
                target=_extract_worker,
                args=(
                    rank, num_gpus, per_gpu[rank],
                    model_path, args.dtype, args.extraction_layer,
                    args.max_new_tokens, output_dir, args.split,
                    thinking_mode, extract_point,
                ),
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        failed = [i for i, p in enumerate(processes) if p.exitcode != 0]
        if failed:
            print(f"ERROR: Workers on GPU(s) {failed} failed!")
            sys.exit(1)

        # Merge worker outputs (workers write ``meta``; merge into ``metadata``)
        print("Merging worker outputs ...")
        hs_chunks: list = []
        gt_chunks: list = []
        h_te_chunks: list = []
        h_act_chunks: list = []
        h_cb_chunks: list = []
        merged_meta: list = []
        total_samples = 0
        multi_point = extract_point == "multi"
        for rank in range(num_gpus):
            worker_dir = output_dir / f"worker_{rank}"
            worker_file = worker_dir / f"{args.split}_hidden_states.pt"
            if worker_file.exists():
                data = load_extracted_samples(worker_file)
                hs_chunks.append(data["hidden_states"])
                gt_chunks.append(data["gt_coords_norm"])
                chunk_meta = worker_meta_list(data)
                merged_meta.extend(chunk_meta)
                total_samples += data["hidden_states"].shape[0]
                if multi_point:
                    for key, bucket in (
                        ("h_thinking_end", h_te_chunks),
                        ("h_action", h_act_chunks),
                        ("h_coord_bracket", h_cb_chunks),
                    ):
                        if key not in data:
                            print(f"ERROR: worker {rank} missing {key}")
                            sys.exit(1)
                        bucket.append(data[key])
                print(
                    f"  GPU {rank}: {data['hidden_states'].shape[0]} samples "
                    f"(meta={len(chunk_meta)}, multi={bool(data.get('multi_point'))})"
                )

        if total_samples > 0:
            if len(merged_meta) != total_samples:
                print(
                    f"ERROR: merged metadata length {len(merged_meta)} != N={total_samples}"
                )
                sys.exit(1)
            merged_data = {
                "hidden_states": torch.cat(hs_chunks, dim=0),
                "gt_coords_norm": torch.cat(gt_chunks, dim=0),
                "metadata": merged_meta,
                "meta": merged_meta,
            }
            if multi_point:
                merged_data["h_thinking_end"] = torch.cat(h_te_chunks, dim=0)
                merged_data["h_action"] = torch.cat(h_act_chunks, dim=0)
                merged_data["h_coord_bracket"] = torch.cat(h_cb_chunks, dim=0)
                merged_data["multi_point"] = True
            final_path = output_dir / f"{args.split}_hidden_states.pt"
            torch.save(merged_data, str(final_path))
            print(
                f"Merged {total_samples} samples → {final_path} "
                f"(len(metadata)={len(merged_meta)}, multi_point={multi_point})"
            )

        summary = {
            "split": args.split,
            "num_gpus": num_gpus,
            "total_episodes": len(all_episodes),
            "total_samples": total_samples,
            "metadata_len": len(merged_meta),
            "model_path": model_path,
            "extraction_layer": args.extraction_layer,
            "thinking_mode": thinking_mode,
            "extract_point": extract_point,
            "multi_point": multi_point,
        }
        (output_dir / "extraction_summary.json").write_text(json.dumps(summary, indent=2))
        print(f"\n=== Multi-GPU extraction complete ===")
        print(json.dumps(summary, indent=2))
        return

    # Single-GPU path
    print(f"Loading model from {model_path} on device {args.device} ...")
    print(f"thinking_mode={thinking_mode} extract_point={extract_point}")
    model, processor, _ = load_model_for_extraction(
        model_path, device=args.device, dtype=args.dtype,
    )
    print("Model loaded.")

    print(f"Loading AndroidControl ({args.split} split) ...")
    dataset = AndroidControlDataset()
    samples = []
    episode_count = 0
    step_count = 0
    skip_count = 0

    for episode in dataset.iter_episodes(split=args.split, limit=args.episode_limit):
        episode_count += 1
        for step in episode.steps:
            action_type = str(step.raw_action.get("action_type", "")).lower()
            if action_type not in COORD_ACTION_TYPES:
                skip_count += 1
                continue

            step_count += 1
            t0 = time.perf_counter()
            sample = extract_hidden_state(
                model, processor, step,
                layer=args.extraction_layer,
                max_new_tokens=args.max_new_tokens,
                thinking_mode=thinking_mode,
                extract_point=extract_point,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            if sample is None:
                print(f"  [SKIP] ep={episode.episode_id} step={step.step_index} "
                      f"action={action_type} — extraction failed")
                continue

            samples.append(sample)
            multi_note = ""
            if sample.extract_point == "multi":
                multi_note = (
                    f" lens=te{sample.prefix_token_len_thinking_end}/"
                    f"act{sample.prefix_token_len_action}/"
                    f"cb{sample.prefix_token_len_coord_bracket}"
                )
            print(
                f"  [{len(samples):4d}] ep={sample.episode_id} step={sample.step_index} "
                f"type={sample.action_type} "
                f"gt=({sample.gt_coord_999[0]},{sample.gt_coord_999[1]}) "
                f"gen_tokens={sample.generated_tokens} "
                f"h_dim={sample.hidden_state.shape[0]}{multi_note} "
                f"{elapsed_ms:.0f}ms"
            )

            if len(samples) % 200 == 0:
                ckpt_path = output_dir / f"checkpoint_{len(samples)}.pt"
                save_extracted_samples(samples, ckpt_path)

        if args.episode_limit and episode_count >= args.episode_limit:
            break

    if samples:
        final_path = output_dir / f"{args.split}_hidden_states.pt"
        save_extracted_samples(samples, final_path)

    summary = {
        "split": args.split,
        "episodes_processed": episode_count,
        "total_steps_seen": step_count + skip_count,
        "coord_steps": step_count,
        "extracted_samples": len(samples),
        "skipped_non_coord": skip_count,
        "extraction_layer": args.extraction_layer,
        "model_path": model_path,
        "thinking_mode": thinking_mode,
        "extract_point": extract_point,
        "multi_point": extract_point == "multi",
        "metadata_len": len(samples),
    }
    summary_path = output_dir / "extraction_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\n=== Extraction complete ===")
    print(json.dumps(summary, indent=2))


# ── Phase 2: Train ───────────────────────────────────────────────────────────

def run_train(args: argparse.Namespace) -> None:
    """Train the CoordRegressionHead MLP on extracted data."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset, random_split

    from guiaccel.model.coord_head import (
        CoordHeadTrainConfig,
        CoordRegressionHead,
        compute_loss,
        save_coord_head,
    )
    from guiaccel.model.hidden_state_extractor import (
        load_extracted_samples,
        select_hidden_states_for_point,
    )

    output_dir = Path(args.output_dir) / "trained"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load extracted data
    extract_dir = Path(args.output_dir) / "extracted"
    data_path = extract_dir / f"{args.train_split}_hidden_states.pt"
    if not data_path.exists():
        # Try to find any .pt file
        candidates = sorted(extract_dir.glob("*.pt"))
        if not candidates:
            print(f"ERROR: No extracted data found in {extract_dir}")
            return
        data_path = candidates[-1]
        print(f"Using extracted data: {data_path}")

    train_point = getattr(args, "extract_point", "action")
    if train_point == "multi":
        print(
            "ERROR: --extract-point multi is extract-only; "
            "pick thinking_end|action|coord_bracket for train"
        )
        sys.exit(1)

    data = load_extracted_samples(data_path)
    hidden_states = select_hidden_states_for_point(data, train_point).float()
    gt_coords = data["gt_coords_norm"].float()      # (N, 2)
    # Drop AndroidControl annotation outliers with coords outside the screenshot.
    in_bounds = (gt_coords >= 0).all(dim=1) & (gt_coords <= 1).all(dim=1)
    n_oob = int((~in_bounds).sum())
    if n_oob:
        print(f"Dropping {n_oob} OOB GT samples (coords outside [0,1])")
        hidden_states = hidden_states[in_bounds]
        gt_coords = gt_coords[in_bounds]
    N = hidden_states.shape[0]
    input_dim = hidden_states.shape[1]
    print(f"Loaded {N} samples, input_dim={input_dim}, extract_point={train_point}")
    print(f"  multi_point_artifact={bool(data.get('multi_point'))}")
    print(f"  h mean={hidden_states.mean():.4f} std={hidden_states.std():.4f} "
          f"absmax={hidden_states.abs().max():.2f}")
    print(f"  gt mean={gt_coords.mean(0).tolist()} std={gt_coords.std(0).tolist()}")

    config = CoordHeadTrainConfig(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        smooth_l1_beta=args.smooth_l1_beta,
        val_fraction=args.val_fraction,
        seed=args.seed,
        grad_clip=getattr(args, "grad_clip", 1.0),
    )

    # Split into train / val
    torch.manual_seed(config.seed)
    full_dataset = TensorDataset(hidden_states, gt_coords)
    val_size = max(1, int(N * config.val_fraction))
    train_size = N - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)

    print(f"Train: {train_size}, Val: {val_size}")
    print(f"Config: hidden_dim={config.hidden_dim}, lr={config.lr}, "
          f"epochs={config.epochs}, beta={config.smooth_l1_beta}")

    # Build model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CoordRegressionHead(
        input_dim=config.input_dim,
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
    ).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay,
    )

    # Training loop
    best_val_loss = float("inf")
    best_epoch = -1
    patience_counter = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, config.epochs + 1):
        # Train
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for h_batch, gt_batch in train_loader:
            h_batch = h_batch.to(device)
            gt_batch = gt_batch.to(device)
            pred = model(h_batch)
            loss = compute_loss(pred, gt_batch, beta=config.smooth_l1_beta)
            optimizer.zero_grad()
            loss.backward()
            # Prevent Sigmoid saturation from a single oversized Adam step.
            if config.grad_clip and config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            train_loss_sum += loss.item() * h_batch.shape[0]
            train_count += h_batch.shape[0]

        train_loss = train_loss_sum / max(1, train_count)

        # Validate
        model.eval()
        val_loss_sum = 0.0
        val_mae_sum = 0.0
        val_count = 0
        with torch.no_grad():
            for h_batch, gt_batch in val_loader:
                h_batch = h_batch.to(device)
                gt_batch = gt_batch.to(device)
                pred = model(h_batch)
                loss = compute_loss(pred, gt_batch, beta=config.smooth_l1_beta)
                val_loss_sum += loss.item() * h_batch.shape[0]
                # MAE in 0-999 scale
                mae_999 = (pred - gt_batch).abs() * 999.0
                val_mae_sum += mae_999.sum().item()
                val_count += h_batch.shape[0]

        val_loss = val_loss_sum / max(1, val_count)
        val_mae = val_mae_sum / max(1, val_count * 2)  # average over x and y

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_mae_999": val_mae,
        }
        history.append(record)

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            save_coord_head(
                model, config,
                output_dir / "coord_head_best.pth",
                extra_meta={
                    "best_epoch": best_epoch,
                    "best_val_loss": best_val_loss,
                    "extract_point": train_point,
                },
            )
        else:
            patience_counter += 1

        marker = " *" if improved else ""
        print(
            f"  Epoch {epoch:3d}/{config.epochs} | "
            f"train_loss={train_loss:.6f} | "
            f"val_loss={val_loss:.6f} | "
            f"val_MAE@999={val_mae:.1f}{marker}"
        )

        if patience_counter >= config.patience:
            print(f"  Early stopping at epoch {epoch} (patience={config.patience})")
            break

    # Save final model + history
    save_coord_head(
        model, config,
        output_dir / "coord_head_final.pth",
        extra_meta={"final_epoch": epoch, "best_epoch": best_epoch},
    )
    history_path = output_dir / "training_history.json"
    history_path.write_text(json.dumps(history, indent=2))

    summary = {
        "total_samples": N,
        "train_size": train_size,
        "val_size": val_size,
        "param_count": param_count,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "final_epoch": epoch,
        "extract_point": train_point,
        "thinking_mode": getattr(args, "thinking_mode", "template"),
        "config": {
            "input_dim": config.input_dim,
            "hidden_dim": config.hidden_dim,
            "lr": config.lr,
            "epochs": config.epochs,
            "dropout": config.dropout,
        },
    }
    (output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n=== Training complete ===")
    print(json.dumps(summary, indent=2))


# ── Phase 3: Decode Eval (AR generate vs CoordHead) ─────────────────────────

def _decode_eval_worker(
    rank: int,
    episodes_for_worker: list,
    model_path: str,
    head_path: str,
    dtype: str,
    extraction_layer: int,
    max_new_tokens: int,
    output_dir: Path,
    split: str,
    thinking_mode: str = "template",
    extract_point: str = "action",
) -> None:
    """Per-GPU worker: AR decode + regression head on assigned episodes."""
    import torch
    from guiaccel.model.coord_decode_eval import (
        autoregressive_decode_coords,
        mae_999,
        regression_predict_coords,
    )
    from guiaccel.model.coord_head import load_coord_head
    from guiaccel.model.hidden_state_extractor import (
        COORD_ACTION_TYPES,
        load_model_for_extraction,
    )

    worker_output = output_dir / f"worker_{rank}"
    worker_output.mkdir(parents=True, exist_ok=True)

    print(
        f"[GPU {rank}] Loading VLM on cuda:{rank} ... "
        f"(thinking_mode={thinking_mode}, extract_point={extract_point})",
        flush=True,
    )
    torch.cuda.set_device(rank)
    model, processor, _ = load_model_for_extraction(
        model_path, device=rank, dtype=dtype,
    )
    device = torch.device(f"cuda:{rank}")
    coord_head, _ = load_coord_head(head_path, device=device)
    print(f"[GPU {rank}] Ready. episodes={len(episodes_for_worker)}", flush=True)

    results: list[dict] = []
    for ep_idx, episode in enumerate(episodes_for_worker):
        for step in episode.steps:
            action_type = str(step.raw_action.get("action_type", "")).lower()
            if action_type not in COORD_ACTION_TYPES:
                continue

            ar = autoregressive_decode_coords(
                model, processor, step,
                max_new_tokens=max_new_tokens,
            )
            if not ar.get("ok"):
                continue

            reg = regression_predict_coords(
                model, processor, coord_head, step,
                layer=extraction_layer,
                device=device,
                thinking_mode=thinking_mode,
                extract_point=extract_point,
            )
            if not reg.get("ok"):
                print(
                    f"  [GPU {rank}][SKIP-REG] ep={episode.episode_id} "
                    f"step={step.step_index}",
                    flush=True,
                )
                continue

            gt_999 = ar["gt_999"]
            row = {
                **{k: v for k, v in ar.items() if k != "raw_output"},
                "raw_output": ar.get("raw_output"),
                "reg_999": reg["reg_999"],
                "reg_abs": reg["reg_abs"],
                "reg_extract_ms": reg["reg_extract_ms"],
                "reg_head_ms": reg["reg_head_ms"],
                "reg_ms": reg["reg_ms"],
                "reg_prefix_tokens": reg["reg_prefix_tokens"],
                "reg_hit_bbox": reg["reg_hit_bbox"],
                "mae_reg_vs_gt": mae_999(reg["reg_999"], gt_999),
                "mae_ar_vs_gt": (
                    mae_999(ar["ar_999"], gt_999) if ar.get("ar_999") else None
                ),
                "mae_reg_vs_ar": (
                    mae_999(reg["reg_999"], ar["ar_999"])
                    if ar.get("ar_999") else None
                ),
                "latency_ratio_reg_over_ar": (
                    round(reg["reg_ms"] / ar["ar_ms"], 4)
                    if ar["ar_ms"] > 0 else None
                ),
            }
            results.append(row)
            ar_s = (
                f"ar=({ar['ar_999'][0]},{ar['ar_999'][1]})"
                if ar.get("ar_999") else "ar=FAIL"
            )
            print(
                f"  [GPU {rank}][{len(results):4d}] ep={row['episode_id']} "
                f"step={row['step_index']} gt=({gt_999[0]},{gt_999[1]}) "
                f"{ar_s} reg=({row['reg_999'][0]},{row['reg_999'][1]}) "
                f"MAE_reg={row['mae_reg_vs_gt']:.1f} "
                f"ar_ms={row['ar_ms']:.0f} reg_ms={row['reg_ms']:.0f}",
                flush=True,
            )

            if len(results) % 50 == 0:
                ckpt = worker_output / f"checkpoint_{len(results)}.json"
                ckpt.write_text(json.dumps(results, indent=2))

        if (ep_idx + 1) % 10 == 0:
            print(
                f"  [GPU {rank}] Progress: {ep_idx + 1}/{len(episodes_for_worker)} "
                f"episodes, {len(results)} samples",
                flush=True,
            )

    out_path = worker_output / f"{split}_decode_eval.json"
    out_path.write_text(json.dumps(results, indent=2))
    summary = {
        "rank": rank,
        "episodes": len(episodes_for_worker),
        "samples": len(results),
    }
    (worker_output / "worker_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[GPU {rank}] Done: {len(results)} samples", flush=True)


def _summarize_decode_results(results: list[dict], *, split: str, num_gpus: int) -> dict:
    def _mean(xs: list[float]) -> float | None:
        return sum(xs) / len(xs) if xs else None

    def _pct(xs: list[bool | None]) -> float | None:
        vals = [x for x in xs if x is not None]
        return sum(1 for x in vals if x) / len(vals) if vals else None

    reg_mae = [r["mae_reg_vs_gt"] for r in results if r.get("mae_reg_vs_gt") is not None]
    ar_mae = [r["mae_ar_vs_gt"] for r in results if r.get("mae_ar_vs_gt") is not None]
    cross = [r["mae_reg_vs_ar"] for r in results if r.get("mae_reg_vs_ar") is not None]
    ar_ms = [r["ar_ms"] for r in results]
    reg_ms = [r["reg_ms"] for r in results]
    ratios = [
        r["latency_ratio_reg_over_ar"]
        for r in results
        if r.get("latency_ratio_reg_over_ar") is not None
    ]
    parse_ok = sum(1 for r in results if r.get("ar_parse_ok"))

    return {
        "split": split,
        "num_gpus": num_gpus,
        "total_samples": len(results),
        "ar_parse_ok_rate": parse_ok / len(results) if results else 0.0,
        "mae_reg_vs_gt_mean": _mean(reg_mae),
        "mae_reg_vs_gt_median": sorted(reg_mae)[len(reg_mae) // 2] if reg_mae else None,
        "mae_ar_vs_gt_mean": _mean(ar_mae),
        "mae_ar_vs_gt_median": sorted(ar_mae)[len(ar_mae) // 2] if ar_mae else None,
        "mae_reg_vs_ar_mean": _mean(cross),
        "reg_within_20": sum(1 for m in reg_mae if m <= 20) / len(reg_mae) if reg_mae else None,
        "reg_within_50": sum(1 for m in reg_mae if m <= 50) / len(reg_mae) if reg_mae else None,
        "ar_within_20": sum(1 for m in ar_mae if m <= 20) / len(ar_mae) if ar_mae else None,
        "ar_within_50": sum(1 for m in ar_mae if m <= 50) / len(ar_mae) if ar_mae else None,
        "reg_bbox_hit_rate": _pct([r.get("reg_hit_bbox") for r in results]),
        "ar_bbox_hit_rate": _pct([r.get("ar_hit_bbox") for r in results]),
        "ar_ms_mean": _mean(ar_ms),
        "reg_ms_mean": _mean(reg_ms),
        "latency_ratio_reg_over_ar_mean": _mean(ratios),
        "tokens_saved_estimate_mean": _mean([
            max(0, r["ar_gen_tokens"] - r["reg_prefix_tokens"])
            for r in results
            if r.get("ar_gen_tokens") is not None and r.get("reg_prefix_tokens") is not None
        ]),
    }


def run_eval(args: argparse.Namespace) -> None:
    """Full decode eval: AR generate vs GT-Forcing + CoordHead on AndroidControl."""
    from guiaccel.data import AndroidControlDataset

    output_dir = Path(args.output_dir) / "eval"
    output_dir.mkdir(parents=True, exist_ok=True)

    head_path = Path(args.output_dir) / "trained" / "coord_head_best.pth"
    if not head_path.exists():
        head_path = Path(args.output_dir) / "trained" / "coord_head_final.pth"
    if not head_path.exists():
        # Allow explicitly pointing at a train run via TRAINED_HEAD / symlink layout
        alt = Path(args.output_dir) / "coord_head_best.pth"
        if alt.exists():
            head_path = alt
        else:
            print("ERROR: No trained model found under output-dir/trained/")
            return

    model_path = os.environ.get(
        "GUIACCEL_BASE_MODEL_PATH",
        str(PROJECT_ROOT / "models" / "Qwen3-VL-8B-Instruct"),
    )
    num_gpus = max(1, int(args.num_gpus))
    split = args.split

    print(f"Decode eval: split={split} num_gpus={num_gpus}")
    print(f"Head: {head_path}")
    print(f"Model: {model_path}")

    print(f"Loading AndroidControl ({split}) episode list ...")
    dataset = AndroidControlDataset()
    all_episodes = list(dataset.iter_episodes(split=split, limit=args.episode_limit))
    print(f"Total episodes: {len(all_episodes)}")

    thinking_mode = getattr(args, "thinking_mode", "template")
    extract_point = getattr(args, "extract_point", "action")
    print(f"thinking_mode={thinking_mode} extract_point={extract_point}")

    if num_gpus == 1:
        _decode_eval_worker(
            0, all_episodes, model_path, str(head_path),
            args.dtype, args.extraction_layer, args.max_new_tokens,
            output_dir, split, thinking_mode, extract_point,
        )
        worker_file = output_dir / "worker_0" / f"{split}_decode_eval.json"
        results = json.loads(worker_file.read_text()) if worker_file.exists() else []
    else:
        per_gpu: list[list] = [[] for _ in range(num_gpus)]
        for i, ep in enumerate(all_episodes):
            per_gpu[i % num_gpus].append(ep)
        for rank in range(num_gpus):
            print(f"  GPU {rank}: {len(per_gpu[rank])} episodes")

        import torch.multiprocessing as mp
        mp.set_start_method("spawn", force=True)
        processes = []
        for rank in range(num_gpus):
            p = mp.Process(
                target=_decode_eval_worker,
                args=(
                    rank, per_gpu[rank], model_path, str(head_path),
                    args.dtype, args.extraction_layer, args.max_new_tokens,
                    output_dir, split, thinking_mode, extract_point,
                ),
            )
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
        failed = [i for i, p in enumerate(processes) if p.exitcode != 0]
        if failed:
            print(f"ERROR: Workers on GPU(s) {failed} failed!")
            sys.exit(1)

        results = []
        for rank in range(num_gpus):
            worker_file = output_dir / f"worker_{rank}" / f"{split}_decode_eval.json"
            if worker_file.exists():
                chunk = json.loads(worker_file.read_text())
                results.extend(chunk)
                print(f"  GPU {rank}: {len(chunk)} samples")

    summary = _summarize_decode_results(results, split=split, num_gpus=num_gpus)
    (output_dir / "results.json").write_text(json.dumps(results, indent=2))
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== Decode evaluation complete ===")
    print(json.dumps(summary, indent=2))


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Coordinate regression head experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--output-dir", required=True, help="Root output directory")
    parser.add_argument(
        "--mode", required=True,
        choices=("extract", "train", "eval"),
        help="extract | train | eval (AR decode vs CoordHead)",
    )

    # Shared
    parser.add_argument("--device", type=int, default=0, help="CUDA device index")
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"))
    parser.add_argument("--num-gpus", type=int, default=1,
                        help="Number of GPUs for parallel extract/eval")

    # Extract / Eval
    parser.add_argument("--split", default="train", help="Dataset split")
    parser.add_argument("--episode-limit", type=int, default=None, help="Max episodes to process")
    parser.add_argument("--extraction-layer", type=int, default=-1,
                        help="Transformer layer to extract from (-1 = last)")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument(
        "--thinking-mode",
        default="template",
        choices=("none", "template", "ar_cache"),
        help=(
            "GT-Forcing thinking prefix: template=E1-A (default), none=legacy no-think, "
            "ar_cache=E1-B/E3 (needs cached thinking_text; not for T1 extract)"
        ),
    )
    parser.add_argument(
        "--extract-point",
        default="action",
        choices=("thinking_end", "action", "coord_bracket", "multi"),
        help=(
            "Hidden-state position within the forced prefix (default: action). "
            "Use 'multi' for E2 one-forward extract of all three points "
            "(train/eval must then pick a single point)."
        ),
    )

    # Train
    parser.add_argument("--train-split", default="train", help="Split name for training data")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--smooth-l1-beta", type=float, default=0.01)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grad-clip", type=float, default=1.0,
                        help="Max grad norm; 0 disables clipping")

    args = parser.parse_args()

    if args.mode == "extract":
        run_extract(args)
    elif args.mode == "train":
        run_train(args)
    elif args.mode == "eval":
        run_eval(args)


if __name__ == "__main__":
    main()
