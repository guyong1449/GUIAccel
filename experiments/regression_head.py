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

def run_extract(args: argparse.Namespace) -> None:
    """Run Qwen3-VL on AndroidControl and save (hidden_state, gt_coord) pairs."""
    import torch
    from guiaccel.data import AndroidControlDataset
    from guiaccel.model.hidden_state_extractor import (
        COORD_ACTION_TYPES,
        extract_hidden_state,
        load_model_for_extraction,
        save_extracted_samples,
    )

    output_dir = Path(args.output_dir) / "extracted"
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = os.environ.get(
        "GUIACCEL_BASE_MODEL_PATH",
        str(PROJECT_ROOT / "models" / "Qwen3-VL-8B-Instruct"),
    )
    print(f"Loading model from {model_path} on device {args.device} ...")
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
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            if sample is None:
                print(f"  [SKIP] ep={episode.episode_id} step={step.step_index} "
                      f"action={action_type} — extraction failed")
                continue

            samples.append(sample)
            print(
                f"  [{len(samples):4d}] ep={sample.episode_id} step={sample.step_index} "
                f"type={sample.action_type} "
                f"gt=({sample.gt_coord_999[0]},{sample.gt_coord_999[1]}) "
                f"gen_tokens={sample.generated_tokens} "
                f"h_dim={sample.hidden_state.shape[0]} "
                f"{elapsed_ms:.0f}ms"
            )

            # Periodic checkpoint every 200 samples
            if len(samples) % 200 == 0:
                ckpt_path = output_dir / f"checkpoint_{len(samples)}.pt"
                save_extracted_samples(samples, ckpt_path)

        if args.episode_limit and episode_count >= args.episode_limit:
            break

    # Final save
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
    from guiaccel.model.hidden_state_extractor import load_extracted_samples

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

    data = load_extracted_samples(data_path)
    hidden_states = data["hidden_states"]   # (N, hidden_dim)
    gt_coords = data["gt_coords_norm"]      # (N, 2)
    N = hidden_states.shape[0]
    input_dim = hidden_states.shape[1]
    print(f"Loaded {N} samples, input_dim={input_dim}")

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
                extra_meta={"best_epoch": best_epoch, "best_val_loss": best_val_loss},
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


# ── Phase 3: Eval ────────────────────────────────────────────────────────────

def run_eval(args: argparse.Namespace) -> None:
    """Evaluate the trained regression head against autoregressive decode."""
    import torch
    from guiaccel.data import AndroidControlDataset
    from guiaccel.model.coord_head import load_coord_head
    from guiaccel.model.hidden_state_extractor import (
        COORD_ACTION_TYPES,
        extract_hidden_state,
        load_model_for_extraction,
    )

    output_dir = Path(args.output_dir) / "eval"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load regression head
    head_path = Path(args.output_dir) / "trained" / "coord_head_best.pth"
    if not head_path.exists():
        head_path = Path(args.output_dir) / "trained" / "coord_head_final.pth"
    if not head_path.exists():
        print(f"ERROR: No trained model found. Run --mode train first.")
        return

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    coord_head, meta = load_coord_head(head_path, device=device)
    print(f"Loaded regression head from {head_path}")

    # Load VLM for extraction
    model_path = os.environ.get(
        "GUIACCEL_BASE_MODEL_PATH",
        str(PROJECT_ROOT / "models" / "Qwen3-VL-8B-Instruct"),
    )
    print(f"Loading Qwen3-VL from {model_path} ...")
    model, processor, _ = load_model_for_extraction(
        model_path, device=args.device, dtype=args.dtype,
    )
    print("Model loaded.")

    # Evaluate
    dataset = AndroidControlDataset()
    results = []
    episode_count = 0

    for episode in dataset.iter_episodes(split=args.split, limit=args.episode_limit):
        episode_count += 1
        for step in episode.steps:
            action_type = str(step.raw_action.get("action_type", "")).lower()
            if action_type not in COORD_ACTION_TYPES:
                continue

            t0 = time.perf_counter()
            sample = extract_hidden_state(
                model, processor, step,
                layer=args.extraction_layer,
                max_new_tokens=args.max_new_tokens,
            )
            extract_ms = (time.perf_counter() - t0) * 1000.0

            if sample is None:
                continue

            # Regression prediction
            h = sample.hidden_state.to(device).unsqueeze(0)
            with torch.no_grad():
                pred_999 = coord_head.predict_999(h)[0].cpu().tolist()

            gt_999 = sample.gt_coord_999
            error_x = abs(pred_999[0] - gt_999[0])
            error_y = abs(pred_999[1] - gt_999[1])
            mae_999 = (error_x + error_y) / 2.0

            result = {
                "episode_id": sample.episode_id,
                "step_index": sample.step_index,
                "action_type": sample.action_type,
                "gt_999": list(gt_999),
                "pred_999": pred_999,
                "error_x": error_x,
                "error_y": error_y,
                "mae_999": mae_999,
                "generated_tokens": sample.generated_tokens,
                "extract_ms": round(extract_ms, 1),
            }
            results.append(result)
            print(
                f"  [{len(results):4d}] ep={sample.episode_id} step={sample.step_index} "
                f"gt=({gt_999[0]},{gt_999[1]}) pred=({pred_999[0]},{pred_999[1]}) "
                f"MAE={mae_999:.1f}"
            )

        if args.episode_limit and episode_count >= args.episode_limit:
            break

    # Summary statistics
    if results:
        all_mae = [r["mae_999"] for r in results]
        all_err_x = [r["error_x"] for r in results]
        all_err_y = [r["error_y"] for r in results]

        summary = {
            "split": args.split,
            "total_samples": len(results),
            "episodes_evaluated": episode_count,
            "mae_999_mean": sum(all_mae) / len(all_mae),
            "mae_999_median": sorted(all_mae)[len(all_mae) // 2],
            "mae_999_p90": sorted(all_mae)[int(len(all_mae) * 0.9)],
            "mae_999_max": max(all_mae),
            "error_x_mean": sum(all_err_x) / len(all_err_x),
            "error_y_mean": sum(all_err_y) / len(all_err_y),
            "within_20": sum(1 for m in all_mae if m <= 20) / len(all_mae),
            "within_50": sum(1 for m in all_mae if m <= 50) / len(all_mae),
        }
    else:
        summary = {"error": "No samples evaluated"}

    # Save
    results_path = output_dir / "results.json"
    results_path.write_text(json.dumps(results, indent=2))
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"\n=== Evaluation complete ===")
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
        help="extract: collect hidden states; train: fit MLP; eval: compare",
    )

    # Shared
    parser.add_argument("--device", type=int, default=0, help="CUDA device index")
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"))

    # Extract / Eval
    parser.add_argument("--split", default="train", help="Dataset split")
    parser.add_argument("--episode-limit", type=int, default=None, help="Max episodes to process")
    parser.add_argument("--extraction-layer", type=int, default=-1,
                        help="Transformer layer to extract from (-1 = last)")
    parser.add_argument("--max-new-tokens", type=int, default=512)

    # Train
    parser.add_argument("--train-split", default="train", help="Split name for training data")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--smooth-l1-beta", type=float, default=0.01)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.mode == "extract":
        run_extract(args)
    elif args.mode == "train":
        run_train(args)
    elif args.mode == "eval":
        run_eval(args)


if __name__ == "__main__":
    main()
