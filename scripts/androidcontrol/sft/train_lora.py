"""Fine-tune MAI-UI-8B or another Qwen3-VL-compatible checkpoint with LoRA."""

from __future__ import annotations

import argparse
import io
import json
import os
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import sys
from pathlib import Path

for _parent in Path(__file__).resolve().parents:
    _lib = _parent / "_lib"
    if (_lib / "repo_path.py").is_file():
        sys.path.insert(0, str(_lib))
        break
else:
    raise RuntimeError(f"Could not locate scripts/_lib from {__file__}")
from repo_path import bootstrap

REPO_ROOT = bootstrap(Path(__file__))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from skillreuse.configuration import load_benchmark_config, resolve_backend_config
from skillreuse.data import AndroidControlDataset, LearnGUIDataset, canonicalize_step
from skillreuse.model.qwen_backend import QwenBackendConfig, resolve_visible_cuda_devices
from skillreuse.model.service_backend import ANDROIDCONTROL_MAIUI_SYSTEM_PROMPT
from skillreuse.routing.execution import StepContext
from skillreuse.routing.fallback import build_full_prompt
from skillreuse.types import CanonicalAction, DatasetEpisode, ScreenshotAsset

DEFAULT_LORA_TARGET_MODULES: tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
PROJECTOR_LINEAR_SUFFIXES: tuple[str, ...] = (
    "merger.linear_fc1",
    "merger.linear_fc2",
)
PROJECTOR_COLLECTION_MARKERS: tuple[str, ...] = (
    ".deepstack_merger_list.",
)
TRAINING_EXAMPLES_CACHE_FORMAT_VERSION = 2
TRAINING_EXAMPLES_CACHE_WAIT_TIMEOUT_SECONDS = 7200
TRAINING_EXAMPLES_CACHE_WAIT_POLL_SECONDS = 5.0
TRAINING_EXAMPLES_CACHE_STATUS_INTERVAL_SECONDS = 30.0


@dataclass(frozen=True)
class TrainingExample:
    """One supervised multimodal next-action example."""

    benchmark: str
    observation_id: str
    screenshot: ScreenshotAsset
    prompt_text: str
    response_text: str


class PromptSupervisionDataset:
    """Minimal indexable dataset wrapper for Trainer."""

    def __init__(self, examples: Sequence[TrainingExample]) -> None:
        self._examples = tuple(examples)

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, index: int) -> TrainingExample:
        return self._examples[index]


class QwenVLChatCollator:
    """Collate multimodal chat-supervision examples for MAI-UI/Qwen3-VL-compatible chat templates."""

    def __init__(
        self,
        *,
        processor: Any,
        max_length: int,
        max_image_pixels: int,
        system_prompt: str,
    ) -> None:
        self.processor = processor
        self.max_length = int(max_length)
        self.max_image_pixels = int(max_image_pixels)
        self.system_prompt = str(system_prompt)
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is not None:
            if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
                tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "right"
            tokenizer.model_max_length = int(max_length)

    def __call__(self, batch: Sequence[TrainingExample]) -> dict[str, Any]:
        prepared_messages = [self._messages_for_example(example) for example in batch]
        messages = [full_messages for full_messages, _ in prepared_messages]
        encoded = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        labels = encoded["input_ids"].new_full(encoded["input_ids"].shape, -100)
        attention_mask = encoded.get("attention_mask")
        for row_index, (_, prompt_messages) in enumerate(prepared_messages):
            prompt_token_ids = self._token_ids_for_messages(
                prompt_messages,
                add_generation_prompt=True,
            )
            sequence_length = self._sequence_length(encoded=encoded, row_index=row_index, attention_mask=attention_mask)
            prefix_length = min(len(prompt_token_ids), sequence_length)
            if tuple(int(token_id) for token_id in encoded["input_ids"][row_index, :prefix_length].tolist()) != prompt_token_ids[
                :prefix_length
            ]:
                raise ValueError("Assistant supervision prefix could not be aligned with the chat template output.")
            labels[row_index, prefix_length:sequence_length] = encoded["input_ids"][row_index, prefix_length:sequence_length]
        encoded["labels"] = labels
        return encoded

    def _messages_for_example(
        self,
        example: TrainingExample,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        image = _resize_image_to_budget(_load_image(example.screenshot), max_pixels=self.max_image_pixels)
        prompt_messages = [
            {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": example.prompt_text},
                ],
            },
        ]
        full_messages = [
            *prompt_messages,
            {
                "role": "assistant",
                "content": [{"type": "text", "text": example.response_text}],
            },
        ]
        return full_messages, prompt_messages

    def _token_ids_for_messages(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        add_generation_prompt: bool,
    ) -> tuple[int, ...]:
        encoded = self.processor.apply_chat_template(
            list(messages),
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            return_dict=True,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        input_ids = encoded["input_ids"]
        if getattr(input_ids, "ndim", 1) == 2:
            if int(input_ids.shape[0]) != 1:
                raise ValueError("Expected a single-example chat template encoding.")
            return tuple(int(token_id) for token_id in input_ids[0].tolist())
        return tuple(int(token_id) for token_id in input_ids.tolist())

    def _sequence_length(
        self,
        *,
        encoded: dict[str, Any],
        row_index: int,
        attention_mask: Any,
    ) -> int:
        if attention_mask is None:
            return int(encoded["input_ids"].shape[-1])
        return int(attention_mask[row_index].sum().item())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune MAI-UI-8B or another Qwen3-VL-compatible checkpoint with LoRA on LearnGUI or AndroidControl."
    )
    parser.add_argument("--benchmark", required=True, choices=("LearnGUI", "AndroidControl"))
    parser.add_argument("--config", help="Optional JSON config path. Defaults to configs/<benchmark>/default.json.")
    parser.add_argument("--dataset-manifest", help="Optional dataset manifest override.")
    parser.add_argument("--model-path", help="Optional MAI-UI-8B / Qwen3-VL-compatible base model override.")
    parser.add_argument(
        "--output-dir",
        help=(
            "Directory for the training run root or final adapter directory. "
            "If omitted, uses paths.expected_lora_adapter_output_path from the selected config."
        ),
    )
    parser.add_argument("--local-rank", "--local_rank", type=int, default=-1, help=argparse.SUPPRESS)
    parser.add_argument("--rank", type=int, default=16, help="LoRA rank. Default: 16.")
    parser.add_argument("--alpha", type=int, default=32, help="LoRA alpha. Default: 32.")
    parser.add_argument("--dropout", type=float, default=0.05, help="LoRA dropout. Default: 0.05.")
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="Learning rate. Default: 2e-4.")
    parser.add_argument(
        "--epochs",
        type=float,
        default=None,
        help="Training epochs. Default: LearnGUI=3, AndroidControl=2.",
    )
    parser.add_argument("--weight-decay", type=float, default=0.1, help="Weight decay. Default: 0.1.")
    parser.add_argument("--warmup-ratio", type=float, default=0.03, help="Warmup ratio. Default: 0.03.")
    parser.add_argument(
        "--max-text-tokens",
        type=int,
        default=4096,
        help="Maximum multimodal token length enforced by the processor. Default: 4096.",
    )
    parser.add_argument(
        "--max-image-pixels",
        type=int,
        default=602112,
        help="Maximum training image pixel budget after resize. Default: 602112 (UI-Genie recipe).",
    )
    parser.add_argument("--per-device-batch-size", type=int, default=1, help="Per-device batch size. Default: 1.")
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=8,
        help="Gradient accumulation steps. Default: 8.",
    )
    parser.add_argument("--logging-steps", type=int, default=10, help="Trainer logging interval. Default: 10.")
    parser.add_argument("--save-strategy", choices=("epoch", "steps", "no"), default="epoch")
    parser.add_argument("--save-steps", type=int, default=500, help="Checkpoint interval when save-strategy=steps.")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="Optional maximum optimizer steps. Overrides epochs when positive.",
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed. Default: 7.")
    parser.add_argument(
        "--example-cache-path",
        help=(
            "Optional cache file for prebuilt training examples. Defaults to "
            "<run_root>/<benchmark>_training_examples.pkl."
        ),
    )
    parser.add_argument(
        "--disable-example-cache",
        action="store_true",
        help="Disable the on-disk training-example cache and rebuild examples in every process.",
    )
    parser.add_argument(
        "--lora-target-modules",
        nargs="+",
        default=None,
        help=(
            "Optional override for the language-model LoRA target module suffixes. "
            "Projector layers are still auto-discovered and appended."
        ),
    )
    args = parser.parse_args()
    config = load_benchmark_config(benchmark=args.benchmark, config_path=args.config)
    backend_config = resolve_backend_config(config)
    epochs = float(args.epochs) if args.epochs is not None else _default_epochs_for_benchmark(args.benchmark)

    resolved_devices, visible_cuda_devices_source = _resolve_training_visible_cuda_devices(backend_config=backend_config)
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(device) for device in resolved_devices)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # No external internet — prevent transformers from hanging on HF downloads
    import os as _os
    _os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    _os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

    import peft
    import torch
    from peft import LoraConfig, TaskType
    from PIL import Image
    from transformers import AutoProcessor, Trainer, TrainingArguments, Qwen3VLForConditionalGeneration

    _require_min_version(
        package_name="transformers",
        version=getattr(__import__("transformers"), "__version__", "0.0.0"),
        minimum=(4, 57, 0),
        requirement="This MAI-UI-8B / Qwen3-VL-compatible training path requires transformers >= 4.57.0.",
    )
    _require_min_version(
        package_name="peft",
        version=getattr(peft, "__version__", "0.0.0"),
        minimum=(0, 17, 1),
        requirement=(
            "This training path requires a PEFT build with MAI-UI-8B / Qwen3-VL add_adapter support. "
            "Python 3.9 environments can satisfy that with peft >= 0.17.1."
        ),
    )

    paths = dict(config.get("paths") or {})
    dataset_manifest = str(Path(args.dataset_manifest).resolve()) if args.dataset_manifest else paths.get("dataset_manifest")
    base_model_path = str(Path(args.model_path).resolve()) if args.model_path else paths.get("base_model_path")
    run_root_dir, output_dir, checkpoints_dir = _resolve_output_paths(config, args.output_dir)

    if not dataset_manifest:
        raise ValueError("A dataset manifest path is required.")
    if not base_model_path:
        raise ValueError("A base model path is required.")

    _log_phase(
        "starting training setup",
        benchmark=args.benchmark,
        dataset_manifest=dataset_manifest,
        base_model_path=base_model_path,
        run_root_dir=str(run_root_dir),
        configured_visible_cuda_devices=list(backend_config.visible_cuda_devices),
        resolved_visible_cuda_devices=list(resolved_devices),
        visible_cuda_devices_source=visible_cuda_devices_source,
    )
    example_cache_path = _resolve_example_cache_path(
        run_root_dir=run_root_dir,
        benchmark=args.benchmark,
        explicit_cache_path=args.example_cache_path,
    )
    examples = _load_or_build_training_examples(
        benchmark=args.benchmark,
        dataset_manifest=dataset_manifest,
        cache_path=example_cache_path,
        disable_cache=bool(args.disable_example_cache),
    )
    if not examples:
        raise ValueError(f"No {args.benchmark} training examples were generated from {dataset_manifest}.")
    _log_phase(
        "training examples ready",
        benchmark=args.benchmark,
        example_count=len(examples),
        cache_path=str(example_cache_path) if example_cache_path is not None else "disabled",
    )

    _log_phase("loading processor", base_model_path=base_model_path)
    processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=False)
    attention_implementation = _resolve_attention_implementation()
    _log_phase(
        "loading base model",
        base_model_path=base_model_path,
        attention_implementation=attention_implementation or "eager",
    )
    model_load_kwargs = {
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": False,
        "low_cpu_mem_usage": True,
    }
    if attention_implementation is not None:
        model_load_kwargs["attn_implementation"] = attention_implementation
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        base_model_path,
        **model_load_kwargs,
    )
    model.config.use_cache = False
    _log_phase("base model loaded")

    projector_targets = _discover_projector_target_modules(model=model, torch_module=torch)
    language_targets = tuple(args.lora_target_modules or DEFAULT_LORA_TARGET_MODULES)
    target_modules = tuple(dict.fromkeys((*language_targets, *projector_targets)))
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=int(args.rank),
        lora_alpha=int(args.alpha),
        lora_dropout=float(args.dropout),
        target_modules=list(target_modules),
        bias="none",
    )
    model.add_adapter(lora_config, adapter_name="default")
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    train_dataset = PromptSupervisionDataset(examples)
    _log_phase(
        "adapter attached",
        language_target_count=len(language_targets),
        projector_target_count=len(projector_targets),
        total_target_count=len(target_modules),
    )

    backend_prompt = (
        ANDROIDCONTROL_MAIUI_SYSTEM_PROMPT
        if args.benchmark == "AndroidControl"
        else backend_config.full_model_system_prompt
    )
    collator = QwenVLChatCollator(
        processor=processor,
        max_length=int(args.max_text_tokens),
        max_image_pixels=int(args.max_image_pixels),
        system_prompt=backend_prompt,
    )

    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(checkpoints_dir),
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        max_steps=int(args.max_steps),
        per_device_train_batch_size=int(args.per_device_batch_size),
        gradient_accumulation_steps=int(args.gradient_accumulation_steps),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        warmup_ratio=float(args.warmup_ratio),
        bf16=True,
        fp16=False,
        logging_steps=int(args.logging_steps),
        logging_first_step=True,
        save_strategy=str(args.save_strategy),
        save_steps=int(args.save_steps),
        save_total_limit=2,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        report_to=[],
        seed=int(args.seed),
        gradient_checkpointing=True,
        ddp_find_unused_parameters=False,
        save_on_each_node=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
        processing_class=processor,
    )
    _log_phase("entering trainer.train()", max_steps=int(args.max_steps), epochs=epochs)
    trainer.train()
    _wait_for_everyone(trainer)
    _log_phase("trainer.train() completed")

    manifest_payload = {
        "benchmark": args.benchmark,
        "dataset_manifest": dataset_manifest,
        "base_model_path": base_model_path,
        "output_dir": str(output_dir),
        "run_root_dir": str(run_root_dir),
        "checkpoints_dir": str(checkpoints_dir),
        "visible_cuda_devices": list(resolved_devices),
        "visible_cuda_devices_source": visible_cuda_devices_source,
        "configured_visible_cuda_devices": list(backend_config.visible_cuda_devices),
        "example_count": len(examples),
        "rank": int(args.rank),
        "alpha": int(args.alpha),
        "dropout": float(args.dropout),
        "learning_rate": float(args.learning_rate),
        "epochs": epochs,
        "weight_decay": float(args.weight_decay),
        "warmup_ratio": float(args.warmup_ratio),
        "attention_implementation": attention_implementation or "eager",
        "max_text_tokens": int(args.max_text_tokens),
        "max_image_pixels": int(args.max_image_pixels),
        "lora_target_modules": list(target_modules),
        "language_target_modules": list(language_targets),
        "projector_target_modules": list(projector_targets),
    }
    if trainer.is_world_process_zero():
        output_dir.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(output_dir))
        processor.save_pretrained(str(output_dir))
        (output_dir / "training_manifest.json").write_text(json.dumps(manifest_payload, indent=2, sort_keys=True))
        print(json.dumps(manifest_payload, indent=2, sort_keys=True))
    _wait_for_everyone(trainer)

    # Keep Pillow imported until the end so the collator can reference it through the processor path.
    _destroy_process_group_if_initialized(torch_module=torch)
    del Image


def _resolve_example_cache_path(
    *,
    run_root_dir: Path,
    benchmark: str,
    explicit_cache_path: str | None,
) -> Path | None:
    if explicit_cache_path:
        return Path(explicit_cache_path).resolve()
    return (run_root_dir / f"{benchmark.lower()}_training_examples.pkl").resolve()


def _load_or_build_training_examples(
    *,
    benchmark: str,
    dataset_manifest: str,
    cache_path: Path | None,
    disable_cache: bool,
) -> tuple[TrainingExample, ...]:
    if disable_cache or cache_path is None:
        _log_phase("building training examples without cache", benchmark=benchmark)
        return _build_training_examples(benchmark=benchmark, dataset_manifest=dataset_manifest)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = cache_path.with_suffix(cache_path.suffix + ".json")
    lock_path = cache_path.with_suffix(cache_path.suffix + ".lock")
    deadline = time.time() + TRAINING_EXAMPLES_CACHE_WAIT_TIMEOUT_SECONDS
    next_status_at = 0.0
    while time.time() < deadline:
        cached_examples = _load_training_examples_cache(
            cache_path=cache_path,
            metadata_path=metadata_path,
            benchmark=benchmark,
            dataset_manifest=dataset_manifest,
        )
        if cached_examples is not None:
            _log_phase("loaded cached training examples", cache_path=str(cache_path), example_count=len(cached_examples))
            return cached_examples

        built_examples = _try_build_training_examples_cache(
            cache_path=cache_path,
            metadata_path=metadata_path,
            lock_path=lock_path,
            benchmark=benchmark,
            dataset_manifest=dataset_manifest,
        )
        if built_examples is not None:
            return built_examples

        if _cleanup_stale_training_examples_lock(lock_path=lock_path, cache_path=cache_path):
            continue

        now = time.time()
        if now >= next_status_at:
            lock_state = "present" if lock_path.exists() else "absent"
            _log_phase("waiting for cached training examples", cache_path=str(cache_path), lock_state=lock_state)
            next_status_at = now + TRAINING_EXAMPLES_CACHE_STATUS_INTERVAL_SECONDS
        time.sleep(TRAINING_EXAMPLES_CACHE_WAIT_POLL_SECONDS)
    raise TimeoutError(
        f"Timed out waiting for cached training examples at {cache_path}. "
        f"Lock file state: {'present' if lock_path.exists() else 'absent'}."
    )


def _load_training_examples_cache(
    *,
    cache_path: Path,
    metadata_path: Path,
    benchmark: str,
    dataset_manifest: str,
) -> tuple[TrainingExample, ...] | None:
    if not cache_path.exists() or not metadata_path.exists():
        return None
    try:
        metadata = json.loads(metadata_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if metadata.get("benchmark") != benchmark or metadata.get("dataset_manifest") != dataset_manifest:
        return None
    if metadata.get("cache_format_version") != TRAINING_EXAMPLES_CACHE_FORMAT_VERSION:
        return None
    try:
        with cache_path.open("rb") as handle:
            loaded = pickle.load(handle)
    except (OSError, EOFError, pickle.PickleError):
        return None
    return tuple(loaded)


def _try_build_training_examples_cache(
    *,
    cache_path: Path,
    metadata_path: Path,
    lock_path: Path,
    benchmark: str,
    dataset_manifest: str,
) -> tuple[TrainingExample, ...] | None:
    lock_fd: int | None = None
    lock_acquired = False
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        lock_acquired = True
        lock_payload = {
            "benchmark": benchmark,
            "dataset_manifest": dataset_manifest,
            "rank": _distributed_rank(),
            "world_size": _distributed_world_size(),
            "pid": os.getpid(),
            "started_at": int(time.time()),
        }
        payload_bytes = json.dumps(lock_payload, sort_keys=True).encode("utf-8")
        if os.write(lock_fd, payload_bytes) != len(payload_bytes):
            raise OSError(f"Short write while creating cache lock at {lock_path}.")
        os.fsync(lock_fd)
        _log_phase("building training examples cache", cache_path=str(cache_path))
        examples = _build_training_examples(benchmark=benchmark, dataset_manifest=dataset_manifest)
        _write_training_examples_cache(
            cache_path=cache_path,
            metadata_path=metadata_path,
            benchmark=benchmark,
            dataset_manifest=dataset_manifest,
            examples=examples,
        )
        _log_phase("training examples cache written", cache_path=str(cache_path), example_count=len(examples))
        return examples
    except FileExistsError:
        return None
    finally:
        if lock_fd is not None:
            os.close(lock_fd)
        if lock_acquired:
            lock_path.unlink(missing_ok=True)


def _write_training_examples_cache(
    *,
    cache_path: Path,
    metadata_path: Path,
    benchmark: str,
    dataset_manifest: str,
    examples: Sequence[TrainingExample],
) -> None:
    tmp_cache_path = cache_path.with_suffix(cache_path.suffix + f".tmp-{os.getpid()}")
    tmp_metadata_path = metadata_path.with_suffix(metadata_path.suffix + f".tmp-{os.getpid()}")
    with tmp_cache_path.open("wb") as handle:
        pickle.dump(tuple(examples), handle, protocol=pickle.HIGHEST_PROTOCOL)
    metadata_payload = {
        "benchmark": benchmark,
        "dataset_manifest": dataset_manifest,
        "cache_format_version": TRAINING_EXAMPLES_CACHE_FORMAT_VERSION,
        "example_count": len(examples),
        "created_at_unix": int(time.time()),
    }
    tmp_metadata_path.write_text(json.dumps(metadata_payload, indent=2, sort_keys=True))
    os.replace(tmp_cache_path, cache_path)
    os.replace(tmp_metadata_path, metadata_path)


def _cleanup_stale_training_examples_lock(
    *,
    lock_path: Path,
    cache_path: Path,
) -> bool:
    if not lock_path.exists():
        return False
    stale_grace_seconds = max(10.0, TRAINING_EXAMPLES_CACHE_WAIT_POLL_SECONDS * 2.0)
    lock_age_seconds = _lock_file_age_seconds(lock_path)
    try:
        payload = json.loads(lock_path.read_text())
    except json.JSONDecodeError:
        payload = None
    except OSError:
        return False
    if not isinstance(payload, dict):
        if lock_age_seconds < stale_grace_seconds:
            return False
        lock_path.unlink(missing_ok=True)
        _log_phase(
            "removed invalid training examples cache lock",
            cache_path=str(cache_path),
            lock_age_seconds=f"{lock_age_seconds:.1f}",
        )
        return True
    pid = payload.get("pid")
    if isinstance(pid, int) and pid > 0 and _pid_is_alive(pid):
        return False
    if (not isinstance(pid, int) or pid <= 0) and lock_age_seconds < stale_grace_seconds:
        return False
    lock_path.unlink(missing_ok=True)
    _log_phase(
        "removed stale training examples cache lock",
        cache_path=str(cache_path),
        lock_age_seconds=f"{lock_age_seconds:.1f}",
        lock_pid=pid if isinstance(pid, int) else "unknown",
        started_at=payload.get("started_at", "unknown"),
    )
    return True


def _lock_file_age_seconds(lock_path: Path) -> float:
    try:
        return max(0.0, time.time() - lock_path.stat().st_mtime)
    except OSError:
        return 0.0


def _pid_is_alive(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _distributed_rank() -> int:
    return int(os.environ.get("RANK", "0"))


def _distributed_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def _log_phase(message: str, **kwargs: Any) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    rank = _distributed_rank()
    world_size = _distributed_world_size()
    payload = " ".join(f"{key}={value}" for key, value in sorted(kwargs.items()))
    line = f"[{timestamp}] [rank {rank}/{world_size}] {message}"
    if payload:
        line = f"{line} {payload}"
    print(line, flush=True)


def _resolve_output_paths(
    config: dict[str, Any],
    explicit_output_dir: str | None,
) -> tuple[Path, Path, Path]:
    if explicit_output_dir is not None:
        requested_path = Path(explicit_output_dir).resolve()
    else:
        paths = dict(config.get("paths") or {})
        expected_path = paths.get("expected_lora_adapter_output_path")
        if not expected_path:
            raise ValueError(
                "No output directory was provided and the selected config does not define "
                "paths.expected_lora_adapter_output_path."
            )
        requested_path = Path(str(expected_path)).resolve()

    if requested_path.name == "final_adapter":
        final_adapter_dir = requested_path
        run_root_dir = requested_path.parent
    else:
        run_root_dir = requested_path
        final_adapter_dir = requested_path / "final_adapter"
    checkpoints_dir = run_root_dir / "checkpoints"
    return run_root_dir, final_adapter_dir, checkpoints_dir


def _resolve_training_visible_cuda_devices(
    *,
    backend_config: QwenBackendConfig,
) -> tuple[tuple[int, ...], str]:
    env_value = os.environ.get("CUDA_VISIBLE_DEVICES")
    # In torchrun/distributed mode (WORLD_SIZE > 1), skip config-range validation:
    # torchrun manages GPU assignment per rank; the config's visible_cuda_devices
    # is only used to restrict single-process local training.
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        if env_value is not None and env_value.strip():
            devices = tuple(int(d.strip()) for d in env_value.split(",") if d.strip())
            return devices, "environment:CUDA_VISIBLE_DEVICES (torchrun)"
        # No env override — infer from world size
        devices = tuple(range(world_size))
        return devices, f"torchrun:world_size={world_size}"
    source = "environment:CUDA_VISIBLE_DEVICES" if env_value is not None and env_value.strip() else "backend.visible_cuda_devices"
    resolved_devices = resolve_visible_cuda_devices(
        backend_config.visible_cuda_devices,
        env_value=env_value,
    )
    return resolved_devices, source


def _wait_for_everyone(trainer: Any) -> None:
    accelerator = getattr(trainer, "accelerator", None)
    if accelerator is not None and hasattr(accelerator, "wait_for_everyone"):
        accelerator.wait_for_everyone()


def _destroy_process_group_if_initialized(*, torch_module: Any) -> None:
    distributed = getattr(torch_module, "distributed", None)
    if distributed is None or not hasattr(distributed, "is_available"):
        return
    if not distributed.is_available() or not hasattr(distributed, "is_initialized"):
        return
    if not distributed.is_initialized():
        return
    distributed.destroy_process_group()


def _build_training_examples(
    *,
    benchmark: str,
    dataset_manifest: str,
) -> tuple[TrainingExample, ...]:
    if benchmark == "LearnGUI":
        return _build_learngui_examples(dataset_manifest)
    if benchmark == "AndroidControl":
        return _build_androidcontrol_examples(dataset_manifest)
    raise ValueError(f"Unsupported benchmark: {benchmark}")


def _default_epochs_for_benchmark(benchmark: str) -> float:
    if benchmark == "LearnGUI":
        return 3.0
    if benchmark == "AndroidControl":
        return 2.0
    raise ValueError(f"Unsupported benchmark: {benchmark}")


def _resolve_attention_implementation() -> str | None:
    try:
        import flash_attn  # noqa: F401
    except Exception:
        return None
    return "flash_attention_2"


def _build_learngui_examples(dataset_manifest: str) -> tuple[TrainingExample, ...]:
    dataset = LearnGUIDataset(manifest_path=dataset_manifest)
    episode_cache: dict[str, DatasetEpisode] = {}
    canonical_cache: dict[str, tuple[Any, ...]] = {}
    examples: list[TrainingExample] = []

    for task in dataset.iter_tasks(split="train"):
        query_episode = _cached_episode(dataset, episode_cache, str(task.query_episode_id))
        query_records = _cached_canonical_records(query_episode, canonical_cache)
        support_episodes = tuple(
            _cached_episode(dataset, episode_cache, str(episode_id))
            for episode_id in task.support_episode_ids
        )
        support_context = {
            "shot_count": int(task.k_shot),
            "support_episode_ids": tuple(str(episode.episode_id) for episode in support_episodes),
            "support_goals": tuple(str(episode.goal) for episode in support_episodes),
            "support_lengths": tuple(len(episode.steps) for episode in support_episodes),
            "support_app": str(task.app),
        }
        for step_index, step in enumerate(query_episode.steps):
            examples.append(
                TrainingExample(
                    benchmark="LearnGUI",
                    observation_id=f"learngui:train:{task.k_shot}-shot:{query_episode.episode_id}:{step_index}",
                    screenshot=step.screenshot,
                    prompt_text=build_full_prompt(
                        StepContext(
                            observation_id=f"learngui:train:{task.k_shot}-shot:{query_episode.episode_id}:{step_index}",
                            record=query_records[step_index],
                            history=tuple(query_records[:step_index]),
                            support_context=support_context,
                        )
                    ),
                    response_text=_action_to_response_text(query_records[step_index].canonical_action),
                )
            )
    return tuple(examples)


def _build_androidcontrol_examples(dataset_manifest: str) -> tuple[TrainingExample, ...]:
    import time as _t
    dataset = AndroidControlDataset(manifest_path=dataset_manifest)
    examples: list[TrainingExample] = []
    episode_count = 0
    t0 = _t.time()
    _log_phase("building AndroidControl training examples", split="train")
    for episode in dataset.iter_episodes(split="train"):
        episode_count += 1
        if episode_count % 200 == 0:
            elapsed = _t.time() - t0
            _log_phase(
                "building examples",
                episodes=episode_count,
                steps=len(examples),
                elapsed_s=f"{elapsed:.0f}",
            )
        canonical_records = tuple(canonicalize_step(step) for step in episode.steps)
        for step_index, step in enumerate(episode.steps):
            observation_id = f"androidcontrol:train:{episode.episode_id}:{step_index}"
            w = int(getattr(step.screenshot, "width", None) or 1080)
            h = int(getattr(step.screenshot, "height", None) or 1920)
            examples.append(
                TrainingExample(
                    benchmark="AndroidControl",
                    observation_id=observation_id,
                    screenshot=step.screenshot,
                    prompt_text=build_full_prompt(
                        StepContext(
                            observation_id=observation_id,
                            record=canonical_records[step_index],
                            history=tuple(canonical_records[:step_index]),
                            support_context={},
                        )
                    ),
                    response_text=_action_to_response_text(
                        canonical_records[step_index].canonical_action,
                        screenshot_width=w,
                        screenshot_height=h,
                    ),
                )
            )
    elapsed = _t.time() - t0
    _log_phase(
        "training examples ready",
        episodes=episode_count,
        steps=len(examples),
        elapsed_s=f"{elapsed:.0f}",
    )
    return tuple(examples)
def _cached_episode(
    dataset: LearnGUIDataset,
    cache: dict[str, DatasetEpisode],
    episode_id: str,
) -> DatasetEpisode:
    episode = cache.get(episode_id)
    if episode is None:
        episode = dataset.get_episode(episode_id)
        cache[episode_id] = episode
    return episode


def _cached_canonical_records(
    episode: DatasetEpisode,
    cache: dict[str, tuple[Any, ...]],
) -> tuple[Any, ...]:
    records = cache.get(str(episode.episode_id))
    if records is None:
        records = tuple(canonicalize_step(step) for step in episode.steps)
        cache[str(episode.episode_id)] = records
    return records


def _pixel_to_scale999(pixel: float, dim: int) -> int:
    """Convert pixel coordinate to 0-999 normalized scale."""
    return max(0, min(999, int(round(pixel / max(1, dim) * 999))))


def _action_to_response_text(
    action: CanonicalAction,
    *,
    screenshot_width: int = 1080,
    screenshot_height: int = 1920,
) -> str:
    """Format a CanonicalAction as MAI-UI mobile_use tool_call response.

    Uses the same <thinking>...<tool_call>...</tool_call> format expected by
    ANDROIDCONTROL_MAIUI_SYSTEM_PROMPT so training and inference are consistent.
    Coordinates are in 0-999 normalized scale matching MAI-UI inference output.
    """
    action_type = str(action.action_type or "").upper()

    if action_type in ("CLICK", "LONG_PRESS"):
        if action.bbox is not None:
            cx = (action.bbox[0] + action.bbox[2]) / 2.0
            cy = (action.bbox[1] + action.bbox[3]) / 2.0
        else:
            cx, cy = screenshot_width / 2.0, screenshot_height / 2.0
        nx = _pixel_to_scale999(cx, screenshot_width)
        ny = _pixel_to_scale999(cy, screenshot_height)
        act_name = "long_press" if action_type == "LONG_PRESS" else "tap"
        arguments = {"action": act_name, "coordinate": [nx, ny]}

    elif action_type == "TYPE":
        arguments = {"action": "type", "text": str(action.argument or "")}

    elif action_type == "SCROLL":
        direction = str(action.direction or "down").lower()
        if action.bbox is not None:
            cx = (action.bbox[0] + action.bbox[2]) / 2.0
            cy = (action.bbox[1] + action.bbox[3]) / 2.0
        else:
            cx, cy = screenshot_width / 2.0, screenshot_height / 2.0
        nx = _pixel_to_scale999(cx, screenshot_width)
        ny = _pixel_to_scale999(cy, screenshot_height)
        arguments = {"action": "scroll", "coordinate": [nx, ny], "direction": direction}

    elif action_type == "NAV":
        nav_arg = str(action.argument or "back").lower()
        nav_map = {"back": "navigate_back", "home": "navigate_home", "recent": "navigate_recent"}
        arguments = {"action": nav_map.get(nav_arg, "navigate_back")}

    elif action_type == "TERMINATE":
        arguments = {"action": "finish_episode"}

    elif action_type == "WAIT":
        arguments = {"action": "wait"}

    else:
        # fallback: tap center
        arguments = {"action": "tap", "coordinate": [499, 499]}

    tool_call_body = json.dumps(
        {"name": "mobile_use", "arguments": arguments},
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=False,
    )
    return f"<thinking>Based on the screen and the instruction, I will perform this action.</thinking><tool_call>{tool_call_body}</tool_call>"


def _discover_projector_target_modules(
    *,
    model: Any,
    torch_module: Any,
) -> tuple[str, ...]:
    linear_cls = getattr(getattr(torch_module, "nn", None), "Linear", None)
    if linear_cls is None:
        raise RuntimeError("torch.nn.Linear is unavailable; cannot inspect MAI-UI-8B / Qwen3-VL projector layers.")

    discovered: list[str] = []
    for module_name, module in model.named_modules():
        if not isinstance(module, linear_cls):
            continue
        if module_name.endswith(PROJECTOR_LINEAR_SUFFIXES):
            discovered.append(module_name)
            continue
        if any(marker in module_name for marker in PROJECTOR_COLLECTION_MARKERS) and module_name.endswith(
            ("linear_fc1", "linear_fc2")
        ):
            discovered.append(module_name)
    resolved = tuple(sorted(dict.fromkeys(discovered)))
    if not resolved:
        raise ValueError(
            "Could not locate any MAI-UI-8B / Qwen3-VL multimodal projector layers. "
            "Expected merger/deepstack merger Linear modules in the loaded model."
        )
    return resolved


def _load_image(asset: ScreenshotAsset) -> Any:
    from PIL import Image

    if asset.path is not None and asset.path.exists():
        return Image.open(asset.path).convert("RGB")
    if asset.png_bytes is not None:
        return Image.open(io.BytesIO(asset.png_bytes)).convert("RGB")
    raise ValueError("ScreenshotAsset has neither a readable path nor inline PNG bytes.")


def _resize_image_to_budget(image: Any, *, max_pixels: int) -> Any:
    width, height = image.size
    max_pixels_value = max(1, int(max_pixels))
    if width * height <= max_pixels_value:
        return image
    scale = (float(max_pixels_value) / float(width * height)) ** 0.5
    resized = (
        max(1, int(round(width * scale))),
        max(1, int(round(height * scale))),
    )
    return image.resize(resized)


def _parse_version(version: str) -> tuple[int, ...]:
    parts: list[int] = []
    for token in str(version).replace("-", ".").split("."):
        digits = "".join(character for character in token if character.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def _require_min_version(
    *,
    package_name: str,
    version: str,
    minimum: tuple[int, int, int],
    requirement: str,
) -> None:
    parsed_version = _parse_version(version)
    if parsed_version < minimum:
        raise RuntimeError(
            f"{package_name} {version} is too old. {requirement}"
        )


if __name__ == "__main__":
    main()
