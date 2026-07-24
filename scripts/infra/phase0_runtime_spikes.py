#!/usr/bin/env python3
"""Phase 0 Runtime Spike verification for DivPrune → Qwen3-VL migration.

Validates 3 critical assumptions before implementing DivPruneQwenBackend (Task 1.2).

Spikes:
  #2  grid/K alignment    — new_grid=(1,2,2K) satisfies prod//merge^2==K==image_pad_count
  #7  monkey-patch bypass  — patched get_image_features (identity) reproduces exact output
  #9  batch repad          — batch_size=2 with different resolutions works after prune+left-pad

Run via (use submit_phase0_spikes.sh or):
  REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
  source "${REPO_ROOT}/scripts/_lib/load_env.sh" && skillreuse_load_env "${REPO_ROOT}"
  srun --partition=h20-gpu --gres=gpu:1 --time=00:30:00 --cpus-per-task=8 --mem=80G \
    "${SKILLREUSE_CONDA_PREFIX}/bin/python" \
    "${REPO_ROOT}/scripts/infra/phase0_runtime_spikes.py"
"""

import sys
import copy
import traceback
from pathlib import Path

import torch
from PIL import Image
import transformers
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

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
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from skillreuse.model.divprune import divprune_select

MODEL_PATH = str(_REPO_ROOT / "models" / "Qwen3-VL-8B-Instruct")
KEEP_RATIO = 0.098


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def load_model_and_processor():
    print(f"transformers version: {transformers.__version__}")
    print(f"torch version: {torch.__version__}")
    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    processor.tokenizer.padding_side = "left"
    print("Loading model (bfloat16, device_map=0)...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        device_map=0,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model.eval()
    spatial_merge = model.model.visual.spatial_merge_size
    img_tok_id = model.config.image_token_id
    print(f"Model loaded. spatial_merge_size={spatial_merge}, image_token_id={img_tok_id}")
    return model, processor


def _get_feat_host(model):
    """Return the object that owns get_image_features (model.model or model)."""
    if hasattr(model.model, "get_image_features"):
        return model.model
    if hasattr(model, "get_image_features"):
        return model
    raise AttributeError(
        "Neither model.model nor model has get_image_features. "
        "Check transformers version or model class."
    )


def _build_single_inputs(processor, image, device):
    """Process one image+text pair and move tensors to device."""
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "Describe this image."},
        ],
    }]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    raw = processor(text=[text], images=[image], return_tensors="pt")
    return {k: v.to(device) for k, v in raw.items()}


def _remove_image_pad_tokens(input_ids_1d, mm_1d, attn_1d, image_token_id, keep_k):
    """Remove the last (N-keep_k) image_pad tokens from a 1-D sequence."""
    pad_pos = (input_ids_1d == image_token_id).nonzero(as_tuple=True)[0]
    positions_to_remove = pad_pos[keep_k:]
    keep = torch.ones(input_ids_1d.shape[0], dtype=torch.bool, device=input_ids_1d.device)
    keep[positions_to_remove] = False
    return input_ids_1d[keep], mm_1d[keep], attn_1d[keep]


def _build_pruned_vision_output(orig_vo, pruned_pooler, pruned_deepstack):
    """Reconstruct a vision output object with replaced pooler/deepstack fields."""
    cls = type(orig_vo)
    if hasattr(orig_vo, "keys"):
        # ModelOutput (OrderedDict subclass): reconstruct from field dict
        kw = {}
        for k in orig_vo.keys():
            if k == "pooler_output":
                kw[k] = pruned_pooler
            elif k == "deepstack_features" and pruned_deepstack is not None:
                kw[k] = pruned_deepstack
            else:
                kw[k] = orig_vo[k]
        return cls(**kw)
    else:
        result = copy.copy(orig_vo)
        result.pooler_output = pruned_pooler
        if pruned_deepstack is not None:
            result.deepstack_features = pruned_deepstack
        return result


# ---------------------------------------------------------------------------
# Spike #2: Grid/K Alignment Verification
# ---------------------------------------------------------------------------

def run_spike_2(model, processor):
    """
    Verify new_grid=(1,2,2K) produces prod(new_grid)//merge^2 == K == image_pad count.
    Falls back to model.generate(max_new_tokens=1) as runtime acceptance check.

    Returns (passed: bool, shared_image, shared_inputs) for reuse by Spike #7.
    """
    print("\n" + "=" * 60)
    print("Spike #2: Grid/K Alignment Verification")
    print("=" * 60)
    try:
        device = next(model.parameters()).device
        sms = model.model.visual.spatial_merge_size      # spatial_merge_size
        image_token_id = model.config.image_token_id

        # --- 1. Create dummy image and process ---
        image = Image.new("RGB", (640, 480), color=(128, 128, 128))
        inputs = _build_single_inputs(processor, image, device)

        input_ids       = inputs["input_ids"]           # (1, seq_len)
        image_grid_thw  = inputs["image_grid_thw"]      # (1, 3)
        mm_token_types  = inputs["mm_token_type_ids"]   # (1, seq_len)
        attention_mask  = inputs["attention_mask"]       # (1, seq_len)

        # --- 2. Compute N and K ---
        t, h, w = image_grid_thw[0].tolist()
        N = int(t * h * w) // (sms ** 2)
        K = max(1, round(KEEP_RATIO * N))
        print(f"  original grid : t={t}, h={h}, w={w}")
        print(f"  N = {t}*{h}*{w} // {sms}^2 = {N}")
        print(f"  K = max(1, round({KEEP_RATIO} * {N})) = {K}")

        # --- 3. Formula check: prod(1,2,2K) // sms^2 must equal K ---
        new_grid = torch.tensor([[1, 2, 2 * K]], device=device)
        formula_K = int(new_grid[0].prod().item()) // (sms ** 2)
        print(f"  new_grid = (1, 2, {2*K}); prod // {sms}^2 = {formula_K}")
        assert formula_K == K, f"Grid formula mismatch: computed {formula_K} != K={K}"

        # --- 4. Verify original image_pad count matches N ---
        orig_pad_count = (input_ids[0] == image_token_id).sum().item()
        print(f"  original image_pad count: {orig_pad_count}")
        assert orig_pad_count == N, (
            f"Original pad count {orig_pad_count} != N={N} — "
            "processor grid mismatch"
        )

        # --- 5. Build modified inputs with exactly K image_pad tokens ---
        new_ids_1d, new_mm_1d, new_attn_1d = _remove_image_pad_tokens(
            input_ids[0], mm_token_types[0], attention_mask[0],
            image_token_id, K
        )
        new_input_ids  = new_ids_1d.unsqueeze(0)
        new_mm_types   = new_mm_1d.unsqueeze(0)
        new_attn_mask  = new_attn_1d.unsqueeze(0)

        new_pad_count = (new_input_ids[0] == image_token_id).sum().item()
        print(f"  modified image_pad count: {new_pad_count}")
        assert new_pad_count == K, (
            f"Modified pad count {new_pad_count} != K={K}"
        )

        # --- 6. Verify M-RoPE acceptance ---
        # Try get_rope_index first (pass new_mm_types, not None); fall back to
        # monkey-patch + generate if it raises (avoids ViT shape mismatch).
        vision_pos_count = None
        for obj, label in [(model, "model"), (model.model, "model.model")]:
            if hasattr(obj, "get_rope_index"):
                try:
                    obj.get_rope_index(
                        new_input_ids, new_grid, new_mm_types, new_attn_mask
                    )
                    vision_pos_count = int(new_mm_types[0].sum().item())
                    print(f"  get_rope_index on {label} succeeded")
                    break
                except Exception as rope_err:
                    print(f"  get_rope_index on {label} failed ({rope_err}), trying next …")

        if vision_pos_count is None:
            # Monkey-patch approach: pre-compute pruned features so the ViT is
            # never called with a mismatched grid during generate().
            feat_host = _get_feat_host(model)
            with torch.no_grad():
                vision_output = feat_host.get_image_features(
                    inputs["pixel_values"], inputs["image_grid_thw"]
                )

            feat_i = vision_output.pooler_output[0]        # (N, D)
            sel = divprune_select(feat_i.float(), KEEP_RATIO)  # (K,)
            pruned_pooler = [feat_i[sel]]

            pruned_deepstack = None
            if (
                hasattr(vision_output, "deepstack_features")
                and vision_output.deepstack_features is not None
            ):
                pruned_deepstack = [
                    layer_feat[sel]
                    for layer_feat in vision_output.deepstack_features
                ]

            pruned_vo = _build_pruned_vision_output(
                vision_output, pruned_pooler, pruned_deepstack
            )

            original_fn = feat_host.get_image_features
            try:
                feat_host.get_image_features = lambda pv, grid, **kw: pruned_vo
                model.model.rope_deltas = None
                with torch.no_grad():
                    model.generate(
                        input_ids=new_input_ids,
                        image_grid_thw=new_grid,
                        mm_token_type_ids=new_mm_types,
                        attention_mask=new_attn_mask,
                        pixel_values=inputs["pixel_values"],
                        max_new_tokens=1,
                        do_sample=False,
                    )
            finally:
                feat_host.get_image_features = original_fn

            vision_pos_count = new_pad_count  # verified indirectly: no crash
            print("  monkey-patch + generate() fallback succeeded (no crash)")

        assert vision_pos_count == K, (
            f"Vision position count {vision_pos_count} != K={K}"
        )

        print(f"\nSpike #2: PASS — N={N}, K={K}, vision_positions={vision_pos_count}")
        return True, image, inputs

    except Exception:
        print(f"\nSpike #2: FAIL")
        traceback.print_exc()
        return False, None, None


# ---------------------------------------------------------------------------
# Spike #7: Monkey-patch + keep_ratio=1.0 Bypass Verification
# ---------------------------------------------------------------------------

def run_spike_7(model, processor, spike2_image, spike2_inputs):
    """
    Verify that monkey-patching get_image_features to return the unmodified output
    produces byte-identical greedy generations.
    """
    print("\n" + "=" * 60)
    print("Spike #7: Monkey-patch + keep_ratio=1.0 Bypass Verification")
    print("=" * 60)
    try:
        device = next(model.parameters()).device

        # Rebuild inputs if Spike #2 did not provide them
        if spike2_image is None:
            spike2_image = Image.new("RGB", (640, 480), color=(128, 128, 128))
            spike2_inputs = _build_single_inputs(processor, spike2_image, device)

        gen_kwargs = dict(max_new_tokens=20, do_sample=False)
        input_len = spike2_inputs["input_ids"].shape[1]

        # --- Standard path ---
        model.model.rope_deltas = None
        with torch.no_grad():
            ref_out = model.generate(**spike2_inputs, **gen_kwargs)
        reference_output = processor.tokenizer.decode(
            ref_out[0][input_len:], skip_special_tokens=True
        )
        print(f"  standard path  : {repr(reference_output[:80])}")

        # --- Monkey-patch path ---
        feat_host = _get_feat_host(model)

        with torch.no_grad():
            vision_output = feat_host.get_image_features(
                spike2_inputs["pixel_values"],
                spike2_inputs["image_grid_thw"],
            )

        original_fn = feat_host.get_image_features
        try:
            feat_host.get_image_features = lambda pv, grid, **kw: vision_output
            model.model.rope_deltas = None
            with torch.no_grad():
                patched_out = model.generate(**spike2_inputs, **gen_kwargs)
            patched_output = processor.tokenizer.decode(
                patched_out[0][input_len:], skip_special_tokens=True
            )
        finally:
            feat_host.get_image_features = original_fn

        print(f"  monkey-patch   : {repr(patched_output[:80])}")

        assert reference_output == patched_output, (
            f"Output mismatch!\n"
            f"  reference : {repr(reference_output)}\n"
            f"  patched   : {repr(patched_output)}"
        )

        print(
            f"\nSpike #7: PASS — outputs identical "
            f"({len(reference_output)} chars, {ref_out.shape[1] - input_len} tokens)"
        )
        return True

    except Exception:
        print(f"\nSpike #7: FAIL")
        traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# Spike #9: Batch Repad Verification
# ---------------------------------------------------------------------------

def run_spike_9(model, processor):
    """
    Verify that batch_size=2 with different-resolution images works correctly
    after DivPrune token pruning and left-pad realignment.
    """
    print("\n" + "=" * 60)
    print("Spike #9: Batch Repad Verification")
    print("=" * 60)
    try:
        device          = next(model.parameters()).device
        sms             = model.model.visual.spatial_merge_size
        image_token_id  = model.config.image_token_id
        pad_token_id    = processor.tokenizer.pad_token_id

        # --- 1. Two images of different sizes ---
        img1 = Image.new("RGB", (640, 480), color=(100, 150, 200))
        img2 = Image.new("RGB", (320, 240), color=(200, 100, 150))

        def _msg(img):
            return [{
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": "Describe this image."},
                ],
            }]

        texts = [
            processor.apply_chat_template(
                _msg(img1), tokenize=False, add_generation_prompt=True
            ),
            processor.apply_chat_template(
                _msg(img2), tokenize=False, add_generation_prompt=True
            ),
        ]
        raw = processor(
            text=texts, images=[img1, img2], return_tensors="pt", padding=True
        )
        inputs = {k: v.to(device) for k, v in raw.items()}

        input_ids       = inputs["input_ids"]           # (2, max_seq)
        image_grid_thw  = inputs["image_grid_thw"]      # (2, 3)
        mm_token_types  = inputs["mm_token_type_ids"]   # (2, max_seq)
        attention_mask  = inputs["attention_mask"]       # (2, max_seq)
        pixel_values    = inputs["pixel_values"]

        # --- 2. Compute N_i and K_i per image ---
        N_values, K_values = [], []
        for i in range(2):
            t, h, w = image_grid_thw[i].tolist()
            N_i = int(t * h * w) // (sms ** 2)
            K_i = max(1, round(KEEP_RATIO * N_i))
            N_values.append(N_i)
            K_values.append(K_i)
            print(f"  image {i+1}: grid=({t},{h},{w}), N={N_i}, K={K_i}")

        # --- 3. Run get_image_features on full batch ---
        feat_host = _get_feat_host(model)
        with torch.no_grad():
            vision_output = feat_host.get_image_features(pixel_values, image_grid_thw)

        print(f"  vision_output type      : {type(vision_output).__name__}")
        print(f"  pooler_output[0].shape  : {vision_output.pooler_output[0].shape}")
        print(f"  pooler_output[1].shape  : {vision_output.pooler_output[1].shape}")

        # --- 4. DivPrune selection per image ---
        pruned_pooler      = []
        selected_idx_list  = []
        for i in range(2):
            feat_i = vision_output.pooler_output[i]          # (N_i, D)
            assert feat_i.shape[0] == N_values[i], (
                f"pooler_output[{i}] shape {feat_i.shape[0]} != N_i={N_values[i]}"
            )
            sel = divprune_select(feat_i.float(), KEEP_RATIO)  # (K_i,)
            assert len(sel) == K_values[i], (
                f"divprune_select returned {len(sel)} != K_i={K_values[i]}"
            )
            pruned_pooler.append(feat_i[sel])
            selected_idx_list.append(sel)
            print(f"  image {i+1}: divprune selected {len(sel)} / {N_values[i]} tokens")

        # --- 5. Sync-prune deepstack_features ---
        pruned_deepstack = None
        if (
            hasattr(vision_output, "deepstack_features")
            and vision_output.deepstack_features is not None
        ):
            pruned_deepstack = []
            for layer_feat in vision_output.deepstack_features:
                # layer_feat: (N_0 + N_1, D_layer)
                splits = torch.split(layer_feat, N_values, dim=0)
                pruned_layer = torch.cat(
                    [splits[i][selected_idx_list[i]] for i in range(2)], dim=0
                )
                pruned_deepstack.append(pruned_layer)
            print(f"  deepstack layers pruned : {len(pruned_deepstack)}")
        else:
            print("  deepstack_features      : None (skipped)")

        pruned_vo = _build_pruned_vision_output(
            vision_output, pruned_pooler, pruned_deepstack
        )

        # --- 6. Prune image_pad tokens per sample, then left-pad ---
        new_ids_list, new_mm_list, new_attn_list = [], [], []
        for i in range(2):
            mask_i = attention_mask[i].bool()
            seq_i  = input_ids[i][mask_i]       # strip existing left-padding
            mm_i   = mm_token_types[i][mask_i]
            attn_i = torch.ones(seq_i.shape[0], dtype=torch.long, device=device)

            pad_count_i = (seq_i == image_token_id).sum().item()
            assert pad_count_i == N_values[i], (
                f"sample {i}: image_pad count {pad_count_i} != N_i={N_values[i]}"
            )

            new_ids_1d, new_mm_1d, new_attn_1d = _remove_image_pad_tokens(
                seq_i, mm_i, attn_i, image_token_id, K_values[i]
            )
            new_ids_list.append(new_ids_1d)
            new_mm_list.append(new_mm_1d)
            new_attn_list.append(new_attn_1d)

        max_len = max(s.shape[0] for s in new_ids_list)

        def _left_pad(seq, pad_val, dtype):
            pad_len = max_len - seq.shape[0]
            return torch.cat([
                torch.full((pad_len,), pad_val, dtype=dtype, device=device),
                seq,
            ])

        new_input_ids_batch = torch.stack([
            _left_pad(s, pad_token_id, s.dtype) for s in new_ids_list
        ])                                                           # (2, max_len)
        new_mm_batch = torch.stack([
            _left_pad(m, 0, m.dtype) for m in new_mm_list
        ])                                                           # (2, max_len)
        new_attn_batch = torch.stack([
            _left_pad(a, 0, torch.long) for a in new_attn_list
        ])                                                           # (2, max_len)

        new_grids = torch.tensor(
            [[1, 2, 2 * K_values[i]] for i in range(2)], device=device
        )                                                            # (2, 3)
        print(f"  new_grids: {new_grids.tolist()}")
        print(f"  padded batch shape: {new_input_ids_batch.shape}")

        # --- 7. Monkey-patch and generate ---
        original_fn = feat_host.get_image_features
        try:
            feat_host.get_image_features = lambda pv, grid, **kw: pruned_vo
            model.model.rope_deltas = None
            with torch.no_grad():
                out = model.generate(
                    input_ids=new_input_ids_batch,
                    image_grid_thw=new_grids,
                    mm_token_type_ids=new_mm_batch,
                    attention_mask=new_attn_batch,
                    pixel_values=pixel_values,
                    max_new_tokens=20,
                    do_sample=False,
                )
        finally:
            feat_host.get_image_features = original_fn

        # --- 8. Assertions ---
        input_len      = new_input_ids_batch.shape[1]
        gen_tok_count  = out.shape[1] - input_len
        assert gen_tok_count > 0, (
            f"No tokens generated: out.shape={out.shape}, input_len={input_len}"
        )

        output_texts = [
            processor.tokenizer.decode(out[i][input_len:], skip_special_tokens=True)
            for i in range(2)
        ]
        assert len(output_texts[0]) > 0, "Sample 0 output text is empty"
        assert len(output_texts[1]) > 0, "Sample 1 output text is empty"

        print(f"  sample 0 output: {repr(output_texts[0][:60])}")
        print(f"  sample 1 output: {repr(output_texts[1][:60])}")

        print(
            f"\nSpike #9: PASS — "
            f"N={N_values}, K={K_values}, "
            f"generated_tokens={gen_tok_count}, "
            f"both outputs non-empty"
        )
        return True

    except Exception:
        print(f"\nSpike #9: FAIL")
        traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Phase 0 Runtime Spike Verification")
    print("DivPrune → Qwen3-VL Migration")
    print("=" * 60)

    model, processor = load_model_and_processor()

    results = {}

    ok2, shared_image, shared_inputs = run_spike_2(model, processor)
    results["Spike #2 (grid/K alignment)"] = ok2

    ok7 = run_spike_7(model, processor, shared_image, shared_inputs)
    results["Spike #7 (monkey-patch bypass)"] = ok7

    ok9 = run_spike_9(model, processor)
    results["Spike #9 (batch repad)"] = ok9

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")

    if all(results.values()):
        print("\nAll spikes PASSED. Proceed to Phase 1 (Task 1.1).")
    else:
        print("\nSome spikes FAILED. Fix issues before proceeding to Phase 1.")
        sys.exit(1)


if __name__ == "__main__":
    main()
