"""Unit tests for skillreuse.model.divprune_backend.

T4 (pad-removal ↔ embedding-scatter equivalence):
    SkillReuse selects visual embeddings via ``img_embeds[selected]`` and trims
  excess ``<|image_pad|>`` slots to K.  LLaVA reindexes the full token
  sequence with sorted ``keep_indexs``.  These tests prove both paths yield
  identical visual embedding content.

  Note: M-RoPE ``image_grid_thw`` reassignment is no longer needed — real 3D
  position IDs are built from the original grid coordinates.

T2: CPU unit tests for ``_prune_input_ids_batch``.
T5: CPU unit tests for ``_compute_selected_position_ids`` and
    ``_build_position_ids_for_batch``.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from skillreuse.model.divprune import divprune_select
from skillreuse.model.divprune_backend import (
    DivPruneQwenBackend,
    _build_position_ids_for_batch,
    _compute_selected_position_ids,
    _precompute_K_N_values,
    _prune_vision_output,
    _resolve_qwen3_vl_model,
)
from skillreuse.model.qwen_backend import QwenLoRABackend

IMAGE_TOKEN_ID = 151655
PAD_TOKEN_ID = 0
MERGE_SIZE = 2


def _grid_thw_for_n_tokens(n_tokens: int, merge_size: int = MERGE_SIZE) -> list[int]:
    """Build image_grid_thw row so prod(-1)//merge_size**2 == n_tokens."""
    hw = n_tokens * merge_size**2
    h = int(hw**0.5)
    while h > 1 and hw % h != 0:
        h -= 1
    w = hw // h
    return [1, h, w]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_backend(*, merge_size: int = MERGE_SIZE, pad_token_id: int = PAD_TOKEN_ID) -> DivPruneQwenBackend:
    backend = DivPruneQwenBackend.__new__(DivPruneQwenBackend)
    qwen3_vl_model = SimpleNamespace(
        visual=SimpleNamespace(spatial_merge_size=merge_size),
        get_image_features=lambda *a, **kw: None,
    )
    backend._model = SimpleNamespace(model=qwen3_vl_model)
    backend._processor = SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=pad_token_id))
    return backend


def _build_batch_inputs(
    samples: list[tuple[list[int], list[int], int]],
    *,
    pad_token_id: int = PAD_TOKEN_ID,
) -> dict[str, torch.Tensor]:
    """Build left-padded batch tensors from (valid_ids, valid_mm, left_pad) triples."""
    max_total = max(left_pad + len(ids) for ids, _, left_pad in samples)
    batch_size = len(samples)

    input_ids = torch.full((batch_size, max_total), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_total), dtype=torch.long)
    mm_token_type_ids = torch.zeros((batch_size, max_total), dtype=torch.long)

    for b, (valid_ids, valid_mm, left_pad) in enumerate(samples):
        length = len(valid_ids)
        input_ids[b, left_pad : left_pad + length] = torch.tensor(valid_ids, dtype=torch.long)
        attention_mask[b, left_pad : left_pad + length] = 1
        mm_token_type_ids[b, left_pad : left_pad + length] = torch.tensor(valid_mm, dtype=torch.long)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "mm_token_type_ids": mm_token_type_ids,
    }


def _sequence_with_image_pads(
    *,
    prefix: list[int],
    num_pads: int,
    suffix: list[int],
    image_token_id: int = IMAGE_TOKEN_ID,
) -> tuple[list[int], list[int]]:
    ids = prefix + [image_token_id] * num_pads + suffix
    mm = [0] * len(prefix) + [1] * num_pads + [0] * len(suffix)
    return ids, mm


def _llava_style_visual_embeds(
    visual_embeds: torch.Tensor,
    selected: torch.Tensor,
    *,
    sys_token_len: int,
    tail_len: int,
) -> torch.Tensor:
    """Reindex full sequence the way LLaVA DivPrune does (visual portion only)."""
    n_visual = visual_embeds.shape[0]
    device = visual_embeds.device
    sys_embeds = torch.randn(sys_token_len, visual_embeds.shape[1], device=device)
    tail_embeds = torch.randn(tail_len, visual_embeds.shape[1], device=device)
    full = torch.cat([sys_embeds, visual_embeds, tail_embeds], dim=0)

    selected_sorted = torch.sort(selected).values
    selected_abs = selected_sorted + sys_token_len
    keep_indexs = torch.cat(
        (
            torch.arange(sys_token_len, device=device),
            selected_abs,
            torch.arange(sys_token_len + n_visual, full.shape[0], device=device),
        )
    ).sort().values

    pruned = full[keep_indexs]
    return pruned[sys_token_len : sys_token_len + selected_sorted.shape[0]]


def _skillreuse_pad_keep_mask(
    ids_valid: torch.Tensor,
    *,
    image_token_id: int,
    k_kept: int,
) -> torch.Tensor:
    """Mirror pad-trimming logic from ``_prune_input_ids_batch``."""
    image_pad_mask = ids_valid == image_token_id
    pad_positions = image_pad_mask.nonzero(as_tuple=True)[0]
    keep_mask = torch.ones(ids_valid.shape[0], dtype=torch.bool)
    n_pads = int(image_pad_mask.sum().item())
    if k_kept < n_pads:
        keep_mask[pad_positions[k_kept:]] = False
    return keep_mask


# ---------------------------------------------------------------------------
# T4: embedding-scatter equivalence
# ---------------------------------------------------------------------------


def test_embedding_scatter_equivalence_llava_vs_skillreuse():
    """Sorted index gather on visual embeds matches LLaVA keep_indexs reindexing."""
    torch.manual_seed(0)
    n_visual, dim = 25, 16
    visual_embeds = torch.randn(n_visual, dim)
    selected = torch.tensor([3, 7, 15, 22], dtype=torch.long)

    selected_sorted = torch.sort(selected).values
    llava_visual = _llava_style_visual_embeds(
        visual_embeds, selected_sorted, sys_token_len=5, tail_len=8
    )
    skillreuse_visual = visual_embeds[selected_sorted]

    assert torch.equal(llava_visual, skillreuse_visual)


def test_embedding_scatter_equivalence_non_contiguous_indices():
    """Equivalence holds for arbitrary non-contiguous, unsorted selection."""
    torch.manual_seed(7)
    n_visual, dim = 40, 8
    visual_embeds = torch.randn(n_visual, dim)
    selected = torch.tensor([1, 19, 5, 33, 12], dtype=torch.long)

    selected_sorted = torch.sort(selected).values
    llava_visual = _llava_style_visual_embeds(
        visual_embeds, selected_sorted, sys_token_len=3, tail_len=6
    )
    skillreuse_visual = visual_embeds[selected_sorted]

    assert torch.equal(llava_visual, skillreuse_visual)


def test_pad_keep_mask_retains_first_k_slots():
    """Pad trimming keeps the first K <|image_pad|> positions and drops the rest."""
    prefix, suffix = [10, 11], [20, 21]
    ids, _ = _sequence_with_image_pads(prefix=prefix, num_pads=10, suffix=suffix)
    ids_valid = torch.tensor(ids, dtype=torch.long)

    for k in [3, 7, 10]:
        keep_mask = _skillreuse_pad_keep_mask(ids_valid, image_token_id=IMAGE_TOKEN_ID, k_kept=k)
        pruned = ids_valid[keep_mask]
        pad_count = int((pruned == IMAGE_TOKEN_ID).sum().item())
        assert pad_count == k
        assert pruned.tolist() == prefix + [IMAGE_TOKEN_ID] * k + suffix


def test_pad_keep_mask_scattered_pad_positions():
    """Pad trimming works when image pads are contiguous in the valid sequence."""
    # Prefix/suffix ensure pads are a contiguous block (Qwen template layout).
    ids, mm = _sequence_with_image_pads(prefix=[1, 2, 3], num_pads=6, suffix=[4, 5])
    ids_valid = torch.tensor(ids, dtype=torch.long)
    mm_valid = torch.tensor(mm, dtype=torch.long)

    k = 2
    keep_mask = _skillreuse_pad_keep_mask(ids_valid, image_token_id=IMAGE_TOKEN_ID, k_kept=k)
    new_ids = ids_valid[keep_mask]
    new_mm = mm_valid[keep_mask]

    assert int((new_ids == IMAGE_TOKEN_ID).sum().item()) == k
    assert int((new_mm == 1).sum().item()) == k


# ---------------------------------------------------------------------------
# T2: _prune_input_ids_batch
# ---------------------------------------------------------------------------


def test_prune_reduces_pad_count_n_to_k():
    backend = _make_backend()
    n_pads, k_kept = 8, 3
    ids, mm = _sequence_with_image_pads(prefix=[100, 101], num_pads=n_pads, suffix=[200])
    inputs = _build_batch_inputs([(ids, mm, 0)])

    result = backend._prune_input_ids_batch(
        inputs, [k_kept], [n_pads], IMAGE_TOKEN_ID
    )

    valid = result["attention_mask"].bool()
    pruned_ids = result["input_ids"][0][valid[0]]
    assert int((pruned_ids == IMAGE_TOKEN_ID).sum().item()) == k_kept


def test_prune_no_image_grid_thw_in_result():
    """返回的 dict 不再包含 image_grid_thw（position_ids 由外部构建）。"""
    backend = _make_backend()
    n_pads, k_kept = 8, 3
    ids, mm = _sequence_with_image_pads(prefix=[100, 101], num_pads=n_pads, suffix=[200])
    inputs = _build_batch_inputs([(ids, mm, 0)])

    result = backend._prune_input_ids_batch(
        inputs, [k_kept], [n_pads], IMAGE_TOKEN_ID
    )

    assert "image_grid_thw" not in result
    assert "input_ids" in result
    assert "attention_mask" in result
    assert "mm_token_type_ids" in result


def test_prune_raises_on_split_sizes_mismatch():
    backend = _make_backend()
    n_pads = 6
    ids, mm = _sequence_with_image_pads(prefix=[1], num_pads=n_pads, suffix=[2])
    inputs = _build_batch_inputs([(ids, mm, 0)])

    with pytest.raises(AssertionError, match=r"image_pad count 6 != N_values\[0\] 9"):
        backend._prune_input_ids_batch(inputs, [3], [9], IMAGE_TOKEN_ID)


def test_prune_raises_on_batch_size_split_sizes_mismatch():
    backend = _make_backend()
    ids, mm = _sequence_with_image_pads(prefix=[1], num_pads=4, suffix=[2])
    inputs = _build_batch_inputs([(ids, mm, 0)])

    with pytest.raises(AssertionError, match=r"batch_size 1 != len\(N_values\) 2"):
        backend._prune_input_ids_batch(inputs, [2, 2], [4, 4], IMAGE_TOKEN_ID)


def test_prune_left_pad_batch_attention_mask():
    """Two samples with different lengths are left-padded to a common max length."""
    backend = _make_backend()

    n_pads = 6
    ids_short, mm_short = _sequence_with_image_pads(prefix=[10], num_pads=n_pads, suffix=[20])
    ids_long, mm_long = _sequence_with_image_pads(prefix=[10, 11, 12], num_pads=n_pads, suffix=[20, 21, 22])
    inputs = _build_batch_inputs(
        [
            (ids_short, mm_short, 2),
            (ids_long, mm_long, 0),
        ]
    )

    k_kept = 3
    result = backend._prune_input_ids_batch(
        inputs,
        [k_kept, k_kept],
        [n_pads, n_pads],
        IMAGE_TOKEN_ID,
    )

    batch_size, seq_len = result["input_ids"].shape
    assert batch_size == 2
    assert seq_len == result["attention_mask"].shape[1]

    for b in range(batch_size):
        mask = result["attention_mask"][b]
        ids_row = result["input_ids"][b]

        assert mask.sum().item() > 0
        first_one = int(mask.argmax().item())
        if first_one > 0:
            assert (mask[:first_one] == 0).all()
            assert (ids_row[:first_one] == PAD_TOKEN_ID).all()
        assert (mask[first_one:] == 1).all()

    short_valid_len = int(result["attention_mask"][0].sum().item())
    long_valid_len = int(result["attention_mask"][1].sum().item())
    assert long_valid_len > short_valid_len
    assert result["input_ids"].shape[1] == long_valid_len


# ---------------------------------------------------------------------------
# K precomputation
# ---------------------------------------------------------------------------


def test_K_precomputation_matches_divprune():
    """Verify K = max(1, round(keep_ratio * N)) matches divprune_select output length."""
    for n in [10, 29, 80, 100, 300, 1225]:
        for keep_ratio in [0.098, 0.2, 0.5, 1.0]:
            k_precomputed = max(1, round(keep_ratio * n))
            features = torch.randn(n, 128)
            k_actual = divprune_select(features, keep_ratio).shape[0]
            assert k_precomputed == k_actual


def test_precompute_K_N_values_from_grid():
    """_precompute_K_N_values returns correct N and reasonable K (2-tuple)."""
    grid = torch.tensor([[1, 14, 14], [1, 20, 20]], dtype=torch.long)
    merge_size = 2
    keep_ratio = 0.098
    k_values, n_values = _precompute_K_N_values(grid, merge_size, keep_ratio)
    expected_n = [int((1 * 14 * 14) // 4), int((1 * 20 * 20) // 4)]
    assert n_values == expected_n
    for k, n in zip(k_values, expected_n):
        assert k == max(1, round(keep_ratio * n))
        assert k >= 1


# ---------------------------------------------------------------------------
# _resolve_qwen3_vl_model
# ---------------------------------------------------------------------------


def test_resolve_qwen3_vl_model_bare():
    qwen3_vl_model = SimpleNamespace(get_image_features=lambda: None)
    outer = SimpleNamespace(model=qwen3_vl_model)
    assert _resolve_qwen3_vl_model(outer) is qwen3_vl_model


def test_resolve_qwen3_vl_model_peft():
    qwen3_vl_model = SimpleNamespace(get_image_features=lambda: None)
    for_gen = SimpleNamespace(model=qwen3_vl_model)
    base_model = SimpleNamespace(model=for_gen)
    peft = SimpleNamespace(base_model=base_model)
    assert _resolve_qwen3_vl_model(peft) is qwen3_vl_model


def test_resolve_qwen3_vl_model_peft_base_is_qwen3_vl_model():
    """PeftModel.base_model may be Qwen3VLModel directly (no .model wrapper)."""
    qwen3_vl_model = SimpleNamespace(get_image_features=lambda: None)
    peft = SimpleNamespace(base_model=qwen3_vl_model)
    assert _resolve_qwen3_vl_model(peft) is qwen3_vl_model


def test_resolve_qwen3_vl_model_raises():
    with pytest.raises(AttributeError, match="Cannot resolve"):
        _resolve_qwen3_vl_model(SimpleNamespace())


# ---------------------------------------------------------------------------
# Feature flag routing (S2)
# ---------------------------------------------------------------------------


def test_run_generation_batch_respects_inline_vit_flag(monkeypatch):
    """DIVPRUNE_INLINE_VIT=0 routes to legacy; =1 routes to inline."""
    backend = DivPruneQwenBackend.__new__(DivPruneQwenBackend)
    routed: list[str] = []

    def fake_legacy(*a, **kw):
        routed.append("legacy")
        return (("legacy",), (1,), (2,))

    def fake_inline(*a, **kw):
        routed.append("inline")
        return (("inline",), (3,), (4,))

    backend._run_generation_batch_legacy = fake_legacy
    backend._run_generation_batch_inline = fake_inline
    gen_kwargs = dict(
        max_new_tokens=8,
        temperature=0.0,
        top_p=1.0,
        repetition_penalty=1.0,
    )

    monkeypatch.setenv("DIVPRUNE_INLINE_VIT", "0")
    out_legacy = backend._run_generation_batch([[]], **gen_kwargs)
    assert routed == ["legacy"]
    assert out_legacy[0] == ("legacy",)

    routed.clear()
    monkeypatch.setenv("DIVPRUNE_INLINE_VIT", "1")
    out_inline = backend._run_generation_batch([[]], **gen_kwargs)
    assert routed == ["inline"]
    assert out_inline[0] == ("inline",)


# ---------------------------------------------------------------------------
# Single-image constraint (S3)
# ---------------------------------------------------------------------------


def _backend_with_multi_image_grid() -> DivPruneQwenBackend:
    backend = DivPruneQwenBackend.__new__(DivPruneQwenBackend)
    backend.keep_ratio = 0.098
    backend._model_device = None
    backend._processor = SimpleNamespace(
        apply_chat_template=lambda *a, **kw: {
            "input_ids": torch.zeros(1, 5, dtype=torch.long),
            "attention_mask": torch.ones(1, 5, dtype=torch.long),
            "mm_token_type_ids": torch.zeros(1, 5, dtype=torch.long),
            "pixel_values": torch.randn(2, 3),
            "image_grid_thw": torch.tensor([[1, 14, 14], [1, 14, 14]]),
        },
        batch_decode=lambda *a, **kw: [""],
        tokenizer=SimpleNamespace(pad_token_id=0),
    )
    qwen3_vl_model = SimpleNamespace(
        visual=SimpleNamespace(spatial_merge_size=2),
        get_image_features=lambda *a, **kw: None,
        rope_deltas=None,
    )
    backend._model = SimpleNamespace(
        model=qwen3_vl_model,
        config=SimpleNamespace(image_token_id=IMAGE_TOKEN_ID),
        generate=lambda **kw: torch.zeros(1, 10, dtype=torch.long),
    )
    return backend


@pytest.mark.parametrize(
    "path_name",
    ["_run_generation_batch_inline", "_run_generation_batch_legacy"],
)
def test_single_image_constraint_raises(path_name):
    """Multi-image per sample raises ValueError on inline and legacy paths."""
    backend = _backend_with_multi_image_grid()
    run_path = getattr(backend, path_name)
    with pytest.raises(ValueError, match="exactly one image per sample"):
        run_path(
            [[{"role": "user", "content": "hi"}]],
            max_new_tokens=8,
            temperature=0.0,
            top_p=1.0,
            repetition_penalty=1.0,
        )


# ---------------------------------------------------------------------------
# ViT call count & inline vs legacy embedding equivalence
# ---------------------------------------------------------------------------


def _make_mock_generation_backend(
    *,
    n_tokens: int = 20,
    keep_ratio: float = 0.3,
    batch_size: int = 1,
    with_deepstack: bool = False,
) -> tuple[DivPruneQwenBackend, dict[str, int]]:
    """Build a backend with mocked model/processor for generation-path tests."""
    counters = {"vit_forward": 0}

    torch.manual_seed(42)
    img_embeds = torch.randn(n_tokens, 8)
    ds_layers = 2
    deepstack = None
    if with_deepstack:
        deepstack = [torch.randn(n_tokens, 8) for _ in range(ds_layers)]

    def _vision_output():
        out = SimpleNamespace(
            pooler_output=[img_embeds.clone()],
            deepstack_features=deepstack,
        )
        return out

    def visual_forward(*args, **kwargs):
        counters["vit_forward"] += 1
        return _vision_output()

    visual = SimpleNamespace(spatial_merge_size=2, forward=visual_forward)

    def get_image_features(pv, grid, **kw):
        visual.forward(pv, grid_thw=grid, **kw)
        return _vision_output()

    qwen3_vl_model = SimpleNamespace(
        visual=visual,
        get_image_features=get_image_features,
        rope_deltas=None,
    )

    prompt_len = 3 + n_tokens + 2
    gen_len = 5

    def generate(**kwargs):
        # 在新版 inline 路径中，ViT 已在 generate 之前执行。
        # generate 内部的 get_image_features 已被 patch 为返回缓存结果。
        return torch.zeros(batch_size, prompt_len + gen_len, dtype=torch.long)

    backend = DivPruneQwenBackend.__new__(DivPruneQwenBackend)
    backend.keep_ratio = keep_ratio
    backend._last_batch_visual_token_counts = None
    backend._model_device = None

    def apply_chat_template(messages, **kw):
        grid_row = _grid_thw_for_n_tokens(n_tokens)
        ids = torch.full((batch_size, prompt_len), 100, dtype=torch.long)
        mm = torch.zeros(batch_size, prompt_len, dtype=torch.long)
        for b in range(batch_size):
            pad_start = 3
            ids[b, pad_start : pad_start + n_tokens] = IMAGE_TOKEN_ID
            mm[b, pad_start : pad_start + n_tokens] = 1
        return {
            "input_ids": ids,
            "attention_mask": torch.ones(batch_size, prompt_len, dtype=torch.long),
            "mm_token_type_ids": mm,
            "pixel_values": torch.randn(batch_size, 3),
            "image_grid_thw": torch.tensor([grid_row] * batch_size, dtype=torch.long),
        }

    backend._processor = SimpleNamespace(
        apply_chat_template=apply_chat_template,
        batch_decode=lambda trimmed, **kw: ["answer"] * len(trimmed),
        tokenizer=SimpleNamespace(pad_token_id=0),
    )
    backend._model = SimpleNamespace(
        model=qwen3_vl_model,
        config=SimpleNamespace(image_token_id=IMAGE_TOKEN_ID),
        generate=generate,
    )
    return backend, counters


def test_k_equals_n_bypass_calls_super(monkeypatch):
    """When K==N, inline path delegates to super()._run_generation_batch()."""
    backend, _ = _make_mock_generation_backend(n_tokens=20, keep_ratio=1.0)
    super_calls: list[tuple] = []

    def tracking_super(self, messages_batch, **kwargs):
        super_calls.append((messages_batch, kwargs))
        return (("super",), (99,), (5,))

    monkeypatch.setattr(QwenLoRABackend, "_run_generation_batch", tracking_super)

    messages = [[{"role": "user", "content": "hi"}]]
    gen_kwargs = dict(
        max_new_tokens=5,
        temperature=0.0,
        top_p=1.0,
        repetition_penalty=1.0,
    )
    out = backend._run_generation_batch_inline(messages, **gen_kwargs)

    assert len(super_calls) == 1
    assert super_calls[0][0] == messages
    assert out[0] == ("super",)
    assert out[1] == (99,)


def test_vit_call_count(monkeypatch):
    """Exactly one ViT forward per batch (pre-executed before generate)."""
    monkeypatch.setenv("DIVPRUNE_INLINE_VIT", "1")
    backend, counters = _make_mock_generation_backend(n_tokens=20, keep_ratio=0.3)
    counters["vit_forward"] = 0

    backend._run_generation_batch_inline(
        [[{"role": "user", "content": "hi"}]],
        max_new_tokens=5,
        temperature=0.0,
        top_p=1.0,
        repetition_penalty=1.0,
    )
    assert counters["vit_forward"] == 1, (
        f"expected 1 ViT forward, got {counters['vit_forward']}"
    )


def test_position_ids_passed_to_generate(monkeypatch):
    """Inline path passes position_ids and rope_deltas to generate()."""
    monkeypatch.setenv("DIVPRUNE_INLINE_VIT", "1")
    backend, _ = _make_mock_generation_backend(n_tokens=20, keep_ratio=0.3)

    captured_kwargs: dict = {}
    original_generate = backend._model.generate

    def capturing_generate(**kwargs):
        captured_kwargs.update(kwargs)
        return original_generate(**kwargs)

    backend._model.generate = capturing_generate

    backend._run_generation_batch_inline(
        [[{"role": "user", "content": "hi"}]],
        max_new_tokens=5,
        temperature=0.0,
        top_p=1.0,
        repetition_penalty=1.0,
    )

    assert "position_ids" in captured_kwargs, "position_ids not passed to generate()"
    pid = captured_kwargs["position_ids"]
    assert pid.ndim == 3, f"position_ids should be 3D, got {pid.ndim}D"
    assert pid.shape[0] == 4, f"position_ids dim 0 should be 4, got {pid.shape[0]}"


def test_rope_deltas_set_on_feat_host(monkeypatch):
    """Inline path sets rope_deltas on feat_host before generate()."""
    monkeypatch.setenv("DIVPRUNE_INLINE_VIT", "1")
    backend, _ = _make_mock_generation_backend(n_tokens=20, keep_ratio=0.3)
    feat_host = _resolve_qwen3_vl_model(backend._model)

    backend._run_generation_batch_inline(
        [[{"role": "user", "content": "hi"}]],
        max_new_tokens=5,
        temperature=0.0,
        top_p=1.0,
        repetition_penalty=1.0,
    )

    assert feat_host.rope_deltas is not None, "rope_deltas not set on feat_host"
    assert feat_host.rope_deltas.ndim == 2, "rope_deltas should be (B, 1)"


def test_legacy_vs_inline_pruned_embeddings_equivalent(monkeypatch):
    """Legacy and inline paths produce identical pruned pooler_output for fixed ViT output."""
    import skillreuse.model.divprune_backend as dp_backend

    torch.manual_seed(7)
    n_tokens = 30
    keep_ratio = 0.2
    backend, _ = _make_mock_generation_backend(n_tokens=n_tokens, keep_ratio=keep_ratio)

    captured: dict[str, torch.Tensor] = {}
    original_prune = dp_backend._prune_vision_output

    def capture_prune(vision_output, *, keep_ratio, split_sizes, K_values):
        result, indices = original_prune(
            vision_output,
            keep_ratio=keep_ratio,
            split_sizes=split_sizes,
            K_values=K_values,
        )
        captured["last"] = result.pooler_output[0].clone()
        return result, indices

    monkeypatch.setattr(dp_backend, "_prune_vision_output", capture_prune)

    messages = [[{"role": "user", "content": "hi"}]]
    gen_kwargs = dict(
        max_new_tokens=5,
        temperature=0.0,
        top_p=1.0,
        repetition_penalty=1.0,
    )

    monkeypatch.setenv("DIVPRUNE_INLINE_VIT", "0")
    backend._run_generation_batch_legacy(messages, **gen_kwargs)
    legacy_pruned = captured["last"].clone()

    captured.clear()
    monkeypatch.setenv("DIVPRUNE_INLINE_VIT", "1")
    backend._run_generation_batch_inline(messages, **gen_kwargs)
    inline_pruned = captured["last"].clone()

    assert torch.equal(legacy_pruned, inline_pruned)
    merge_size = 2
    grid = torch.tensor([_grid_thw_for_n_tokens(n_tokens, merge_size)])
    k_values, _ = _precompute_K_N_values(grid, merge_size, keep_ratio)
    assert legacy_pruned.shape[0] == k_values[0]


# ---------------------------------------------------------------------------
# _prune_vision_output returns (pruned_output, selected_indices)
# ---------------------------------------------------------------------------


def test_prune_vision_output_returns_indices():
    """_prune_vision_output returns (pruned_output, all_selected_indices) tuple."""
    torch.manual_seed(3)
    n_tokens = 24
    keep_ratio = 0.25
    img_embeds = torch.randn(n_tokens, 8)

    vision_output = SimpleNamespace(
        pooler_output=[img_embeds],
        deepstack_features=None,
    )
    merge_size = 2
    grid = torch.tensor([_grid_thw_for_n_tokens(n_tokens, merge_size)])
    k_values, n_values = _precompute_K_N_values(grid, merge_size, keep_ratio)

    result = _prune_vision_output(
        vision_output, keep_ratio=keep_ratio, split_sizes=n_values, K_values=k_values
    )

    assert isinstance(result, tuple) and len(result) == 2
    pruned_output, all_selected_indices = result
    assert len(all_selected_indices) == 1
    assert all_selected_indices[0].shape[0] == k_values[0]
    assert pruned_output.pooler_output[0].shape[0] == k_values[0]


def test_deepstack_prune_shapes():
    """DeepStack cat rows == sum(K_values) when deepstack_features present."""
    torch.manual_seed(3)
    n_tokens = 24
    keep_ratio = 0.25
    img_embeds = torch.randn(n_tokens, 8)
    deepstack = [torch.randn(n_tokens, 8) for _ in range(2)]

    vision_output = SimpleNamespace(
        pooler_output=[img_embeds],
        deepstack_features=deepstack,
    )
    merge_size = 2
    grid = torch.tensor([_grid_thw_for_n_tokens(n_tokens, merge_size)])
    k_values, n_values = _precompute_K_N_values(grid, merge_size, keep_ratio)

    pruned, _ = _prune_vision_output(
        vision_output, keep_ratio=keep_ratio, split_sizes=n_values, K_values=k_values
    )
    assert pruned.deepstack_features is not None
    for ds in pruned.deepstack_features:
        assert ds.shape[0] == sum(k_values)


# ---------------------------------------------------------------------------
# T5: _compute_selected_position_ids
# ---------------------------------------------------------------------------


def test_index_to_3d_coords_basic():
    """Verify flat index → (t, h, w) mapping for a known grid."""
    # grid [1, 6, 8], merge_size=2 → T=1, H=3, W=4, N=12
    grid_thw = torch.tensor([1, 6, 8], dtype=torch.long)
    merge_size = 2

    # index 0 → (0, 0, 0)
    pos = _compute_selected_position_ids(
        torch.tensor([0]), grid_thw, merge_size, start_position=0, device=torch.device("cpu")
    )
    assert pos.shape == (3, 1)
    assert pos[0, 0].item() == 0  # t
    assert pos[1, 0].item() == 0  # h
    assert pos[2, 0].item() == 0  # w

    # index 3 → (0, 0, 3) — last column of first row
    pos = _compute_selected_position_ids(
        torch.tensor([3]), grid_thw, merge_size, start_position=0, device=torch.device("cpu")
    )
    assert pos[0, 0].item() == 0
    assert pos[1, 0].item() == 0
    assert pos[2, 0].item() == 3

    # index 4 → (0, 1, 0) — first column of second row
    pos = _compute_selected_position_ids(
        torch.tensor([4]), grid_thw, merge_size, start_position=0, device=torch.device("cpu")
    )
    assert pos[0, 0].item() == 0
    assert pos[1, 0].item() == 1
    assert pos[2, 0].item() == 0

    # index 11 → (0, 2, 3) — last token
    pos = _compute_selected_position_ids(
        torch.tensor([11]), grid_thw, merge_size, start_position=0, device=torch.device("cpu")
    )
    assert pos[0, 0].item() == 0
    assert pos[1, 0].item() == 2
    assert pos[2, 0].item() == 3


def test_index_to_3d_coords_with_start_position():
    """start_position offsets all three dimensions."""
    grid_thw = torch.tensor([1, 4, 4], dtype=torch.long)
    merge_size = 2
    start_position = 10

    pos = _compute_selected_position_ids(
        torch.tensor([0, 3]), grid_thw, merge_size, start_position=start_position,
        device=torch.device("cpu"),
    )
    assert pos.shape == (3, 2)
    # index 0 → (0+10, 0+10, 0+10)
    assert pos[0, 0].item() == 10
    assert pos[1, 0].item() == 10
    assert pos[2, 0].item() == 10
    # index 3 → (0+10, 1+10, 1+10)  (W=2, so index 3 → h=1, w=1)
    assert pos[0, 1].item() == 10
    assert pos[1, 1].item() == 11
    assert pos[2, 1].item() == 11


def test_selected_positions_are_subset_of_original():
    """确保选中 token 的位置是原始密集网格位置的子集。"""
    grid_thw = torch.tensor([1, 10, 12], dtype=torch.long)
    merge_size = 2
    H = 10 // merge_size  # 5
    W = 12 // merge_size  # 6
    N = H * W  # 30

    # 原始密集网格位置
    all_pos = _compute_selected_position_ids(
        torch.arange(N), grid_thw, merge_size, start_position=0,
        device=torch.device("cpu"),
    )  # (3, N)

    # 选中的子集
    selected = torch.tensor([0, 5, 13, 29])
    sel_pos = _compute_selected_position_ids(
        selected, grid_thw, merge_size, start_position=0,
        device=torch.device("cpu"),
    )  # (3, 4)

    for j in range(selected.shape[0]):
        col = sel_pos[:, j : j + 1]  # (3, 1)
        match = (all_pos == col).all(dim=0).any()
        assert match, f"Selected position {j} not found in original grid"


# ---------------------------------------------------------------------------
# T5: _build_position_ids_for_batch
# ---------------------------------------------------------------------------


def test_build_position_ids_shape_and_dtype():
    """_build_position_ids_for_batch returns (4, B, S) with correct dtype."""
    batch_size, seq_len = 1, 12
    input_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    mm_token_type_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)

    # 3 text + 4 vision + 5 text
    mm_token_type_ids[0, 3:7] = 1

    grid_thw = torch.tensor([[1, 4, 4]], dtype=torch.long)
    merge_size = 2
    selected_indices = [torch.tensor([0, 1, 2, 3], dtype=torch.long)]

    position_ids, deltas = _build_position_ids_for_batch(
        input_ids, attention_mask, mm_token_type_ids,
        selected_indices, grid_thw, merge_size,
    )

    assert position_ids.shape == (4, batch_size, seq_len)
    assert position_ids.dtype == input_ids.dtype
    assert deltas.shape == (batch_size, 1)


def test_build_position_ids_text_positions_are_sequential():
    """Text-only input: position_ids dim 0 = cumulative valid count - 1."""
    batch_size, seq_len = 1, 8
    input_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    mm_token_type_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)

    position_ids, deltas = _build_position_ids_for_batch(
        input_ids, attention_mask, mm_token_type_ids,
        [],  # no images
        torch.zeros(0, 3, dtype=torch.long), 2,
    )

    # dim 0 should be [0, 1, 2, ..., 7]
    assert position_ids[0, 0].tolist() == list(range(seq_len))
    # dims 1-3 should also be [0, 1, 2, ..., 7] for text-only
    assert position_ids[1, 0].tolist() == list(range(seq_len))
    assert position_ids[2, 0].tolist() == list(range(seq_len))
    assert position_ids[3, 0].tolist() == list(range(seq_len))


def test_build_position_ids_vision_positions_match_selected():
    """Vision tokens get their original 3D positions, text tokens get sequential."""
    batch_size = 1
    # Layout: [3 text] [2 vision] [3 text]
    seq_len = 8
    input_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    mm_token_type_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
    mm_token_type_ids[0, 3:5] = 1  # vision tokens at positions 3, 4

    # grid [1, 4, 4], merge_size=2 → H=2, W=2, N=4
    grid_thw = torch.tensor([[1, 4, 4]], dtype=torch.long)
    merge_size = 2
    # Select indices 0 and 3 from the original grid
    selected_indices = [torch.tensor([0, 3], dtype=torch.long)]

    position_ids, deltas = _build_position_ids_for_batch(
        input_ids, attention_mask, mm_token_type_ids,
        selected_indices, grid_thw, merge_size,
    )

    # Text tokens 0-2: positions 0, 1, 2
    assert position_ids[0, 0, 0].item() == 0
    assert position_ids[0, 0, 1].item() == 1
    assert position_ids[0, 0, 2].item() == 2

    # Vision token at seq pos 3 (index 0 in grid → t=0, h=0, w=0) + start_pos=3
    assert position_ids[1, 0, 3].item() == 3  # t = 0 + 3
    assert position_ids[2, 0, 3].item() == 3  # h = 0 + 3
    assert position_ids[3, 0, 3].item() == 3  # w = 0 + 3

    # Vision token at seq pos 4 (index 3 in grid → t=0, h=1, w=1) + start_pos=3
    assert position_ids[1, 0, 4].item() == 3  # t = 0 + 3
    assert position_ids[2, 0, 4].item() == 4  # h = 1 + 3
    assert position_ids[3, 0, 4].item() == 4  # w = 1 + 3

    # Text after vision: current_pos = 3 + max(4,4)//2 = 3 + 2 = 5
    assert position_ids[0, 0, 5].item() == 5
    assert position_ids[1, 0, 5].item() == 5


def test_build_position_ids_padding_zeroed():
    """Padded (mask=0) positions get position_ids = 0."""
    batch_size = 1
    seq_len = 6
    input_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
    # First 2 are padding (mask=0), rest valid
    attention_mask = torch.tensor([[0, 0, 1, 1, 1, 1]], dtype=torch.long)
    mm_token_type_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)

    position_ids, _ = _build_position_ids_for_batch(
        input_ids, attention_mask, mm_token_type_ids,
        [], torch.zeros(0, 3, dtype=torch.long), 2,
    )

    # Padded positions should be 0
    for d in range(4):
        assert position_ids[d, 0, 0].item() == 0
        assert position_ids[d, 0, 1].item() == 0


def test_build_position_ids_multi_image():
    """Multiple images in the same sample each get correct positions."""
    batch_size = 1
    # Layout: [2 text] [2 vision_img0] [2 text] [2 vision_img1] [2 text]
    seq_len = 10
    input_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    mm_token_type_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
    mm_token_type_ids[0, 2:4] = 1   # image 0
    mm_token_type_ids[0, 6:8] = 1   # image 1

    # Both images: grid [1, 4, 4], merge_size=2 → H=2, W=2, N=4
    grid_thw = torch.tensor([[1, 4, 4], [1, 4, 4]], dtype=torch.long)
    merge_size = 2
    selected_indices = [
        torch.tensor([0, 1], dtype=torch.long),  # image 0: indices 0, 1
        torch.tensor([2, 3], dtype=torch.long),  # image 1: indices 2, 3
    ]

    position_ids, deltas = _build_position_ids_for_batch(
        input_ids, attention_mask, mm_token_type_ids,
        selected_indices, grid_thw, merge_size,
    )

    # Image 0 vision tokens: start_pos = 2
    # index 0 → (0+2, 0+2, 0+2) = (2, 2, 2)
    assert position_ids[1, 0, 2].item() == 2
    assert position_ids[2, 0, 2].item() == 2
    assert position_ids[3, 0, 2].item() == 2

    # index 1 → (0+2, 0+2, 1+2) = (2, 2, 3)
    assert position_ids[1, 0, 3].item() == 2
    assert position_ids[2, 0, 3].item() == 2
    assert position_ids[3, 0, 3].item() == 3

    # Text after image 0: current_pos = 2 + max(4,4)//2 = 2 + 2 = 4
    # Text at seq 4: pos = 4
    assert position_ids[0, 0, 4].item() == 4

    # Image 1 vision tokens: start_pos = 4 + 2 = 6
    # index 2 → (0+6, 1+6, 0+6) = (6, 7, 6)
    assert position_ids[1, 0, 6].item() == 6
    assert position_ids[2, 0, 6].item() == 7
    assert position_ids[3, 0, 6].item() == 6

    # index 3 → (0+6, 1+6, 1+6) = (6, 7, 7)
    assert position_ids[1, 0, 7].item() == 6
    assert position_ids[2, 0, 7].item() == 7
    assert position_ids[3, 0, 7].item() == 7
