# Vendored MSD/EAGLE inference core

Source:

- Repository: `https://github.com/Lyn-Lucy/MSD`
- Commit: `fd76a5ee2bd107a5a04f05afd0651aab7ba6fab4`
- Original subtree: `EAGLE/eagle/model/`

The Qwen2-VL inference files were copied without functional changes first, and
then package imports and model dispatch were narrowed to the Qwen2-VL-only
GUIAccel integration. The upstream Apache 2.0 license is preserved in
`LICENSE.EAGLE`.

Vendored files:

- `ea_model.py`
- `cnets.py`
- `utils.py`
- `kv_cache.py`
- `configs.py`
- `choices.py`
- `modeling_qwen2vl_kv.py`
