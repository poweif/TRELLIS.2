# Vendored: MeshAnythingV2

- **Upstream**: https://github.com/buaacyw/MeshAnythingV2
- **Base commit**: `461d3b6ed750ab3443281b2e4a0e30e8ee98097e` (2025-04-29). Source only, no `.git`;
  one 16 MB demo clip removed.
- **Weights**: `Yiwen-ntu/meshanythingv2` (loaded by `trellis2/utils/meshanything_bridge.py`).

## Changes (for transformers ≥ 5; installed: 5.12.x)

- `MeshAnything/models/shape_opt.py`: `ShapeOPTDecoder.forward` rewritten for the `Cache` /
  `DynamicCache` API (upstream assumed tuple-of-tuples `past_key_values` and a 3-tuple layer
  return). `layer_idx` passed to `OPTDecoderLayer` (KV caching silently breaks without it).
- Removed calls to APIs that no longer exist: the `use_flash_attention_2=True` kwarg and
  `.to_bettertransformer()` (superseded by `attn_implementation` / SDPA, already set).
- `MeshAnything/miche/encode.py`: config YAML path made `__file__`-relative.
- Calling convention: the checkpoint loads in fp32, but the reference `main.py` feeds fp16. Call
  `.half()` on the model explicitly (the bridge does).

Full account: `docs/archive/history.md`, section "MeshAnything V2 vendored and patched".
