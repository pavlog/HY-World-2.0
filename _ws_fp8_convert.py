"""Offline: quantize the WorldStereo transformer (34GB bf16) -> ~18GB weight-only FP8 on disk.
Streams the source tensor-by-tensor (low RAM read); big 2D .weight -> float8_e4m3fn, everything else
(norms, 1D, biases, conv/patch-embed, modulation) stays bf16 — the GGUF-Q8 convention. Lets the loader
skip the 34GB bf16 peak (load the 18GB file straight into a meta-init arch).

Peak RAM ~= the 18GB output dict (held until save_file). Safe with ~50GB free.
"""
import torch
from safetensors import safe_open
from safetensors.torch import save_file

SRC = r"D:/HF_MODELS/WorldStereo/worldstereo-memory-dmd/model.safetensors"
DST = r"D:/HF_MODELS/WorldStereo/worldstereo-memory-dmd/model_fp8.safetensors"

out, n_fp8, n_keep, fp8_keys = {}, 0, 0, []
with safe_open(SRC, framework="pt", device="cpu") as f:
    src_meta = f.metadata() or {}
    keys = list(f.keys())
    for i, k in enumerate(keys):
        t = f.get_tensor(k)
        if t.ndim == 2 and t.numel() > 1_000_000 and k.endswith(".weight"):
            out[k] = t.to(torch.float8_e4m3fn)          # weight-only fp8 (no scale: direct e4m3, ComfyUI-style)
            n_fp8 += 1; fp8_keys.append(k)
        else:
            out[k] = t.to(torch.bfloat16)
        n_keep = len(out) - n_fp8
        del t
        if i % 300 == 0:
            print(f"  {i}/{len(keys)} ... fp8={n_fp8} bf16={n_keep}", flush=True)

meta = {k: str(v) for k, v in src_meta.items()}
meta["ws_fp8"] = "weight-only float8_e4m3fn on 2D .weight numel>1M; rest bf16"
print(f"writing {len(out)} tensors (fp8={n_fp8}, bf16={n_keep}) -> {DST}", flush=True)
save_file(out, DST, metadata=meta)
import os
print(f"DONE: {os.path.getsize(DST)/1e9:.2f} GB | fp8 layers: {n_fp8}")
