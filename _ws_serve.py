"""WorldStereo RESIDENT SERVER — load the 17B (+aux) ONCE, then serve many clips from stdin.
The graph showed ~10s of real compute buried in ~3min of per-run model loading; this kills the reload.

  python _ws_serve.py            # then type/pipe requests, one per line:
     <scene_dir>|<prompt>        # -> writes <scene>/render_results/view0/traj0/<model>_result.mp4
     quit                        # exit

Models stay resident in RAM (encoders) / GPU (fp8 transformer via model-offload); each request is just
encode+denoise+decode (~10-20s). Same winning recipe as _ws_core: set WS_FP8_FILE, WS_OFFLOAD=1,
WS_SAGE=1, WS_FP8_UMT5=1 before launching. See [[worldstereo-runs-on-4090]].
"""
import os, sys, json, shutil, time
sys.path.insert(0, r"D:/HY-World-2.0/hyworld2/worldgen")

# ---- component-reuse env-gating (identical to _ws_core) ----
os.environ["HF_HOME"] = r"D:/HF_MODELS"; os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["WORLDSTEREO_TRANSFORMER_CONFIG"] = r"D:/HF_MODELS/wan21_i2v_cfg/transformer/config.json"
os.environ["WORLDSTEREO_TOKENIZER"] = "google/umt5-xxl"
_UMT5 = (r"D:/Models/models/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors"
         if os.environ.get("WS_FP8_UMT5") == "1" else
         r"D:/Models/models/text_encoders/umt5-xxl-enc-bf16.safetensors")
os.environ["WORLDSTEREO_LOCAL_UMT5"] = _UMT5
os.environ["WORLDSTEREO_LOCAL_VAE"]  = r"D:/Models/models/vae/Wan2_1_VAE_bf16.safetensors"
os.environ["WORLDSTEREO_LOCAL_CLIP"] = r"D:/Models/models/clip_vision/model.safetensors"

import types, torch
import torch.nn.functional as F
import torch.distributed as dist
torch.compile = lambda model=None, *a, **k: model

# SDPA -> SageAttention (triton int8) — same patch as _ws_core
if os.environ.get("WS_SAGE") == "1":
    from sageattention import sageattn as _sage   # auto: fp8 CUDA kernel on sm89 (5.5ms, no JIT, 3x SDPA)
    _orig_sdpa = F.scaled_dot_product_attention
    def _sage_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **kw):
        if (attn_mask is None and scale is None and query.shape[-1] in (64, 128)
                and query.dtype in (torch.float16, torch.bfloat16) and query.shape[-2] > 1):
            try:
                return _sage(query, key, value, tensor_layout="HND", is_causal=is_causal)
            except Exception:
                pass
        return _orig_sdpa(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p,
                          is_causal=is_causal, scale=scale, **kw)
    F.scaled_dot_product_attention = _sage_sdpa
    torch.nn.functional.scaled_dot_product_attention = _sage_sdpa
    print("[ws_serve] SDPA -> SageAttention (auto/fp8-cuda sm89)")

from torch.distributed.device_mesh import init_device_mesh
from diffusers.utils import export_to_video
from models.worldstereo_wrapper import WorldStereo
from src.data_utils import load_mutli_traj_dataset
from src.sp_utils.parallel_states import initialize_parallel_state

VIEW, TRAJ = "view0", "traj0"
MODEL_TYPE = "worldstereo-memory-dmd"
_REF_IDX = torch.tensor([0], dtype=torch.long)

# ================= LOAD ONCE =================
os.environ["MASTER_ADDR"] = "127.0.0.1"; os.environ["MASTER_PORT"] = "29501"; os.environ["USE_LIBUV"] = "0"
for k, v in dict(RANK="0", WORLD_SIZE="1", LOCAL_RANK="0").items():
    os.environ.setdefault(k, v)
device = torch.device("cuda:0"); torch.cuda.set_device(0)
dist.init_process_group(backend="gloo", rank=0, world_size=1)
device_mesh = init_device_mesh("cuda", (1, 1), mesh_dim_names=("rep", "shard"))
initialize_parallel_state(sp=1)

print("[ws_serve] loading WorldStereo (once)…", flush=True)
_t0 = time.time()
ws = WorldStereo.from_pretrained(r"D:/HF_MODELS/WorldStereo", subfolder=MODEL_TYPE,
                                 local_files_only=True, sp_world_size=1, fsdp=False,
                                 device_mesh=device_mesh, device=device)
# MANUAL placement (NO model_cpu_offload — its per-module forward hooks serialize the GPU to ~115W).
# vae is tiny (~0.5GB) -> resident on GPU always. umt5/clip -> GPU only for the encode phase, then forced
# back to CPU. transformer -> GPU only for denoise. So encode peak = umt5(11)+clip(1.2)+vae = ~13GB; denoise
# peak = transformer(17.45)+vae+activation(6) = ~24GB. Both fit 24GB; denoise runs hook-free -> saturates.
PIPE = ws.pipeline
PIPE.vae.to(device)
PIPE.image_encoder.to(dtype=torch.bfloat16).to(device)   # clip is small (~1.2GB bf16) -> resident; pipeline encodes image internally
PIPE.transformer.to("cpu"); PIPE.text_encoder.to("cpu")  # umt5(11GB) + transformer(17GB) swap per phase

def _to_gpu(m):
    return m.to(device)             # pinning the full 17GB+11GB host copies OOMs CUDA pinned mem on Windows;
def _to_cpu_pinned(m):             # keep pageable swaps (~3s, not the dominant cost; denoise 12.9s @430W is).
    m.to("cpu")
torch.cuda.empty_cache()
print(f"[ws_serve] ready in {time.time()-_t0:.1f}s — send '<scene_dir>|<prompt>' per line, 'quit' to exit", flush=True)


import hashlib, glob
PROMPT_CACHE = r"D:/HY-World-2.0/_ws_prompt_cache"; os.makedirs(PROMPT_CACHE, exist_ok=True)
CACHE_LIMIT = 500               # LRU: keep the 500 most-recently-used prompt_embeds, evict the oldest
_STATE = {"tr_on_gpu": False}   # keep the transformer RESIDENT across requests; only swap it off when umt5 is needed

def _cache_evict():
    files = sorted(glob.glob(f"{PROMPT_CACHE}/*.pt"), key=os.path.getmtime)
    for f in files[:-CACHE_LIMIT]:                      # everything older than the newest CACHE_LIMIT
        try: os.remove(f)
        except OSError: pass

def run_one(scene, prompt, seed=1024):
    rr = f"{scene}/render_results"
    mem = f"{rr}/{VIEW}/{TRAJ}/memory_inputs"; os.makedirs(mem, exist_ok=True)
    if not os.path.exists(f"{mem}/{MODEL_TYPE}.mp4"):
        shutil.copy(f"{rr}/{VIEW}/{TRAJ}/render.mp4", f"{mem}/{MODEL_TYPE}.mp4")
    meta = load_mutli_traj_dataset(cfg=ws.cfg, input_path=rr, output_path=rr, view_id=VIEW, traj_id=TRAJ,
                                   device=device, ref_index=_REF_IDX, model_type=MODEL_TYPE, task_type="panorama")
    neg = ws.cfg.get("negative_prompt", "")
    cpath = f"{PROMPT_CACHE}/{hashlib.md5((prompt + '||' + neg).encode()).hexdigest()}.pt"

    # ---- ENCODE: cached prompt_embeds -> skip umt5 entirely (transformer stays resident). Else swap-encode-cache.
    swp_te = swp_tr = enc_dt = 0.0
    if os.path.exists(cpath):
        os.utime(cpath, None)                            # LRU touch: mark recently used
        d = torch.load(cpath, map_location=device)
        pe, ne = d["pe"], d["ne"]
    else:
        if _STATE["tr_on_gpu"]:                                   # need umt5 room -> evict transformer
            PIPE.transformer.to("cpu"); _STATE["tr_on_gpu"] = False; torch.cuda.empty_cache()
        ts = time.time(); _to_gpu(PIPE.text_encoder); torch.cuda.synchronize(); swp_te = time.time() - ts
        t0 = time.time()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            pe, ne = PIPE.encode_prompt(prompt=prompt, negative_prompt=neg, do_classifier_free_guidance=False,
                                        num_videos_per_prompt=1, max_sequence_length=512, device=device)
        enc_dt = time.time() - t0
        torch.save({"pe": pe.cpu(), "ne": (ne.cpu() if ne is not None else None)}, cpath)
        _cache_evict()                                   # keep cache <= CACHE_LIMIT (LRU)
        PIPE.text_encoder.to("cpu"); torch.cuda.empty_cache()

    # ---- DENOISE: ensure transformer resident (swap in only if it was evicted), keep it on GPU afterwards ----
    if not _STATE["tr_on_gpu"]:
        ts = time.time(); _to_gpu(PIPE.transformer); torch.cuda.synchronize(); swp_tr = time.time() - ts
        _STATE["tr_on_gpu"] = True
    pe = pe.to(device); ne = ne.to(device) if ne is not None else None
    kwargs = {k: v for k, v in meta.items() if v is not None}
    kwargs.update(prompt=None, prompt_embeds=pe, negative_prompt_embeds=ne,
                  generator=torch.Generator(device=device).manual_seed(seed), output_type="pt",
                  latent_cond_mode=ws.cfg.latent_cond_mode, mode="test",
                  num_frames=ws.cfg.nframe, ref_index=_REF_IDX.to(device))
    torch.cuda.reset_peak_memory_stats()
    t_den = time.time()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = PIPE(**kwargs).frames[0].float()
    den_dt = time.time() - t_den                                 # transformer stays resident for the next request

    nan = torch.isnan(out).sum().item() + torch.isinf(out).sum().item()
    arr = out.permute(0, 2, 3, 1).cpu().numpy()
    res = f"{rr}/{VIEW}/{TRAJ}/{MODEL_TYPE}_result.mp4"
    export_to_video(arr, res, fps=16)
    peak = torch.cuda.max_memory_allocated() / 1e9
    cached = "CACHED" if swp_te == 0 and enc_dt == 0 else "fresh"
    print(f"[ws_serve] DONE [{cached}] swap(umt5={swp_te:.1f}s tr={swp_tr:.1f}s) enc={enc_dt:.1f}s "
          f"denoise={den_dt:.1f}s peak={peak:.1f}GB NaN={nan} -> {res}", flush=True)
    return res


# ================= SERVE LOOP (only when run as a script; importable as a module for chained drivers) ====
if __name__ == "__main__":
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        if line.lower() in ("quit", "exit", "q"):
            break
        try:
            scene, _, prompt = line.partition("|")
            scene = scene.strip() or r"D:/HY-World-2.0/_ws_scene/case_cockpit"
            prompt = prompt.strip() or "interior of a sci-fi spaceship cockpit, glowing blue consoles, cinematic"
            run_one(scene, prompt)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"[ws_serve] ERROR: {e}", flush=True)
    if dist.is_initialized():
        dist.destroy_process_group()
