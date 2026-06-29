"""WorldStereo CORE standalone driver — one trajectory, NO memory bank / SAM3 / WorldMirror.
Calls worldstereo.pipeline on a hand-made scene-dir (from _ws_scene_build.py). Per-clip text prompt is NATIVE
(pipeline __call__ accepts `prompt`). A per-frame prompt-travel patch point is marked below.

  python _ws_core.py [scene_dir] [prompt]

⚠️ DO NOT run unattended. The 17B Wan transformer in bf16 = ~34 GB > 24 GB VRAM — it WILL OOM as-is.
   Before a real run you must EITHER FP8-quantize the transformer OR enable cpu_offload/block-swap
   (Lyra playbook). This script is the wiring; the VRAM fit is the open item (see [[lyra2-4090-vram-reality]]).

VERIFY-ON-FIRST-RUN flags are marked `# VERIFY:` inline.
"""
import os, sys, json, shutil
sys.path.insert(0, r"D:/HY-World-2.0/hyworld2/worldgen")

# ---- component-reuse env-gating (skip 28GB transformer DL + use LOCAL Wan aux) ----
os.environ["HF_HOME"] = r"D:/HF_MODELS"
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["WORLDSTEREO_TRANSFORMER_CONFIG"] = r"D:/HF_MODELS/wan21_i2v_cfg/transformer/config.json"   # VERIFY: 466-byte Wan-I2V transformer config present
os.environ["WORLDSTEREO_TOKENIZER"]   = "google/umt5-xxl"
# VRAM levers (env): WS_OFFLOAD=1 -> pipeline cpu-offload; WS_FP8_UMT5=1 -> use the FP8 UMT5 (saves ~5GB)
_UMT5 = (r"D:/Models/models/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors"
         if os.environ.get("WS_FP8_UMT5") == "1" else
         r"D:/Models/models/text_encoders/umt5-xxl-enc-bf16.safetensors")
os.environ["WORLDSTEREO_LOCAL_UMT5"]  = _UMT5
os.environ["WORLDSTEREO_LOCAL_VAE"]   = r"D:/Models/models/vae/Wan2_1_VAE_bf16.safetensors"
os.environ["WORLDSTEREO_LOCAL_CLIP"]  = r"D:/Models/models/clip_vision/model.safetensors"               # VERIFY: CLIP-H vision path

import types
import torch
import torch.nn.functional as F
import torch.distributed as dist

# torch.compile crashes on this Windows build (inductor/triton "duplicate template name") — make it a no-op.
torch.compile = lambda model=None, *a, **k: model

# WS_SAGE=1: route unmasked SDPA -> SageAttention triton int8 kernel (JITs via triton, bypasses the broken
# CUDA .pyd built against 12.5). err vs SDPA ~1e-4. Falls back to torch SDPA for masked/odd-dim/fp32 calls.
if os.environ.get("WS_SAGE") == "1":
    from sageattention import sageattn as _sage   # auto: fp8 CUDA kernel on sm89 (no JIT, ~3x SDPA)
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
    print("[ws_core] SDPA -> SageAttention (triton int8) for unmasked attention")
from torch.distributed.device_mesh import init_device_mesh
from diffusers.utils import export_to_video
from models.worldstereo_wrapper import WorldStereo
from src.data_utils import load_mutli_traj_dataset
from src.sp_utils.parallel_states import initialize_parallel_state

SCENE      = sys.argv[1] if len(sys.argv) > 1 else r"D:/HY-World-2.0/_ws_scene/case_cockpit"
PROMPT     = sys.argv[2] if len(sys.argv) > 2 else "interior of a sci-fi spaceship cockpit, glowing consoles, cinematic, high detail"
VIEW, TRAJ = "view0", "traj0"
MODEL_TYPE = "worldstereo-memory-dmd"
RR = f"{SCENE}/render_results"

# ---- single-GPU distributed init (the code is wired for SP/FSDP; world_size=1 works) ----
# Windows: no NCCL (Linux-only) -> gloo backend; force localhost (avoid kubernetes.docker.internal resolution).
os.environ["MASTER_ADDR"] = "127.0.0.1"; os.environ["MASTER_PORT"] = "29501"; os.environ["USE_LIBUV"] = "0"
for k, v in dict(RANK="0", WORLD_SIZE="1", LOCAL_RANK="0").items():
    os.environ.setdefault(k, v)
device = torch.device("cuda:0"); torch.cuda.set_device(0)
dist.init_process_group(backend="gloo", rank=0, world_size=1)
device_mesh = init_device_mesh("cuda", (1, 1), mesh_dim_names=("rep", "shard"))
initialize_parallel_state(sp=1)

# ---- load WorldStereo (component reuse, no FSDP) ----
# VERIFY: 17B fit — add FP8 quant or cpu_offload here before a real run, else OOM.
ws = WorldStereo.from_pretrained(r"D:/HF_MODELS/WorldStereo", subfolder=MODEL_TYPE,   # local dir -> reads {dir}/{subfolder}/{config,model}
                                 local_files_only=True, sp_world_size=1, fsdp=False,
                                 device_mesh=device_mesh, device=device)

# ---- WEIGHT-ONLY FP8: cast big Linear weights 34GB->17GB resident (norms/embeds stay bf16);
#      forward upcasts weight->activation dtype, so compute stays bf16 (no scaled_mm needed). ----
if os.environ.get("WS_FP8_TRANSFORMER") == "1" and not os.environ.get("WS_FP8_FILE"):
    def _fp8_fwd(self, x):
        return F.linear(x, self.weight.to(x.dtype), self.bias)
    _n = 0
    for _m in ws.pipeline.transformer.modules():
        if isinstance(_m, torch.nn.Linear) and _m.weight.ndim == 2 and _m.weight.numel() > 1_000_000:
            _m.weight.data = _m.weight.data.to(torch.float8_e4m3fn)     # ~half the bytes
            _m.forward = types.MethodType(_fp8_fwd, _m)
            _n += 1
    import gc as _gc; _gc.collect()
    print(f"[ws_core] FP8 weight-only on {_n} Linear layers (norms/embeds kept bf16)")

# ---- VRAM fit for 24GB: the 17B bf16 transformer (~34GB) does NOT fit. Pick a lever (VERIFY which the wrapper honors): ----
if os.environ.get("WS_OFFLOAD") == "1":
    # SEQUENTIAL cpu-offload = submodule(block)-granularity: the 34GB transformer's blocks stream to GPU one
    # at a time during denoise (model-offload would move the WHOLE 34GB transformer to GPU -> OOM>24GB).
    # Pair with WS_LOWMEM_LOAD=1 (streamed load, transformer stays on CPU). Slower but fits 24GB.
    # MODEL offload (whole-module): the fp8 transformer (17GB) moves to GPU as ONE unit for denoise -> FITS 24GB,
    # no shared-memory spill / PCIe paging. umt5/vae move in for their own phases. (sequential = submodule churn
    # that overflowed dedicated VRAM into shared.) Use WS_OFFLOAD_MODE=sequential to force the old path.
    _mode = os.environ.get("WS_OFFLOAD_MODE", "model")
    try:
        if _mode == "sequential":
            ws.pipeline.enable_sequential_cpu_offload(device=device)
            print("[ws_core] enabled SEQUENTIAL cpu-offload")
        else:
            ws.pipeline.enable_model_cpu_offload(device=device)
            print("[ws_core] enabled MODEL cpu-offload (whole-module; fp8 transformer fits 24GB)")
    except Exception as e:
        print(f"[ws_core] offload '{_mode}' failed ({e})")
# (b) FP8 transformer: stream-quantize D:/HF_MODELS/WorldStereo/.../model.safetensors -> float8_e4m3fn_scaled
#     ONCE to disk (low peak RAM via safetensors lazy load), then load that (17GB, fits 24GB). TODO worker.
# PREREQ: free CPU RAM + VRAM first — the box currently has ComfyUI (PID holding ~30GB RAM + VRAM) loaded.

# ---- STUB the memory reference (no memory bank): load_mutli_traj_dataset loads reference_video
#      unconditionally from memory_inputs/{model_type}.mp4. Use the render itself as a neutral stub. ----
mem = f"{RR}/{VIEW}/{TRAJ}/memory_inputs"; os.makedirs(mem, exist_ok=True)
if not os.path.exists(f"{mem}/{MODEL_TYPE}.mp4"):
    shutil.copy(f"{RR}/{VIEW}/{TRAJ}/render.mp4", f"{mem}/{MODEL_TYPE}.mp4")   # VERIFY: first-traj may prefer an empty ref

# WS_NFRAME: shrink the clip length to fit activations in 24GB dedicated VRAM (no shared spill / PCIe paging).
# nframe-1 must be divisible by 4 (VAE temporal). 21->6 latent-frames; 9->3; 13->4. Cuts activation ~linearly.
_nf = os.environ.get("WS_NFRAME")
if _nf:
    ws.cfg.nframe = int(_nf)
    print(f"[ws_core] nframe override -> {ws.cfg.nframe}")

# ref_index: the memory-dmd variant ALWAYS computes ref_rotary_emb[:, ref_index+1] (worldstereo.py:675),
# so ref_index can't be None even without a memory bank. post_patch_num_frames = latent frames = 6 here,
# so valid ref_index values are [0..4] (+1 -> [1..5]). Use a single self-reference frame [0]. CPU tensor for
# the reference_video[ref_index] filter (data_utils:187, CPU); device copy goes into kwargs for the GPU rope.
_REF_IDX = torch.tensor([0], dtype=torch.long)
meta = load_mutli_traj_dataset(cfg=ws.cfg, input_path=RR, output_path=RR, view_id=VIEW, traj_id=TRAJ,
                               device=device, ref_index=_REF_IDX, model_type=MODEL_TYPE, task_type="panorama")

# ---- pipeline kwargs + PER-CLIP PROMPT (native) ----
kwargs = {k: v for k, v in meta.items() if v is not None}
kwargs.update(
    prompt=PROMPT,                                          # <-- per-clip text control (native)
    negative_prompt=ws.cfg.get("negative_prompt", ""),
    generator=torch.Generator(device=device).manual_seed(1024),
    output_type="pt",
    latent_cond_mode=ws.cfg.latent_cond_mode,
    mode="test",                                            # DMD 4-step
    num_frames=ws.cfg.nframe,                               # match the (possibly overridden) clip length
    ref_index=_REF_IDX.to(device),                          # GPU copy: indexes ref_rotary_emb on cuda
)
# === PER-FRAME PROMPT-TRAVEL patch point ===
# To vary the prompt across the F frames: encode prompt_A & prompt_B (ws.pipeline.encode_prompt),
# slerp/lerp -> prompt_embeds [B, F, seq, dim], pass prompt_embeds=... instead of prompt, AND patch
# models/controlnet.py attn2 to index encoder_hidden_states by the frame axis `f`. Infra (frame tokens +
# cross-attn) is already there. Per-denoise variant: swap prompt_embeds between the 4 DMD steps in the loop.

print(f"[ws_core] scene={SCENE} prompt={PROMPT!r}")
def _vram(tag):
    a = torch.cuda.memory_allocated()/1e9; r = torch.cuda.memory_reserved()/1e9
    p = torch.cuda.max_memory_allocated()/1e9
    print(f"[vram] {tag}: alloc={a:.2f}GB reserved={r:.2f}GB peak_alloc={p:.2f}GB", flush=True)
_vram("pre-pipeline (resident model on GPU)")
torch.cuda.reset_peak_memory_stats()
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    out = ws.pipeline(**kwargs).frames[0].float()
_vram("post-pipeline (peak = denoise activation demand)")

# NaN guard (Lyra2 lesson): scale-less fp8 can overflow (e4m3 max ~448) -> inf -> NaN, silently black output.
_nan = torch.isnan(out).sum().item(); _inf = torch.isinf(out).sum().item()
_rng = (out.min().item(), out.max().item()) if _nan == 0 else (float("nan"), float("nan"))
print(f"[nan-guard] out NaN={_nan} Inf={_inf} range={_rng} shape={tuple(out.shape)}", flush=True)
if _nan or _inf:
    print("[nan-guard] !! OUTPUT CORRUPT — fp8 likely overflowed. Need per-tensor-scaled fp8 (C).", flush=True)

out = out.permute(0, 2, 3, 1).cpu().numpy()
res = f"{RR}/{VIEW}/{TRAJ}/{MODEL_TYPE}_result.mp4"
export_to_video(out, res, fps=16)
print(f"[ws_core] WS result -> {res}")
if dist.is_initialized():
    dist.destroy_process_group()
