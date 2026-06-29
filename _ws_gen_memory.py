"""Memory-bank generation over OUR authored clips (one scene, traj0..N): HW2 PanoramaMemoryBank
retrieval + update_memory so appearance is held across clips. Reuses _ws_serve's loaded model.

Pipeline (plan Phase 0.2 — geometry=our prior skeleton, appearance=generation only):
  1. seed GEOMETRY scaffold from our cloud (global_pcd.ply + full_depth.pt)            [_ws_seed_bank]
  2. WARM-UP: generate traj0 self-ref (start_frame seed) -> result.mp4                  [_ws_serve.run_one]
  3. seed APPEARANCE store pano_bank from the GENERATED traj0 frames (NOT the prior)    [_ws_seed_pano_bank]
  4. construct PanoramaMemoryBank
  5. loop traj1..N: retrieval (real ref_index) -> dump _mem_debug PNGs -> memory-conditioned generate
     -> update_memory  (appearance held across clips)

  <WS env> worldmirror_py _ws_gen_memory.py <scene_dir> "<prompt>" --cloud <world_cloud.npz>
"""
import sys, os, json, shutil, time, glob as _glob
_ARGV = sys.argv[1:]; sys.argv = [sys.argv[0]]                 # so importing _ws_serve doesn't see our args
SCENE = _ARGV[0]
PROMPT = _ARGV[1] if len(_ARGV) > 1 and not _ARGV[1].startswith("--") else "sci-fi spaceship hangar interior, cinematic, high detail"
CLOUD = _ARGV[_ARGV.index("--cloud") + 1] if "--cloud" in _ARGV else None

import torch, numpy as np, cv2
import _ws_serve as WS                                          # loads WS model ONCE
sys.path.insert(0, r"D:/HY-World-2.0/hyworld2/worldgen")
sys.path.insert(0, r"D:/HY-World-2.0")
import _ws_seed_bank, _ws_seed_pano_bank
from src.retrieval_wm import PanoramaMemoryBank
from src.data_utils import load_mutli_traj_dataset, sort_trajs
from diffusers.utils import export_to_video, load_video
import imagesize
from moge.model.v2 import MoGeModel

dev, cfg, PIPE, MODEL = WS.device, WS.ws.cfg, WS.PIPE, WS.MODEL_TYPE
rr = f"{SCENE}/render_results"
w, h = imagesize.get(f"{rr}/view0/start_frame.png")
render_list = sorted(_glob.glob(f"{rr}/view*/traj*/render.mp4"))   # sort_trajs() is Windows-path-broken (bslash) + drops traj3
print(f"[mem] {len(render_list)} trajs, {w}x{h}, cloud={CLOUD}", flush=True)

# ---- 1. geometry scaffold (skeleton = our prior) ----
if CLOUD and not os.path.exists(f"{rr}/global_pcd.ply"):
    print("[mem] seeding geometry scaffold…", flush=True)
    _ws_seed_bank.main(SCENE, CLOUD)

import gc
from PIL import Image
all_trajs = sorted(int(p.replace("\\", "/").split("/")[-2].replace("traj", "")) for p in render_list)
SEED_EVERY = 8
SEED_TRAJS = sorted(set(all_trajs[::SEED_EVERY] + [all_trajs[-1]]))
print(f"[mem] SEED_TRAJS ({len(SEED_TRAJS)}): {SEED_TRAJS}", flush=True)

# ---- ENCODE PROMPT ONCE, THEN DROP umt5 ENTIRELY — before ANY generation ----
# CRITICAL: umt5 (11GB) + transformer (17.45GB) co-resident = 30.8GB -> pages/stalls. Encoding up front and
# freeing umt5 means every generation (seeds AND loop) runs transformer-only (19.5GB) -> fits 24GB. (Old run_one
# swapped per-call and could momentarily co-locate them -> the seed-phase stall.)
neg = cfg.get("negative_prompt", "")
PIPE.transformer.to("cpu"); WS._STATE["tr_on_gpu"] = False; torch.cuda.empty_cache()
PIPE.text_encoder.to(dev)
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    pe, ne = PIPE.encode_prompt(prompt=PROMPT, negative_prompt=neg, do_classifier_free_guidance=False,
                                num_videos_per_prompt=1, max_sequence_length=512, device=dev)
PIPE.text_encoder.to("cpu"); PIPE.text_encoder = None    # drop umt5 entirely
gc.collect(); torch.cuda.empty_cache()
PIPE.transformer.to(dev); WS._STATE["tr_on_gpu"] = True
print(f"[mem] umt5 dropped; transformer resident. {torch.cuda.mem_get_info()[0]/1e9:.1f}GB free", flush=True)


def gen_clip(traj_id, ref_index, color_anchor=None):
    """Generate ONE clip via PIPE (transformer-only, cached pe/ne). memory_inputs must already be written.
    color_anchor (np [n,H,W,3]) -> match per-channel mean/std to it (used in the loop to lock to the clean seed)."""
    meta = load_mutli_traj_dataset(cfg=cfg, input_path=rr, output_path=rr, view_id="view0", traj_id=traj_id,
                                   device=dev, ref_index=ref_index, model_type=MODEL, task_type="panorama")
    kwargs = {k: v for k, v in meta.items() if v is not None}
    # FREE CLIP (1.2GB) for the denoise: pre-encode image_embeds, then park image_encoder on CPU. The
    # memory-conditioned denoise (with reference_video) is ~1GB over 24GB; CLIP's weights are idle during the
    # 4 DMD steps, so this recovers exactly the headroom we need. (image is still passed for prepare_latents.)
    if getattr(PIPE, "image_encoder", None) is not None and kwargs.get("image") is not None:
        PIPE.image_encoder.to(dev)
        with torch.no_grad():
            kwargs["image_embeds"] = PIPE.encode_image(kwargs["image"], dev)
        PIPE.image_encoder.to("cpu"); torch.cuda.empty_cache()
    ri = ref_index.to(dev) if torch.is_tensor(ref_index) else ref_index
    kwargs.update(prompt=None, prompt_embeds=pe, negative_prompt_embeds=ne,
                  generator=torch.Generator(device=dev).manual_seed(1024), output_type="pt",
                  latent_cond_mode=cfg.latent_cond_mode, mode="test", num_frames=cfg.nframe, ref_index=ri)
    torch.cuda.reset_peak_memory_stats(); _td = time.time()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = PIPE(**kwargs).frames[0].float()
    peak = torch.cuda.max_memory_allocated() / 1e9
    arr = out.permute(0, 2, 3, 1).clamp(0, 1).cpu().numpy().astype(np.float32)   # FLOAT [0,1] RGB — export_to_video SWAPS R<->B on uint8 input, correct on float!
    if color_anchor is not None and len(color_anchor):
        ref = np.asarray(color_anchor).reshape(-1, 3).astype(np.float32) / 255.0   # anchor [0,255] -> [0,1]
        a = arr.reshape(-1, 3)
        for ch in range(3):
            rm, rs = ref[:, ch].mean(), ref[:, ch].std() + 1e-3
            am, asd = a[:, ch].mean(), a[:, ch].std() + 1e-3
            a[:, ch] = (a[:, ch] - am) / asd * rs + rm
        arr = a.clip(0, 1).reshape(arr.shape)
    res = f"{rr}/view0/{traj_id}/{MODEL}_result.mp4"
    export_to_video(arr, res, fps=16)   # float -> no swap
    return arr, peak, res, time.time() - _td

# ---- 2. SPREAD SEEDS: clean self-ref clips (ref_index=[0]) = GLOBAL anchor; transformer-only -> fits ----
_REF1 = torch.tensor([0], dtype=torch.long)
seeds = []
for k in SEED_TRAJS:
    tid = f"traj{k}"; mem = f"{rr}/view0/{tid}/memory_inputs"; os.makedirs(mem, exist_ok=True)
    if not os.path.exists(f"{mem}/{MODEL}.mp4"):
        shutil.copy(f"{rr}/view0/{tid}/render.mp4", f"{mem}/{MODEL}.mp4")   # self-ref conditioning
    _, peak, res_k, dt = gen_clip(tid, _REF1)
    seeds.append((k, res_k)); print(f"[mem]   seed {tid} {dt:.0f}s peak={peak:.1f}GB", flush=True)

# ---- 3. seed pano_bank from ALL clean seed clips (global coverage anchor) ----
shutil.rmtree(f"{rr}/pano_bank", ignore_errors=True)
print("[mem] seeding pano_bank from clean spread seeds…", flush=True)
PIPE.transformer.to("cpu"); WS._STATE["tr_on_gpu"] = False; torch.cuda.empty_cache()   # free GPU: seed_multi renders depth from the 3.77M cloud (point_rendering) -> needs room; transformer (19.5GB) would push it >24GB
_ws_seed_pano_bank.seed_multi(SCENE, CLOUD, seeds, "view0")
torch.cuda.empty_cache()

# ---- 4. construct the bank ----
print("[mem] loading MoGe + constructing bank…", flush=True)
moge = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to(dev)
class _Dummy:                                                  # non-None -> bank skips SAM3 (indoor never calls it)
    def eval(self): return self
    def to(self, *a, **k): return self
bank = PanoramaMemoryBank(root_path=SCENE, image_width=w, image_height=h, device=dev, nframe=cfg.nframe,
                          max_reference=1, align_nframe=4, rank=0, world_size=1, moge_model=moge,  # max_reference=1: 1 reference latent (= the working full-C ref_index=[0]) -> fits 23.5GB. 2 refs -> >24GB spill. nearest clean seed is enough anchor.
                          sam3_model=_Dummy(), sam3_processor=_Dummy(), results_name=MODEL, valid_threshold=0.15,
                          pts_num=2_000_000, kb_anomaly_percentile=90, pcd_nb_neighbors=10, pcd_std_ratio=2.0)
print(f"[mem] bank ready: mem_size={bank.mem_size} depth_median={bank.depth_median:.2f} max_d={bank.max_d:.2f}", flush=True)
del moge; bank.moge_model = None     # MoGe (~3GB) unused in loop (no update_memory/apply_worldmirror) -> free it.
gc.collect(); torch.cuda.empty_cache()
PIPE.transformer.to(dev); WS._STATE["tr_on_gpu"] = True; torch.cuda.empty_cache()   # transformer back on GPU for the loop (was off during seed_multi depth-render)
_free, _tot = torch.cuda.mem_get_info()
print(f"[mem] GPU before loop: free={_free/1e9:.1f}GB / {_tot/1e9:.1f}GB (alloc={torch.cuda.memory_allocated()/1e9:.1f}GB)", flush=True)

# ---- 5. loop NON-seed trajs: each conditioned on the nearest CLEAN seed; NO update_memory (memory stays = the
# fixed clean seed anchor -> no autoregressive chain -> no drift). Seed trajs already have their self-ref result.
for render_path in render_list:
    p = render_path.replace("\\", "/").split("/"); view_id, traj_id = p[-3], p[-2]
    if int(traj_id.replace("traj", "")) in SEED_TRAJS:        # seed clips already generated (clean), keep them
        continue
    print(f"[mem] {view_id}/{traj_id} retrieval…", flush=True)
    cam = json.load(open(f"{rr}/{view_id}/{traj_id}/camera.json"))
    tar_w2cs = torch.from_numpy(np.array(cam["extrinsic"])).to(dtype=torch.float32, device=dev)
    tar_Ks = torch.from_numpy(np.array(cam["intrinsic"])).to(dtype=torch.float32, device=dev)
    PIPE.transformer.to("cpu"); WS._STATE["tr_on_gpu"] = False; torch.cuda.empty_cache()   # free GPU: retrieval runs camera_selector(DINOv2) which OOM-pages on top of the resident 17GB transformer
    print(f"  [vram] after tr->cpu: free={torch.cuda.mem_get_info()[0]/1e9:.1f}GB alloc={torch.cuda.memory_allocated()/1e9:.1f}GB", flush=True)
    retrieved, ref_index, ref_index_dict, ref_w2cs, _ = bank.retrieval(tar_w2cs, tar_Ks, view_id=view_id, traj_id=traj_id)
    print(f"  [vram] after retrieval: free={torch.cuda.mem_get_info()[0]/1e9:.1f}GB alloc={torch.cuda.memory_allocated()/1e9:.1f}GB  ref_index={ref_index.tolist() if torch.is_tensor(ref_index) else ref_index}", flush=True)
    mem = f"{rr}/{view_id}/{traj_id}/memory_inputs"; os.makedirs(mem, exist_ok=True)
    export_to_video((retrieved / 255), f"{mem}/{MODEL}.mp4", fps=16)
    if ref_index_dict is not None: json.dump(ref_index_dict, open(f"{mem}/{MODEL}_ref_index.json", "w"), indent=2)
    if ref_w2cs is not None: json.dump(ref_w2cs.cpu().numpy().tolist(), open(f"{mem}/{MODEL}_ref_w2cs.json", "w"), indent=2)
    # dump intermediate memory PNGs (what the bank reprojects/selects into THIS clip)
    dbg = f"{rr}/{view_id}/{traj_id}/_mem_debug"; os.makedirs(dbg, exist_ok=True)
    rfa = np.asarray(retrieved)
    for k in range(rfa.shape[0]):
        cv2.imwrite(f"{dbg}/retrieved_{k:02d}.png", cv2.cvtColor(rfa[k].astype(np.uint8), cv2.COLOR_RGB2BGR))
    json.dump({"ref_index": ref_index.tolist() if torch.is_tensor(ref_index) else ref_index},
              open(f"{dbg}/ref_index.json", "w"))

    torch.cuda.empty_cache()    # release retrieval's transient DINOv2 allocations
    PIPE.transformer.to(dev); WS._STATE["tr_on_gpu"] = True    # transformer back on GPU for the denoise (off during retrieval)

    # generate via the shared helper (transformer-only) + colour-anchor to the CLEAN retrieved seed
    _, _peak, res, dt = gen_clip(traj_id, ref_index, color_anchor=retrieved)
    print(f"[mem]   -> {res}  denoise={dt:.0f}s peak={_peak:.1f}GB (ref_index={ref_index.tolist() if torch.is_tensor(ref_index) else ref_index})", flush=True)
    # NO update_memory: memory stays = the fixed clean spread seeds. Each clip anchors to a clean seed, never to a
    # prior loop output -> no chain -> no drift. (Colour-normalize above now matches to a CLEAN seed, reinforcing it.)
print("[mem] ALL DONE", flush=True)
