"""Multi-VIEW memory-bank generation over the LEAN pano-exploration scene (16 views x N trajs).

Adapted from _ws_gen_memory.py (single-view B-mesh) for the pano exploration scene built by _ws_lean_traj.py.
Same 24GB VRAM recipe: encode prompt once + drop umt5; park CLIP image_encoder; swap transformer to CPU around
seed-depth-render / retrieval. Seeds = one clean self-ref clip per view (global appearance anchor). NO update_memory
in the loop (memory = the fixed clean spread seeds -> no autoregressive drift). pano_bank keyed by view+traj.

  <WS env> worldmirror_py _ws_gen_lean.py <scene_dir> "<prompt>"
"""
import sys, os, json, shutil, time, glob as _glob
_ARGV = sys.argv[1:]; sys.argv = [sys.argv[0]]
SCENE = _ARGV[0]
PROMPT = _ARGV[1] if len(_ARGV) > 1 and not _ARGV[1].startswith("--") else "sci-fi spaceship hangar interior, cinematic, high detail, sharp"

import torch, numpy as np, cv2, gc, trimesh
import _ws_serve as WS
sys.path.insert(0, r"D:/HY-World-2.0/hyworld2/worldgen")
sys.path.insert(0, r"D:/HY-World-2.0")
from src.retrieval_wm import PanoramaMemoryBank
from src.data_utils import load_mutli_traj_dataset
from src.general_utils import save_16bit_png_depth
from src.pointcloud import point_rendering
from diffusers.utils import export_to_video
import imagesize
from moge.model.v2 import MoGeModel
from PIL import Image

dev, cfg, PIPE, MODEL = WS.device, WS.ws.cfg, WS.PIPE, WS.MODEL_TYPE
rr = f"{SCENE}/render_results"

# ---- xyz npz from global_pcd (pano_bank metric-depth render needs xyz) ----
NPZ = f"{rr}/_global_xyz.npz"
if not os.path.exists(NPZ):
    g = trimesh.load(f"{rr}/global_pcd.ply")
    np.savez(NPZ, xyz=np.asarray(g.vertices, np.float32))
    print(f"[lean-gen] wrote {NPZ} ({len(g.vertices)} pts)", flush=True)

cams = sorted(_glob.glob(f"{rr}/view*/traj*/camera.json"),
              key=lambda p: (int(p.replace("\\", "/").split("/")[-3].replace("view", "")),
                             int(p.replace("\\", "/").split("/")[-2].replace("traj", ""))))
PAIRS = [(p.replace("\\", "/").split("/")[-3], p.replace("\\", "/").split("/")[-2]) for p in cams]
w, h = imagesize.get(f"{rr}/view0/start_frame.png")
views = sorted({v for v, _ in PAIRS}, key=lambda s: int(s.replace("view", "")))
SEED_PAIRS = [(v, "traj0") for v in views if (v, "traj0") in PAIRS]   # one clean seed per view = global anchor
SEED_SET = set(SEED_PAIRS)
print(f"[lean-gen] {len(PAIRS)} (view,traj) pairs over {len(views)} views, {w}x{h}; {len(SEED_PAIRS)} seeds", flush=True)

# ---- encode prompt ONCE, drop umt5 (umt5 11GB + transformer 17.45GB co-resident = 30.8GB -> stall) ----
neg = cfg.get("negative_prompt", "")
PIPE.transformer.to("cpu"); WS._STATE["tr_on_gpu"] = False; torch.cuda.empty_cache()
PIPE.text_encoder.to(dev)
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    pe, ne = PIPE.encode_prompt(prompt=PROMPT, negative_prompt=neg, do_classifier_free_guidance=False,
                                num_videos_per_prompt=1, max_sequence_length=512, device=dev)
PIPE.text_encoder.to("cpu"); PIPE.text_encoder = None
gc.collect(); torch.cuda.empty_cache()
PIPE.transformer.to(dev); WS._STATE["tr_on_gpu"] = True
print(f"[lean-gen] umt5 dropped; transformer resident. {torch.cuda.mem_get_info()[0]/1e9:.1f}GB free", flush=True)


def gen_clip(view_id, traj_id, ref_index, color_anchor=None):
    meta = load_mutli_traj_dataset(cfg=cfg, input_path=rr, output_path=rr, view_id=view_id, traj_id=traj_id,
                                   device=dev, ref_index=ref_index, model_type=MODEL, task_type="panorama")
    kwargs = {k: v for k, v in meta.items() if v is not None}
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
    arr = out.permute(0, 2, 3, 1).clamp(0, 1).cpu().numpy().astype(np.float32)   # FLOAT [0,1] RGB (export_to_video swaps R<->B on uint8!)
    if color_anchor is not None and len(color_anchor):
        ref = np.asarray(color_anchor).reshape(-1, 3).astype(np.float32) / 255.0
        a = arr.reshape(-1, 3)
        for ch in range(3):
            rm, rs = ref[:, ch].mean(), ref[:, ch].std() + 1e-3
            am, asd = a[:, ch].mean(), a[:, ch].std() + 1e-3
            a[:, ch] = (a[:, ch] - am) / asd * rs + rm
        arr = a.clip(0, 1).reshape(arr.shape)
    res = f"{rr}/{view_id}/{traj_id}/{MODEL}_result.mp4"
    export_to_video(arr, res, fps=16)
    return arr, peak, res, time.time() - _td


def seed_pano_bank_mv(seeds):
    """Seed pano_bank from clean self-ref clips across views. seeds=[(view,traj,mp4)]. Keyed v{V}t{T}f{i}.
    Metric depth rendered from global_pcd at each clip's cameras (consistent with skeleton)."""
    pb = f"{rr}/pano_bank"; shutil.rmtree(pb, ignore_errors=True)
    os.makedirs(f"{pb}/images", exist_ok=True); os.makedirs(f"{pb}/depths", exist_ok=True)
    pts = torch.from_numpy(np.load(NPZ)["xyz"].astype(np.float32))
    allcams = {}; total = 0
    for view_id, traj_id, mp4 in seeds:
        cap = cv2.VideoCapture(mp4); frames = []
        while True:
            ok, f = cap.read()
            if not ok: break
            frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        cap.release()
        cam = json.load(open(f"{rr}/{view_id}/{traj_id}/camera.json")); ext, intr = cam["extrinsic"], cam["intrinsic"]
        n = min(len(frames), len(ext)); H, W = frames[0].shape[:2]
        Ks = torch.tensor(np.stack([intr[i] for i in range(n)]), dtype=torch.float32)
        w2cs = torch.tensor(np.stack([ext[i] for i in range(n)]), dtype=torch.float32)
        _, depth = point_rendering(Ks, w2cs, pts, torch.zeros((pts.shape[0], 3)), dev, H, W,
                                   render_radius=0.012, return_depth=True)
        depth = depth[:, 0].cpu().numpy()
        vn, tn = int(view_id.replace("view", "")), int(traj_id.replace("traj", ""))
        for i in range(n):
            key = f"v{vn:02d}t{tn:02d}f{i:02d}"
            cv2.imwrite(f"{pb}/images/{key}.png", cv2.cvtColor(frames[i], cv2.COLOR_RGB2BGR))
            d = depth[i].copy(); d[d < 0] = 0.0
            save_16bit_png_depth(d.astype(np.float32), f"{pb}/depths/{key}.png")
            allcams[key] = {"extrinsic": np.array(ext[i], float).tolist(), "intrinsic": np.array(intr[i], float).tolist()}
            total += 1
    json.dump(allcams, open(f"{pb}/cameras.json", "w"))
    print(f"  pano_bank <- {total} frames from {len(seeds)} seed clips across {len(views)} views", flush=True)


# ---- 1. SEEDS: clean self-ref clips (ref_index=[0]), one per view ----
_REF1 = torch.tensor([0], dtype=torch.long)
seeds = []
for v, t in SEED_PAIRS:
    mem = f"{rr}/{v}/{t}/memory_inputs"; os.makedirs(mem, exist_ok=True)
    if not os.path.exists(f"{mem}/{MODEL}.mp4"):
        shutil.copy(f"{rr}/{v}/{t}/render.mp4", f"{mem}/{MODEL}.mp4")
    res_k = f"{rr}/{v}/{t}/{MODEL}_result.mp4"
    if os.path.exists(res_k):                                   # deterministic seed (seed=1024) -> reuse
        seeds.append((v, t, res_k)); print(f"[lean-gen]   seed {v}/{t} CACHED", flush=True); continue
    _, peak, res_k, dt = gen_clip(v, t, _REF1)
    seeds.append((v, t, res_k)); print(f"[lean-gen]   seed {v}/{t} {dt:.0f}s peak={peak:.1f}GB", flush=True)

# ---- 2. seed pano_bank from all clean seeds ----
print("[lean-gen] seeding pano_bank…", flush=True)
PIPE.transformer.to("cpu"); WS._STATE["tr_on_gpu"] = False; torch.cuda.empty_cache()
seed_pano_bank_mv(seeds)
torch.cuda.empty_cache()

# ---- 3. construct bank ----
print("[lean-gen] loading MoGe + constructing bank…", flush=True)
moge = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to(dev)
class _Dummy:
    def eval(self): return self
    def to(self, *a, **k): return self
bank = PanoramaMemoryBank(root_path=SCENE, image_width=w, image_height=h, device=dev, nframe=cfg.nframe,
                          max_reference=1, align_nframe=4, rank=0, world_size=1, moge_model=moge,
                          sam3_model=_Dummy(), sam3_processor=_Dummy(), results_name=MODEL, valid_threshold=0.15,
                          pts_num=2_000_000, kb_anomaly_percentile=90, pcd_nb_neighbors=10, pcd_std_ratio=2.0)
print(f"[lean-gen] bank ready: mem_size={bank.mem_size} depth_median={bank.depth_median:.2f} max_d={bank.max_d:.2f}", flush=True)
del moge; bank.moge_model = None
gc.collect(); torch.cuda.empty_cache()
PIPE.transformer.to(dev); WS._STATE["tr_on_gpu"] = True; torch.cuda.empty_cache()

# ---- 4. loop non-seed (view,traj): retrieval (nearest clean seed) -> memory-conditioned gen ----
done = 0
for v, t in PAIRS:
    if (v, t) in SEED_SET:
        continue
    cam = json.load(open(f"{rr}/{v}/{t}/camera.json"))
    tar_w2cs = torch.from_numpy(np.array(cam["extrinsic"])).to(dtype=torch.float32, device=dev)
    tar_Ks = torch.from_numpy(np.array(cam["intrinsic"])).to(dtype=torch.float32, device=dev)
    PIPE.transformer.to("cpu"); WS._STATE["tr_on_gpu"] = False; torch.cuda.empty_cache()
    retrieved, ref_index, ref_index_dict, ref_w2cs, _ = bank.retrieval(tar_w2cs, tar_Ks, view_id=v, traj_id=t)
    mem = f"{rr}/{v}/{t}/memory_inputs"; os.makedirs(mem, exist_ok=True)
    export_to_video((retrieved / 255), f"{mem}/{MODEL}.mp4", fps=16)
    if ref_index_dict is not None: json.dump(ref_index_dict, open(f"{mem}/{MODEL}_ref_index.json", "w"), indent=2)
    if ref_w2cs is not None: json.dump(ref_w2cs.cpu().numpy().tolist(), open(f"{mem}/{MODEL}_ref_w2cs.json", "w"), indent=2)
    torch.cuda.empty_cache()
    PIPE.transformer.to(dev); WS._STATE["tr_on_gpu"] = True
    _, _peak, res, dt = gen_clip(v, t, ref_index, color_anchor=retrieved)
    done += 1
    print(f"[lean-gen] {done}/{len(PAIRS)-len(SEED_PAIRS)}  {v}/{t} -> {dt:.0f}s peak={_peak:.1f}GB ref={ref_index.tolist() if torch.is_tensor(ref_index) else ref_index}", flush=True)
print("[lean-gen] ALL DONE", flush=True)
