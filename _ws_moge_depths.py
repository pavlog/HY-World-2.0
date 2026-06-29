"""Precompute MoGe-2 depth (PER-FRAME aligned to the cloud scaffold via scale+shift -> cross-view consistent) +
camera-space NORMALS, for depth+normal-supervised 3DGS (HW2-style).

  worldmirror_py _ws_moge_depths.py <lean_scene>
"""
import sys, os, json, glob
import numpy as np, cv2, torch
sys.path.insert(0, "D:/HY-World-2.0/hyworld2/worldgen")
from moge.model.v2 import MoGeModel
from src.pointcloud import point_rendering
import trimesh

SCENE = sys.argv[1]; rr = f"{SCENE}/render_results"; M = "worldstereo-memory-dmd"
dev = "cuda"; W, H = 832, 480
moge = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to(dev).eval()
gpts = torch.as_tensor(np.asarray(trimesh.load(f"{rr}/global_pcd.ply").vertices, np.float32), device=dev)
zeros = torch.zeros((len(gpts), 3), device=dev)

def read_frames(mp4):
    cap = cv2.VideoCapture(mp4); fr = []
    while True:
        ok, f = cap.read()
        if not ok: break
        fr.append(cv2.cvtColor(cv2.resize(f, (W, H)), cv2.COLOR_RGB2BGR) if False else cv2.cvtColor(cv2.resize(f, (W, H)), cv2.COLOR_BGR2RGB))
    cap.release(); return fr

cams = sorted(glob.glob(f"{rr}/view*/traj*/camera.json"))
done = 0
for cj in cams:
    td = os.path.dirname(cj); cam = json.load(open(cj)); ext = cam["extrinsic"]; intr = cam["intrinsic"]
    fr = read_frames(f"{td}/{M}_result.mp4"); n = min(len(fr), len(ext))
    if n == 0: continue
    Ks = np.array([intr[i] for i in range(n)], np.float32); w2cs = np.array([ext[i] for i in range(n)], np.float32)
    _, cdep = point_rendering(Ks, w2cs, gpts, zeros, dev, H, W, render_radius=0.012, return_depth=True)
    cdep = cdep[:, 0].cpu().numpy()                                  # cloud depth per frame (consistent scaffold)
    deps = np.zeros((n, H, W), np.float32); nrms = np.zeros((n, H, W, 3), np.float16)
    for i in range(n):
        img = torch.as_tensor(fr[i] / 255.0, dtype=torch.float32, device=dev).permute(2, 0, 1)
        with torch.no_grad():
            o = moge.infer(img, use_fp16=True)
        md = o["depth"].cpu().numpy(); m = o["mask"].cpu().numpy(); nr = o["normal"].cpu().numpy() if "normal" in o else np.zeros((H, W, 3), np.float32)
        cd = cdep[i]; valid = (cd > 1e-3) & (md > 1e-3) & m & np.isfinite(md)
        if valid.sum() > 500:                                       # per-frame align MoGe depth to cloud (a*md+b)
            a, b = np.polyfit(md[valid], cd[valid], 1)
        else:
            a, b = 0.258, 0.0
        d = a * md + b; d[~m] = 0; d[~np.isfinite(d)] = 0; d[d < 0] = 0
        deps[i] = d; nrms[i] = nr.astype(np.float16)
    np.save(f"{td}/moge_depth.npy", deps.astype(np.float16))
    np.save(f"{td}/moge_normal.npy", nrms)
    done += 1
    if done % 8 == 0: print(f"[moge] {done}/{len(cams)} trajs (per-frame aligned + normals)", flush=True)
print(f"[moge] DONE: {done} trajs", flush=True)
