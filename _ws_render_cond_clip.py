"""Render the global_pcd as the WS conditioning along ONE trajectory via our pytorch3d-free z-buffer splat.
  worldmirror_py _ws_render_cond_clip.py <render_results_dir> <view{i}/traj{j}> [render_radius=0.012]
"""
import os, sys, json
import numpy as np, torch, trimesh, cv2
sys.path.insert(0, "D:/HY-World-2.0/hyworld2/worldgen")
from src.pointcloud import point_rendering

RR   = sys.argv[1]
TRAJ = sys.argv[2]
RAD  = float(sys.argv[3]) if len(sys.argv) > 3 else 0.012
dev  = "cuda"

pcd = trimesh.load(f"{RR}/global_pcd.ply")
pts = np.asarray(pcd.vertices, np.float32)
cols = np.asarray(pcd.visual.vertex_colors)[:, :3].astype(np.float32) / 255.0
cj = json.load(open(f"{RR}/{TRAJ}/camera.json"))
ext = np.array(cj["extrinsic"], np.float32)
K   = np.array(cj["intrinsic"], np.float32)
H, W = cj["height"], cj["width"]
print(f"[cond] {TRAJ} type={cj['type']}  frames={len(ext)}  pts={len(pts)}  rad={RAD}", flush=True)

rgbs, _ = point_rendering(K, ext, pts, cols, dev, H, W, render_radius=RAD, return_depth=False)
arr = (rgbs.permute(0, 2, 3, 1).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)   # [F,H,W,3] RGB

tag = TRAJ.replace("/", "_")
out = f"D:/_world_hangar/_ws_lean_cond_{tag}.mp4"
vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), 10, (W, H))
for f in arr:
    vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
vw.release()
# coverage diagnostic: fraction of non-background pixels per frame
cov = (arr.sum(-1) > 0).mean(axis=(1, 2))
print(f"[cond] coverage first={cov[0]:.2f} mid={cov[len(cov)//2]:.2f} last={cov[-1]:.2f}", flush=True)
strip = np.concatenate([arr[0], arr[len(arr)//2], arr[-1]], axis=1)
cv2.imwrite(out.replace(".mp4", "_strip.png"), cv2.cvtColor(strip, cv2.COLOR_RGB2BGR))
print(f"[cond] saved {out} + _strip.png  {arr.shape}", flush=True)
