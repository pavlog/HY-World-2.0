"""Batch-render the WS conditioning (global_pcd z-buffer splat) for ALL view*/traj* in a lean scene.
Writes render.mp4 (cloud RGB) + render_mask.mp4 (white = empty/to-fill) into each traj dir, as the gen driver expects.
  worldmirror_py _ws_render_cond_all.py <render_results_dir> [render_radius=0.012]
"""
import os, sys, json, glob
import numpy as np, torch, trimesh, cv2
sys.path.insert(0, "D:/HY-World-2.0/hyworld2/worldgen")
from src.pointcloud import point_rendering

RR  = sys.argv[1]
RAD = float(sys.argv[2]) if len(sys.argv) > 2 else 0.012
dev = "cuda"

pcd = trimesh.load(f"{RR}/global_pcd.ply")
pts = torch.as_tensor(np.asarray(pcd.vertices, np.float32), device=dev)
cols = torch.as_tensor(np.asarray(pcd.visual.vertex_colors)[:, :3].astype(np.float32) / 255.0, device=dev)
print(f"[cond-all] {len(pts)} pts, rad={RAD}", flush=True)

cams = sorted(glob.glob(f"{RR}/view*/traj*/camera.json"),
              key=lambda p: (int(p.replace("\\", "/").split("/")[-3].replace("view", "")),
                             int(p.replace("\\", "/").split("/")[-2].replace("traj", ""))))
n = 0
for cj_path in cams:
    td = os.path.dirname(cj_path)
    cj = json.load(open(cj_path))
    ext = np.array(cj["extrinsic"], np.float32); K = np.array(cj["intrinsic"], np.float32)
    H, W = cj["height"], cj["width"]
    rgbs, masks = point_rendering(K, ext, pts, cols, dev, H, W, render_radius=RAD, return_depth=False)
    rgb = (rgbs.permute(0, 2, 3, 1).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)   # [F,H,W,3] RGB
    msk = (masks[:, 0].cpu().numpy() * 255).astype(np.uint8)                              # [F,H,W] 255=empty
    vw = cv2.VideoWriter(f"{td}/render.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 16, (W, H))
    for f in rgb:
        vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    vw.release()
    vm = cv2.VideoWriter(f"{td}/render_mask.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 16, (W, H))
    for f in msk:
        vm.write(cv2.cvtColor(f, cv2.COLOR_GRAY2BGR))
    vm.release()
    n += 1
    if n % 8 == 0:
        print(f"[cond-all] {n}/{len(cams)} (last empty={msk.mean()/255:.2f})", flush=True)
print(f"[cond-all] DONE {n} trajectories -> render.mp4 + render_mask.mp4", flush=True)
