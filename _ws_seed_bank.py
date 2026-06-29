"""Seed the GEOMETRY scaffold a PanoramaMemoryBank.__init__ needs, from OUR world cloud (prior metric).

This writes ONLY the geometry store (skeleton = ours); the appearance store (pano_bank) is seeded separately
from a warm-up GENERATION (never from the prior — see plan Phase 0.2 "two separate stores").

Emits under <scene>/render_results/:
  global_pcd.ply              non-sky points (verts+colors)  -> bank geometry + depth-stats source
  sky_pcd.ply                 sky points (optional; bank loads if present)
  full_depth_prediction.pt    {"distance": torch[H,W]} matching sky_mask.png; only min/max/median over the
                              ground_mask are read (retrieval fov-overlap far-plane), so we fill ground pixels
                              with the cloud's real distance distribution from the scene/camera centre.
sky_mask.png is assumed already present (written by _ws_to_scene.py).

  python _ws_seed_bank.py <scene_dir> <cloud.npz>
"""
import sys, os, json, glob
import numpy as np
import torch
import trimesh
from PIL import Image


def cam_centers(scene):
    """Mean camera centre over all trajs (≈ scene/pano centre) from camera.json w2c extrinsics."""
    cs = []
    for cj in glob.glob(f"{scene}/render_results/view*/traj*/camera.json"):
        ext = json.load(open(cj)).get("extrinsic", [])
        for M in ext:
            w2c = np.array(M, np.float32)
            if w2c.shape == (4, 4):
                cs.append((np.linalg.inv(w2c))[:3, 3])
    if not cs:
        return None
    return np.mean(np.stack(cs, 0), 0)


def main(scene, cloud_npz):
    rr = f"{scene}/render_results"
    os.makedirs(rr, exist_ok=True)
    z = np.load(cloud_npz)
    xyz = z["xyz"].astype(np.float64)
    col = z["color"][:, :3].astype(np.uint8)
    sky = z["is_sky"].astype(bool) if "is_sky" in z else np.zeros(len(xyz), bool)
    g = ~sky

    # 1) global_pcd.ply = scene geometry (non-sky)
    trimesh.PointCloud(vertices=xyz[g], colors=col[g]).export(f"{rr}/global_pcd.ply")
    print(f"  global_pcd.ply  <- {int(g.sum())} non-sky pts")
    # 2) sky_pcd.ply (optional) = sky backdrop
    if sky.any():
        trimesh.PointCloud(vertices=xyz[sky], colors=col[sky]).export(f"{rr}/sky_pcd.ply")
        print(f"  sky_pcd.ply     <- {int(sky.sum())} sky pts")

    # 3) full_depth_prediction.pt — distance map matching sky_mask, faithful ground stats
    sky_mask = np.array(Image.open(f"{rr}/sky_mask.png"))
    if sky_mask.ndim == 3:
        sky_mask = sky_mask[..., 0]
    H, W = sky_mask.shape
    ground = sky_mask >= 128                                  # 255 = ground/coverage, 0 = sky
    center = cam_centers(scene)
    if center is None:
        center = xyz[g].mean(0)
        print("  (no camera.json centre — using non-sky centroid)")
    d = np.linalg.norm(xyz[g] - center[None], axis=1)          # real per-point ray distances
    d.sort()
    n_ground = int(ground.sum())
    dist = np.zeros((H, W), np.float32)
    if n_ground > 0 and len(d) > 0:
        idx = np.linspace(0, len(d) - 1, n_ground).astype(np.int64)   # preserves min/max/median exactly
        dist[ground] = d[idx].astype(np.float32)
    torch.save({"distance": torch.from_numpy(dist).float()}, f"{rr}/full_depth_prediction.pt")
    sel = dist[ground]
    print(f"  full_depth.pt   <- distance[{H}x{W}] ground={n_ground}  "
          f"min={sel.min():.2f} med={np.median(sel):.2f} max={sel.max():.2f}")
    print("SEED-BANK geometry scaffold DONE (skeleton = our prior metric).")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
