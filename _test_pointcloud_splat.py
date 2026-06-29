"""Self-test for the pytorch3d-free pointcloud.point_rendering (Path B).
Verifies: (1) module imports with NO pytorch3d, (2) shapes match the original contract,
(3) a known 3D point lands on the right pixel with the right camera-space depth,
(4) mask polarity (1=empty), (5) depth=-1 where empty."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "hyworld2", "worldgen"))
import numpy as np, torch
from src.pointcloud import point_rendering   # must import without pytorch3d

dev = "cuda" if torch.cuda.is_available() else "cpu"
H, W = 480, 832
fx = fy = 500.0; cx, cy = W / 2.0, H / 2.0
K = torch.tensor([[[fx, 0, cx], [0, fy, cy], [0, 0, 1]]], dtype=torch.float32)   # [1,3,3]
w2c = torch.eye(4)[None]                                                          # [1,4,4] identity (cam=world)

# one point straight ahead at z=2m, slightly right/up; expected pixel:
Z = 2.0; Xp = 0.10; Yp = -0.05
pts = torch.tensor([[Xp, Yp, Z]], dtype=torch.float32)
cols = torch.tensor([[0.7, 0.2, -0.3]], dtype=torch.float32)                      # arbitrary feature range
exp_u = int(round(fx * Xp / Z + cx)); exp_v = int(round(fy * Yp / Z + cy))

# --- return_depth=False contract ---
rgbs, masks = point_rendering(K, w2c, pts, cols, dev, H, W, render_radius=0.008, return_depth=False)
assert tuple(rgbs.shape) == (1, 3, H, W), rgbs.shape
assert tuple(masks.shape) == (1, 1, H, W), masks.shape
covered = (masks[0, 0] == 0)
n_cov = int(covered.sum())
assert n_cov > 0, "nothing rendered"
# the covered centroid should be near expected pixel
ys, xs = torch.where(covered.cpu())
cu, cv = float(xs.float().mean()), float(ys.float().mean())
assert abs(cu - exp_u) <= 2 and abs(cv - exp_v) <= 2, f"pixel off: got ({cu:.1f},{cv:.1f}) exp ({exp_u},{exp_v})"
# color at the winning pixel
col_at = rgbs[0, :, exp_v, exp_u].cpu().numpy()
assert np.allclose(col_at, [0.7, 0.2, -0.3], atol=1e-4), col_at
# mask polarity: most pixels empty -> mask 1
assert masks.mean() > 0.9, f"mask mean {masks.mean():.3f} (expected mostly empty=1)"

# --- return_depth=True contract ---
rgbs2, depth = point_rendering(K, w2c, pts, cols, dev, H, W, render_radius=0.008, return_depth=True)
assert tuple(rgbs2.shape) == (1, H, W, 3), rgbs2.shape          # un-rearranged
assert tuple(depth.shape) == (1, 1, H, W), depth.shape
d_at = float(depth[0, 0, exp_v, exp_u])
assert abs(d_at - Z) < 1e-3, f"depth {d_at} != {Z}"
assert float(depth[0, 0, 0, 0]) == -1.0, "empty corner should be -1"

# --- z-buffer: nearer point wins over farther at same pixel ---
pts2 = torch.tensor([[0., 0., 5.0], [0., 0., 2.0]], dtype=torch.float32)          # far then near
cols2 = torch.tensor([[1., 0., 0.], [0., 1., 0.]], dtype=torch.float32)           # red far, green near
rgbs3, depth3 = point_rendering(K, w2c, pts2, cols2, dev, H, W, render_radius=0.004, return_depth=True)
cpix = rgbs3[0, int(cy), int(cx)].cpu().numpy()
assert np.allclose(cpix, [0., 1., 0.], atol=1e-4), f"z-buffer wrong, got {cpix} (want green/near)"
assert abs(float(depth3[0, 0, int(cy), int(cx)]) - 2.0) < 1e-3

print(f"OK  device={dev}  covered_px={n_cov}  centroid=({cu:.1f},{cv:.1f}) exp=({exp_u},{exp_v})  depth={d_at:.3f}  zbuf=near-wins")
print("ALL CONTRACT CHECKS PASSED — pytorch3d-free point_rendering is byte-compatible.")
