"""DEPTH-SUPERVISED 3DGS (HW2's essence): train gaussians with RGB + MoGe-depth supervision so they lie ON the
surface (not floating) -> clean gsplat depth -> Open3D TSDF -> clean mesh. Run in TRELLIS env (gsplat + open3d 0.19).

  trellis_py _ws_gs_train_depth.py <lean_scene> <out.ply> [iters=12000] [lambda_depth=0.6]
"""
import sys, os, json, glob, random, time
import numpy as np, cv2, torch
from gsplat import rasterization
import trimesh, open3d as o3d

import torch.nn.functional as F
SCENE, OUT = sys.argv[1], sys.argv[2]
ITERS = int(sys.argv[3]) if len(sys.argv) > 3 else 12000
LD = float(sys.argv[4]) if len(sys.argv) > 4 else 0.6
LN = float(sys.argv[5]) if len(sys.argv) > 5 else 0.05   # normal-loss weight
LF = float(sys.argv[6]) if len(sys.argv) > 6 else 0.01   # flatten (thin-surfel) weight
WARM = 2000                                               # iters before normal loss kicks in (depth must settle)
dev = "cuda"; M = "worldstereo-memory-dmd"; W, H = 832, 480
rr = f"{SCENE}/render_results"

# ---- init gaussians from the pano cloud ----
g = trimesh.load(f"{rr}/global_pcd.ply")
V = np.asarray(g.vertices, np.float32); C = np.asarray(g.visual.vertex_colors)[:, :3].astype(np.float32) / 255
if len(V) > 300000:
    i = np.random.choice(len(V), 300000, replace=False); V = V[i]; C = C[i]
N = len(V)
means = torch.tensor(V, device=dev, requires_grad=True)
log_scales = torch.full((N, 3), float(np.log(0.02)), device=dev, requires_grad=True)
quats = torch.zeros(N, 4, device=dev); quats[:, 0] = 1; quats.requires_grad_(True)
logit_op = torch.full((N,), float(np.log(0.5 / 0.5)), device=dev, requires_grad=True)
colors = torch.tensor(C, device=dev, requires_grad=True)
opt = torch.optim.Adam([
    {"params": [means], "lr": 1.6e-4}, {"params": [log_scales], "lr": 5e-3},
    {"params": [quats], "lr": 1e-3}, {"params": [logit_op], "lr": 5e-2}, {"params": [colors], "lr": 2.5e-3}])

# ---- preload views (rgb uint8, moge depth f16, moge normal f16, camera) ----
cams = sorted(glob.glob(f"{rr}/view*/traj*/camera.json"))
RGB, DEP, NRM, W2C, KS = [], [], [], [], []
for cj in cams:
    td = os.path.dirname(cj); cam = json.load(open(cj)); ext = cam["extrinsic"]
    mp = f"{td}/moge_depth.npy"
    if not os.path.exists(mp): continue
    deps = np.load(mp); nrms = np.load(f"{td}/moge_normal.npy") if os.path.exists(f"{td}/moge_normal.npy") else None
    cap = cv2.VideoCapture(f"{td}/{M}_result.mp4"); fr = []
    while True:
        ok, f = cap.read()
        if not ok: break
        fr.append(cv2.cvtColor(cv2.resize(f, (W, H)), cv2.COLOR_BGR2RGB))
    cap.release()
    n = min(len(fr), len(ext), len(deps))
    for i in range(n):
        RGB.append(fr[i]); DEP.append(deps[i]); NRM.append(nrms[i] if nrms is not None else None)
        W2C.append(np.array(ext[i], np.float32)); KS.append(np.array(cam["intrinsic"][i], np.float32))
NV = len(RGB)
print(f"[train] {N} gaussians, {NV} views, iters={ITERS}, lambda_depth={LD}", flush=True)

def render(w2c, K):
    vm = torch.as_tensor(w2c[None], dtype=torch.float32, device=dev); Ks = torch.as_tensor(K[None], dtype=torch.float32, device=dev)
    out, alpha, _ = rasterization(means, torch.nn.functional.normalize(quats, dim=-1), torch.exp(log_scales),
                                  torch.sigmoid(logit_op), colors.clamp(0, 1), vm, Ks, W, H, render_mode="RGB+ED")
    return out[0, ..., :3], out[0, ..., 3]

_vv, _uu = torch.meshgrid(torch.arange(H, device=dev, dtype=torch.float32),
                          torch.arange(W, device=dev, dtype=torch.float32), indexing="ij")
def depth_to_normal(d, K):
    """camera-space normals from rendered depth via cross of central-diff tangents; faces camera (n_z<0)."""
    X = (_uu - K[0, 2]) * d / K[0, 0]; Y = (_vv - K[1, 2]) * d / K[1, 1]
    P = torch.stack([X, Y, d], -1)                                          # [H,W,3]
    du = P[1:-1, 2:, :] - P[1:-1, :-2, :]; dv = P[2:, 1:-1, :] - P[:-2, 1:-1, :]
    n = F.normalize(torch.cross(du, dv, dim=-1), dim=-1)                    # [H-2,W-2,3]
    n = torch.where(n[..., 2:3] > 0, -n, n)                                 # face the camera (consistent sign)
    return n

t0 = time.time()
for it in range(ITERS):
    j = random.randrange(NV)
    rgb_p, d_p = render(W2C[j], KS[j])
    g_rgb = torch.as_tensor(RGB[j], dtype=torch.float32, device=dev) / 255.0
    g_d = torch.as_tensor(DEP[j].astype(np.float32), device=dev)
    lrgb = (rgb_p - g_rgb).abs().mean()
    dm = g_d > 1e-3
    ld = ((d_p - g_d).abs() * dm).sum() / dm.sum().clamp(min=1)
    loss = lrgb + LD * ld
    # flatten reg: push the thinnest axis small -> surfel-like gaussians lie ON the surface (HW2/2DGS idea)
    lf = torch.exp(log_scales).min(dim=1).values.mean()
    loss = loss + LF * lf
    # normal loss: rendered-depth normals vs MoGe normals (camera space), after depth has settled
    ln = torch.zeros((), device=dev)
    if it >= WARM and NRM[j] is not None:
        n_p = depth_to_normal(d_p, KS[j])                                   # [H-2,W-2,3]
        g_n = torch.as_tensor(NRM[j].astype(np.float32), device=dev)[1:-1, 1:-1, :]
        g_n = F.normalize(g_n, dim=-1)
        g_n = torch.where(g_n[..., 2:3] > 0, -g_n, g_n)                     # same camera-facing convention
        nm = (dm[1:-1, 1:-1] & (g_n.abs().sum(-1) > 1e-3))
        ln = ((1.0 - (n_p * g_n).sum(-1)) * nm).sum() / nm.sum().clamp(min=1)
        loss = loss + LN * ln
    opt.zero_grad(); loss.backward(); opt.step()
    if it % 500 == 0:
        print(f"[train] it {it}/{ITERS} rgb {lrgb.item():.4f} depth {ld.item():.4f} nrm {ln.item():.4f} flat {lf.item():.4f} {time.time()-t0:.0f}s", flush=True)

# ---- TSDF from the trained (surface-aligned) gaussians ----
print("[train] TSDF fusing trained-gaussian depth…", flush=True)
vol = o3d.pipelines.integration.ScalableTSDFVolume(voxel_length=0.012, sdf_trunc=0.048, color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
with torch.no_grad():
    for j in range(NV):
        _, d_p = render(W2C[j], KS[j])
        d = d_p.cpu().numpy(); d[d <= 1e-3] = 0.0
        rgb = np.ascontiguousarray(RGB[j])
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(rgb), o3d.geometry.Image(d.astype(np.float32)),
            depth_scale=1.0, depth_trunc=10.0, convert_rgb_to_intensity=False)
        K = KS[j]; pin = o3d.camera.PinholeCameraIntrinsic(W, H, K[0, 0], K[1, 1], K[0, 2], K[1, 2])
        vol.integrate(rgbd, pin, W2C[j].astype(np.float64))
mesh = vol.extract_triangle_mesh(); mesh.compute_vertex_normals()
mesh.remove_degenerate_triangles(); mesh.remove_unreferenced_vertices()
o3d.io.write_triangle_mesh(OUT, mesh)
print(f"[train] DONE: {len(mesh.vertices)} verts, {len(mesh.triangles)} tris -> {OUT}", flush=True)
