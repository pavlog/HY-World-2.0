"""HW2-style mesh: render DEPTH from the 3DGS gaussian centers (the clean optimized cloud) per camera, then
Open3D ScalableTSDFVolume fuse with the generated RGB -> watertight mesh. Cleaner than Poisson/MoGe because TSDF
averages rendered depth across all views (HW2's exact extract_mesh_bounded method).

  worldmirror_py _ws_tsdf_gaussians.py <3dgs_point_cloud.ply> <lean_scene> <out.ply> [voxel=0.01]
"""
import sys, os, json, glob
import numpy as np, cv2, torch, open3d as o3d

# inlined point_rendering (pytorch3d-free z-buffer splat; pure torch — avoids src.pointcloud's einops/imageio chain)
def _disk_offsets(rad_px, device):
    if rad_px <= 0:
        return torch.zeros((1, 2), dtype=torch.long, device=device)
    a = torch.arange(-rad_px, rad_px + 1, device=device)
    ys, xs = torch.meshgrid(a, a, indexing='ij')
    keep = (xs * xs + ys * ys) <= rad_px * rad_px
    return torch.stack([xs[keep], ys[keep]], dim=1).long()

def point_rendering(K, w2cs, points, colors, device, h, w, render_radius=0.008, return_depth=False):
    K = torch.as_tensor(K, dtype=torch.float32, device=device); w2cs = torch.as_tensor(w2cs, dtype=torch.float32, device=device)
    pts = torch.as_tensor(points, dtype=torch.float32, device=device); cols = torch.as_tensor(colors, dtype=torch.float32, device=device)
    if cols.ndim == 1: cols = cols[:, None]
    F = w2cs.shape[0]; N = pts.shape[0]; C = cols.shape[1]
    bg = torch.zeros(C, dtype=torch.float32, device=device)
    rad_px = int(round(render_radius * min(h, w) / 2.0)); off = _disk_offsets(rad_px, device); D = off.shape[0]
    pts_h = torch.cat([pts, torch.ones((N, 1), device=device)], dim=1)
    rgbs_out = bg.view(1, 1, 1, C).expand(F, h, w, C).clone()
    depth_out = torch.full((F, h, w), -1.0, dtype=torch.float32, device=device)
    for f in range(F):
        Pc = (w2cs[f] @ pts_h.T).T[:, :3]; z = Pc[:, 2]
        u = K[f, 0, 0] * Pc[:, 0] / z + K[f, 0, 2]; v = K[f, 1, 1] * Pc[:, 1] / z + K[f, 1, 2]
        ui = (u.round().long()[:, None] + off[None, :, 0]).reshape(-1); vi = (v.round().long()[:, None] + off[None, :, 1]).reshape(-1)
        zz = z[:, None].expand(N, D).reshape(-1); pid = torch.arange(N, device=device)[:, None].expand(N, D).reshape(-1)
        ok = (zz > 1e-4) & (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
        ui, vi, zz, pid = ui[ok], vi[ok], zz[ok], pid[ok]
        if ui.numel() == 0: continue
        flat = vi * w + ui
        depth_buf = torch.full((h * w,), float('inf'), device=device); depth_buf.scatter_reduce_(0, flat, zz, reduce='amin', include_self=True)
        won = zz <= depth_buf[flat] + 1e-6
        col_buf = bg.view(1, C).expand(h * w, C).clone(); col_buf[flat[won]] = cols[pid[won]]
        rgbs_out[f] = col_buf.reshape(h, w, C)
        df = depth_buf.clone(); df[~torch.isfinite(depth_buf)] = -1.0; depth_out[f] = df.reshape(h, w)
    if return_depth:
        return rgbs_out, depth_out[:, None].contiguous()
    return rgbs_out.permute(0, 3, 1, 2).contiguous(), (depth_out == -1).float()[:, None]

PLY = sys.argv[1]
SCENE = sys.argv[2]
OUT = sys.argv[3]
VOXEL = float(sys.argv[4]) if len(sys.argv) > 4 else 0.01
dev = "cuda"; M = "worldstereo-memory-dmd"; W, H = 832, 480

# ---- load gaussian centers + base color (SH DC) ----
try:
    from plyfile import PlyData
    v = PlyData.read(PLY)['vertex']
    xyz = np.stack([v['x'], v['y'], v['z']], 1).astype(np.float32)
    SH = 0.28209479177387814
    col = np.clip(np.stack([v['f_dc_0'], v['f_dc_1'], v['f_dc_2']], 1) * SH + 0.5, 0, 1).astype(np.float32)
except Exception as e:
    import trimesh
    g = trimesh.load(PLY); xyz = np.asarray(g.vertices, np.float32); col = np.full((len(xyz), 3), 0.5, np.float32)
    print("[tsdf] plyfile failed, using trimesh xyz only:", e)
if len(xyz) > 800000:                                   # cap: huge N x disk-offsets segfaults point_rendering
    idx = np.random.choice(len(xyz), 800000, replace=False); xyz = xyz[idx]; col = col[idx]
gp = torch.as_tensor(xyz, device=dev)
print(f"[tsdf] {len(xyz)} gaussian centers (capped), voxel={VOXEL}", flush=True)

rr = f"{SCENE}/render_results"
vol = o3d.pipelines.integration.ScalableTSDFVolume(
    voxel_length=VOXEL, sdf_trunc=VOXEL * 4,
    color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

cams = sorted(glob.glob(f"{rr}/view*/traj*/camera.json"),
              key=lambda p: (int(p.replace("\\", "/").split("/")[-3].replace("view", "")),
                             int(p.replace("\\", "/").split("/")[-2].replace("traj", ""))))
nint = 0
for ci, cj in enumerate(cams):
    td = os.path.dirname(cj); cam = json.load(open(cj)); ext = cam["extrinsic"]; intr = cam["intrinsic"]
    cap = cv2.VideoCapture(f"{td}/{M}_result.mp4"); fr = []
    while True:
        ok, f = cap.read()
        if not ok: break
        fr.append(f)
    cap.release()
    n = min(len(fr), len(ext))
    if n == 0: continue
    Ks = np.array([intr[i] for i in range(n)], np.float32); w2cs = np.array([ext[i] for i in range(n)], np.float32)
    _, depth = point_rendering(Ks, w2cs, gp, torch.zeros((len(gp), 3), device=dev), dev, H, W, render_radius=0.012, return_depth=True)
    depth = depth[:, 0].cpu().numpy()
    for i in range(n):
        d = depth[i].copy(); d[d < 0] = 0.0
        rgb = cv2.cvtColor(cv2.resize(fr[i], (W, H)), cv2.COLOR_BGR2RGB)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.ascontiguousarray(rgb)), o3d.geometry.Image(d.astype(np.float32)),
            depth_scale=1.0, depth_trunc=10.0, convert_rgb_to_intensity=False)
        K = Ks[i]; pin = o3d.camera.PinholeCameraIntrinsic(W, H, K[0, 0], K[1, 1], K[0, 2], K[1, 2])
        vol.integrate(rgbd, pin, w2cs[i].astype(np.float64))
        nint += 1
    if (ci + 1) % 8 == 0:
        print(f"[tsdf] {ci+1}/{len(cams)} clips integrated ({nint} frames)", flush=True)

print("[tsdf] extracting mesh…", flush=True)
mesh = vol.extract_triangle_mesh(); mesh.compute_vertex_normals()
mesh.remove_degenerate_triangles(); mesh.remove_unreferenced_vertices()
o3d.io.write_triangle_mesh(OUT, mesh)
print(f"[tsdf] DONE: {len(mesh.vertices)} verts, {len(mesh.triangles)} tris -> {OUT}", flush=True)
