"""HW2-faithful mesh: render SMOOTH depth from the trained 3DGS gaussians via gsplat (RGB+ED, alpha-blended depth
-- not a center z-buffer), then Open3D TSDF fuse with generated RGB. This is HW2's render_views_gsplat + TSDF.
Run in the TRELLIS env (gsplat 1.4.0 + open3d 0.19 both work).

  trellis_py _ws_tsdf_gsplat.py <3dgs_point_cloud.ply> <lean_scene> <out.ply> [voxel=0.012]
"""
import sys, os, json, glob
import numpy as np, cv2, torch, open3d as o3d
from gsplat import rasterization

PLY, SCENE, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
VOXEL = float(sys.argv[4]) if len(sys.argv) > 4 else 0.012
dev = "cuda"; M = "worldstereo-memory-dmd"; W, H = 832, 480

# ---- parse 3DGS ply -> gsplat tensors ----
from plyfile import PlyData
v = PlyData.read(PLY)['vertex']
means = torch.tensor(np.stack([v['x'], v['y'], v['z']], 1), dtype=torch.float32, device=dev)
scales = torch.tensor(np.exp(np.stack([v['scale_0'], v['scale_1'], v['scale_2']], 1)), dtype=torch.float32, device=dev)
quats = torch.tensor(np.stack([v['rot_0'], v['rot_1'], v['rot_2'], v['rot_3']], 1), dtype=torch.float32, device=dev)
quats = quats / quats.norm(dim=-1, keepdim=True)
opac = torch.sigmoid(torch.tensor(np.array(v['opacity']), dtype=torch.float32, device=dev))
SH = 0.28209479177387814
colors = torch.tensor(np.clip(np.stack([v['f_dc_0'], v['f_dc_1'], v['f_dc_2']], 1) * SH + 0.5, 0, 1), dtype=torch.float32, device=dev)
print(f"[tsdf-gs] {len(means)} gaussians, voxel={VOXEL}", flush=True)

rr = f"{SCENE}/render_results"
vol = o3d.pipelines.integration.ScalableTSDFVolume(
    voxel_length=VOXEL, sdf_trunc=VOXEL * 4, color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
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
    vm = torch.tensor(np.array([ext[i] for i in range(n)]), dtype=torch.float32, device=dev)   # w2c [n,4,4]
    Ks = torch.tensor(np.array([intr[i] for i in range(n)]), dtype=torch.float32, device=dev)
    with torch.no_grad():
        out, alpha, _ = rasterization(means, quats, scales, opac, colors, vm, Ks, W, H, render_mode="RGB+ED")
    depth = out[..., 3].cpu().numpy()      # [n,H,W] expected depth
    for i in range(n):
        d = depth[i].copy(); d[d <= 1e-3] = 0.0
        rgb = cv2.cvtColor(cv2.resize(fr[i], (W, H)), cv2.COLOR_BGR2RGB)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.ascontiguousarray(rgb)), o3d.geometry.Image(d.astype(np.float32)),
            depth_scale=1.0, depth_trunc=10.0, convert_rgb_to_intensity=False)
        K = Ks[i].cpu().numpy(); pin = o3d.camera.PinholeCameraIntrinsic(W, H, K[0, 0], K[1, 1], K[0, 2], K[1, 2])
        vol.integrate(rgbd, pin, np.array(ext[i], np.float64))
        nint += 1
    if (ci + 1) % 8 == 0:
        print(f"[tsdf-gs] {ci+1}/{len(cams)} clips ({nint} frames)", flush=True)

print("[tsdf-gs] extracting mesh…", flush=True)
mesh = vol.extract_triangle_mesh(); mesh.compute_vertex_normals()
mesh.remove_degenerate_triangles(); mesh.remove_unreferenced_vertices()
o3d.io.write_triangle_mesh(OUT, mesh)
print(f"[tsdf-gs] DONE: {len(mesh.vertices)} verts, {len(mesh.triangles)} tris -> {OUT}", flush=True)
