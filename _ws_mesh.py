"""Fused cloud -> clean Poisson mesh with DENSE vertex colors -> GLB + PLY (Blender-ready).

Statistical outlier removal + normal estimation/orientation + screened Poisson + density-trim (cuts the
low-confidence ballooned shell Poisson adds in unseen regions). Vertex colors carry through from the cloud.

  worldmirror_py _ws_mesh.py <fused.npz|.ply> <out_prefix> [poisson_depth=10] [density_trim_q=0.04]
"""
import sys, os
import numpy as np, open3d as o3d, trimesh

INP, OUT = sys.argv[1], sys.argv[2]
DEPTH = int(sys.argv[3]) if len(sys.argv) > 3 else 10
TRIMQ = float(sys.argv[4]) if len(sys.argv) > 4 else 0.04

if INP.endswith(".npz"):
    z = np.load(INP); P = z["xyz"].astype(np.float64); C = z["color"].astype(np.float64) / 255.0
else:
    m = trimesh.load(INP); P = np.asarray(m.vertices, np.float64); C = np.asarray(m.visual.vertex_colors)[:, :3] / 255.0
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(P)
pcd.colors = o3d.utility.Vector3dVector(np.clip(C, 0, 1))
print(f"[mesh] {len(P)} points  bounds={ (P.max(0)-P.min(0)).round(2) }", flush=True)

diag = np.linalg.norm(P.max(0) - P.min(0))
if len(pcd.points) > 2_000_000:                                 # cap for speed (already voxel-fused)
    pcd = pcd.voxel_down_sample(diag * 0.004)
pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=diag * 0.02, max_nn=30))
pcd.orient_normals_consistent_tangent_plane(15)   # towards_camera_location segfaults in o3d 0.18; this is stable post-downsample
print(f"[mesh] {len(pcd.points)} after downsample+outlier-removal; normals oriented", flush=True)

mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=DEPTH, linear_fit=True)
dens = np.asarray(dens)
keep = dens > np.quantile(dens, TRIMQ)
mesh.remove_vertices_by_mask(~keep)
mesh.remove_degenerate_triangles(); mesh.remove_unreferenced_vertices()
mesh.compute_vertex_normals()
print(f"[mesh] poisson depth={DEPTH} -> {len(mesh.vertices)} verts {len(mesh.triangles)} tris (trimmed {(~keep).mean():.0%})", flush=True)

o3d.io.write_triangle_mesh(OUT + ".ply", mesh)
V = np.asarray(mesh.vertices); Fc = np.asarray(mesh.triangles)
VC = (np.clip(np.asarray(mesh.vertex_colors), 0, 1) * 255).astype(np.uint8)
VC = np.concatenate([VC, np.full((len(VC), 1), 255, np.uint8)], 1)
trimesh.Trimesh(vertices=V, faces=Fc, vertex_colors=VC, process=False).export(OUT + ".glb")
print(f"[mesh] DONE -> {OUT}.glb / .ply  size={ (V.max(0)-V.min(0)).round(2) }", flush=True)
