"""Standalone open3d Poisson mesher — run in a SUBPROCESS by the step server so an
open3d C++ segfault (which no try/except can catch) kills only this child, not the
resident server. Reads a point cloud from an .npz, writes <out>.glb + <out>.meshinfo.json.

  python _ws_mesher.py <in.npz(V,C)> <out_base> <depth> <fast:0|1> <voxel> <trim_quantile>
"""
import sys, json, numpy as np, open3d as o3d, trimesh

inp, outbase = sys.argv[1], sys.argv[2]
depth = int(sys.argv[3]); fast = sys.argv[4] == '1'
voxel = float(sys.argv[5]); trimq = float(sys.argv[6]) if len(sys.argv) > 6 else 0.02

d = np.load(inp)
V = d['V'].astype(np.float64); C = d['C'].astype(np.float64)

# --- sanitize: open3d Poisson segfaults on non-finite points and on NaN normals; the fast
#     path orients normals toward the camera origin (0,0,0), so a point AT the origin gives a
#     zero direction -> NaN normal -> crash. Drop both. ---
finite = np.isfinite(V).all(1) & np.isfinite(C).all(1)
if fast:
    finite &= (np.linalg.norm(V, axis=1) > 1e-4)
V = V[finite]; C = np.clip(C[finite], 0, 1)
if len(V) < 100:
    json.dump({"error": "too few valid points"}, open(outbase + ".meshinfo.json", "w")); sys.exit(0)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(V); pcd.colors = o3d.utility.Vector3dVector(C)
diag = float(np.linalg.norm(V.max(0) - V.min(0)))
if voxel and voxel > 0:
    pcd = pcd.voxel_down_sample(voxel)
pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=(8 if fast else 20), std_ratio=2.5)
pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=diag * 0.02, max_nn=(16 if fast else 30)))
# Orient normals toward the camera origin (0,0,0) — our scenes are interior, camera at origin.
# Done in numpy: open3d's orient_normals_towards_camera_location / consistent_tangent_plane
# SEGFAULT on some clouds in this build; a manual sign-flip is crash-free and equivalent for interiors.
_n = np.asarray(pcd.normals); _p = np.asarray(pcd.points)
if _n.shape == _p.shape and len(_n):
    _flip = np.einsum('ij,ij->i', _n, _p) > 0        # normal points away from origin -> flip it toward the camera
    _n[_flip] *= -1.0
    pcd.normals = o3d.utility.Vector3dVector(_n)

# guard: any non-finite normal would crash Poisson
nrm = np.asarray(pcd.normals)
if nrm.size and not np.isfinite(nrm).all():
    keep = np.isfinite(nrm).all(1)
    pcd = pcd.select_by_index(np.where(keep)[0])

m, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth, linear_fit=(not fast))
dens = np.asarray(dens)
m.remove_vertices_by_mask(dens < np.quantile(dens, trimq))
m.remove_degenerate_triangles(); m.remove_unreferenced_vertices(); m.compute_vertex_normals()

MV = np.asarray(m.vertices); Fc = np.asarray(m.triangles)
MC = (np.clip(np.asarray(m.vertex_colors), 0, 1) * 255).astype(np.uint8)
trimesh.Trimesh(vertices=MV, faces=Fc,
                vertex_colors=np.concatenate([MC, np.full((len(MC), 1), 255, np.uint8)], 1),
                process=False).export(outbase + ".glb")
o3d.io.write_triangle_mesh(outbase + ".ply", m)
json.dump({"nverts": int(len(MV)), "ntris": int(len(Fc))}, open(outbase + ".meshinfo.json", "w"))
