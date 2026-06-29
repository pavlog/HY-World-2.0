"""Shaded grey render of meshes (geometry only, no texture) for side-by-side quality compare.
Legacy Open3D Visualizer (hidden OpenGL window — works headless on Windows, unlike filament EGL).

  trellis_py _ws_shade_compare.py <out_dir> <label1=mesh1.ply> <label2=mesh2.ply> ...
"""
import sys, os
import numpy as np, open3d as o3d

OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
items = [a.split("=", 1) for a in sys.argv[2:]]
W, Hh = 960, 640
YAWS = [0, 90, 200]
PITCH = -0.35
VFOV = 60.0
fy = 0.5 * Hh / np.tan(np.radians(VFOV) / 2); fx = fy
K = np.array([[fx, 0, W / 2 - 0.5], [0, fy, Hh / 2 - 0.5], [0, 0, 1]])

def look_at_extrinsic(eye, center, world_up=(0, 0, 1)):
    fwd = center - eye; fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, world_up); right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    R = np.stack([right, down, fwd], 0)            # rows: cam x,y(down),z(fwd) in world
    t = -R @ eye
    E = np.eye(4); E[:3, :3] = R; E[:3, 3] = t
    return E

vis = o3d.visualization.Visualizer()
vis.create_window(width=W, height=Hh, visible=False)
opt = vis.get_render_option(); opt.background_color = np.array([1, 1, 1]); opt.light_on = True; opt.mesh_show_back_face = True
intr = o3d.camera.PinholeCameraIntrinsic(W, Hh, fx, fy, K[0, 2], K[1, 2])

for label, path in items:
    if not os.path.exists(path):
        print(f"[shade] MISSING {label}: {path}", flush=True); continue
    m = o3d.io.read_triangle_mesh(path); m.compute_vertex_normals()
    m.paint_uniform_color([0.72, 0.72, 0.74])
    bb = m.get_axis_aligned_bounding_box(); c = bb.get_center(); ext = float(np.linalg.norm(bb.get_extent()))
    vis.clear_geometries(); vis.add_geometry(m, reset_bounding_box=True)
    vc = vis.get_view_control()
    for yaw in YAWS:
        a = np.radians(yaw)
        d = np.array([np.cos(a), np.sin(a), -np.tan(PITCH)]); d = d / np.linalg.norm(d)
        eye = c + d * ext * 0.85
        param = o3d.camera.PinholeCameraParameters()
        param.intrinsic = intr
        param.extrinsic = look_at_extrinsic(eye, c)
        vc.convert_from_pinhole_camera_parameters(param, allow_arbitrary=True)
        vis.poll_events(); vis.update_renderer()
        fn = f"{OUT}/{label}_yaw{yaw:03d}.png"
        vis.capture_screen_image(fn, do_render=True)
        print(f"[shade] {fn}  (tris={len(m.triangles)})", flush=True)
vis.destroy_window()
print("[shade] DONE", flush=True)
