import os, numpy as np, open3d as o3d
OUT = r'D:/HY-World-2.0/_pano_out'
GLB = os.path.join(OUT, 'pano_moge_surround.glb')
mesh = o3d.io.read_triangle_mesh(GLB)
mesh.compute_vertex_normals()
W = H = 900
ren = o3d.visualization.rendering.OffscreenRenderer(W, H)
ren.scene.set_background([0.05, 0.05, 0.07, 1.0])
mat = o3d.visualization.rendering.MaterialRecord(); mat.shader = 'defaultLit'
ren.scene.add_geometry('m', mesh, mat)
# camera at origin, look outward; mesh built with cam at origin (rays from origin)
dirs = {'front': [0, 0, 1], 'right': [1, 0, 0], 'back': [0, 0, -1], 'left': [-1, 0, 0],
        'down': [0, -1, 0.001], 'up': [0, 1, 0.001]}
eye = np.array([0.0, 0.0, 0.0])
up = np.array([0.0, 1.0, 0.0])
ren.scene.camera.set_projection(90.0, W / H, 0.01, 1000.0, o3d.visualization.rendering.Camera.FovType.Vertical)
for name, d in dirs.items():
    d = np.array(d, float); center = eye + d
    u = up if abs(d[1]) < 0.9 else np.array([0.0, 0.0, 1.0])
    ren.scene.camera.look_at(center.tolist(), eye.tolist(), u.tolist())
    img = ren.render_to_image()
    p = os.path.join(OUT, f'moge_interior_{name}.png')
    o3d.io.write_image(p, img)
    print('rendered', p)
print('done')
