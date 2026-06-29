import os, numpy as np, trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = r'D:/HY-World-2.0/_pano_out'
m = trimesh.load(os.path.join(OUT, 'game007_pano1_surround.glb'), force='mesh')
V = np.asarray(m.vertices, np.float32)
C = np.asarray(m.visual.vertex_colors[:, :3], np.float32) / 255.0
C = np.clip(C ** 0.45 * 1.35, 0, 1)            # gamma/exposure boost (dim library scene)
# subsample for speed
n = V.shape[0]
idx = np.random.RandomState(0).choice(n, size=min(400000, n), replace=False)
V, C = V[idx], C[idx]

def look_render(name, fwd):
    fwd = np.array(fwd, float); fwd /= np.linalg.norm(fwd)
    up0 = np.array([0, 1, 0.0]) if abs(fwd[1]) < 0.9 else np.array([0, 0, 1.0])
    right = np.cross(fwd, up0); right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    R = np.stack([right, up, fwd], 0)          # world->cam rows
    P = (R @ V.T).T                            # camera coords
    z = P[:, 2]
    front = z > 0.05
    Pf, Cf, zf = P[front], C[front], z[front]
    f = 0.9
    u = f * Pf[:, 0] / zf; v = f * Pf[:, 1] / zf
    keep = (np.abs(u) < 1) & (np.abs(v) < 1)
    u, v, Cf, zf = u[keep], v[keep], Cf[keep], zf[keep]
    order = np.argsort(-zf)                     # far first
    u, v, Cf = u[order], v[order], Cf[order]
    fig = plt.figure(figsize=(6, 6), dpi=150); ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor('#0d0d12')
    ax.scatter(u, v, c=np.clip(Cf, 0, 1), s=3.5, marker='o', linewidths=0, edgecolors='none')
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.axis('off')
    p = os.path.join(OUT, f'g007_interior_{name}.png')
    fig.savefig(p, facecolor='#0d0d12'); plt.close(fig)
    print('rendered', p, '| pts', len(u))

for nm, d in {'front': [0, 0, 1], 'right': [1, 0, 0], 'back': [0, 0, -1], 'left': [-1, 0, 0]}.items():
    look_render(nm, d)
print('done')
