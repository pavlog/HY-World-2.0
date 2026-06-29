"""Geometry-anchored voxel fuse using the MESH-render EXACT depth (polygon prior) + the GENERATED rgb.
For each frame: read the mesh Z-pass depth (exact, from the authored mesh) + the generated result frame, unproject
to world, voxel-hash mean colour. Structure = our exact polygon geometry; appearance = generation. No cloud.

  worldmirror_py _ws_fuse_mesh.py <scene_dir> <out_prefix> [t0] [t1]
"""
import sys, os, json, glob
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import numpy as np, cv2

SCENE, OUTP = sys.argv[1], sys.argv[2]
T0 = int(sys.argv[3]) if len(sys.argv) > 3 else 0
T1 = int(sys.argv[4]) if len(sys.argv) > 4 else 9999
W, H, V = 832, 480, 0.04
OFF = 1 << 20
MODEL = "worldstereo-memory-dmd"
uu, vv = np.meshgrid(np.arange(W), np.arange(H)); uu = uu.ravel().astype(np.float32); vv = vv.ravel().astype(np.float32)
G = None; buf_xyz, buf_rgb = [], []

def encode(vk):
    return ((vk[:, 0] + OFF).astype(np.int64) << 42) | ((vk[:, 1] + OFF).astype(np.int64) << 21) | (vk[:, 2] + OFF).astype(np.int64)

def flush():
    global G, buf_xyz, buf_rgb
    if not buf_xyz: return
    xyz = np.concatenate(buf_xyz); rgb = np.concatenate(buf_rgb).astype(np.int64); buf_xyz, buf_rgb = [], []
    key = encode(np.floor(xyz / V).astype(np.int64))
    ck, cx, cy, cz = key, xyz[:, 0].astype(np.float64), xyz[:, 1].astype(np.float64), xyz[:, 2].astype(np.float64)
    cr, cg, cb, cc = rgb[:, 0], rgb[:, 1], rgb[:, 2], np.ones(len(key), np.int64)
    if G is not None:
        ck = np.concatenate([G['k'], ck]); cx = np.concatenate([G['x'], cx]); cy = np.concatenate([G['y'], cy]); cz = np.concatenate([G['z'], cz])
        cr = np.concatenate([G['r'], cr]); cg = np.concatenate([G['g'], cg]); cb = np.concatenate([G['b'], cb]); cc = np.concatenate([G['c'], cc])
    uk, inv = np.unique(ck, return_inverse=True)
    def s(a, dt): o = np.zeros(len(uk), dt); np.add.at(o, inv, a); return o
    G = {'k': uk, 'x': s(cx, np.float64), 'y': s(cy, np.float64), 'z': s(cz, np.float64),
         'r': s(cr, np.int64), 'g': s(cg, np.int64), 'b': s(cb, np.int64), 'c': s(cc, np.int64)}

base = f"{SCENE}/render_results/view0"
trajs = sorted(int(os.path.basename(d).replace('traj', '')) for d in glob.glob(f"{base}/traj*") if os.path.isdir(d))
trajs = [t for t in trajs if T0 <= t <= T1]
nv = 0
for t in trajs:
    cj = json.load(open(f"{base}/traj{t}/camera.json")); extr, intr = cj["extrinsic"], cj["intrinsic"]
    deps = sorted(glob.glob(f"{SCENE}/_meshrender/traj{t}/depth_*.exr"))
    cap = cv2.VideoCapture(f"{base}/traj{t}/{MODEL}_result.mp4"); gen = []
    while True:
        r, f = cap.read()
        if not r: break
        gen.append(f)
    cap.release()
    nframes = min(len(deps), len(gen), len(extr))
    for i in range(nframes):
        dep = cv2.imread(deps[i], cv2.IMREAD_UNCHANGED)
        z = (dep[:, :, 0] if dep.ndim == 3 else dep).astype(np.float32).ravel()
        valid = (z > 1e-3) & (z < 1e8)
        if not valid.any(): continue
        K = intr[i]; M = np.asarray(extr[i], np.float64); c2w = np.linalg.inv(M)
        fx, fy, cx, cy = K[0][0], K[1][1], K[0][2], K[1][2]
        zz = z[valid]
        camx = (uu[valid] - cx) * zz / fx; camy = (vv[valid] - cy) * zz / fy
        camP = np.stack([camx, camy, zz, np.ones_like(zz)], 1)
        world = (camP @ c2w.T)[:, :3].astype(np.float32)
        col = cv2.cvtColor(cv2.resize(gen[i], (W, H)), cv2.COLOR_BGR2RGB).reshape(-1, 3)[valid]
        buf_xyz.append(world); buf_rgb.append(col); nv += 1
    flush(); print(f"  traj{t}: {nframes} frames  voxels={0 if G is None else len(G['k'])}", flush=True)

c = G['c'].astype(np.float64)
pos = np.stack([G['x'] / c, G['y'] / c, G['z'] / c], 1).astype(np.float32)
rgb = np.stack([G['r'] / c, G['g'] / c, G['b'] / c], 1).clip(0, 255).astype(np.uint8)
np.savez_compressed(OUTP + ".npz", xyz=pos, color=rgb)
with open(OUTP + ".ply", "wb") as f:
    f.write(("ply\nformat binary_little_endian 1.0\nelement vertex %d\n"
             "property float x\nproperty float y\nproperty float z\n"
             "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n" % len(pos)).encode())
    rec = np.empty(len(pos), dtype=[('x','<f4'),('y','<f4'),('z','<f4'),('r','u1'),('g','u1'),('b','u1')])
    rec['x'], rec['y'], rec['z'] = pos[:,0], pos[:,1], pos[:,2]; rec['r'], rec['g'], rec['b'] = rgb[:,0], rgb[:,1], rgb[:,2]
    f.write(rec.tobytes())
print(f"FUSED (mesh-depth) trajs {trajs[0]}..{trajs[-1]}  voxels={len(pos)}  views={nv} -> {OUTP}.ply", flush=True)
