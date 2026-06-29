"""Geometry-anchored voxel fuse of a traj-range from ONE bank scene (view0/traj0..N).
Render EXACT depth from our cloud per camera -> unproject the GENERATED rgb -> voxel-hash mean colour.
Structure = our exact geometry; appearance = generation; multi-view averaged.

  worldmirror_py _ws_fuse_cseg.py <scene_dir> <cloud.npz> <out_prefix> <t0> <t1>   (trajs t0..t1 inclusive)
"""
import sys, json, os, numpy as np, cv2

SCENE, CLOUD, OUTP = sys.argv[1], sys.argv[2], sys.argv[3]
T0 = int(sys.argv[4]) if len(sys.argv) > 4 else 0
T1 = int(sys.argv[5]) if len(sys.argv) > 5 else 999
W, H, V = 832, 480, 0.04
OFF = 1 << 20
MODEL = "worldstereo-memory-dmd"

xyz_cloud = np.load(CLOUD)["xyz"].astype(np.float32)
Ph = np.c_[xyz_cloud, np.ones(len(xyz_cloud), np.float32)]
uu, vv = np.meshgrid(np.arange(W), np.arange(H)); uu = uu.ravel().astype(np.float32); vv = vv.ravel().astype(np.float32)
G = None; buf_xyz, buf_rgb = [], []

def render_depth(K, M):
    cam = (Ph @ np.asarray(M, np.float32).T)[:, :3]; zc = cam[:, 2]; m = zc > 1e-3
    cam, zc = cam[m], zc[m]
    fx, fy, cx, cy = K[0][0], K[1][1], K[0][2], K[1][2]
    u = (fx * cam[:, 0] / zc + cx).astype(np.int32); v = (fy * cam[:, 1] / zc + cy).astype(np.int32)
    ok = (u >= 0) & (u < W) & (v >= 0) & (v < H); u, v, zc = u[ok], v[ok], zc[ok]
    depth = np.full(H * W, np.inf, np.float32); lin = v * W + u
    o = np.argsort(-zc); depth[lin[o]] = zc[o]; depth[~np.isfinite(depth)] = 0.0
    return depth.reshape(H, W)

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
trajs = sorted([int(d.split('traj')[-1]) for d in os.listdir(base) if d.startswith('traj')])
trajs = [t for t in trajs if T0 <= t <= T1]
nv = 0
for t in trajs:
    cj = f"{base}/traj{t}/camera.json"; mp4 = f"{base}/traj{t}/{MODEL}_result.mp4"
    if not (os.path.exists(cj) and os.path.exists(mp4)): continue
    cam = json.load(open(cj)); extr, intr = cam["extrinsic"], cam["intrinsic"]
    cap = cv2.VideoCapture(mp4); frames = []
    while True:
        r, f = cap.read()
        if not r: break
        frames.append(f)
    cap.release()
    for i in range(min(len(frames), len(extr))):
        K = intr[i]; M = np.asarray(extr[i], np.float64); c2w = np.linalg.inv(M)
        depth = render_depth(K, M).ravel(); valid = depth > 1e-3; z = depth[valid]
        fx, fy, cx, cy = K[0][0], K[1][1], K[0][2], K[1][2]
        camx = (uu[valid] - cx) * z / fx; camy = (vv[valid] - cy) * z / fy
        camP = np.stack([camx, camy, z, np.ones_like(z)], 1)
        world = (camP @ c2w.T)[:, :3].astype(np.float32)
        col = cv2.cvtColor(cv2.resize(frames[i], (W, H)), cv2.COLOR_BGR2RGB).reshape(-1, 3)[valid]
        buf_xyz.append(world); buf_rgb.append(col); nv += 1
    flush(); print(f"  traj{t}: voxels={0 if G is None else len(G['k'])}", flush=True)

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
print(f"FUSED trajs {trajs[0]}..{trajs[-1]}  voxels={len(pos)}  views={nv} -> {OUTP}.ply", flush=True)
