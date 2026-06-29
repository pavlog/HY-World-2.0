"""Fuse the GENERATED exploration videos into one colored world cloud (path (i), HW2-native geometry).

For each generated clip (*_result.mp4): MoGe-2 metric depth/points per frame (using OUR known fov_x), scaled by a
SINGLE global factor s that aligns MoGe-metric to the pano-camera scale (estimated on the seed clips by comparing
MoGe depth vs the global_pcd-rendered depth), then transformed to world by the EXACT camera c2w and voxel-accumulated
with the generated RGB. Revealed/occluded geometry comes from MoGe-on-generation, NOT the centre-only pano cloud.

  worldmirror_py _ws_fuse_gen.py <scene_dir> <out_prefix> [voxel=0.015]
"""
import sys, os, json, glob
import numpy as np, cv2, torch, trimesh
sys.path.insert(0, "D:/HY-World-2.0/hyworld2/worldgen")
from moge.model.v2 import MoGeModel
from src.pointcloud import point_rendering

SCENE, OUTP = sys.argv[1], sys.argv[2]
V = float(sys.argv[3]) if len(sys.argv) > 3 else 0.015
MODEL = "worldstereo-memory-dmd"
dev = "cuda"
rr = f"{SCENE}/render_results"
OFF = 1 << 20

moge = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to(dev).eval()
gpts = torch.as_tensor(np.asarray(trimesh.load(f"{rr}/global_pcd.ply").vertices, np.float32), device=dev)

def read_frames(mp4):
    cap = cv2.VideoCapture(mp4); fr = []
    while True:
        ok, f = cap.read()
        if not ok: break
        fr.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    return fr

def moge_depth(frame_rgb, fov_x=None):
    # NOTE: do NOT force fov_x — MoGe degenerates to a flat wall at our 120deg split FOV; auto-estimate (~96deg)
    # gives real structure + self-consistent points/intrinsics. We use points + our extrinsic to place in world.
    img = torch.as_tensor(frame_rgb / 255.0, dtype=torch.float32, device=dev).permute(2, 0, 1)
    with torch.no_grad():
        out = moge.infer(img, use_fp16=True)
    return out["points"], out["depth"], out["mask"]   # points[H,W,3] cam, depth[H,W], mask[H,W]

clips = sorted(glob.glob(f"{rr}/view*/traj*/{MODEL}_result.mp4"),
               key=lambda p: (int(p.replace("\\", "/").split("/")[-3].replace("view", "")),
                              int(p.replace("\\", "/").split("/")[-2].replace("traj", ""))))
print(f"[fuse] {len(clips)} generated clips, voxel={V}", flush=True)

# ---- 1. global scale s: MoGe vs cloud-rendered depth on seed (traj0) first frames ----
ratios = []
for c in clips:
    if "/traj0/" not in c.replace("\\", "/"):
        continue
    td = os.path.dirname(c); cam = json.load(open(f"{td}/camera.json"))
    fov_x = 2 * np.arctan(cam["width"] / (2 * cam["intrinsic"][0][0][0]))
    fr = read_frames(c)
    if not fr: continue
    _, mdep, mmask = moge_depth(fr[0], fov_x)
    K = torch.as_tensor([cam["intrinsic"][0]], dtype=torch.float32, device=dev)
    w2c = torch.as_tensor([cam["extrinsic"][0]], dtype=torch.float32, device=dev)
    _, cdep = point_rendering(K, w2c, gpts, torch.zeros((len(gpts), 3), device=dev),
                              dev, cam["height"], cam["width"], render_radius=0.012, return_depth=True)
    cdep = cdep[0, 0]; md = mdep
    valid = (cdep > 1e-3) & torch.isfinite(md) & (md > 1e-3) & mmask
    if valid.sum() > 500:
        ratios.append((cdep[valid] / md[valid]).median().item())
s = float(np.median(ratios)) if ratios else 1.0
print(f"[fuse] global scale s={s:.4f} (from {len(ratios)} seed frames)", flush=True)

# ---- 2. fuse all clips: MoGe points * s -> world (exact c2w) -> voxel mean color ----
G = None; bx, bc = [], []
def encode(vk):
    return ((vk[:, 0] + OFF).astype(np.int64) << 42) | ((vk[:, 1] + OFF).astype(np.int64) << 21) | (vk[:, 2] + OFF).astype(np.int64)
def flush():
    global G, bx, bc
    if not bx: return
    xyz = np.concatenate(bx); rgb = np.concatenate(bc).astype(np.int64); bx, bc = [], []
    key = encode(np.floor(xyz / V).astype(np.int64))
    ck, cx, cy, cz = key, xyz[:, 0].astype(np.float64), xyz[:, 1].astype(np.float64), xyz[:, 2].astype(np.float64)
    cr, cg, cb, cc = rgb[:, 0], rgb[:, 1], rgb[:, 2], np.ones(len(key), np.int64)
    if G is not None:
        ck = np.concatenate([G['k'], ck]); cx = np.concatenate([G['x'], cx]); cy = np.concatenate([G['y'], cy]); cz = np.concatenate([G['z'], cz])
        cr = np.concatenate([G['r'], cr]); cg = np.concatenate([G['g'], cg]); cb = np.concatenate([G['b'], cb]); cc = np.concatenate([G['c'], cc])
    uk, inv = np.unique(ck, return_inverse=True)
    def sm(a, dt): o = np.zeros(len(uk), dt); np.add.at(o, inv, a); return o
    G = {'k': uk, 'x': sm(cx, np.float64), 'y': sm(cy, np.float64), 'z': sm(cz, np.float64),
         'r': sm(cr, np.int64), 'g': sm(cg, np.int64), 'b': sm(cb, np.int64), 'c': sm(cc, np.int64)}

nf = 0
for ci, c in enumerate(clips):
    td = os.path.dirname(c); cam = json.load(open(f"{td}/camera.json"))
    fov_x = 2 * np.arctan(cam["width"] / (2 * cam["intrinsic"][0][0][0]))
    fr = read_frames(c); ext = cam["extrinsic"]
    n = min(len(fr), len(ext))
    for i in range(n):
        pts_cam, _, mmask = moge_depth(fr[i], fov_x)
        m = mmask.cpu().numpy().reshape(-1)
        pc = (pts_cam.cpu().numpy().reshape(-1, 3) * s)[m]
        if len(pc) == 0: continue
        c2w = np.linalg.inv(np.array(ext[i], np.float64))
        ph = np.concatenate([pc, np.ones((len(pc), 1))], 1)
        world = (ph @ c2w.T)[:, :3].astype(np.float32)
        col = (fr[i].reshape(-1, 3))[m]
        bx.append(world); bc.append(col); nf += 1
    flush()
    if (ci + 1) % 8 == 0:
        print(f"[fuse] {ci+1}/{len(clips)} clips  voxels={0 if G is None else len(G['k'])}", flush=True)

c = G['c'].astype(np.float64)
pos = np.stack([G['x'] / c, G['y'] / c, G['z'] / c], 1).astype(np.float32)
rgb = np.stack([G['r'] / c, G['g'] / c, G['b'] / c], 1).clip(0, 255).astype(np.uint8)
np.savez_compressed(OUTP + ".npz", xyz=pos, color=rgb)
with open(OUTP + ".ply", "wb") as f:
    f.write(("ply\nformat binary_little_endian 1.0\nelement vertex %d\n"
             "property float x\nproperty float y\nproperty float z\n"
             "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n" % len(pos)).encode())
    rec = np.empty(len(pos), dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('r', 'u1'), ('g', 'u1'), ('b', 'u1')])
    rec['x'], rec['y'], rec['z'] = pos[:, 0], pos[:, 1], pos[:, 2]; rec['r'], rec['g'], rec['b'] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    f.write(rec.tobytes())
print(f"[fuse] DONE {nf} frames -> {len(pos)} voxels -> {OUTP}.ply / .npz", flush=True)
