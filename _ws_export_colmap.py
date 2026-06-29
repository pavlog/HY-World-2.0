"""Export a project's generated frames + known cameras + scaffold cloud as a COLMAP dataset for SuGaR/3DGS.
Our cameras are OpenCV w2c == COLMAP convention (+X right, +Y down, +Z forward), so w2c -> (quat,t) directly.

  worldmirror_py _ws_export_colmap.py <project_dir> <out_dir>
e.g. _ws_export_colmap.py D:/_world_hangar/_ws_projects/hangar D:/SuGaR/data/hangar
"""
import sys, os, json, glob, shutil
import numpy as np, cv2, trimesh

PROJ = sys.argv[1]
OUT = sys.argv[2]
rr = f"{PROJ}/render_results" if os.path.exists(f"{PROJ}/render_results") else f"{PROJ}/scene/render_results"
FR = f"{PROJ}/frames"
os.makedirs(f"{OUT}/images", exist_ok=True)
os.makedirs(f"{OUT}/sparse/0", exist_ok=True)


def mat2quat(R):
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2; qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s; qy = (R[0, 2] - R[2, 0]) / s; qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2; qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s; qy = (R[0, 1] + R[1, 0]) / s; qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2; qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s; qy = 0.25 * s; qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2; qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s; qy = (R[1, 2] + R[2, 1]) / s; qz = 0.25 * s
    return qw, qx, qy, qz


# gather (bgr_frame, w2c, K) — project (chain.json) OR lean scene (view*/traj* result mp4s)
items = []  # (frame_bgr, w2c, K)
if os.path.exists(f"{PROJ}/chain.json"):
    hist = json.load(open(f"{PROJ}/chain.json"))["history"]
    for si, h in enumerate(hist):
        poses = h["poses"]; K = np.array(h["intrinsic"], np.float32)
        for fi in range(len(poses)):
            fp = f"{FR}/s{si:03d}_f{fi}.png"
            if os.path.exists(fp):
                items.append((cv2.imread(fp), np.array(poses[fi], np.float64), K))
    print(f"[colmap] {len(items)} frames from {len(hist)} chain steps")
else:                                                       # lean scene: render_results/view*/traj*/result.mp4
    M = "worldstereo-memory-dmd"
    cams = sorted(glob.glob(f"{rr}/view*/traj*/camera.json"),
                  key=lambda p: (int(p.replace("\\", "/").split("/")[-3].replace("view", "")),
                                 int(p.replace("\\", "/").split("/")[-2].replace("traj", ""))))
    for cj in cams:
        td = os.path.dirname(cj); cam = json.load(open(cj)); ext = cam["extrinsic"]; K = np.array(cam["intrinsic"][0], np.float32)
        res = f"{td}/{M}_result.mp4"
        if not os.path.exists(res): continue
        cap = cv2.VideoCapture(res); fr = []
        while True:
            ok, f = cap.read()
            if not ok: break
            fr.append(f)
        cap.release()
        for i in range(min(len(fr), len(ext))):
            items.append((fr[i], np.array(ext[i], np.float64), K))
    print(f"[colmap] {len(items)} frames from {len(cams)} lean trajs")
if not items:
    print("[colmap] no frames"); sys.exit(1)

H, W = items[0][0].shape[:2]
K0 = items[0][2]
# cameras.txt (one shared PINHOLE)
with open(f"{OUT}/sparse/0/cameras.txt", "w") as f:
    f.write("# Camera list\n")
    f.write(f"1 PINHOLE {W} {H} {K0[0,0]:.6f} {K0[1,1]:.6f} {K0[0,2]:.6f} {K0[1,2]:.6f}\n")

# images.txt
with open(f"{OUT}/sparse/0/images.txt", "w") as f:
    f.write("# Image list\n")
    for idx, (frame, w2c, K) in enumerate(items, 1):
        R = w2c[:3, :3]; t = w2c[:3, 3]
        qw, qx, qy, qz = mat2quat(R)
        name = f"{idx:05d}.png"
        cv2.imwrite(f"{OUT}/images/{name}", frame)
        f.write(f"{idx} {qw:.8f} {qx:.8f} {qy:.8f} {qz:.8f} {t[0]:.8f} {t[1]:.8f} {t[2]:.8f} 1 {name}\n\n")

# points3D.txt from scaffold cloud (init points)
g = trimesh.load(f"{rr}/global_pcd.ply")
V = np.asarray(g.vertices); C = np.asarray(g.visual.vertex_colors)[:, :3]
n = min(100000, len(V)); idx = np.random.choice(len(V), n, replace=False)
with open(f"{OUT}/sparse/0/points3D.txt", "w") as f:
    f.write("# 3D point list\n")
    for pid, i in enumerate(idx, 1):
        f.write(f"{pid} {V[i,0]:.6f} {V[i,1]:.6f} {V[i,2]:.6f} {int(C[i,0])} {int(C[i,1])} {int(C[i,2])} 0\n")
print(f"[colmap] DONE -> {OUT}  ({len(items)} images, {n} init points, {W}x{H})")
