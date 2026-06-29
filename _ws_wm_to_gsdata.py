"""Convert WorldMirror reconstruction output -> world_gs_trainer gs_data format.
Real-video path (no pano/polar). Uses WM's actual metric depth (depth_*.npy, NOT the normalized png),
w2c extrinsics + K from camera_params.json, computes camera-space normals from depth (what the trainer wants),
and WM points.ply as SfM init. Output feeds world_gs_trainer exactly like the generated-world gs_data.

  trellis_py _ws_wm_to_gsdata.py <wm_outdir> <gs_data_out>
"""
import sys, os, json, glob, shutil
import numpy as np, cv2
from PIL import Image

WM, OUT = sys.argv[1], sys.argv[2]
for d in ("images", "depths", "normals"):
    os.makedirs(f"{OUT}/{d}", exist_ok=True)

cam = json.load(open(f"{WM}/camera_params.json"))
extr = [np.array(e["matrix"], np.float64) for e in cam["extrinsics"]]   # w2c [4,4]
intr = [np.array(k["matrix"], np.float64) for k in cam["intrinsics"]]   # K [3,3]
N = len(extr)

def find(patterns):
    for p in patterns:
        f = sorted(glob.glob(f"{WM}/{p}"))
        if f: return f
    return []
imgs = find(["images/*.png", "images/*.jpg", "rgb/*.png", "*.png"])
deps = find(["depth/depth_*.npy", "depths/*.npy", "depth/*.npy", "**/depth_*.npy"])
assert len(deps) >= N, f"found {len(deps)} depth npy for {N} cameras"

def save_16bit(depth, path):                       # HW2 load_16bit_png_depth encoding
    u16 = np.array(depth, np.float32).astype(np.float16).view(np.uint16)
    Image.fromarray(u16).save(path)

def depth_to_normal(d, K):                         # camera-space normals from depth gradients
    H, W = d.shape; fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    vv, uu = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    X = (uu - cx) * d / fx; Y = (vv - cy) * d / fy
    P = np.stack([X, Y, d], -1)
    du = np.zeros_like(P); dv = np.zeros_like(P)
    du[1:-1, 1:-1] = P[1:-1, 2:] - P[1:-1, :-2]; dv[1:-1, 1:-1] = P[2:, 1:-1] - P[:-2, 1:-1]
    n = np.cross(du, dv); ln = np.linalg.norm(n, axis=-1, keepdims=True)
    n = np.divide(n, ln, out=np.zeros_like(n), where=ln > 1e-8)
    n = np.where(n[..., 2:3] > 0, -n, n)           # face camera
    return n

cams = {}
W0 = H0 = None
for i in range(N):
    name = f"frame_{i:06d}"
    rgb = cv2.cvtColor(cv2.imread(imgs[i]), cv2.COLOR_BGR2RGB)
    d = np.load(deps[i]).astype(np.float32)
    H, W = d.shape; W0, H0 = W, H        # WM processed resolution = the resolution the intrinsics are for
    if rgb.shape[:2] != (H, W):          # downscale the full-res frame to match depth/intrinsics
        rgb = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_AREA)
    Image.fromarray(rgb).save(f"{OUT}/images/{name}.png")
    save_16bit(d, f"{OUT}/depths/{name}.png")
    nrm = depth_to_normal(d, intr[i])
    Image.fromarray((((nrm + 1) / 2) * 255).astype(np.uint8)).save(f"{OUT}/normals/{name}.png")
    cams[name] = {"extrinsic": extr[i].tolist(), "intrinsic": intr[i].tolist()}

cams["width"], cams["height"] = W0, H0
json.dump(cams, open(f"{OUT}/cameras.json", "w"), indent=2)

pts = find(["points.ply", "*points*.ply", "**/points.ply"])
if pts: shutil.copy(pts[0], f"{OUT}/points.ply")
else:   print("[warn] no points.ply found in WM output — set init from gs.ply or random")
print(f"[wm2gs] {N} frames -> {OUT}  (images+depths+normals+cameras.json{' +points.ply' if pts else ''})", flush=True)
