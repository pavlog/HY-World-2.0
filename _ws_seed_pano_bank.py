"""Seed the APPEARANCE store (pano_bank) of a PanoramaMemoryBank from a clip's FRAMES.

pano_bank layout the bank expects (retrieval_wm.py __init__ :776-807, flat globs):
  pano_bank/images/<key>.png     RGB frames
  pano_bank/depths/<key>.png      16-bit (float16-as-uint16) metric depth, same H,W
  pano_bank/cameras.json          {key: {"extrinsic": w2c[4x4], "intrinsic": K[3x3]}}

Depth is rendered from OUR cloud (metric) at the clip cameras via the pytorch3d-free point_rendering — so the
seed depth is consistent with global_pcd (our skeleton). RGB frames come from `frames_mp4`.

IMPORTANT (plan Phase 0.2): for the REAL run `frames_mp4` MUST be a GENERATED clip (worldstereo result), never
our prior conditioning render — else the prior appearance leaks into the memory. A prior render is acceptable
ONLY as a throwaway CONSTRUCTION-TEST placeholder (pass --placeholder to label it).

  python _ws_seed_pano_bank.py <scene> <cloud.npz> <frames_mp4> [view0] [traj0] [--placeholder]
"""
import sys, os, json, glob
import numpy as np, cv2, torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "hyworld2", "worldgen"))
from src.general_utils import save_16bit_png_depth
from src.pointcloud import point_rendering


def read_frames(mp4):
    cap = cv2.VideoCapture(mp4); fr = []
    while True:
        ok, f = cap.read()
        if not ok: break
        fr.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    return fr


def seed(scene, cloud_npz, frames_mp4, view="view0", traj="traj0", placeholder=False):
    rr = f"{scene}/render_results"
    pb = f"{rr}/pano_bank"
    os.makedirs(f"{pb}/images", exist_ok=True); os.makedirs(f"{pb}/depths", exist_ok=True)
    if placeholder:
        print("  [PLACEHOLDER seed — construction test only, NOT a real generated memory]")

    frames = read_frames(frames_mp4)
    cam = json.load(open(f"{rr}/{view}/{traj}/camera.json"))
    ext = cam["extrinsic"]; intr = cam["intrinsic"]
    n = min(len(frames), len(ext))
    H, W = frames[0].shape[:2]

    z = np.load(cloud_npz)
    sky = z["is_sky"].astype(bool) if "is_sky" in z else np.zeros(len(z["xyz"]), bool)
    pts = torch.from_numpy(z["xyz"][~sky].astype(np.float32))      # geometry (non-sky) for metric depth
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Ks = torch.tensor(np.stack([intr[i] for i in range(n)]), dtype=torch.float32)
    w2cs = torch.tensor(np.stack([ext[i] for i in range(n)]), dtype=torch.float32)
    # render metric depth from cloud at all clip cameras (colors unused -> zeros)
    _, depth = point_rendering(Ks, w2cs, pts, torch.zeros((pts.shape[0], 3)), dev, H, W,
                               render_radius=0.012, return_depth=True)   # [n,1,H,W], z, -1 empty
    depth = depth[:, 0].cpu().numpy()

    cams = {}
    for i in range(n):
        key = f"{i:05d}"
        cv2.imwrite(f"{pb}/images/{key}.png", cv2.cvtColor(frames[i], cv2.COLOR_RGB2BGR))
        d = depth[i].copy(); d[d < 0] = 0.0
        save_16bit_png_depth(d.astype(np.float32), f"{pb}/depths/{key}.png")
        cams[key] = {"extrinsic": np.array(ext[i], float).tolist(),
                     "intrinsic": np.array(intr[i], float).tolist()}
    json.dump(cams, open(f"{pb}/cameras.json", "w"))
    cov = float((depth.reshape(n, -1) > 0).mean())
    print(f"  pano_bank <- {n} frames  images+depths+cameras.json  depth-coverage={cov:.2%}")
    print("SEED pano_bank DONE" + ("  (PLACEHOLDER)" if placeholder else ""))


def seed_multi(scene, cloud_npz, seeds, view="view0"):
    """Seed pano_bank from MULTIPLE clean self-ref clips = a GLOBAL coverage anchor (our panorama-equivalent,
    HW2-style). seeds = [(traj_idx, frames_mp4), ...]. Each clip's frames get unique keys + cloud-rendered metric
    depth + its camera. Memory then covers the whole path -> retrieval always returns a clean spread anchor."""
    rr = f"{scene}/render_results"; pb = f"{rr}/pano_bank"
    os.makedirs(f"{pb}/images", exist_ok=True); os.makedirs(f"{pb}/depths", exist_ok=True)
    z = np.load(cloud_npz)
    sky = z["is_sky"].astype(bool) if "is_sky" in z else np.zeros(len(z["xyz"]), bool)
    pts = torch.from_numpy(z["xyz"][~sky].astype(np.float32))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cams = {}; total = 0
    for traj, mp4 in seeds:
        frames = read_frames(mp4)
        cam = json.load(open(f"{rr}/{view}/traj{traj}/camera.json")); ext, intr = cam["extrinsic"], cam["intrinsic"]
        n = min(len(frames), len(ext)); H, W = frames[0].shape[:2]
        Ks = torch.tensor(np.stack([intr[i] for i in range(n)]), dtype=torch.float32)
        w2cs = torch.tensor(np.stack([ext[i] for i in range(n)]), dtype=torch.float32)
        _, depth = point_rendering(Ks, w2cs, pts, torch.zeros((pts.shape[0], 3)), dev, H, W,
                                   render_radius=0.012, return_depth=True)
        depth = depth[:, 0].cpu().numpy()
        for i in range(n):
            key = f"t{traj:03d}f{i:02d}"                      # unique across seed trajs
            cv2.imwrite(f"{pb}/images/{key}.png", cv2.cvtColor(frames[i], cv2.COLOR_RGB2BGR))
            d = depth[i].copy(); d[d < 0] = 0.0
            save_16bit_png_depth(d.astype(np.float32), f"{pb}/depths/{key}.png")
            cams[key] = {"extrinsic": np.array(ext[i], float).tolist(), "intrinsic": np.array(intr[i], float).tolist()}
            total += 1
    json.dump(cams, open(f"{pb}/cameras.json", "w"))
    print(f"  pano_bank <- {total} frames from {len(seeds)} clean seed clips (global anchor)")
    print("SEED_MULTI pano_bank DONE")


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    seed(a[0], a[1], a[2], a[3] if len(a) > 3 else "view0", a[4] if len(a) > 4 else "traj0",
         placeholder="--placeholder" in sys.argv)
