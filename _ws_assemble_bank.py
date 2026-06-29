"""Assemble the generation_bank_ layout that HW2 Stage-3 (video_gen.py) would have produced, from our
real lean-generation outputs, so HW2's own gen_gs_data.py (Stage 4) runs unmodified. Pure plumbing:
- global_pcd.ply -> generation_bank/global_pcd.ply + aligned_pcd.ply
- per-frame moge_depth.npy -> generation_bank/{view}/{traj}/depths/{i:04d}.png (16-bit, HW2 encoding)
- copy source panorama.png to scene root

  python _ws_assemble_bank.py <scene_dir> <source_panorama.png> [result_name=worldstereo-memory-dmd]
"""
import sys, os, glob, shutil
import numpy as np
from PIL import Image

SCENE = sys.argv[1]; PANO_SRC = sys.argv[2]
RN = sys.argv[3] if len(sys.argv) > 3 else "worldstereo-memory-dmd"
rr = f"{SCENE}/render_results"; gb = f"{rr}/generation_bank_{RN}"
os.makedirs(gb, exist_ok=True)

def save_16bit_png_depth(depth, path):  # exact HW2 encoding (src/general_utils.py)
    u16 = np.array(depth, dtype=np.float32).astype(np.float16).view(np.uint16)
    Image.fromarray(u16).save(path)

# 1) point clouds
shutil.copy(f"{rr}/global_pcd.ply", f"{gb}/global_pcd.ply")
shutil.copy(f"{rr}/global_pcd.ply", f"{gb}/aligned_pcd.ply")
print(f"[bank] global_pcd + aligned_pcd -> {gb}", flush=True)

# 2) per-frame video depths from our precomputed aligned MoGe depth
cams = sorted(glob.glob(f"{rr}/view*/traj*/moge_depth.npy"))
nframes = 0
for mp in cams:
    td = os.path.dirname(mp); view = td.replace("\\", "/").split("/")[-2]; traj = td.replace("\\", "/").split("/")[-1]
    deps = np.load(mp)  # [n,H,W]
    dd = f"{gb}/{view}/{traj}/depths"; os.makedirs(dd, exist_ok=True)
    for i in range(len(deps)):
        save_16bit_png_depth(deps[i], f"{dd}/{i:04d}.png"); nframes += 1
print(f"[bank] wrote {nframes} depth PNGs across {len(cams)} trajs", flush=True)

# 3) source panorama at scene root (gen_gs_data reads {scene}/panorama.png)
shutil.copy(PANO_SRC, f"{SCENE}/panorama.png")
print(f"[bank] panorama.png -> {SCENE}", flush=True)
print("[bank] DONE", flush=True)
