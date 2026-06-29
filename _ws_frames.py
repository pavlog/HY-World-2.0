"""STEP: export a video (or stitched clip set) to numbered per-frame PNGs in ONE folder.
  python _ws_frames.py <in.mp4> <out_dir>                 # single mp4
  python _ws_frames.py --clips <out_dir> <clip0> <clip1>  # stitch clip result mp4s (drop boundary dup)
"""
import sys, os, cv2

def read(p):
    c = cv2.VideoCapture(p); fr = []
    while True:
        r, f = c.read()
        if not r: break
        fr.append(f)
    c.release(); return fr

if sys.argv[1] == "--clips":
    OUT = sys.argv[2]; clips = sys.argv[3:]
    allf = []
    for i, scene in enumerate(clips):
        g = read(f"{scene}/render_results/view0/traj0/worldstereo-memory-dmd_result.mp4")
        allf += g if i == 0 else g[1:]            # drop duplicate boundary frame
else:
    OUT = sys.argv[2]; allf = read(sys.argv[1])

os.makedirs(OUT, exist_ok=True)
for i, f in enumerate(allf):
    cv2.imwrite(f"{OUT}/frame_{i:04d}.png", f)
print(f"exported {len(allf)} frames -> {OUT}")
