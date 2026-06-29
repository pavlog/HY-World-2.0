"""Restructure our per-clip scene dirs into ONE scene with traj0..trajN (what PanoramaMemoryBank spans).
  worldmirror_py _ws_to_scene.py <OUT> <clip0> <clip1> ...
Also writes a sky_mask.png (union of per-traj render_mask coverage; white=ground, black=space)."""
import sys, os, json, shutil, numpy as np, cv2
OUT = sys.argv[1]; clips = sys.argv[2:]
v = f"{OUT}/render_results/view0"; os.makedirs(v, exist_ok=True)
W = H = None
cover = None
for i, c in enumerate(clips):
    src = f"{c}/render_results/view0/traj0"; dst = f"{v}/traj{i}"; os.makedirs(dst, exist_ok=True)
    for f in ("render.mp4", "render_mask.mp4", "camera.json", "traj_caption.json"):
        if os.path.exists(f"{src}/{f}"): shutil.copy(f"{src}/{f}", f"{dst}/{f}")
    if i == 0 and os.path.exists(f"{c}/render_results/view0/start_frame.png"):
        shutil.copy(f"{c}/render_results/view0/start_frame.png", f"{v}/start_frame.png")
    if os.path.exists(f"{c}/panorama.png") and not os.path.exists(f"{OUT}/panorama.png"):
        shutil.copy(f"{c}/panorama.png", f"{OUT}/panorama.png")
    # accumulate coverage for the sky_mask (ground = anywhere geometry projects in ANY traj)
    cap = cv2.VideoCapture(f"{src}/render_mask.mp4")
    while True:
        ok, fr = cap.read()
        if not ok: break
        if cover is None: H, W = fr.shape[:2]; cover = np.zeros((H, W), bool)
        cover |= (fr[:, :, 0] > 10)
    cap.release()
json.dump({"scene_type": "indoor"}, open(f"{OUT}/meta_info.json", "w"))
if cover is not None:
    cv2.imwrite(f"{v.rsplit('/view0',1)[0]}/sky_mask.png", (cover * 255).astype(np.uint8))  # render_results/sky_mask.png
print(f"scene {OUT}: {len(clips)} trajs, sky_mask cover={cover.mean()*100:.0f}% ground" if cover is not None else f"scene {OUT}: {len(clips)} trajs")
