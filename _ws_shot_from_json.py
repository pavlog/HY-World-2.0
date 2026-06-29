"""Turn an authored shot_cameras.json (from the web app OR Blender export) into a WS shot scene.
Splits the per-frame w2c poses into 6-frame clips, copies the pano scaffold, writes camera.json + caption + meta.

  worldmirror_py _ws_shot_from_json.py <shot_cameras.json> <shot_scene_dir> [lean_scene]
Then: _ws_render_cond_all.py <shot>/render_results ; _ws_gen_lean.py <shot> "<prompt>"
"""
import sys, os, json, shutil

SHOT_JSON = sys.argv[1]
SCENE = sys.argv[2]
LEAN = sys.argv[3] if len(sys.argv) > 3 else "D:/_world_hangar/_ws_lean/scene"
CLIP = 6
PROMPT = "sci-fi spaceship hangar interior, cinematic, high detail, sharp panels and skylights"

d = json.load(open(SHOT_JSON))
frames = d["frames"]; K = d["intrinsic"]; W, H = d["width"], d["height"]
rr = f"{SCENE}/render_results"; os.makedirs(f"{rr}/view0", exist_ok=True)
json.dump({"scene_type": "indoor"}, open(f"{SCENE}/meta_info.json", "w"))
for fn in ("global_pcd.ply", "full_depth_prediction.pt", "sky_mask.png"):
    s = f"{LEAN}/render_results/{fn}"
    if os.path.exists(s): shutil.copy(s, f"{rr}/{fn}")
sf = f"{LEAN}/render_results/view0/start_frame.png"
if os.path.exists(sf): shutil.copy(sf, f"{rr}/view0/start_frame.png")

n = 0
for ci in range(0, len(frames), CLIP):
    chunk = frames[ci:ci + CLIP]
    if len(chunk) < CLIP: chunk = chunk + [chunk[-1]] * (CLIP - len(chunk))
    td = f"{rr}/view0/traj{n}"; os.makedirs(td, exist_ok=True)
    json.dump({"extrinsic": chunk, "intrinsic": [K] * CLIP, "width": W, "height": H, "type": "shot"},
              open(f"{td}/camera.json", "w"), indent=2)
    json.dump({"prompt": PROMPT}, open(f"{td}/traj_caption.json", "w"))
    n += 1
print(f"[shot] {len(frames)} frames -> {n} clips -> {rr}")
