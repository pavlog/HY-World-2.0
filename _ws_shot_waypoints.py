"""Author a directed shot from WAYPOINTS (programmatic) -> WS shot scene (6-frame clips).
Each waypoint = (eye[x,y,z], azimuth_deg, elev_deg). Azimuth: 0=+X (space), 180=-X (door); around +Z, Z-up.
Segments interpolate eye (linear) + az/elev (linear); per-SEGMENT prompt -> each clip gets its own traj_caption.json
(so you can re-gen one clip with a changed prompt). Auto-splits the frame stream into 6-frame clips.

  worldmirror_py _ws_shot_waypoints.py <shot_scene_dir> [lean_scene]
Edit WAYPOINTS below. Then: _ws_render_cond_all.py <shot>/render_results ; _ws_gen_lean.py <shot> "<prompt>"
"""
import sys, os, json, shutil
import numpy as np
sys.path.insert(0, "D:/HY-World-2.0/hyworld2/worldgen")
import utils3d

SCENE = sys.argv[1]
LEAN = sys.argv[2] if len(sys.argv) > 2 else "D:/_world_hangar/_ws_lean/scene"
W, H, CLIP = 832, 480, 6
FOV_X, FOV_Y = 120.0, 90.0

# (eye, azimuth_deg, elev_deg, frames_to_next, prompt_for_this_segment)
BASE = "sci-fi spaceship hangar interior, cinematic, high detail, sharp panels and skylights"
WAYPOINTS = [
    ([ 0.00, 0, 0], 180, 0, 12, BASE + ", approaching the lit doorway"),   # dolly to door (-X)
    ([-1.27, 0, 0], 180, 0,  6, BASE + ", turning around at the doorway"), # arrive door -> turn
    ([-1.27, 0, 0], 360, 0, 24, BASE + ", flying out toward dark space opening, stars"),  # turned 180 -> dolly to +X
    ([ 0.90, 0, 0], 360, 0,  0, BASE),                                      # end (space)
]

def dir_from(az, el):
    a, e = np.deg2rad(az), np.deg2rad(el)
    return np.array([np.cos(e)*np.cos(a), np.cos(e)*np.sin(a), np.sin(e)], np.float32)

# build per-frame poses + per-frame prompt
intr = utils3d.numpy.intrinsics_from_fov(fov_x=np.deg2rad(FOV_X), fov_y=np.deg2rad(FOV_Y)).astype(np.float32)
K = intr.copy(); K[0, :] *= W; K[1, :] *= H
poses, prompts = [], []
for i in range(len(WAYPOINTS) - 1):
    e0, az0, el0, nf, pr = WAYPOINTS[i]
    e1, az1, el1 = WAYPOINTS[i+1][0], WAYPOINTS[i+1][1], WAYPOINTS[i+1][2]
    for f in range(nf):
        t = f / nf
        eye = np.array(e0)*(1-t) + np.array(e1)*t
        az = az0*(1-t) + az1*t; el = el0*(1-t) + el1*t
        d = dir_from(az, el)
        w2c = utils3d.numpy.extrinsics_look_at(eye.astype(np.float32), (eye+d).astype(np.float32), np.array([0,0,1.0],np.float32)).astype(np.float32)
        poses.append(w2c.tolist()); prompts.append(pr)

rr = f"{SCENE}/render_results"; os.makedirs(f"{rr}/view0", exist_ok=True)
json.dump({"scene_type": "indoor"}, open(f"{SCENE}/meta_info.json", "w"))
for fn in ("global_pcd.ply", "full_depth_prediction.pt", "sky_mask.png"):
    s = f"{LEAN}/render_results/{fn}"
    if os.path.exists(s): shutil.copy(s, f"{rr}/{fn}")
sf = f"{LEAN}/render_results/view0/start_frame.png"
if os.path.exists(sf): shutil.copy(sf, f"{rr}/view0/start_frame.png")

n = 0
for ci in range(0, len(poses), CLIP):
    chunk = poses[ci:ci+CLIP]; pr = prompts[ci]
    if len(chunk) < CLIP: chunk = chunk + [chunk[-1]]*(CLIP-len(chunk))
    td = f"{rr}/view0/traj{n}"; os.makedirs(td, exist_ok=True)
    json.dump({"extrinsic": chunk, "intrinsic": [K.tolist()]*CLIP, "width": W, "height": H, "type": "shot"},
              open(f"{td}/camera.json", "w"), indent=2)
    json.dump({"prompt": pr}, open(f"{td}/traj_caption.json", "w"))
    n += 1
print(f"[shot] {len(poses)} frames -> {n} clips (6f each) -> {rr}")
