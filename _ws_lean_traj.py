"""LEAN pano->world exploration trajectory generator — HW2's GEOMETRIC core, NO VLM/SAM3/navmesh.

Builds the scaffold from a panorama (MoGe-2 full_depth + global_pcd), polar-splits into perspective
views, and plans obstacle-aware EXPLORATION trajectories (rotation L/R + up-aerial + forward-wonder) via
HW2's own get_c2w (cKDTree obstacle avoidance over global_pcd). Output scene dir is consumable by our
_ws gen: render_results/{global_pcd.ply, full_depth_prediction.pt, sky_mask.png, view{i}/traj{j}/camera.json}.

This is the part of traj_generate.py that fixes the user's "fly-through bakes pano artifacts" problem —
it explores the NAVIGABLE space instead of diving into the single-pano mesh. VLM only picked smart targets.

  worldmirror_py _ws_lean_traj.py <panorama.png> <scene_out_dir> [nframe=21] [move_dist=8.0] [sky_mask.png]
"""
import os, sys, json
import numpy as np, torch, trimesh, cv2
from PIL import Image
from scipy.spatial import cKDTree
import torch.nn.functional as F
import utils3d

WG = "D:/HY-World-2.0/hyworld2/worldgen"
sys.path.insert(0, WG)

# pytorch3d shim — camera_utils imports look_at_rotation (only used by the unused 'eloop' mode)
import types as _types
def _look_at_rotation(camera_position, at=((0, 0, 0),), up=((0, 1, 0),), device="cpu"):
    cp = torch.as_tensor(camera_position, dtype=torch.float32).reshape(-1, 3)
    at_ = torch.as_tensor(at, dtype=torch.float32).reshape(-1, 3)
    up_ = torch.as_tensor(up, dtype=torch.float32).reshape(-1, 3)
    z = torch.nn.functional.normalize(at_ - cp, dim=-1)
    x = torch.nn.functional.normalize(torch.cross(up_.expand_as(z), z, dim=-1), dim=-1)
    y = torch.cross(z, x, dim=-1)
    return torch.stack([x, y, z], dim=2)
for _n in ("pytorch3d", "pytorch3d.renderer", "pytorch3d.renderer.cameras"):
    sys.modules.setdefault(_n, _types.ModuleType(_n))
sys.modules["pytorch3d.renderer.cameras"].look_at_rotation = _look_at_rotation

from moge.model.v2 import MoGeModel
from src.camera_utils import get_c2w
from src.general_utils import adjust_image_size
from src.panorama_utils import (split_panorama_image, split_panorama_depth, rotate_around_z_axis,
                                 pred_pano_depth, convert_rgbd2pcd_panorama)

PANO  = sys.argv[1]
SCENE = sys.argv[2]
NFRAME    = int(sys.argv[3]) if len(sys.argv) > 3 else 21
# forward-wonder magnitude as a FRACTION of median_depth (air_bound = md*0.5, so fwd = md*0.5*MOVE_FWD)
MOVE_FWD  = float(sys.argv[4]) if len(sys.argv) > 4 else 0.8   # -> ~0.4 * median_depth forward
SKY_IN    = sys.argv[5] if len(sys.argv) > 5 else None

# HW2 defaults (traj_generate.py)
FOV_X, FOV_Y = 120.0, 90.0
SPLIT_RES    = 480
ROTATION_DEG = 120.0
ROTATION_UP  = 45.0
UP_RIGHT     = 60.0
DIST_THRESH  = 0.1
OBS_DECAY    = 2.0 / 3.0
OBS_LIMIT    = 3
MOGE_ID = "Ruicheng/moge-2-vitl-normal"
dev = "cuda"

rr = f"{SCENE}/render_results"
os.makedirs(rr, exist_ok=True)
json.dump({"scene_type": "indoor"}, open(f"{SCENE}/meta_info.json", "w"))   # bank reads scene_type only

full_img = Image.open(PANO).convert("RGB")
if full_img.size[1] > 1920:
    full_img = full_img.resize((3840, 1920), resample=Image.Resampling.BICUBIC)
W0, H0 = full_img.size
print(f"[lean] pano {W0}x{H0}", flush=True)

# ---- 1. MoGe-2 panorama depth ----
depth_model = MoGeModel.from_pretrained(MOGE_ID).to(dev).eval()
if os.path.exists(f"{rr}/full_depth_prediction.pt"):
    full_depth = torch.load(f"{rr}/full_depth_prediction.pt", weights_only=False)
    print("[lean] loaded cached full_depth", flush=True)
else:
    full_depth = pred_pano_depth(depth_model, full_img)
    print("[lean] MoGe pano depth done", flush=True)
full_depth["distance"] = full_depth["distance"].to(dev)
full_depth["rays"]     = full_depth["rays"].to(dev)

# ---- sky/edge mask (sky stub = none unless provided) ----
edge_mask = torch.from_numpy(utils3d.numpy.depth_edge(full_depth["distance"].cpu().numpy(), rtol=0.1)).bool()
if SKY_IN and os.path.exists(SKY_IN):
    sm = Image.open(SKY_IN).convert("L").resize((W0, H0), Image.Resampling.NEAREST)
    sky_mask = torch.from_numpy(np.array(sm) > 127).bool()       # white = sky
    print(f"[lean] sky mask from {SKY_IN} ({sky_mask.float().mean():.2f})", flush=True)
else:
    sky_mask = torch.zeros((H0, W0)).bool()                       # no sky
    print("[lean] sky mask STUB (all non-sky)", flush=True)
sky_for_depth = sky_mask
if sky_for_depth.shape != edge_mask.shape:
    sky_for_depth = F.interpolate(sky_for_depth[None, None].float(), size=edge_mask.shape, mode="nearest")[0, 0].bool()
full_mask = (sky_for_depth | edge_mask).to(dev)

# clip far/sky points (HW2 line 314)
max_d = torch.quantile(full_depth["distance"][~full_mask], q=0.99).item()
full_depth["distance"] = torch.clip(full_depth["distance"], 0, max_d)
if not os.path.exists(f"{rr}/full_depth_prediction.pt"):
    torch.save({"distance": full_depth["distance"].cpu(), "rays": full_depth["rays"].cpu()}, f"{rr}/full_depth_prediction.pt")
Image.fromarray(((~sky_for_depth).cpu().numpy() * 255).astype(np.uint8)).save(f"{rr}/sky_mask.png")

# ---- 2. global point cloud ----
if os.path.exists(f"{rr}/global_pcd.ply"):
    global_pcd = trimesh.load(f"{rr}/global_pcd.ply")
else:
    dh, dw = full_depth["distance"].shape
    pcd_img = full_img.resize((dw, dh), resample=Image.Resampling.BICUBIC) if full_img.size != (dw, dh) else full_img
    global_pcd = convert_rgbd2pcd_panorama(
        rgb=torch.tensor(np.array(pcd_img) / 255, dtype=torch.float32),
        distance=full_depth["distance"], rays=full_depth["rays"],
        excluded_region_mask=full_mask, dropout_pcd=False)
    global_pcd.export(f"{rr}/global_pcd.ply")
verts = np.asarray(global_pcd.vertices)
print(f"[lean] global_pcd {len(verts)} pts  max_d={max_d:.2f}", flush=True)
kdtree = cKDTree(verts)

# ---- 3. polar split into perspective views (HW2 lines 403-427) ----
image_h = SPLIT_RES
image_w = int(round(np.tan(np.deg2rad(FOV_X / 2)) / np.tan(np.deg2rad(FOV_Y / 2)) * image_h))
image_h, image_w = adjust_image_size(image_h, image_w)
polar_points = [np.array([-1, 0, 1.0], np.float32), np.array([-1, 0, -1.0], np.float32),
                np.array([0.1, 0, -1.0], np.float32), np.array([0.1, 0, 1.0], np.float32)]
direct_points = polar_points.copy()
rot_deg = 90; N_view = int(360 / rot_deg)
for pp in polar_points:
    for i in range(1, N_view):
        direct_points.append(rotate_around_z_axis(pp.reshape(1, 3), rot_deg * i)[0])
direct_points = np.stack(direct_points, axis=0)
intr = utils3d.numpy.intrinsics_from_fov(fov_x=np.deg2rad(FOV_X), fov_y=np.deg2rad(FOV_Y))
splitted_intrinsics = [intr] * len(direct_points)
splitted_extrinsics = utils3d.numpy.extrinsics_look_at(np.array([0, 0, 0]), direct_points, np.array([0, 0, 1])).astype(np.float32)
splitted_images = split_panorama_image(np.array(full_img), splitted_extrinsics, splitted_intrinsics, h=image_h, w=image_w, interp=cv2.INTER_AREA)
splitted_depths = split_panorama_depth(np.array(full_depth["distance"].cpu()), splitted_extrinsics, splitted_intrinsics, h=image_h, w=image_w, distance_to_depth=True)
splitted_masks  = split_panorama_depth(~np.array(full_mask.cpu()), splitted_extrinsics, splitted_intrinsics, h=image_h, w=image_w)
print(f"[lean] {len(splitted_images)} views @ {image_h}x{image_w}", flush=True)

# ---- 4. exploration candidates: look L / look R / up-aerial / move-forward (wonder) ----
candidates = [
    {"type": "normal", "backward-forward": 0,         "left-right": 0, "rotation": [-ROTATION_DEG, 0],          "name": "right-rotation"},
    {"type": "normal", "backward-forward": 0,         "left-right": 0, "rotation": [ ROTATION_DEG, 0],          "name": "left-rotation"},
    {"type": "aerial", "backward-forward": 0,         "left-right": 0, "rotation": [-UP_RIGHT, -ROTATION_UP],   "name": "up-right-aerial"},
    {"type": "normal", "backward-forward": MOVE_FWD,  "left-right": 0, "rotation": [0, 0],                      "name": "forward-wonder"},
]

# ---- 5. plan trajectories per view ----
ntraj = 0
for vi in range(len(splitted_images)):
    dmask = splitted_masks[vi].bool()
    if dmask.sum() == 0:
        continue
    depth = splitted_depths[vi]
    median_depth = torch.median(depth[dmask]).item()
    if not np.isfinite(median_depth) or median_depth <= 0:
        continue
    c2w_start = np.linalg.inv(np.array(splitted_extrinsics[vi]))
    K = splitted_intrinsics[vi].copy(); K[0, :] *= image_w; K[1, :] *= image_h
    os.makedirs(f"{rr}/view{vi}", exist_ok=True)                  # clean split view = appearance seed
    Image.fromarray(splitted_images[vi]).save(f"{rr}/view{vi}/start_frame.png")
    for ti, move0 in enumerate(candidates):
        move = {k: (list(v) if isinstance(v, list) else v) for k, v in move0.items()}  # fresh copy (get_c2w mutates)
        olim = 6 if move0["name"] == "forward-wonder" else OBS_LIMIT   # let forward halve down to a safe short step
        c2ws, obs = get_c2w(c2w_start.copy(), move, median_depth, air_bound=median_depth * 0.5,
                            n_inter=NFRAME - 1, kdtree=kdtree, mesh=None, distance_threshold=DIST_THRESH,
                            obs_decay=OBS_DECAY, obs_limit=olim)
        if obs > olim:
            print(f"  view{vi}/traj{ti} ({move0['name']}) too many collisions, skip", flush=True)
            continue
        c2ws = np.concatenate([c2w_start[None], c2ws], axis=0)
        w2cs = np.linalg.inv(c2ws)
        td = f"{rr}/view{vi}/traj{ti}"; os.makedirs(td, exist_ok=True)
        json.dump({"extrinsic": w2cs.tolist(), "intrinsic": [K.tolist()] * len(w2cs),
                   "width": image_w, "height": image_h, "type": move0["name"]},
                  open(f"{td}/camera.json", "w"), indent=2)
        json.dump({"prompt": os.environ.get("LEAN_PROMPT", "sci-fi spaceship hangar interior, cinematic, high detail")},
                  open(f"{td}/traj_caption.json", "w"))           # per-traj prompt (no VLM captioner)
        ntraj += 1
print(f"[lean] DONE {ntraj} trajectories over {len(splitted_images)} views -> {rr}", flush=True)
