# HunyuanWorld-Mirror 2.0 (WorldMirror) — bring-up & reference (2026-06-23)

## What it is
**Feed-forward multi-view → 3D reconstruction.** One forward pass turns a set of images
(posed OR unposed, 1..N) into a full 3D representation. NOT a diffusion/optimization loop —
it's a single VGGT-style transformer with task heads, so it's **fast** (seconds, not minutes).

Outputs in one shot (each head independently toggleable):
- **3DGS** (Gaussian splats) — the "fast 3DGS from multi-images" the user asked about.
- **depth** maps (per view)
- **points** (fused point cloud, world coords)
- **normals**
- **camera** poses + intrinsics (estimated if not given)
- optional **COLMAP** export, **confidence**, **sky mask**

Heads: `DPTHead` (dense: depth/points/normals/gaussians) + `CameraHead` (poses/intrinsics).
Model ≈ 1.2B. Subfolder on HF: `HY-WorldMirror-2.0` inside repo `tencent/HY-World-2.0`.

## Why we care (our angle)
We do NOT need its 3DGS renderer. We care about **mesh/DEPTH quality** for the
geometry-anchored pipeline (depth → Canny/normal → Wan-VACE restyle). Two real uses:
1. **Multi-view → good depth/points/mesh** when we have several real views (phone video of a
   real location, or Lyra MV frames). This is where it beats single-frame DA3 (cross-view geometry).
2. **★ Prior bridge**: `--prior_cam_path` + `--prior_depth_path` let us FEED our own exact
   rendered depth + known cameras as priors. So WorldMirror can fuse *with* our authored
   geometry instead of guessing — a direct hook into our canonical-mesh approach. Worth testing.

For a SINGLE image it degenerates to monocular (its DPT head) ≈ DA3 class — no win there.

## Inputs
- `--input_path DIR` — directory of images (the multi-view set). Also accepts a video? (feed
  extracted frames to be safe). 1..N views; more views w/ parallax = better fusion.
- Optional priors: `--prior_cam_path FILE` (extrinsics+intrinsics), `--prior_depth_path DIR`
  (per-view depth). Loaded via `load_prior_camera` / `load_prior_depth`.

## Parameter reference (CLI `python -m hyworld2.worldrecon.pipeline`)
**Core**
- `--input_path` (req) — image dir.
- `--output_path` (def `inference_output`) — results dir. `--strict_output_path` to force exact.
- `--pretrained_model_name_or_path` (def `tencent/HY-World-2.0`), `--subfolder` (def `HY-WorldMirror-2.0`).
- `--config_path` / `--ckpt_path` — override local config/checkpoint instead of HF download.
- `--target_size` (def **952**) — internal working resolution (long side). Bigger = sharper depth, more VRAM.
- `--max_resolution` (def 1920) — input cap.

**Multi-GPU / precision**
- `--use_fsdp` — shard across GPUs (torchrun). `--enable_bf16` — bf16 inference. `--fsdp_cpu_offload`.
  (Single 4090: omit fsdp; bf16 optional. We run single-GPU.)

**Output toggles** (all saved by default unless `--no_save_*`)
- `--no_save_depth / _normal / _gs / _camera / _points` — skip that head's output.
- `--save_colmap` — also write COLMAP sparse (cameras+points) for downstream SfM tools.
- `--save_conf` — dump per-pixel confidence. `--save_sky_mask` — write sky masks.
- `--disable_heads camera depth normal points gs` — actually skip running a head (faster). Use to
  run depth-only when that's all we want.

**Masking / cleanup** (these clean the point/GS cloud — same ideas as our DA3 sky/edge masks)
- `--apply_sky_mask` (def ON; `--no_sky_mask` off) — drop sky points.
- `--sky_mask_source {auto,model,onnx}` (def auto), `--model_sky_threshold` (def 0.45) — sky detector + cutoff.
- `--apply_edge_mask` (def ON; `--no_edge_mask`) — drop depth-discontinuity edge fringe.
  `--edge_normal_threshold` (def 1.0), `--edge_depth_threshold` (def 0.03) — edge sensitivity.
- `--apply_confidence_mask` (def OFF) + `--confidence_percentile` (def 10.0) — drop lowest-conf X%.

**Compression**
- `--compress_pts` (def ON) — voxel-downsample the point cloud. `--compress_pts_voxel_size` (def 0.002),
  `--compress_pts_max_points` (def 2,000,000).
- `--compress_gs_max_points` (def 5,000,000) — cap on Gaussian count.

**Video render (from the 3DGS)**
- `--save_rendered` — render an interpolated fly-through video from the splats.
- `--render_interp_per_pair` (def 15) — interpolated frames between each camera pair.
- `--render_depth` — also render a depth video.
- `--fps` (def 1), `--video_strategy {old,new}` (def new), `--video_min_frames`/`--video_max_frames` (1/32).

**Misc**: `--log_time` (def ON) timing breakdown; `--no_interactive`.

## Our bring-up state (conda env `worldmirror`, torch 2.7.1, py3.10) — NO CUDA builds
`IMPORT OK` achieved with **guards** (none of these are needed for our depth/mesh path):
1. `hyworldmirror/models/models/rasterization.py:8-9` — `from gsplat...` wrapped in try/except → None.
   gsplat is only the 3DGS rasterizer (render path). Mesh/depth extraction doesn't need it.
   *If we later want `--save_rendered`, build gsplat (reuse a CUDA build).*
2. `hyworldmirror/models/layers/attention.py:10-21` — flash_attn import → nested try/except `_HAS_FLASH`
   flag; line 64 gated on `_HAS_FLASH` → falls back to `F.scaled_dot_product_attention` (SDPA).
3. pip-added: `matplotlib==3.10.3 decord imagesize` (were trimmed from the curated install).

Weights: `from_pretrained` is **subfolder-scoped** (`_resolve_model_dir`, allow_patterns=`HY-WorldMirror-2.0/*`)
→ it will NOT pull the 80B HY-Pano. Safe. Downloading to `D:\HF_MODELS` (HF_HOME). [in progress]

## How to run (single GPU)
```bash
PYTHONPATH=D:/HY-World-2.0 HF_HOME=D:/HF_MODELS \
conda run -n worldmirror python -m hyworld2.worldrecon.pipeline \
  --input_path E:/MyGame/Game007Trailer/wm_input_lyra \
  --output_path D:/HY-World-2.0/_out_lyra \
  --target_size 952
```
- Multi-view test set ready: `E:/MyGame/Game007Trailer/wm_input_lyra` (17 harbor frames from Lyra render).
- Single-view baseline: `E:/MyGame/Game007Trailer/wm_input` (1 cockpit image) — expect ≈ DA3 monocular.
- To get a clean point cloud / depth only: add `--disable_heads gs normal camera` (depth+points only).
- To bridge our geometry: `--prior_depth_path <our rendered depth dir> --prior_cam_path <cams>`.

## TODO when we return
- Run multi-view harbor set → judge **mesh/point quality vs our DA3 dense mesh**.
- Judge **depth quality** specifically (that's what we need good — for Canny/normal control).
- Test the **prior_depth/prior_cam bridge** with our exact rendered depth.
- If we want the fly-through video or 3DGS, build gsplat (then un-guard rasterization).
- Retopology of WM output: same downstream options as cockpit (DeepMesh / decimate-in-Blender).

---
## RESULTS — first run on Lyra harbor multiview (2026-06-23)
Input: 17 frames from a Lyra zoomgs render (pirate harbor) → `E:/MyGame/Game007Trailer/wm_input_lyra`.
Run: `--target_size 952` (adaptive res dropped to **504**), single 4090.

**Bring-up fixes needed for a clean run (beyond the IMPORT-OK guards):**
- **`hf_xet` MUST be removed** from the env (`pip uninstall hf_xet`) + `HF_HUB_DISABLE_XET=1`, else
  weights AND the auto-pulled `Efficient-Large-Model/SANA-WM_bidirectional` (13GB t2i bundle,
  transitively imported, unused by reconstruction) stall at 0 bytes. See [[hf-xet-download-stall-fix]].
- `pip install onnxruntime` — needed by `compute_sky_mask` (downloads `skyseg.onnx`). Without it the
  whole run crashes AFTER inference, before saving.
- `--save_colmap` needs `pycolmap` (missing) → it crashes at the very last step but AFTER depth/normal/
  gaussians/camera are already saved. Either `pip install pycolmap` or just omit `--save_colmap`.

**Performance:** model load 7.4s; **inference 17×504² in ~217s** feed-forward. Fast.

**Outputs** (`_out_lyra/.../`): depth (npy+png ×17), normal (png ×17), `points.ply` (2.0M after
sky+filter+voxel prune from 4.32M), `gaussians.ply` (2.17M splats, 145MB), `camera_params.json`.

**Quality verdict:**
- **Normal maps = excellent** — crisp, detailed (barrel cylinder, dock planks, ship rigging all clean). Directly usable for normal-control.
- **Depth = good, multi-view consistent** but slightly soft/foggy; sky needs the mask (worked).
- **Fused point cloud**: coherent near-field (dock plane, foreground) but **curtain/floater artifacts**
  in ambiguous regions (water surface, distant ships, sky edge) — typical feed-forward MV-depth fusion.
- **No triangle mesh output** — WM gives points+gaussians+depth; a mesh needs a downstream Poisson/TSDF
  fuse of points.ply (would be decent in the structured region, messy on water/far).

**vs our DA3:** DA3 = single-image monocular, sharper per-frame, no cross-view 3D. WM = true multi-view
fuse (consistent across views + camera poses) — the right tool for the **phone-capture / multi-view**
case; DA3 still wins for a **single authored frame** (sharper + our exact-depth path). Confirms the
split in [strategy-capture-and-consistency.md].

---
## TSDF → mesh from WM depths (2026-06-23)
Script: `D:/HY-World-2.0/_tsdf_fuse.py` (Open3D ScalableTSDFVolume, voxel 0.0075, sdf_trunc 0.03,
depth_trunc 5.0). Reads `camera_params.json` (intrinsics 3x3 per-cam; extrinsics 4x4 = **c2w** →
invert to w2c for Open3D), pairs each `depth_000i.npy` with the resized input frame for color,
integrates, extract_triangle_mesh, then `cluster_connected_triangles` to drop floaters.
Render: `_render_tsdf.py` (Blender workbench, FLAT + VERTEX color). Out: `_tsdf_mesh_clean.ply`.

Result on the harbor set: RAW 277k v / 492k t → keep 2 largest clusters → 226k v / 430k t (5580
tiny floater clusters dropped). Mesh is **recognizable** (wooden pier planks + teal water, correct
colors) BUT a **curved 2.5D shell, not a solid 3D model**.

★ ROOT CAUSE = the INPUT, not WM/TSDF: the Lyra zoomgs frames are a **slow forward zoom** — camera
translates only ~0.4 units in +Z over 17 frames (extrinsic t: cam0≈0 → cam16 z=0.40), i.e. **near-zero
parallax**. So the multi-view fuse degenerates to a single-viewpoint depth surface; ships/buildings
become thin sheets (seen from one angle, no thickness). TSDF pipeline itself works fine.

**Takeaway:** a real WM 3D-mesh test needs genuine parallax (lateral / orbit motion), e.g. the
phone-capture idea or rendering an ORBIT of our DA3 cockpit mesh (controlled, known GT) → WM → TSDF.
The zoomgs set was a poor multi-view subject. Depth/normal per-view quality is still good (see above).
