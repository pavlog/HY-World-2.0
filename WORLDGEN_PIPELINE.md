# Worldgen Pipeline — Working Status & Reference
_Last updated: 2026-06-28. Consolidated from the build session. Goal: walkable/textured 3D environment meshes for Blender, from either a generated world (panorama) or real video._

---

## 0. Environments (conda, `%USERPROFILE%\miniconda3\envs\`)
| env | python | role |
|---|---|---|
| **worldmirror** | `envs/worldmirror/python.exe` | WorldStereo step-server + WorldMirror recon + MoGe + gen_gs_data. Has sageattention. |
| **trellis** | `envs/trellis/python.exe` | HW2 native trainer (gsplat fork 1.5.3 + fused_ssim), extract_mesh, texture bake. open3d 0.19, nvdiffrast, xatlas. |
| sugar | torch 2.0.1 cu118 | SuGaR (superseded) |

CUDA toolkit for builds: **v12.8** (`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8`). Build flag for VS18: `NVCC_PREPEND_FLAGS="-allow-unsupported-compiler"`.

---

## 1. TWO front-ends → ONE shared core

### Core (universal, builds the mesh) — HW2 native, in `trellis` env
`hyworld2/worldgen/world_gs_trainer.py` (depth + normal + **dist** + MaskGaussian + DefaultStrategy densification + anchor) → `gs/extract_mesh.py` (TSDF) → our texture bake.
- gsplat fork built in-place: `hyworld2/worldgen/third_party/gsplat_maskgaussian` (`pip install --no-build-isolation .`, needed glm copied in from stock gsplat). v1.5.3, has `distloss`.
- deps: fused_ssim (HW2 pins `git+rahul-goel/fused-ssim@328dc983`), tyro viser nerfview torchmetrics lpips imagesize splines tensorboard scikit-learn.
- Input = gs_data dir: `images/ depths/(16-bit) normals/ points.ply cameras.json` (extrinsic = **w2c**, intrinsic = K).

Trainer command (1-GPU, indoor; drop `--sky_depth_from_pcd`/`--convert_to_spz`):
```
trellis_python -m world_gs_trainer default --data_dir <gs_data> --result_dir <out> \
  --disable_viewer --disable_video --save_ply \
  --max_steps 8000 --eval_steps 8000 --save_steps 8000 --ply_steps 8000 \
  --use_scale_regularization --antialiased --depth_loss --normal_loss \
  --use_mask_gaussian --mask_export_stochastic --no-mask-export-anchor-protection \
  --use_anchor_protection --export_mesh \
  --strategy.refine-start-iter 800 --strategy.refine-stop-iter 4000 --strategy.refine-every 500 \
  --strategy.refine-scale2d-stop-iter 4000 --strategy.reset-every 99990 \
  --strategy.grow-grad2d 0.0001 --strategy.prune-scale3d 0.1
```
Output: `<out>/ply/fuse_post.ply` (full mesh) + `fuse_simplified.ply` (10%) + `point_cloud_*.ply` (gaussians) + `position_meta_info.json` (metric bbox/transform). Mesh is in NORMALIZED space → rescale to metric via the bbox in position_meta_info.
KNOWN BUG: trainer line ~1713 `dist.barrier()` is UnboundLocalError on 1-GPU → guarded with `if self.world_size>1`. (Fires AFTER mesh is saved, so harmless even unpatched.)

### Front-end A — Generated world (panorama)
`gen_gs_data.py` (Stage 4) builds gs_data from a generated world: video frames + pano + **polar_down (floor)** views. Needs `generation_bank_<rn>/` (global_pcd + aligned_pcd + per-frame depth PNGs) + `panorama.png` + `full_depth_prediction.pt` + `sky_mask.png`.
- Our lean runs skip the bank → `_ws_assemble_bank.py` rebuilds it (copies global_pcd→global+aligned, writes per-frame depths from moge_depth.npy, copies pano).
- Windows fixes in gen_gs_data: NCCL→gloo (`backend="gloo" if os.name=="nt"`), glob `\`→`/` path bug (else all trajs collapse to 6 frames).
- VALIDATED on hangar: **460k-tri watertight mesh, clean** (smooth panels + floor). Beats all hand-rolled approaches.

### Front-end B — Real video (WorldMirror)  ← general path, multi-room
`_ws_wm_recon.py <video|imgdir> <out> [fps=2] [max_frames=64] [target=952]` (worldmirror env)
- `from hyworld2.worldrecon.pipeline import WorldMirrorPipeline` (run from `D:/HY-World-2.0`; relative imports need package form).
- weights local: `D:/HF_MODELS/HY-World-2.0/HY-WorldMirror-2.0/` (subfolder), `enable_bf16=True`.
- **≤64 frames per run** (smart DIS-flow frame selection). Multi-room → split into ≤64-frame chunks (one volume each).
- Output: `camera_params.json` (extrinsics = **w2c** `{camera_id,matrix}`), `depth/depth_*.npy` (USE .npy, the .png is normalized viz only), `normal/*.png`, `points.ply`, `gaussians.ply`. **Images NOT saved** to outdir → they're in the temp dir `D:/tmp/frames_<name>/frame_*.jpg` (full-res); copy those in.
`_ws_wm_to_gsdata.py <wm_out> <gs_data>` (trellis env): builds gs_data. RESIZE IMAGE→DEPTH res (intrinsics are for WM's processed res, e.g. 952×532), 16-bit depth from npy, camera-space normals from depth, copy points.ply.
- STATUS: pipeline runs end-to-end (validated). First test video (`D:/Victoria/IMG_1416.MOV` — outdoor handheld orbit of a small object) gave a POOR mesh (18k tris, PSNR 12.8) — wrong input class (object+moving background+textureless ground). Pipeline is for ROOM-scale interior walks. Need a representative indoor-walk video to judge real quality. OPEN: sanity-check WM `points.ply` quality vs my converter (depth units) on a good input.

---

## 2. Texture (mesh → textured GLB)
`_ws_bake_texture_glb.py <mesh.ply> <gs_data> <out.glb> [tex=2048]` (trellis env):
- cameras via HW2 `gs.opencv.Parser(normalize=True)` (same space as mesh) → xatlas UV unwrap → nvdiffrast rasterizes UV→per-texel 3D → each of the N source frames projected with per-camera z-buffer (occlusion) + front-facing weight → unseen texels fall back to TSDF vertex color → 48-iter island edge-padding.
- Then rescale to metric (position_meta_info bbox) + **flip UV V** (`uv[:,1]=1-uv[:,1]`) for Blender's glTF importer (else packed-island scramble).
- KEY FINDING: source-frame projection IS pixel-exact (nvdiffrast-verified). The "chaos" was the Blender V-flip, NOT misregistration.
- Deliverables (hangar): `D:/_world_hangar/_ws_gs_native/hangar_textured_blender.glb` (460k, sharp) + `hangar_textured_decim_blender.glb` (46k) + vertex-color variants `hangar_vcol_full/decim.glb`.
- Project's own baker `backend/app/tools/mesh_bake.py` = vertex-color only (poisson/marching/tsdf → to_glb).
- Headless render: Blender 4.1, `--background --factory-startup --python`, EEVEE, factory scene, matte Principled + grey world. (open3d filament EGL offscreen FAILS on Windows → use legacy Visualizer for shaded ply compares.)

---

## 3. Interactive authoring tool — step server + client

### Launch (CRITICAL — `WS_FP8_FILE` is make-or-break)
```
cd D:\HY-World-2.0
set PYTHONUTF8=1                 (HW2 prints emoji; cp1252 → UnicodeEncodeError 500 without this)
set WS_SAGE=1 & set WS_FP8_UMT5=1 & set WS_OFFLOAD=1 & set WS_FP8_TRANSFORMER=1
set WS_FP8_FILE=D:/HF_MODELS/WorldStereo/worldstereo-memory-dmd/model_fp8.safetensors
worldmirror_python -u _ws_step_server.py
```
→ `http://127.0.0.1:5005`. Peak ~22.96GB, loads in ~1.5min. WITHOUT WS_FP8_FILE it loads bf16 (34GB peak) → silent crash at "Loading transformer dtype=bfloat16". **`startAll.bat` does all of this** (auto-kills old :5005 instance). Flask has no hot-reload → restart after server edits; client HTML is re-read per page load.

### `_ws_step_server.py` (worldmirror env) — endpoints
- `/create_project {name, pano}` → build_scaffold (MoGe pano depth → global_pcd). **Reads pano ALPHA** (RGBA, alpha<128 = erased) + `pano_mask.png` into excluded_region_mask (source-level cleanup).
- `/step {frames(w2c), intrinsic, prompt, w, h}` → 1 autoregressive 6-frame clip (cloud-conditioned). Appends to history + `camera_paths`. Snapshots for undo.
- `/accumulate {depth}` → incrementally MoGe-fuse NEW steps into the coverage cloud (undo-aware) + return fast Poisson preview.
- `/fastmesh {depth=8}` → fast Poisson (camera-oriented normals) vertex-color preview.
- `/mesh`, `/world_mesh` → Poisson depth-10 meshes (full quality).
- `/erase_points {cam,f,r,d,fx,cx,cy,cw,ch, boxes[], brushes[], slab}` → projects FULL cloud, deletes front surface (z-buffer + slab), scaffold deletions → `pano_mask.png` (dir→equirect uv), re-saves global_pcd. Snapshots.
- `/undo` (unified timeline: pops last op = step/accumulate/erase, full-state restore, cap 12). `/undo_erase` aliases it.
- `/reset` → clears steps + coverage + undo stack (back to cleaned scaffold). Client gates with a download-reminder modal.
- `/download_cameras` → Blender-ready JSON (`matrix_world = inv(w2c) @ diag(1,-1,-1,1)` per frame + w2c + fov).
- `/download_pano` → tweaked pano RGBA PNG (erased = transparent; round-trips back into the tool).
- `/browse` → native Windows OpenFileDialog (PowerShell -STA) → returns path. `/cloud`, `/frames`, `/log`, `/projects`, `/open_project`.

### `_ws_client.html` (served at :5005) — Canvas2D viewer
- Nav: **drag-to-look** (LMB, cursor stays — no pointer-lock), WASD move, Q/E down/up.
- Controls: point size, point count, **bg color** (dark/grey/white for dark points), show mesh / shaded.
- **Erase points** panel: Nav/Box/Brush modes, brush size, **cut** (slab) depth, **▣ Push delete** (commit), Clear, Undo del. Red overlay preview. Soft warning when editing with generated clips present.
- Buttons: **⬇ Camera path** (Blender JSON), **⬇ Pano (PNG)**, Undo, Reset (modal), Generate Pano/World/Fast mesh.

PROMPT CACHE = built into `_ws_serve.py` (`_ws_prompt_cache/`, LRU 500, md5 key) — NOT a separate server. NOTE: step server's `encode_prompt_cached` uses only in-memory PE_CACHE (per-session), NOT the disk cache → first use of each prompt after restart = ~40s umt5 swap. TODO: wire disk cache into the step server.

---

## 4. Sky handling (HW2 reference, for outdoor)
Sky = "no surface" signature: **normal≈0** (`normals.norm < sky_normal_threshold`) + no valid depth + model sky-mask. Sky is excluded from solid scaffold, reconstructed as a separate FAR `sky_pcd` shell, marked ANCHOR (protected from prune/densify), given a far depth target via `sky_depth_from_pcd` (triple-condition merge). For our alpha cleanup: alpha = "delete" (→ hole, fine indoor); sky needs a DIFFERENT label (far shell), so don't paint sky with alpha — auto-detect by normal or use a separate sky label.

---

## 5. Open items / next steps
- [ ] Real-video quality: run on a representative **indoor room walk** video (not the outdoor object orbit). Sanity-check WM points.ply vs converter depth-units.
- [ ] Multi-room: split long walks into ≤64-frame WorldMirror chunks = volumes; per-chunk mesh+atlas at fixed texels/metre; stitch on overlap. (Atlas can't just grow — chunk it.)
- [ ] Texel/metre invariant for large worlds (volume chunking, UDIM as interim).
- [ ] Wire disk prompt-cache into the step server (instant prompts across restarts).
- [ ] Preview vs final trainer profiles (drop LPIPS + 1500 steps + vertex-color = ~3-5min preview).
- [ ] Sharp texture from gaussian renders (mesh-consistent) if raw-frame projection ever misregisters.

## 6. Key paths
- Scripts: `D:/HY-World-2.0/_ws_*.py`, `_ws_client.html`, `startAll.bat`, `WORLDGEN_PIPELINE.md` (this file).
- HW2 repo: `D:/HY-World-2.0/hyworld2/{worldgen,worldrecon}`.
- Weights: `D:/HF_MODELS/WorldStereo/...model_fp8.safetensors`, `D:/HF_MODELS/HY-World-2.0/HY-WorldMirror-2.0/`.
- Hangar deliverables: `D:/_world_hangar/_ws_gs_native/*.glb`. Hangar gs_data: `D:/_world_hangar/_ws_lean/scene/gs_data`.
- Real-video test: `D:/_world_real/{wm_IMG1416, gs_IMG1416, gs_native_IMG1416}`.
- Projects (step server): `D:/_world_hangar/_ws_projects/`.
