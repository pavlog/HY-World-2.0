"""Real-video front-end: WorldMirror reconstruction (poses + per-frame depth + points) from a video/imagedir.
Runs HW2's own WorldMirrorPipeline. Single-GPU, local weights, bf16. Output -> <outdir> (camera_params.json,
depth/*.npy, normal/*.png, points.ply, images, gs.ply). Feed to _ws_wm_to_gsdata.py -> world_gs_trainer.

  worldmirror_py _ws_wm_recon.py <video_or_imgdir> <outdir> [fps=2] [max_frames=64] [target=952]

SPEED LEVERS (preview vs final):
  fps         frames/sec sampled from video (lower = fewer views = faster)
  max_frames  HARD CAP (WM itself clamps to 64); preview ~24, final 48-64
  target      processing resolution (preview 518, final 952)
WM reconstructs <=64 frames per run; for multi-room walkthroughs, split into <=64-frame segments (one per volume).
"""
import sys, os
sys.path.insert(0, "D:/HY-World-2.0")          # so hyworld2.worldrecon is importable as a package (relative imports)
from hyworld2.worldrecon.pipeline import WorldMirrorPipeline

VIDEO, OUT = sys.argv[1], sys.argv[2]
FPS  = int(sys.argv[3]) if len(sys.argv) > 3 else 2
MAXF = int(sys.argv[4]) if len(sys.argv) > 4 else 64
TGT  = int(sys.argv[5]) if len(sys.argv) > 5 else 952
os.makedirs(OUT, exist_ok=True)

pipe = WorldMirrorPipeline.from_pretrained(
    "D:/HF_MODELS/HY-World-2.0", subfolder="HY-WorldMirror-2.0", enable_bf16=True)

outdir = pipe(
    VIDEO, output_path=OUT, strict_output_path=OUT,
    fps=FPS, video_max_frames=MAXF, target_size=TGT,
    save_depth=True, save_normal=True, save_points=True, save_camera=True, save_gs=True,
    save_colmap=False, apply_sky_mask=True, apply_edge_mask=True, log_time=True)
print(f"\n[wm] reconstruction -> {outdir}", flush=True)
