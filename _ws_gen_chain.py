"""AUTOREGRESSIVE directed-shot gen: clip N continues clip N-1 (start_frame = prev last frame, reference = prev
clip), controlnet motion from each clip's cloud render, PER-CLIP prompt. Foundation of the interactive step mode.
No memory-bank/retrieval/seed-spread (that pins everything to one anchor -> static shot). Sequential traj0..N.

  <WS env> worldmirror_py _ws_gen_chain.py <scene_dir> [pano.png]
"""
import sys, os, json, shutil, time, glob as _glob
_ARGV = sys.argv[1:]; sys.argv = [sys.argv[0]]
SCENE = _ARGV[0]
PANO = _ARGV[1] if len(_ARGV) > 1 else "D:/_world_hangar/panorama.png"

import torch, numpy as np, cv2, gc
import _ws_serve as WS
sys.path.insert(0, r"D:/HY-World-2.0/hyworld2/worldgen")
sys.path.insert(0, r"D:/HY-World-2.0")
import utils3d
from src.data_utils import load_mutli_traj_dataset, get_last_video_frame
from src.panorama_utils import split_panorama_image
from diffusers.utils import export_to_video
from PIL import Image

dev, cfg, PIPE, MODEL = WS.device, WS.ws.cfg, WS.PIPE, WS.MODEL_TYPE
rr = f"{SCENE}/render_results"
clips = sorted(_glob.glob(f"{rr}/view0/traj*/camera.json"),
               key=lambda p: int(p.replace("\\", "/").split("/")[-2].replace("traj", "")))
trajs = [p.replace("\\", "/").split("/")[-2] for p in clips]
prompts = [json.load(open(f"{rr}/view0/{t}/traj_caption.json"))["prompt"] for t in trajs]
cam0 = json.load(open(f"{rr}/view0/{trajs[0]}/camera.json"))
FRAMES = len(cam0["extrinsic"])
print(f"[chain] {len(trajs)} clips x {FRAMES} frames", flush=True)

# ---- encode all unique per-clip prompts once, then drop umt5 ----
neg = cfg.get("negative_prompt", "")
PIPE.transformer.to("cpu"); WS._STATE["tr_on_gpu"] = False; torch.cuda.empty_cache()
PIPE.text_encoder.to(dev)
EMB = {}
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    for p in sorted(set(prompts)):
        EMB[p] = PIPE.encode_prompt(prompt=p, negative_prompt=neg, do_classifier_free_guidance=False,
                                    num_videos_per_prompt=1, max_sequence_length=512, device=dev)
PIPE.text_encoder.to("cpu"); PIPE.text_encoder = None
gc.collect(); torch.cuda.empty_cache()
PIPE.transformer.to(dev); WS._STATE["tr_on_gpu"] = True
print(f"[chain] {len(EMB)} unique prompts encoded; umt5 dropped", flush=True)

# ---- clip0 seed image = clean pano perspective at clip0's first pose ----
W, H = cam0["width"], cam0["height"]
ext0 = np.array(cam0["extrinsic"][0], np.float32)
K0 = np.array(cam0["intrinsic"][0], np.float32); Kn = K0.copy(); Kn[0] /= W; Kn[1] /= H
pano = np.array(Image.open(PANO).convert("RGB"))
seed_img = split_panorama_image(pano, ext0[None], [Kn], h=H, w=W, interp=cv2.INTER_AREA)[0]
Image.fromarray(seed_img).save(f"{rr}/view0/start_frame.png")


def gen(traj_id, prompt, ref_index):
    pe, ne = EMB[prompt]
    meta = load_mutli_traj_dataset(cfg=cfg, input_path=rr, output_path=rr, view_id="view0", traj_id=traj_id,
                                   device=dev, ref_index=ref_index, model_type=MODEL, task_type="panorama")
    kwargs = {k: v for k, v in meta.items() if v is not None}
    if getattr(PIPE, "image_encoder", None) is not None and kwargs.get("image") is not None:
        PIPE.image_encoder.to(dev)
        with torch.no_grad():
            kwargs["image_embeds"] = PIPE.encode_image(kwargs["image"], dev)
        PIPE.image_encoder.to("cpu"); torch.cuda.empty_cache()
    ri = ref_index.to(dev) if torch.is_tensor(ref_index) else ref_index
    kwargs.update(prompt=None, prompt_embeds=pe, negative_prompt_embeds=ne,
                  generator=torch.Generator(device=dev).manual_seed(1024), output_type="pt",
                  latent_cond_mode=cfg.latent_cond_mode, mode="test", num_frames=cfg.nframe, ref_index=ri)
    t = time.time()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = PIPE(**kwargs).frames[0].float()
    arr = out.permute(0, 2, 3, 1).clamp(0, 1).cpu().numpy().astype(np.float32)
    res = f"{rr}/view0/{traj_id}/{MODEL}_result.mp4"
    export_to_video(arr, res, fps=16)
    fd = f"{rr}/view0/{traj_id}/frames"; os.makedirs(fd, exist_ok=True)   # LOSSLESS PNG straight from the model
    for i, fr in enumerate(arr):
        cv2.imwrite(f"{fd}/f{i:02d}.png", cv2.cvtColor((fr * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    return res, time.time() - t

# ---- clip0: self-ref seed ----
m0 = f"{rr}/view0/{trajs[0]}/memory_inputs"; os.makedirs(m0, exist_ok=True)
shutil.copy(f"{rr}/view0/{trajs[0]}/render.mp4", f"{m0}/{MODEL}.mp4")
_, dt = gen(trajs[0], prompts[0], torch.tensor([0], dtype=torch.long))
print(f"[chain] 0/{len(trajs)-1} {trajs[0]} seed {dt:.0f}s", flush=True)

# ---- autoregressive chain: each clip starts where the previous ended ----
for i in range(1, len(trajs)):
    t, prev = trajs[i], trajs[i-1]
    shutil.copy(f"{rr}/view0/{prev}/frames/f{FRAMES-1:02d}.png", f"{rr}/view0/start_frame.png")   # LOSSLESS continuity
    mem = f"{rr}/view0/{t}/memory_inputs"; os.makedirs(mem, exist_ok=True)
    shutil.copy(f"{rr}/view0/{prev}/{MODEL}_result.mp4", f"{mem}/{MODEL}.mp4")
    _, dt = gen(t, prompts[i], torch.tensor([0], dtype=torch.long))   # ref=[0]; continuity via lossless start_frame
    print(f"[chain] {i}/{len(trajs)-1} {t} {dt:.0f}s", flush=True)

# ---- overlap-trimmed LOSSLESS sequence: clip0 all + each chained clip's frames[1:] (drop the redundant join) ----
outdir = "D:/_world_hangar/_ws_shot_frames"; shutil.rmtree(outdir, ignore_errors=True); os.makedirs(outdir)
seq = []
for ci, t in enumerate(trajs):
    fs = sorted(_glob.glob(f"{rr}/view0/{t}/frames/f*.png"))
    seq += fs[(0 if ci == 0 else 1):]
for gi, fp in enumerate(seq):
    shutil.copy(fp, f"{outdir}/frame_{gi:03d}.png")
fr0 = cv2.imread(seq[0]); vw = cv2.VideoWriter("D:/_world_hangar/_ws_shot_CHAIN_clean.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 10, (fr0.shape[1], fr0.shape[0]))
for fp in seq: vw.write(cv2.imread(fp))
vw.release()
print(f"[chain] ALL DONE — {len(seq)} trimmed lossless frames -> {outdir} + _ws_shot_CHAIN_clean.mp4", flush=True)
