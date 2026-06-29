"""Train a 3D Gaussian Splat on OUR data — init from the fused colour cloud (geometry = our prior), supervise
with the GENERATED frames + cameras. Renders dense/smooth (HW-quality video) from any path. No HW gen_gs_data.

  worldmirror_py _ws_gs_train.py <fused.npz> <scene_dir> <out_dir> [iters] [max_pts]
"""
import sys, os, json, glob, math, time
import numpy as np, cv2, torch
from gsplat import rasterization

FUSED, SCENE, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
ITERS = int(sys.argv[4]) if len(sys.argv) > 4 else 2000
MAX_PTS = int(sys.argv[5]) if len(sys.argv) > 5 else 1_500_000
W, H = 832, 480
dev = "cuda"
os.makedirs(OUT, exist_ok=True)
MODEL = "worldstereo-memory-dmd"

# ---- init gaussians from the fused colour cloud (our geometry + generated colour) ----
z = np.load(FUSED); P = z["xyz"].astype(np.float32); C = z["color"].astype(np.float32) / 255.0
if len(P) > MAX_PTS:
    idx = np.random.choice(len(P), MAX_PTS, replace=False); P, C = P[idx], C[idx]
print(f"[gs] init {len(P)} gaussians from {FUSED}", flush=True)
means = torch.tensor(P, device=dev, requires_grad=True)
rgbs = torch.tensor(C, device=dev, requires_grad=True)
N = len(P)
scales = torch.full((N, 3), math.log(0.03), device=dev, requires_grad=True)     # ~3cm
quats = torch.zeros((N, 4), device=dev); quats[:, 0] = 1; quats.requires_grad_(True)
opac = torch.full((N,), 2.0, device=dev, requires_grad=True)                      # sigmoid(2)=0.88

# ---- load cameras + GENERATED frames as ground truth ----
cams = []  # (viewmat[4,4] w2c, K[3,3], gt[H,W,3] in [0,1] RGB)
for cj in sorted(glob.glob(f"{SCENE}/render_results/view0/traj*/camera.json"),
                 key=lambda p: int(p.replace("\\", "/").split("/")[-2].replace("traj", ""))):
    tdir = os.path.dirname(cj); cam = json.load(open(cj)); ext, intr = cam["extrinsic"], cam["intrinsic"]
    cap = cv2.VideoCapture(f"{tdir}/{MODEL}_result.mp4"); frames = []
    while True:
        r, f = cap.read()
        if not r: break
        frames.append(cv2.cvtColor(cv2.resize(f, (W, H)), cv2.COLOR_BGR2RGB))
    cap.release()
    for i in range(min(len(frames), len(ext))):
        cams.append((np.array(ext[i], np.float32), np.array(intr[i], np.float32),
                     frames[i].astype(np.float32) / 255.0))
print(f"[gs] {len(cams)} supervision views", flush=True)

opt = torch.optim.Adam([
    {"params": [means], "lr": 1.6e-4}, {"params": [rgbs], "lr": 2.5e-3},
    {"params": [scales], "lr": 5e-3}, {"params": [quats], "lr": 1e-3}, {"params": [opac], "lr": 5e-2}])

def render(viewmat, K):
    img, alpha, _ = rasterization(means, torch.nn.functional.normalize(quats, dim=-1), torch.exp(scales),
                                  torch.sigmoid(opac), rgbs.clamp(0, 1),
                                  torch.tensor(viewmat, device=dev)[None], torch.tensor(K, device=dev)[None],
                                  W, H, render_mode="RGB", packed=True)
    return img[0].clamp(0, 1)

# ---- train ----
t0 = time.time()
for it in range(ITERS):
    vm, K, gt = cams[np.random.randint(len(cams))]
    pred = render(vm, K)
    g = torch.tensor(gt, device=dev)
    loss = (pred - g).abs().mean()
    opt.zero_grad(); loss.backward(); opt.step()
    if it % 200 == 0 or it == ITERS - 1:
        print(f"[gs] iter {it:4d}/{ITERS}  L1={loss.item():.4f}  {time.time()-t0:.0f}s", flush=True)
print(f"[gs] trained {ITERS} iters in {time.time()-t0:.0f}s", flush=True)

# ---- save gaussians (npz) ----
torch.save({"means": means.detach().cpu(), "scales": scales.detach().cpu(), "quats": quats.detach().cpu(),
            "opac": opac.detach().cpu(), "rgbs": rgbs.detach().cpu()}, f"{OUT}/gs.pt")

# ---- render a flythrough along the scene cameras (DENSE, smooth) ----
out_mp4 = f"{OUT}/gs_flythrough.mp4"; vw = cv2.VideoWriter(out_mp4, cv2.VideoWriter_fourcc(*"mp4v"), 16, (W, H))
with torch.no_grad():
    for vm, K, _ in cams:
        img = (render(vm, K).cpu().numpy() * 255).astype(np.uint8)
        vw.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
vw.release()
print(f"[gs] DONE -> {out_mp4}  ({os.path.getsize(out_mp4)//1024}KB) + {OUT}/gs.pt", flush=True)
