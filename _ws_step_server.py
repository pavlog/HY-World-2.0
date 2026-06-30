"""Interactive STEP server, MULTI-PROJECT. Resident WorldStereo (loaded once) behind a tiny HTTP API.
Each PROJECT = a panorama -> its own scaffold (MoGe pano depth + global_pcd) -> its own autoregressive step chain
(persisted). Switch projects freely. Each /step generates one 6-frame clip continuing the current project's chain,
given client camera poses + prompt (prompt changeable per step). SERVER half of the client-server authoring tool.

  <WS env> worldmirror_py _ws_step_server.py        # loads model (~35s), serves 127.0.0.1:5005
Endpoints (all JSON):
  GET  /projects                              -> {projects:[{name,steps}]}
  POST /create_project {name, pano}           -> build scaffold from pano; -> {cloud,bounds,steps,history}
  POST /open_project   {name}                 -> load scaffold+chain;       -> {cloud,bounds,steps,history}
  POST /step {frames:[w2c..],intrinsic,prompt,width,height} -> {frames:[b64png], step, new_pose, secs}
  POST /reset / POST /undo                    -> clear / pop last step
"""
import sys, os, json, time, base64, shutil, gc, glob, threading, functools, hashlib
import numpy as np, cv2, torch, trimesh
sys.argv = [sys.argv[0]]
import _ws_serve as WS
sys.path.insert(0, r"D:/HY-World-2.0/hyworld2/worldgen")
import utils3d
import torch.nn.functional as F
from src.pointcloud import point_rendering
from src.panorama_utils import split_panorama_image, pred_pano_depth, convert_rgbd2pcd_panorama
from src.data_utils import load_mutli_traj_dataset
from diffusers.utils import export_to_video
from PIL import Image
from flask import Flask, request, jsonify

dev, MODEL, PIPE, cfg = WS.device, WS.MODEL_TYPE, WS.PIPE, WS.ws.cfg
NEG = cfg.get("negative_prompt", "")
PROJ_ROOT = r"D:/_world_hangar/_ws_projects"; os.makedirs(PROJ_ROOT, exist_ok=True)
MOGE_ID = "Ruicheng/moge-2-vitl-normal"
PE_CACHE = {}
from collections import deque
LOG = deque(maxlen=300)
def logln(m):
    LOG.append(m); print("[srv] " + m, flush=True)
S = {"proj": None, "pano": None, "gpts": None, "gcol": None, "last_png": None, "last_result": None, "step": 0, "history": [],
     # incremental, undo-aware coverage accumulation (scaffold points stay at indices [0:scaffold_n]):
     "scaffold_n": 0, "step_pts": [], "accum_step": 0, "scale": None}

# transformer RESIDENT on GPU, CLIP parked (the proven 24GB recipe — NOT run_one which spills CLIP+umt5)
PIPE.image_encoder.to("cpu")
PIPE.transformer.to(dev); WS._STATE["tr_on_gpu"] = True; torch.cuda.empty_cache()

# ---- idle GPU auto-eviction: after IDLE_SECONDS with no GPU request, park transformer+VAE on CPU + empty_cache
#      (frees VRAM for other tasks). Next GPU request lazily restores. Uses the same .to() moves /step already does.
IDLE_SECONDS = 60
GPU = {"loaded": True, "busy": 0, "last": time.time()}
GPU_LOCK = threading.Lock()

def _gpu_ensure():
    with GPU_LOCK:
        GPU["busy"] += 1; GPU["last"] = time.time()
        if not GPU["loaded"]:
            PIPE.vae.to(dev)                       # transformer is restored by the gen path (tr_on_gpu flag)
            GPU["loaded"] = True; logln("GPU: resumed (VAE back on GPU)")

def _gpu_done():
    with GPU_LOCK:
        GPU["busy"] = max(0, GPU["busy"] - 1); GPU["last"] = time.time()

def gpu_endpoint(fn):
    @functools.wraps(fn)
    def w(*a, **k):
        if request.method == "OPTIONS":
            return fn(*a, **k)
        _gpu_ensure()
        try:
            return fn(*a, **k)
        finally:
            _gpu_done()
    return w

def _gpu_watchdog():
    while True:
        time.sleep(5)
        with GPU_LOCK:
            if GPU["loaded"] and GPU["busy"] == 0 and (time.time() - GPU["last"]) > IDLE_SECONDS:
                try:
                    PIPE.transformer.to("cpu"); WS._STATE["tr_on_gpu"] = False
                    PIPE.vae.to("cpu"); torch.cuda.empty_cache()
                    GPU["loaded"] = False
                    logln(f"GPU idle {IDLE_SECONDS}s -> parked transformer+VAE on CPU (VRAM freed)")
                except Exception as e:
                    logln(f"GPU evict error: {e}")
threading.Thread(target=_gpu_watchdog, daemon=True).start()


def encode_prompt_cached(prompt):
    cpath = f"{WS.PROMPT_CACHE}/{hashlib.md5((prompt + '||' + NEG).encode()).hexdigest()}.pt"
    if prompt in PE_CACHE:                                       # in-memory (CPU) hit
        pe, ne = PE_CACHE[prompt]
        logln("prompt cached (mem) → skip encode")
        return pe.to(dev), (ne.to(dev) if ne is not None else None)
    if os.path.exists(cpath):                                   # DISK hit -> no umt5 swap (survives restart; shared with run_one)
        try:
            os.utime(cpath, None)                                # LRU touch
            d = torch.load(cpath, map_location="cpu")
            pe, ne = d["pe"], d["ne"]
            PE_CACHE[prompt] = (pe, ne)
            logln("prompt cached (disk) → skip umt5 swap")
            return pe.to(dev), (ne.to(dev) if ne is not None else None)
        except Exception as e:
            logln(f"disk prompt-cache load failed ({e}) → re-encoding")
    logln("new prompt → encoding (umt5 swap, ~40s)…")
    PIPE.transformer.to("cpu"); WS._STATE["tr_on_gpu"] = False; torch.cuda.empty_cache()   # umt5(11)+transformer(17)=30.8 -> swap
    PIPE.text_encoder.to(dev)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        pe, ne = PIPE.encode_prompt(prompt=prompt, negative_prompt=NEG, do_classifier_free_guidance=False,
                                    num_videos_per_prompt=1, max_sequence_length=512, device=dev)
    PIPE.text_encoder.to("cpu"); torch.cuda.empty_cache()
    PIPE.transformer.to(dev); WS._STATE["tr_on_gpu"] = True
    try:
        torch.save({"pe": pe.cpu(), "ne": (ne.cpu() if ne is not None else None)}, cpath); WS._cache_evict()
        logln("prompt encoded → saved to disk cache")
    except Exception as e:
        logln(f"disk prompt-cache save failed: {e}")
    PE_CACHE[prompt] = (pe.cpu(), ne.cpu() if ne is not None else None)
    return pe, ne


def gen_clip(rr, prompt, ref_index):
    pe, ne = encode_prompt_cached(prompt)
    meta = load_mutli_traj_dataset(cfg=cfg, input_path=rr, output_path=rr, view_id="view0", traj_id="traj0",
                                   device=dev, ref_index=ref_index, model_type=MODEL, task_type="panorama")
    kwargs = {k: v for k, v in meta.items() if v is not None}
    if getattr(PIPE, "image_encoder", None) is not None and kwargs.get("image") is not None:
        PIPE.image_encoder.to(dev)
        with torch.no_grad():
            kwargs["image_embeds"] = PIPE.encode_image(kwargs["image"], dev)
        PIPE.image_encoder.to("cpu"); torch.cuda.empty_cache()
    ri = ref_index.to(dev)
    kwargs.update(prompt=None, prompt_embeds=pe, negative_prompt_embeds=ne,
                  generator=torch.Generator(device=dev).manual_seed(1024), output_type="pt",
                  latent_cond_mode=cfg.latent_cond_mode, mode="test", num_frames=cfg.nframe, ref_index=ri)
    logln("denoising (4-step DMD)…")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = PIPE(**kwargs).frames[0].float()
    arr = out.permute(0, 2, 3, 1).clamp(0, 1).cpu().numpy().astype(np.float32)
    res = f"{rr}/view0/traj0/{MODEL}_result.mp4"
    export_to_video(arr, res, fps=16)
    return res


def build_scaffold(pano_path, rr):
    _rawA = Image.open(pano_path).convert("RGBA")
    _a = np.array(_rawA)[..., 3]
    pano_alpha = _a if (_a < 250).any() else None      # RGBA pano: alpha<128 = erased region to drop from scaffold
    full_img = _rawA.convert("RGB")
    if full_img.size[1] > 1920:
        full_img = full_img.resize((3840, 1920), Image.Resampling.BICUBIC)
    W0, H0 = full_img.size
    logln("MoGe-2: loading + predicting panorama depth…")
    from moge.model.v2 import MoGeModel
    moge = MoGeModel.from_pretrained(MOGE_ID).to(dev).eval()
    fd = pred_pano_depth(moge, full_img)
    logln("unprojecting to global point cloud…")
    fd["distance"] = fd["distance"].to(dev); fd["rays"] = fd["rays"].to(dev)
    edge = torch.from_numpy(utils3d.numpy.depth_edge(fd["distance"].cpu().numpy(), rtol=0.1)).bool()
    sky = torch.zeros((H0, W0)).bool()
    sfd = sky if sky.shape == edge.shape else F.interpolate(sky[None, None].float(), size=edge.shape, mode="nearest")[0, 0].bool()
    full_mask = (sfd | edge).to(dev)
    if pano_alpha is not None:                          # paint-with-alpha cleanup: drop erased pano regions at the source
        am = np.array(Image.fromarray(pano_alpha).resize((full_mask.shape[1], full_mask.shape[0]), Image.NEAREST)) < 128
        full_mask = full_mask | torch.as_tensor(am, device=dev)
        logln(f"pano alpha-mask: excluded {int(am.sum())} px from scaffold")
    _pm = os.path.join(os.path.dirname(pano_path), "pano_mask.png")  # accumulated in-tool point deletions (/erase_points)
    if os.path.exists(_pm):
        em = np.array(Image.open(_pm).convert("L").resize((full_mask.shape[1], full_mask.shape[0]), Image.NEAREST)) < 128
        full_mask = full_mask | torch.as_tensor(em, device=dev)
        logln(f"pano_mask: excluded {int(em.sum())} px from scaffold")
    max_d = torch.quantile(fd["distance"][~full_mask], 0.99).item()
    fd["distance"] = torch.clip(fd["distance"], 0, max_d)
    torch.save({"distance": fd["distance"].cpu(), "rays": fd["rays"].cpu()}, f"{rr}/full_depth_prediction.pt")
    Image.fromarray(((~sfd).cpu().numpy() * 255).astype(np.uint8)).save(f"{rr}/sky_mask.png")
    dh, dw = fd["distance"].shape
    pim = full_img.resize((dw, dh), Image.Resampling.BICUBIC) if full_img.size != (dw, dh) else full_img
    gp = convert_rgbd2pcd_panorama(rgb=torch.tensor(np.array(pim) / 255, dtype=torch.float32),
                                   distance=fd["distance"], rays=fd["rays"], excluded_region_mask=full_mask, dropout_pcd=False)
    gp.export(f"{rr}/global_pcd.ply")
    del moge; gc.collect(); torch.cuda.empty_cache()
    logln(f"scaffold built: {len(gp.vertices)} pts (max_d={max_d:.2f})")


def set_current(name):
    proj = f"{PROJ_ROOT}/{name}"; rr = f"{proj}/scene/render_results"
    g = trimesh.load(f"{rr}/global_pcd.ply")
    S["proj"] = proj; S["pano"] = f"{proj}/pano.png"
    sv = np.asarray(g.vertices, np.float32); sc = np.asarray(g.visual.vertex_colors)[:, :3].astype(np.float32) / 255
    S["scaffold_n"] = len(sv); S["step_pts"] = []; S["accum_step"] = 0; S["scale"] = None
    S["gpts"] = torch.as_tensor(sv, device=dev); S["gcol"] = torch.as_tensor(sc, device=dev)
    cj = f"{proj}/chain.json"
    S["history"] = json.load(open(cj))["history"] if os.path.exists(cj) else []
    S["step"] = len(S["history"])
    S["last_png"] = S["history"][-1]["last_png"] if S["history"] else None
    S["last_result"] = S["history"][-1]["result"] if S["history"] else None
    _accum_load()                       # restore previously-accumulated coverage (if any)


def save_chain():
    json.dump({"history": S["history"]}, open(f"{S['proj']}/chain.json", "w"))


# ---- undo-aware coverage accumulation (scaffold = gpts[:scaffold_n], then one block per accumulated step) ----
def _accum_paths():
    return f"{S['proj']}/accum_pcd.ply", f"{S['proj']}/accum_meta.json"

def _accum_save():
    ply, meta = _accum_paths()
    n0 = S["scaffold_n"]; V = S["gpts"][n0:].cpu().numpy(); C = (S["gcol"][n0:].cpu().numpy() * 255).astype(np.uint8)
    if len(V):
        trimesh.PointCloud(vertices=V, colors=np.concatenate([C, np.full((len(C), 1), 255, np.uint8)], 1)).export(ply)
    elif os.path.exists(ply):
        os.remove(ply)
    json.dump({"step_pts": S["step_pts"], "accum_step": S["accum_step"], "scale": S["scale"]}, open(meta, "w"))

def _accum_load():
    ply, meta = _accum_paths()
    if not os.path.exists(meta):
        return
    m = json.load(open(meta)); S["step_pts"] = m.get("step_pts", []); S["accum_step"] = m.get("accum_step", 0); S["scale"] = m.get("scale")
    if os.path.exists(ply) and sum(S["step_pts"]):
        g = trimesh.load(ply)
        V = torch.as_tensor(np.asarray(g.vertices, np.float32), device=dev)
        C = torch.as_tensor(np.asarray(g.visual.vertex_colors)[:, :3].astype(np.float32) / 255, device=dev)
        S["gpts"] = torch.cat([S["gpts"], V]); S["gcol"] = torch.cat([S["gcol"], C])
    logln(f"restored coverage: {S['accum_step']} steps, {sum(S['step_pts'])} pts")

def _accum_truncate(keep_steps):
    """Drop accumulated points for steps beyond keep_steps (used by undo/reset)."""
    if S["accum_step"] <= keep_steps:
        return
    keep_pts = S["scaffold_n"] + sum(S["step_pts"][:keep_steps])
    S["gpts"] = S["gpts"][:keep_pts]; S["gcol"] = S["gcol"][:keep_pts]
    S["step_pts"] = S["step_pts"][:keep_steps]; S["accum_step"] = keep_steps
    if S["proj"]:
        _accum_save()


def cloud_payload(n=300000):
    V = S["gpts"].cpu().numpy(); C = (S["gcol"].cpu().numpy() * 255).astype(int)
    if n is None or n >= len(V):
        idx = np.arange(len(V))
    else:
        idx = np.random.choice(len(V), n, replace=False)
    n = len(idx)
    flat = np.empty(n * 6)
    flat[0::6] = np.round(V[idx, 0], 2); flat[1::6] = np.round(V[idx, 1], 2); flat[2::6] = np.round(V[idx, 2], 2)
    flat[3::6] = C[idx, 0]; flat[4::6] = C[idx, 1]; flat[5::6] = C[idx, 2]
    return {"cloud": flat.tolist(), "bounds": [V.min(0).tolist(), V.max(0).tolist()],
            "steps": S["step"], "history": [{"prompt": h["prompt"]} for h in S["history"]]}


app = Flask(__name__)
@app.after_request
def cors(r):
    r.headers['Access-Control-Allow-Origin'] = '*'; r.headers['Access-Control-Allow-Headers'] = '*'
    r.headers['Access-Control-Allow-Methods'] = '*'; return r


@app.route('/')
def index():
    return open(r"D:/HY-World-2.0/_ws_client.html", encoding="utf-8").read()


@app.route('/vendor/<path:p>')
def vendor(p):
    """Serve three.js locally (same-origin → no CSP/CDN block) from the frontend node_modules."""
    from flask import send_from_directory
    return send_from_directory(r"E:/MyGame/AIWorldStudio/frontend/node_modules/three", p)


@app.route('/pfile/<path:name>')
def pfile(name):
    """Serve a file from the current project dir (e.g. mesh.glb / fastmesh.glb) for the WebGL viewer (GLTFLoader)."""
    if not S["proj"]:
        return ('no project', 404)
    from flask import send_file
    fp = os.path.join(S["proj"], name)
    if not os.path.exists(fp):
        return ('not found', 404)
    return send_file(fp)


@app.route('/browse', methods=['GET', 'OPTIONS'])
def browse():
    """Open a native Windows file dialog on the server machine (local tool) -> return the chosen path."""
    if request.method == 'OPTIONS':
        return ('', 204)
    import subprocess
    ps = ("Add-Type -AssemblyName System.Windows.Forms;"
          "$f=New-Object System.Windows.Forms.OpenFileDialog;"
          "$f.Filter='Images|*.png;*.jpg;*.jpeg;*.webp;*.tif;*.tiff|All files|*.*';"
          "$f.Title='Select panorama (RGBA PNG = alpha mask)';"
          # owner form forced TopMost so the dialog appears IN FRONT of the fullscreen browser
          "$o=New-Object System.Windows.Forms.Form;"
          "$o.TopMost=$true;$o.ShowInTaskbar=$false;$o.Opacity=0;$o.Show();$o.Activate();"
          "$res=$f.ShowDialog($o);$o.Close();"
          "if($res -eq [System.Windows.Forms.DialogResult]::OK){[Console]::Out.Write($f.FileName)}")
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-STA", "-Command", ps],
                             capture_output=True, text=True, timeout=300)
        return jsonify({"path": out.stdout.strip().replace("\\", "/")})
    except Exception as e:
        return jsonify({"path": "", "error": str(e)})


@app.route('/log')
def getlog():
    return jsonify({"log": list(LOG)})


@app.route('/frames', methods=['POST', 'OPTIONS'])
def frames_all():
    if request.method == 'OPTIONS':
        return ('', 204)
    if not S["proj"]:
        return jsonify({"frames": []})
    FR = f"{S['proj']}/frames"; seq = []
    for si in range(len(S["history"])):
        fs = sorted(glob.glob(f"{FR}/s{si:03d}_f*.png"))
        seq += fs[(0 if si == 0 else 1):]          # overlap-trim: drop redundant join frame
    out = []
    for fp in seq:
        with open(fp, "rb") as f:
            out.append(base64.b64encode(f.read()).decode())
    return jsonify({"frames": out})


@app.route('/mesh', methods=['POST', 'OPTIONS'])
def mesh():
    if request.method == 'OPTIONS':
        return ('', 204)
    if S["gpts"] is None:
        return jsonify({"error": "no project open"}), 200
    import open3d as o3d
    logln("meshing: preparing cloud…")
    V = S["gpts"].cpu().numpy().astype(np.float64); C = np.clip(S["gcol"].cpu().numpy().astype(np.float64), 0, 1)
    pcd = o3d.geometry.PointCloud(); pcd.points = o3d.utility.Vector3dVector(V); pcd.colors = o3d.utility.Vector3dVector(C)
    diag = float(np.linalg.norm(V.max(0) - V.min(0)))
    if len(V) > 1_500_000:
        pcd = pcd.voxel_down_sample(diag * 0.004)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    logln("meshing: normals + Poisson…")
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=diag * 0.02, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(15)
    m, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=10, linear_fit=True)
    dens = np.asarray(dens); m.remove_vertices_by_mask(dens < np.quantile(dens, 0.04))
    m.remove_degenerate_triangles(); m.remove_unreferenced_vertices(); m.compute_vertex_normals()
    out = f"{S['proj']}/mesh"
    o3d.io.write_triangle_mesh(out + ".ply", m)
    MV = np.asarray(m.vertices); Fc = np.asarray(m.triangles)
    MC = (np.clip(np.asarray(m.vertex_colors), 0, 1) * 255).astype(np.uint8)
    VC = np.concatenate([MC, np.full((len(MC), 1), 255, np.uint8)], 1)
    trimesh.Trimesh(vertices=MV, faces=Fc, vertex_colors=VC, process=False).export(out + ".glb")
    logln(f"meshing DONE: {len(MV)} verts, {len(Fc)} tris -> mesh.glb / .ply")
    n = min(300000, len(MV)); idx = np.random.choice(len(MV), n, replace=False) if len(MV) > n else np.arange(len(MV))
    Ci = np.clip(np.asarray(m.vertex_colors), 0, 1) * 255
    flat = np.empty(len(idx) * 6)
    flat[0::6] = np.round(MV[idx, 0], 3); flat[1::6] = np.round(MV[idx, 1], 3); flat[2::6] = np.round(MV[idx, 2], 3)
    flat[3::6] = Ci[idx, 0]; flat[4::6] = Ci[idx, 1]; flat[5::6] = Ci[idx, 2]
    NRM = np.asarray(m.vertex_normals)
    nf = np.empty(len(idx) * 3)
    nf[0::3] = np.round(NRM[idx, 0], 3); nf[1::3] = np.round(NRM[idx, 1], 3); nf[2::3] = np.round(NRM[idx, 2], 3)
    return jsonify({"verts": flat.tolist(), "normals": nf.tolist(), "nverts": int(len(MV)), "ntris": int(len(Fc)), "glb": out + ".glb"})


def _mesh_payload(pcd, outname, voxel, depth=10, fast=False):
    import open3d as o3d
    pts = np.asarray(pcd.points)
    diag = float(np.linalg.norm(pts.max(0) - pts.min(0)))
    if voxel and voxel > 0:
        pcd = pcd.voxel_down_sample(voxel)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=(8 if fast else 20), std_ratio=2.0)
    logln(f"normals + Poisson (depth={depth}{', fast' if fast else ''})…")
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=diag * 0.02, max_nn=(16 if fast else 30)))
    if fast:
        pcd.orient_normals_towards_camera_location(np.array([0.0, 0.0, 0.0]))  # room interior ~origin: instant
    else:
        pcd.orient_normals_consistent_tangent_plane(15)                         # slow but globally consistent
    m, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth, linear_fit=(not fast))
    dens = np.asarray(dens); m.remove_vertices_by_mask(dens < np.quantile(dens, 0.04))
    m.remove_degenerate_triangles(); m.remove_unreferenced_vertices(); m.compute_vertex_normals()
    o3d.io.write_triangle_mesh(outname + ".ply", m)
    MV = np.asarray(m.vertices); Fc = np.asarray(m.triangles)
    MC = (np.clip(np.asarray(m.vertex_colors), 0, 1) * 255).astype(np.uint8)
    trimesh.Trimesh(vertices=MV, faces=Fc, vertex_colors=np.concatenate([MC, np.full((len(MC), 1), 255, np.uint8)], 1),
                    process=False).export(outname + ".glb")
    logln(f"mesh DONE: {len(MV)} verts, {len(Fc)} tris -> {os.path.basename(outname)}.glb")
    n = min(300000, len(MV)); idx = np.random.choice(len(MV), n, replace=False) if len(MV) > n else np.arange(len(MV))
    Ci = np.clip(np.asarray(m.vertex_colors), 0, 1) * 255; NRM = np.asarray(m.vertex_normals)
    flat = np.empty(len(idx) * 6)
    flat[0::6] = np.round(MV[idx, 0], 3); flat[1::6] = np.round(MV[idx, 1], 3); flat[2::6] = np.round(MV[idx, 2], 3)
    flat[3::6] = Ci[idx, 0]; flat[4::6] = Ci[idx, 1]; flat[5::6] = Ci[idx, 2]
    nf = np.empty(len(idx) * 3)
    nf[0::3] = np.round(NRM[idx, 0], 3); nf[1::3] = np.round(NRM[idx, 1], 3); nf[2::3] = np.round(NRM[idx, 2], 3)
    return {"verts": flat.tolist(), "normals": nf.tolist(), "nverts": int(len(MV)), "ntris": int(len(Fc)), "glb": outname + ".glb"}


@app.route('/fastmesh', methods=['POST', 'OPTIONS'])
def fastmesh():
    """FAST Poisson preview (vertex color) on the accumulated cloud — for live authoring context.
    Low octree depth + camera-oriented normals (no slow global orientation). Body: {depth:8}."""
    if request.method == 'OPTIONS':
        return ('', 204)
    if S["gpts"] is None:
        return jsonify({"error": "no project open"}), 200
    import open3d as o3d
    depth = int((request.get_json(silent=True) or {}).get("depth", 8))
    V = S["gpts"].cpu().numpy().astype(np.float64); C = np.clip(S["gcol"].cpu().numpy().astype(np.float64), 0, 1)
    pcd = o3d.geometry.PointCloud(); pcd.points = o3d.utility.Vector3dVector(V); pcd.colors = o3d.utility.Vector3dVector(C)
    diag = float(np.linalg.norm(V.max(0) - V.min(0)))
    t = time.time()
    pl = _mesh_payload(pcd, f"{S['proj']}/fastmesh", diag * 0.006, depth=depth, fast=True)
    logln(f"fastmesh DONE in {time.time()-t:.1f}s")
    return jsonify(pl)


@app.route('/accumulate', methods=['POST', 'OPTIONS'])
@gpu_endpoint
def accumulate():
    """Incrementally fuse ONLY the steps not yet accumulated into the coverage cloud (undo-aware),
    then return a fast Poisson preview. Cheap vs /world_mesh (which re-fuses every step). Body: {depth:8}."""
    if request.method == 'OPTIONS':
        return ('', 204)
    if not S["proj"]:
        return jsonify({"error": "no project open"}), 200
    import open3d as o3d
    from moge.model.v2 import MoGeModel
    depth = int((request.get_json(silent=True) or {}).get("depth", 8))
    start, end = S["accum_step"], len(S["history"])
    FR = f"{S['proj']}/frames"
    if start < end:
        _snapshot()                                       # unified undo: snapshot before accumulating new steps
        logln(f"accumulate: steps {start}..{end-1} (freeing GPU + MoGe)…")
        PIPE.transformer.to("cpu"); WS._STATE["tr_on_gpu"] = False; torch.cuda.empty_cache()
        moge = MoGeModel.from_pretrained(MOGE_ID).to(dev).eval()
        if S["scale"] is None:                          # global MoGe<->scaffold scale, once (step0 frame0)
            h0 = S["history"][0]; K0 = np.array(h0["intrinsic"], np.float32); pose0 = np.array(h0["poses"][0], np.float32)
            fr0 = cv2.cvtColor(cv2.imread(f"{FR}/s000_f0.png"), cv2.COLOR_BGR2RGB); H, W = fr0.shape[:2]
            with torch.no_grad():
                o0 = moge.infer(torch.as_tensor(fr0 / 255.0, dtype=torch.float32, device=dev).permute(2, 0, 1), use_fp16=True)
            base = S["gpts"][:S["scaffold_n"]]
            _, cdep = point_rendering(K0[None], pose0[None], base, torch.zeros((len(base), 3), device=dev),
                                      dev, H, W, render_radius=0.012, return_depth=True)
            cdep = cdep[0, 0]; mdep = o0["depth"]; mmask = o0["mask"]; valid = (cdep > 1e-3) & mmask & (mdep > 1e-3)
            S["scale"] = float((cdep[valid] / mdep[valid]).median().item()) if valid.sum() > 500 else 1.0
            logln(f"accumulate: scale={S['scale']:.4f}")
        scale = S["scale"]
        for si in range(start, end):
            poses = S["history"][si]["poses"]; pstep, cstep = [], []
            for fi in range(len(poses)):
                fp = f"{FR}/s{si:03d}_f{fi}.png"
                if not os.path.exists(fp):
                    continue
                fr = cv2.cvtColor(cv2.imread(fp), cv2.COLOR_BGR2RGB)
                with torch.no_grad():
                    oo = moge.infer(torch.as_tensor(fr / 255.0, dtype=torch.float32, device=dev).permute(2, 0, 1), use_fp16=True)
                mk = oo["mask"].cpu().numpy().reshape(-1); pc = (oo["points"].cpu().numpy().reshape(-1, 3) * scale)[mk]
                if len(pc) == 0:
                    continue
                c2w = np.linalg.inv(np.array(poses[fi], np.float64))
                world = (np.concatenate([pc, np.ones((len(pc), 1))], 1) @ c2w.T)[:, :3].astype(np.float32)
                pstep.append(world); cstep.append(fr.reshape(-1, 3)[mk].astype(np.float32) / 255)
            if pstep:                                   # voxel-thin each step so the cloud stays light
                P = np.concatenate(pstep); Cc = np.concatenate(cstep)
                pc = o3d.geometry.PointCloud(); pc.points = o3d.utility.Vector3dVector(P.astype(np.float64)); pc.colors = o3d.utility.Vector3dVector(Cc.astype(np.float64))
                dg = float(np.linalg.norm(P.max(0) - P.min(0))); pc = pc.voxel_down_sample(max(0.005, dg * 0.004))
                Pv = np.asarray(pc.points, np.float32); Cv = np.asarray(pc.colors, np.float32)
                S["gpts"] = torch.cat([S["gpts"], torch.as_tensor(Pv, device=dev)])
                S["gcol"] = torch.cat([S["gcol"], torch.as_tensor(Cv, device=dev)])
                S["step_pts"].append(len(Pv))
            else:
                S["step_pts"].append(0)
            S["accum_step"] = si + 1
        del moge; gc.collect(); torch.cuda.empty_cache()
        PIPE.transformer.to(dev); WS._STATE["tr_on_gpu"] = True
        _accum_save()
        logln(f"accumulate DONE: {S['accum_step']} steps, {sum(S['step_pts'])} accumulated pts")
    # fast Poisson preview over scaffold + all accumulated coverage
    V = S["gpts"].cpu().numpy().astype(np.float64); C = np.clip(S["gcol"].cpu().numpy().astype(np.float64), 0, 1)
    pcd = o3d.geometry.PointCloud(); pcd.points = o3d.utility.Vector3dVector(V); pcd.colors = o3d.utility.Vector3dVector(C)
    diag = float(np.linalg.norm(V.max(0) - V.min(0)))
    pl = _mesh_payload(pcd, f"{S['proj']}/coverage", diag * 0.006, depth=depth, fast=True)
    pl["accum_step"] = S["accum_step"]; pl["accum_pts"] = int(sum(S["step_pts"]))
    return jsonify(pl)


@app.route('/world_mesh', methods=['POST', 'OPTIONS'])
@gpu_endpoint
def world_mesh():
    if request.method == 'OPTIONS':
        return ('', 204)
    if not S["proj"] or not S["history"]:
        return jsonify({"error": "no generated steps yet"}), 200
    import open3d as o3d
    from moge.model.v2 import MoGeModel
    FR = f"{S['proj']}/frames"
    logln("world-mesh: freeing GPU + loading MoGe-2…")
    PIPE.transformer.to("cpu"); WS._STATE["tr_on_gpu"] = False; torch.cuda.empty_cache()
    moge = MoGeModel.from_pretrained(MOGE_ID).to(dev).eval()

    # global scale: MoGe depth vs scaffold-rendered depth on step0 frame0
    h0 = S["history"][0]; K0 = np.array(h0["intrinsic"], np.float32); pose0 = np.array(h0["poses"][0], np.float32)
    fr0 = cv2.cvtColor(cv2.imread(f"{FR}/s000_f0.png"), cv2.COLOR_BGR2RGB)
    H, W = fr0.shape[:2]
    with torch.no_grad():
        o0 = moge.infer(torch.as_tensor(fr0 / 255.0, dtype=torch.float32, device=dev).permute(2, 0, 1), use_fp16=True)
    _, cdep = point_rendering(K0[None], pose0[None], S["gpts"], torch.zeros((len(S["gpts"]), 3), device=dev),
                              dev, H, W, render_radius=0.012, return_depth=True)
    cdep = cdep[0, 0]; mdep = o0["depth"]; mmask = o0["mask"]
    valid = (cdep > 1e-3) & mmask & (mdep > 1e-3)
    scale = float((cdep[valid] / mdep[valid]).median().item()) if valid.sum() > 500 else 1.0
    logln(f"world-mesh: scale={scale:.4f}; fusing {len(S['history'])} steps (MoGe per frame)…")

    pts_all, col_all = [], []
    for si, h in enumerate(S["history"]):
        poses = h["poses"]
        for fi in range(len(poses)):
            fp = f"{FR}/s{si:03d}_f{fi}.png"
            if not os.path.exists(fp):
                continue
            fr = cv2.cvtColor(cv2.imread(fp), cv2.COLOR_BGR2RGB)
            with torch.no_grad():
                oo = moge.infer(torch.as_tensor(fr / 255.0, dtype=torch.float32, device=dev).permute(2, 0, 1), use_fp16=True)
            m = oo["mask"].cpu().numpy().reshape(-1)
            pc = (oo["points"].cpu().numpy().reshape(-1, 3) * scale)[m]
            if len(pc) == 0:
                continue
            c2w = np.linalg.inv(np.array(poses[fi], np.float64))
            world = (np.concatenate([pc, np.ones((len(pc), 1))], 1) @ c2w.T)[:, :3].astype(np.float32)
            pts_all.append(world); col_all.append(fr.reshape(-1, 3)[m])
    del moge; gc.collect(); torch.cuda.empty_cache()
    PIPE.transformer.to(dev); WS._STATE["tr_on_gpu"] = True
    if not pts_all:
        return jsonify({"error": "no valid geometry"}), 200
    P = np.concatenate(pts_all).astype(np.float64); C = np.concatenate(col_all).astype(np.float64) / 255.0
    logln(f"world-mesh: {len(P)} fused pts → mesh…")
    pcd = o3d.geometry.PointCloud(); pcd.points = o3d.utility.Vector3dVector(P); pcd.colors = o3d.utility.Vector3dVector(np.clip(C, 0, 1))
    diag = float(np.linalg.norm(P.max(0) - P.min(0)))
    return jsonify(_mesh_payload(pcd, f"{S['proj']}/world_mesh", max(0.005, diag * 0.003)))


@app.route('/cloud', methods=['POST', 'OPTIONS'])
def cloud():
    if request.method == 'OPTIONS':
        return ('', 204)
    if S["gpts"] is None:
        return jsonify({"error": "no project open"}), 200
    n = (request.get_json(silent=True) or {}).get("n")
    if n in (None, "all", 0, "0"):
        n = None
    return jsonify(cloud_payload(int(n) if n else None))


@app.route('/projects', methods=['GET', 'OPTIONS'])
def projects():
    if request.method == 'OPTIONS':
        return ('', 204)
    out = []
    for nm in sorted(os.listdir(PROJ_ROOT)):
        cj = f"{PROJ_ROOT}/{nm}/chain.json"
        steps = len(json.load(open(cj))["history"]) if os.path.exists(cj) else 0
        if os.path.isdir(f"{PROJ_ROOT}/{nm}"):
            out.append({"name": nm, "steps": steps})
    return jsonify({"projects": out, "current": os.path.basename(S["proj"]) if S["proj"] else None})


@app.route('/create_project', methods=['POST', 'OPTIONS'])
@gpu_endpoint
def create_project():
    if request.method == 'OPTIONS':
        return ('', 204)
    up = request.files.get('pano')                       # browser file-upload (multipart)
    if up is not None:
        name = (request.form.get('name') or '').strip().strip('"').strip("'")
        if not name:
            return jsonify({"error": "no project name"}), 200
        proj = f"{PROJ_ROOT}/{name}"; rr = f"{proj}/scene/render_results"
        os.makedirs(f"{rr}/view0/traj0", exist_ok=True); os.makedirs(f"{proj}/frames", exist_ok=True)
        os.makedirs(f"{proj}/scene", exist_ok=True)
        try:
            Image.open(up.stream).save(f"{proj}/pano.png")   # normalize any format -> PNG (keeps alpha)
        except Exception as e:
            return jsonify({"error": f"bad image: {e}"}), 200
    else:                                                # legacy: server-side path
        d = request.get_json()
        name = d["name"].strip().strip('"').strip("'")
        pano = d.get("pano", "D:/_world_hangar/panorama.png").strip().strip('"').strip("'")
        if not os.path.exists(pano):
            return jsonify({"error": f"pano not found: {pano}"}), 200
        proj = f"{PROJ_ROOT}/{name}"; rr = f"{proj}/scene/render_results"
        os.makedirs(f"{rr}/view0/traj0", exist_ok=True); os.makedirs(f"{proj}/frames", exist_ok=True)
        shutil.copy(pano, f"{proj}/pano.png")
    json.dump({"scene_type": "indoor"}, open(f"{proj}/scene/meta_info.json", "w"))
    PIPE.transformer.to("cpu"); WS._STATE["tr_on_gpu"] = False; torch.cuda.empty_cache()   # free GPU for MoGe
    build_scaffold(f"{proj}/pano.png", rr)
    PIPE.transformer.to(dev); WS._STATE["tr_on_gpu"] = True; torch.cuda.empty_cache()       # transformer resident again
    set_current(name)
    return jsonify(cloud_payload())


@app.route('/open_project', methods=['POST', 'OPTIONS'])
def open_project():
    if request.method == 'OPTIONS':
        return ('', 204)
    set_current(request.get_json()["name"])
    return jsonify(cloud_payload())


@app.route('/reset', methods=['POST', 'OPTIONS'])
def reset():
    if request.method == 'OPTIONS':
        return ('', 204)
    S["last_png"] = None; S["last_result"] = None; S["step"] = 0; S["history"] = []
    _accum_truncate(0)                   # clear all accumulated coverage back to scaffold
    OPS.clear()                          # reset wipes the undo timeline too (user warned via modal to download first)
    if S["proj"]:
        save_chain()
    return jsonify({"ok": True})


@app.route('/download_cameras', methods=['GET', 'OPTIONS'])
def download_cameras():
    """Blender-ready camera path of the current trajectory (per generated frame)."""
    if request.method == 'OPTIONS':
        return ('', 204)
    if not S["proj"] or not S["history"]:
        return jsonify({"frames": [], "note": "no generated steps"})
    flip = np.diag([1.0, -1.0, -1.0, 1.0])   # OpenCV cam (+Z fwd,+Y down) -> Blender/OpenGL cam (-Z fwd,+Y up)
    K0 = np.array(S["history"][0]["intrinsic"], float)
    fovx = float(2 * np.arctan((832 / 2) / K0[0][0]) * 180 / np.pi)
    frames = []
    for si, h in enumerate(S["history"]):
        for fi, w2c in enumerate(h["poses"]):
            w2c = np.array(w2c, float); mw = np.linalg.inv(w2c) @ flip
            frames.append({"step": si, "frame": fi, "prompt": h.get("prompt", ""),
                           "matrix_world": mw.tolist(), "w2c": w2c.tolist()})
    return jsonify({"convention": "matrix_world = inv(w2c) @ diag(1,-1,-1,1); set as Blender camera.matrix_world",
                    "fov_x_deg": fovx, "width": 832, "height": 480, "fps": 16, "frames": frames})


@app.route('/download_pano', methods=['GET', 'OPTIONS'])
def download_pano():
    """The tweaked panorama as RGBA PNG: erased regions (alpha<128 or pano_mask) become transparent."""
    if request.method == 'OPTIONS':
        return ('', 204)
    if not S["proj"]:
        return ('', 404)
    import io
    from flask import send_file
    arr = np.array(Image.open(S["pano"]).convert("RGBA"))
    alpha = arr[..., 3].copy()
    pm = f"{S['proj']}/pano_mask.png"
    if os.path.exists(pm):
        m = np.array(Image.open(pm).convert("L").resize((arr.shape[1], arr.shape[0]), Image.NEAREST))
        alpha[m < 128] = 0
    arr[..., 3] = alpha
    buf = io.BytesIO(); Image.fromarray(arr).save(buf, format="PNG"); buf.seek(0)
    return send_file(buf, mimetype="image/png", as_attachment=True, download_name="pano_tweaked.png")


@app.route('/download_mesh', methods=['GET', 'OPTIONS'])
def download_mesh():
    """Download the HIGH-QUALITY pano mesh (GLB). Generates it (Poisson depth-10) if not present, else sends cached.
    ?force=1 regenerates from the current cloud (use after editing)."""
    if request.method == 'OPTIONS':
        return ('', 204)
    if S["gpts"] is None:
        return ('no project open', 404)
    from flask import send_file
    glb = f"{S['proj']}/mesh.glb"
    force = request.args.get("force") in ("1", "true")
    if force or not os.path.exists(glb):
        import open3d as o3d
        logln("download_mesh: generating HQ pano mesh (Poisson depth=10)…")
        V = S["gpts"].cpu().numpy().astype(np.float64); C = np.clip(S["gcol"].cpu().numpy().astype(np.float64), 0, 1)
        pcd = o3d.geometry.PointCloud(); pcd.points = o3d.utility.Vector3dVector(V); pcd.colors = o3d.utility.Vector3dVector(C)
        diag = float(np.linalg.norm(V.max(0) - V.min(0)))
        _mesh_payload(pcd, f"{S['proj']}/mesh", diag * 0.004, depth=10, fast=False)
    return send_file(glb, mimetype="model/gltf-binary", as_attachment=True, download_name="pano_mesh.glb")


@app.route('/undo', methods=['POST', 'OPTIONS'])
def undo():
    if request.method == 'OPTIONS':
        return ('', 204)
    if S["gpts"] is None:
        return jsonify({"error": "no project open"})
    ok = _restore()                     # unified timeline: pop the last op (step / accumulate / erase)
    logln("undo" if ok else "undo: nothing to undo")
    return jsonify({"undone": ok, **cloud_payload()})


OPS = []   # UNIFIED undo timeline: one full-state snapshot before each op (step / accumulate / erase). Capped.

def _resave_scaffold():
    sc = S["gpts"][:S["scaffold_n"]].cpu().numpy(); scc = (S["gcol"][:S["scaffold_n"]].cpu().numpy() * 255).astype(np.uint8)
    trimesh.PointCloud(vertices=sc, colors=np.concatenate([scc, np.full((len(scc), 1), 255, np.uint8)], 1)
                       ).export(f"{S['proj']}/scene/render_results/global_pcd.ply")

def _snapshot():
    if S["gpts"] is None:
        return
    pm = f"{S['proj']}/pano_mask.png"
    OPS.append({
        "gpts": S["gpts"].cpu().clone(), "gcol": S["gcol"].cpu().clone(),
        "scaffold_n": S["scaffold_n"], "step_pts": list(S["step_pts"]),
        "accum_step": S["accum_step"], "scale": S["scale"],
        "history": json.loads(json.dumps(S["history"])), "step": S["step"],
        "last_png": S["last_png"], "last_result": S["last_result"],
        "pano_mask": (np.array(Image.open(pm).convert("L")).copy() if os.path.exists(pm) else None),
    })
    while len(OPS) > 12:
        OPS.pop(0)

def _restore():
    if not OPS:
        return False
    s = OPS.pop()
    S["gpts"] = s["gpts"].to(dev); S["gcol"] = s["gcol"].to(dev)
    S["scaffold_n"] = s["scaffold_n"]; S["step_pts"] = s["step_pts"]; S["accum_step"] = s["accum_step"]; S["scale"] = s["scale"]
    S["history"] = s["history"]; S["step"] = s["step"]; S["last_png"] = s["last_png"]; S["last_result"] = s["last_result"]
    pm = f"{S['proj']}/pano_mask.png"
    if s["pano_mask"] is not None:
        Image.fromarray(s["pano_mask"]).save(pm)
    elif os.path.exists(pm):
        os.remove(pm)
    save_chain(); _resave_scaffold(); _accum_save()
    return True

@app.route('/erase_points', methods=['POST', 'OPTIONS'])
def erase_points():
    """Delete points the client selected (box(es) + brush stroke(s)) — committed on the client's Push button.
    Projects the FULL server cloud with the client's camera, keeps the front surface (per-pixel z-buffer + slab),
    removes them, writes erased SCAFFOLD points back to pano_mask.png (dir->equirect uv), re-saves global_pcd."""
    if request.method == 'OPTIONS':
        return ('', 204)
    if S["gpts"] is None:
        return jsonify({"error": "no project open"}), 200
    d = request.get_json()
    cam = np.array(d["cam"], np.float64); f = np.array(d["f"], np.float64); r = np.array(d["r"], np.float64); dn = np.array(d["d"], np.float64)
    fx = float(d["fx"]); cx = float(d["cx"]); cy = float(d["cy"]); CWc = int(d["cw"]); CHc = int(d["ch"])
    boxes = d.get("boxes", []); brushes = d.get("brushes", []); slab = float(d.get("slab", 0.2))
    V = S["gpts"].cpu().numpy().astype(np.float64)
    x = V - cam; zc = x @ f
    inv = np.where(np.abs(zc) < 1e-6, 1e-6, zc)
    sx = cx + (x @ r) * fx / inv; sy = cy + (x @ dn) * fx / inv
    front = zc > 0.05
    sel = np.zeros(len(V), bool)
    for b in boxes:
        x0, y0, x1, y1 = b
        sel |= front & (sx >= min(x0, x1)) & (sx <= max(x0, x1)) & (sy >= min(y0, y1)) & (sy <= max(y0, y1))
    for bx, by, rad in brushes:
        sel |= front & (((sx - bx) ** 2 + (sy - by) ** 2) <= rad * rad)
    si = np.where(sel)[0]
    if len(si) == 0:
        return jsonify({"erased": 0, **cloud_payload()})
    px = np.clip(sx[si].astype(int), 0, CWc - 1); py = np.clip(sy[si].astype(int), 0, CHc - 1)
    flat = py * CWc + px
    zbuf = np.full(CWc * CHc, np.inf); np.minimum.at(zbuf, flat, zc[si])
    er = si[zc[si] <= zbuf[flat] + slab]                 # front surface only (slab = how deep to cut through)
    pm_path = f"{S['proj']}/pano_mask.png"
    _snapshot()                                           # unified undo: full state before this erase
    keep = np.ones(len(V), bool); keep[er] = False        # order-preserving: scaffold/step blocks stay grouped
    n0 = S["scaffold_n"]; new_n0 = int(keep[:n0].sum())
    new_step = []; off = n0
    for c in S["step_pts"]:
        new_step.append(int(keep[off:off + c].sum())); off += c
    kt = torch.as_tensor(keep, device=dev)
    S["gpts"] = S["gpts"][kt]; S["gcol"] = S["gcol"][kt]
    S["scaffold_n"] = new_n0; S["step_pts"] = new_step
    es = er[er < n0]                                       # scaffold deletions -> persist into pano_mask via dir->uv
    if len(es):
        PW, PH = Image.open(S["pano"]).size
        dirs = V[es]; dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
        u = 1 - (np.arctan2(dirs[:, 1], dirs[:, 0]) / (2 * np.pi)) % 1.0
        v = np.arccos(np.clip(dirs[:, 2], -1, 1)) / np.pi
        pxp = np.clip((u * PW).astype(int), 0, PW - 1); pyp = np.clip((v * PH).astype(int), 0, PH - 1)
        pm = np.array(Image.open(pm_path).convert("L")) if os.path.exists(pm_path) else np.full((PH, PW), 255, np.uint8)
        dot = np.zeros((PH, PW), np.uint8); dot[pyp, pxp] = 255; dot = cv2.dilate(dot, np.ones((5, 5), np.uint8))
        pm[dot > 0] = 0; Image.fromarray(pm).save(pm_path)
    _resave_scaffold(); _accum_save()
    logln(f"erase: removed {len(er)} pts ({len(es)} scaffold -> pano_mask)")
    return jsonify({"erased": int(len(er)), **cloud_payload()})

@app.route('/undo_erase', methods=['POST', 'OPTIONS'])
def undo_erase():
    return undo()                                          # unified timeline: erase-undo == undo


@app.route('/step', methods=['POST', 'OPTIONS'])
@gpu_endpoint
def step():
    if request.method == 'OPTIONS':
        return ('', 204)
    if not S["proj"]:
        return jsonify({"error": "no project open"}), 400
    _snapshot()                                           # unified undo: snapshot before generating this step
    d = request.get_json()
    poses = d["frames"]; K = d["intrinsic"]; prompt = d.get("prompt", "sci-fi spaceship hangar interior, cinematic")
    W = d.get("width", 832); H = d.get("height", 480)
    rr = f"{S['proj']}/scene/render_results"; FR = f"{S['proj']}/frames"
    td = f"{rr}/view0/traj0"; os.makedirs(td, exist_ok=True)
    json.dump({"extrinsic": poses, "intrinsic": [K] * len(poses), "width": W, "height": H, "type": "step"}, open(f"{td}/camera.json", "w"))
    json.dump({"prompt": prompt}, open(f"{td}/traj_caption.json", "w"))

    logln(f"step {S['step']+1}: rendering cloud conditioning…")
    ext = np.array(poses, np.float32); Karr = np.array(K, np.float32)   # transformer stays RESIDENT (cond render fits ~17.6GB)
    if Karr.ndim == 2:
        Karr = np.repeat(Karr[None], len(ext), axis=0)
    rgbs, masks = point_rendering(Karr, ext, S["gpts"], S["gcol"], dev, H, W, render_radius=0.012, return_depth=False)
    rgb = (rgbs.permute(0, 2, 3, 1).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    msk = (masks[:, 0].cpu().numpy() * 255).astype(np.uint8)
    vw = cv2.VideoWriter(f"{td}/render.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 16, (W, H))
    for f in rgb: vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    vw.release()
    vm = cv2.VideoWriter(f"{td}/render_mask.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 16, (W, H))
    for f in msk: vm.write(cv2.cvtColor(f, cv2.COLOR_GRAY2BGR))
    vm.release()

    mem = f"{td}/memory_inputs"; os.makedirs(mem, exist_ok=True)
    if S["last_png"] and os.path.exists(S["last_png"]):
        shutil.copy(S["last_png"], f"{rr}/view0/start_frame.png")
        shutil.copy(S["last_result"], f"{mem}/{MODEL}.mp4")
    else:
        Kn = Karr[0].copy(); Kn[0] /= W; Kn[1] /= H
        pano = np.array(Image.open(S["pano"]).convert("RGB"))
        si = split_panorama_image(pano, ext[:1], [Kn], h=H, w=W, interp=cv2.INTER_AREA)[0]
        Image.fromarray(si).save(f"{rr}/view0/start_frame.png")
        shutil.copy(f"{td}/render.mp4", f"{mem}/{MODEL}.mp4")        # first step: self-ref seed (= cloud render)

    t = time.time()
    res = gen_clip(rr, prompt, torch.tensor([0], dtype=torch.long))
    dt = time.time() - t

    cap = cv2.VideoCapture(res); frames = []
    while True:
        ok, f = cap.read()
        if not ok: break
        frames.append(f)
    cap.release()
    si = S["step"]; b64 = []
    for i, f in enumerate(frames):
        cv2.imwrite(f"{FR}/s{si:03d}_f{i}.png", f)
        _, buf = cv2.imencode('.png', f); b64.append(base64.b64encode(buf).decode())
    lastp = f"{FR}/s{si:03d}_last.png"; cv2.imwrite(lastp, frames[-1])
    resp = f"{FR}/s{si:03d}_result.mp4"; shutil.copy(res, resp)
    S["last_png"] = lastp; S["last_result"] = resp
    S["history"].append({"prompt": prompt, "poses": poses, "intrinsic": K, "last_png": lastp, "result": resp})
    S["step"] = len(S["history"]); save_chain()
    logln(f"step {S['step']} DONE ({dt:.0f}s) — '{prompt[:38]}'")
    return jsonify({"frames": b64, "step": S["step"], "new_pose": poses[-1], "secs": round(dt, 1)})


if __name__ == "__main__":
    print("[server] ready on http://127.0.0.1:5005 (multi-project)", flush=True)
    app.run(host='127.0.0.1', port=5005, threaded=False)
