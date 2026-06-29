"""Interpolate the glb's named cameras into smooth per-segment CLIPS (slerp rot + lerp pos), build a WS scene
per clip, and a combined smooth conditioning preview. Each original segment A_i->A_{i+1} -> one 6-keyframe clip.

  python _ws_interp_clips.py <glb> <pano> <scene_root> [N_per_seg=6] [prompt]
Outputs: <scene_root>/case_hangar_c{0..S-1}/  (each a WS scene)  +  <scene_root>/_cond_smooth.mp4 (preview)
"""
import sys, os, json, struct, numpy as np, cv2
from scipy.spatial.transform import Rotation, Slerp

GLB, PANO, ROOT = sys.argv[1], sys.argv[2], sys.argv[3]
NSEG = int(sys.argv[4]) if len(sys.argv) > 4 else 6
PROMPT = sys.argv[5] if len(sys.argv) > 5 else "industrial spaceship hangar bay, orange-lit metal panels, grated floor, cinematic, high detail"
W, H = 832, 480

with open(GLB, "rb") as f:
    f.read(12); clen, _ = struct.unpack("<II", f.read(8)); js = json.loads(f.read(clen))
    blen, _ = struct.unpack("<II", f.read(8)); bin_ = f.read(blen)
def acc(i):
    a = js["accessors"][i]; bv = js["bufferViews"][a["bufferView"]]
    off = bv.get("byteOffset", 0) + a.get("byteOffset", 0)
    return np.frombuffer(bin_, np.float32, a["count"] * 3, off).reshape(a["count"], 3)
def quat2R(q):
    x, y, z, w = q
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
def local(n):
    if "matrix" in n: return np.array(n["matrix"], float).reshape(4, 4).T
    M = np.eye(4)
    if "rotation" in n: M[:3, :3] = quat2R(n["rotation"])
    if "scale" in n: M[:3, :3] = M[:3, :3] @ np.diag(n["scale"])
    if "translation" in n: M[:3, 3] = n["translation"]
    return M
nodes = js["nodes"]; world = {}
def walk(i, par):
    world[i] = par @ local(nodes[i])
    for c in nodes[i].get("children", []): walk(c, world[i])
for r in js["scenes"][0]["nodes"]: walk(r, np.eye(4))
mn = [i for i, n in enumerate(nodes) if "mesh" in n][0]
at = js["meshes"][nodes[mn]["mesh"]]["primitives"][0]["attributes"]
Vh = np.c_[acc(at["POSITION"]), np.ones(js["accessors"][at["POSITION"]]["count"])]
xyz = (Vh @ world[mn].T)[:, :3].astype(np.float32)
col = (acc(at["COLOR_0"])[:, :3] * 255).clip(0, 255).astype(np.uint8)[:, ::-1]
cams = sorted([(nodes[i].get("name", f"n{i}"), i, nodes[i]["camera"]) for i, n in enumerate(nodes) if "camera" in n])
c2ws = [world[ni] for _, ni, _ in cams]
yfov = js["cameras"][cams[0][2]]["perspective"]["yfov"]
print(f"cameras {[c[0] for c in cams]}  yfov={np.degrees(yfov):.0f}  -> {len(cams)-1} segments x {NSEG} frames")

def render(c2w):
    fy = (H/2)/np.tan(yfov/2); fx = fy; cx, cy = W/2, H/2
    pos, right, up, fwd = c2w[:3,3], c2w[:3,0], c2w[:3,1], -c2w[:3,2]
    R = np.stack([right, -up, fwd], 0); P = (xyz - pos) @ R.T
    z = P[:,2]; m = z > 0.05; Pm, Cm, zm = P[m], col[m], z[m]
    u = (fx*Pm[:,0]/zm+cx).astype(np.int32); v = (fy*Pm[:,1]/zm+cy).astype(np.int32)
    ok = (u>=0)&(u<W)&(v>=0)&(v<H); u,v,Cm,zm = u[ok],v[ok],Cm[ok],zm[ok]
    o = np.argsort(-zm); img = np.zeros((H,W,3),np.uint8); msk = np.zeros((H,W),np.uint8)
    img[v[o],u[o]] = Cm[o]; msk[v[o],u[o]] = 255
    img = cv2.dilate(img,np.ones((2,2),np.uint8)); msk = cv2.dilate(msk,np.ones((2,2),np.uint8))
    K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]],np.float32)
    M = np.eye(4,dtype=np.float32); M[:3,:3]=R; M[:3,3]=-R@pos
    return img, msk, M.tolist(), K.tolist()

KF = 6   # WS requires EXACTLY 6 keyframes per clip (num_frames//4+1). Smoother = more CLIPS, not more frames/clip.
def interp_t(A, B, t0, t1, n):
    rots = Rotation.from_matrix([A[:3,:3], B[:3,:3]]); sl = Slerp([0,1], rots)
    out = []
    for t in np.linspace(t0, t1, n):
        C = np.eye(4); C[:3,:3] = sl([t]).as_matrix()[0]; C[:3,3] = (1-t)*A[:3,3] + t*B[:3,3]; out.append(C)
    return out

os.makedirs(ROOT, exist_ok=True)
smooth = cv2.VideoWriter(f"{ROOT}/_cond_smooth.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 12, (W, H))
clips = []
SUB = NSEG   # sub-clips per original segment (each a 6-keyframe clip over a sub-piece -> finer camera motion)
ci = 0
for s in range(len(c2ws)-1):
  for j in range(SUB):
    seq = interp_t(c2ws[s], c2ws[s+1], j/SUB, (j+1)/SUB, KF)
    scene = f"{ROOT}/case_hangar_c{ci}"; TD = f"{scene}/render_results/view0/traj0"; os.makedirs(TD, exist_ok=True)
    vw = cv2.VideoWriter(f"{TD}/render.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 16, (W,H))
    mw = cv2.VideoWriter(f"{TD}/render_mask.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 16, (W,H))
    extr, intr = [], []
    for k, C in enumerate(seq):
        img, msk, M, K = render(C)
        if k == 0: cv2.imwrite(f"{scene}/render_results/view0/start_frame.png", img)
        vw.write(img); mw.write(cv2.cvtColor(msk, cv2.COLOR_GRAY2BGR)); extr.append(M); intr.append(K)
        if ci == 0 or k > 0: smooth.write(img)        # skip duplicate boundary frame across clips
    vw.release(); mw.release()
    json.dump({"extrinsic": extr, "intrinsic": intr}, open(f"{TD}/camera.json","w"))
    json.dump({"prompt": PROMPT}, open(f"{TD}/traj_caption.json","w"))
    json.dump({"scene_type":"indoor"}, open(f"{scene}/meta_info.json","w"))
    cv2.imwrite(f"{scene}/panorama.png", cv2.imread(PANO))
    clips.append(scene); ci += 1
smooth.release()
print(f"built {len(clips)} clips -> {ROOT}/case_hangar_c0..{len(clips)-1}")
print(f"smooth conditioning preview -> {ROOT}/_cond_smooth.mp4")
for c in clips: print("  ", c)
