"""Build a WS scene dir from an EDITED glb (mesh + named cameras) — Step 2 for authored paths.
Reads world-space verts + COLOR_0 (direct accessor; trimesh mishandles the PBR material) + camera nodes
sorted by name (A01..A06 = the 6 keyframes). Renders the splat-conditioning from each camera at 832x480 with
the cameras' own FOV. Emits render.mp4/render_mask.mp4/camera.json/start_frame + panorama/meta/caption.

  python _ws_scene_from_glb.py <glb> <pano> <out_scene_dir> [prompt]
"""
import sys, os, json, struct, numpy as np, cv2

GLB   = sys.argv[1]
PANO  = sys.argv[2]
SCENE = sys.argv[3]
PROMPT= sys.argv[4] if len(sys.argv) > 4 else "industrial spaceship hangar bay, orange-lit metal panels, grated floor, cinematic, high detail"
VIEW, TRAJ, W, H = "view0", "traj0", 832, 480
TD = f"{SCENE}/render_results/{VIEW}/{TRAJ}"; os.makedirs(TD, exist_ok=True)

with open(GLB, "rb") as f:
    f.read(12); clen, _ = struct.unpack("<II", f.read(8)); js = json.loads(f.read(clen))
    blen, _ = struct.unpack("<II", f.read(8)); bin_ = f.read(blen)

def acc(i):
    a = js["accessors"][i]; bv = js["bufferViews"][a["bufferView"]]
    off = bv.get("byteOffset", 0) + a.get("byteOffset", 0); n = a["count"]
    return np.frombuffer(bin_, np.float32, n * 3, off).reshape(n, 3)

def quat2R(q):
    x, y, z, w = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])

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

mesh_node = [i for i, n in enumerate(nodes) if "mesh" in n][0]
at = js["meshes"][nodes[mesh_node]["mesh"]]["primitives"][0]["attributes"]
Vh = np.c_[acc(at["POSITION"]), np.ones(js["accessors"][at["POSITION"]]["count"])]
xyz = (Vh @ world[mesh_node].T)[:, :3].astype(np.float32)
col = (acc(at["COLOR_0"])[:, :3] * 255).clip(0, 255).astype(np.uint8)[:, ::-1]   # RGB->BGR

cams = sorted([(nodes[i].get("name", f"n{i}"), i, nodes[i]["camera"]) for i, n in enumerate(nodes) if "camera" in n])
print(f"glb: verts={len(xyz)}  cameras={[c[0] for c in cams]}")
assert cams, "no cameras in glb"

def render(c2w, yfov):
    fy = (H / 2) / np.tan(yfov / 2); fx = fy; cx, cy = W / 2, H / 2
    pos, right, up, fwd = c2w[:3, 3], c2w[:3, 0], c2w[:3, 1], -c2w[:3, 2]
    R = np.stack([right, -up, fwd], 0); P = (xyz - pos) @ R.T
    z = P[:, 2]; m = z > 0.05; Pm, Cm, zm = P[m], col[m], z[m]
    u = (fx * Pm[:, 0] / zm + cx).astype(np.int32); v = (fy * Pm[:, 1] / zm + cy).astype(np.int32)
    ok = (u >= 0) & (u < W) & (v >= 0) & (v < H); u, v, Cm, zm = u[ok], v[ok], Cm[ok], zm[ok]
    o = np.argsort(-zm)
    img = np.zeros((H, W, 3), np.uint8); msk = np.zeros((H, W), np.uint8)
    img[v[o], u[o]] = Cm[o]; msk[v[o], u[o]] = 255
    img = cv2.dilate(img, np.ones((2, 2), np.uint8)); msk = cv2.dilate(msk, np.ones((2, 2), np.uint8))
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float32)
    M = np.eye(4, dtype=np.float32); M[:3, :3] = R; M[:3, 3] = -R @ pos     # OpenCV w2c
    return img, msk, M, K

vw = cv2.VideoWriter(f"{TD}/render.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 16, (W, H))
mw = cv2.VideoWriter(f"{TD}/render_mask.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 16, (W, H))
extr, intr = [], []
for k, (name, ni, ci) in enumerate(cams):
    yf = js["cameras"][ci]["perspective"]["yfov"]
    img, msk, M, K = render(world[ni], yf)
    if k == 0: cv2.imwrite(f"{SCENE}/render_results/{VIEW}/start_frame.png", img)
    vw.write(img); mw.write(cv2.cvtColor(msk, cv2.COLOR_GRAY2BGR))
    extr.append(M.tolist()); intr.append(K.tolist())
    print(f"  {name}: {np.degrees(yf):.0f}V cov {(msk>0).mean()*100:.0f}%")
vw.release(); mw.release()

json.dump({"extrinsic": extr, "intrinsic": intr}, open(f"{TD}/camera.json", "w"))
json.dump({"prompt": PROMPT}, open(f"{TD}/traj_caption.json", "w"))
json.dump({"scene_type": "indoor"}, open(f"{SCENE}/meta_info.json", "w"))
cv2.imwrite(f"{SCENE}/panorama.png", cv2.imread(PANO))
print(f"SCENE built -> {SCENE}  ({len(cams)} keyframes)  prompt={PROMPT!r}")
