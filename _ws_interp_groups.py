"""Build WS clips from a glb with MULTIPLE camera passes (letters A.. B.. C.. D..). Interpolate ONLY within
a letter group (slerp rot + lerp pos), split each segment by CLIP_METERS into 6-keyframe clips. Per-glb prefix.

  python _ws_interp_groups.py <glb> <pano> <ROOT> <prefix> <clip_meters> [groups e.g. A,B or ALL] [prompt]
Clips: <ROOT>/<prefix>_<G><idx>/   (G = group letter). Prints the clip list grouped by letter (for chaining).
"""
import sys, os, json, struct, re, numpy as np, cv2
from scipy.spatial.transform import Rotation, Slerp

argv = sys.argv[1:]
CLOUD = None                                  # optional rich cloud (.npz from stage_glb_points): splat ITS
if "--cloud" in argv:                         # texture-aware xyz+color (ALL objects) instead of mesh COLOR_0
    _i = argv.index("--cloud"); CLOUD = argv[_i + 1]; del argv[_i:_i + 2]
GLB, PANO, ROOT, PREFIX = argv[0], argv[1], argv[2], argv[3]
CLIP_M = float(argv[4])
ONLY = argv[5].split(",") if len(argv) > 5 and argv[5] != "ALL" else None
PROMPT = argv[6] if len(argv) > 6 else "industrial spaceship hangar bay, orange-lit metal panels, grated floor, cinematic, high detail"
W, H, KF = 832, 480, 6

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
if CLOUD:                                      # texture-aware world cloud (all objects), already world-frame
    _z = np.load(CLOUD)
    xyz = _z["xyz"].astype(np.float32)
    col = _z["color"][:, :3].astype(np.uint8)[:, ::-1]           # RGB -> BGR for cv2
    sky = _z["is_sky"].astype(bool) if "is_sky" in _z else np.zeros(len(xyz), bool)  # Sky = sparse big-radius backdrop
    print(f"  cloud: {CLOUD}  ({len(xyz)} pts, {int(sky.sum())} sky, texture-aware)", flush=True)
else:                                          # legacy: first mesh's vertex colors only (single-material)
    mn = [i for i, n in enumerate(nodes) if "mesh" in n][0]
    at = js["meshes"][nodes[mn]["mesh"]]["primitives"][0]["attributes"]
    Vh = np.c_[acc(at["POSITION"]), np.ones(js["accessors"][at["POSITION"]]["count"])]
    xyz = (Vh @ world[mn].T)[:, :3].astype(np.float32)
    col = (acc(at["COLOR_0"])[:, :3] * 255).clip(0, 255).astype(np.uint8)[:, ::-1]
    sky = np.zeros(len(xyz), bool)
cams = sorted([(nodes[i].get("name", f"n{i}"), i) for i, n in enumerate(nodes) if "camera" in n])
if not cams:
    raise SystemExit(f"!! glb has NO cameras: {GLB}\n"
                     f"   export the scene WITH the camera passes (one camera per keyframe, named per pass: A.., B..).")
yfov = js["cameras"][nodes[cams[0][1]]["camera"]]["perspective"]["yfov"]
groups = {}
for name, ni in cams:
    m = re.match(r"([A-Za-z]+)", name)
    g = m.group(1) if m else "G"          # numeric / unprefixed names -> one default group 'G'
    groups.setdefault(g, []).append(world[ni])
print(f"{PREFIX}: groups={ {g:len(v) for g,v in groups.items()} } yfov={np.degrees(yfov):.0f} clip={CLIP_M}m")

def _splat(u, v, C, z, kernel):
    """z-buffered splat (far->near) + dilate by `kernel`. Returns (img, mask)."""
    img = np.zeros((H, W, 3), np.uint8); msk = np.zeros((H, W), np.uint8)
    o = np.argsort(-z); img[v[o], u[o]] = C[o]; msk[v[o], u[o]] = 255
    k = np.ones((kernel, kernel), np.uint8)
    return cv2.dilate(img, k), cv2.dilate(msk, k)

def render(c2w):
    fy = (H/2)/np.tan(yfov/2); fx = fy; cx, cy = W/2, H/2
    pos, right, up, fwd = c2w[:3,3], c2w[:3,0], c2w[:3,1], -c2w[:3,2]
    R = np.stack([right,-up,fwd],0); P = (xyz-pos)@R.T
    z = P[:,2]; m = z>0.05
    Pm, Cm, zm, sm = P[m], col[m], z[m], sky[m]
    u = (fx*Pm[:,0]/zm+cx).astype(np.int32); v = (fy*Pm[:,1]/zm+cy).astype(np.int32)
    ok = (u>=0)&(u<W)&(v>=0)&(v<H); u,v,Cm,zm,sm = u[ok],v[ok],Cm[ok],zm[ok],sm[ok]
    img = np.zeros((H,W,3),np.uint8); msk = np.zeros((H,W),np.uint8)
    if sm.any():                                       # TIER 1: SKY = sparse backdrop, BIG dilate to fill, painted first
        si, ss = _splat(u[sm], v[sm], Cm[sm], zm[sm], 9)
        img[ss>0] = si[ss>0]; msk[ss>0] = 255
    g = ~sm                                            # TIER 2: everything else z-buffered OVER the sky, crisp (2px)
    if g.any():
        gi, gs = _splat(u[g], v[g], Cm[g], zm[g], 2)
        img[gs>0] = gi[gs>0]; msk[gs>0] = 255
    K=np.array([[fx,0,cx],[0,fy,cy],[0,0,1]],np.float32); M=np.eye(4,dtype=np.float32); M[:3,:3]=R; M[:3,3]=-R@pos
    return img,msk,M.tolist(),K.tolist()
def interp_t(A,B,t0,t1,n):
    sl=Slerp([0,1],Rotation.from_matrix([A[:3,:3],B[:3,:3]])); out=[]
    for t in np.linspace(t0,t1,n):
        C=np.eye(4); C[:3,:3]=sl([t]).as_matrix()[0]; C[:3,3]=(1-t)*A[:3,3]+t*B[:3,3]; out.append(C)
    return out

def build(scene, seq):
    TD=f"{scene}/render_results/view0/traj0"; os.makedirs(TD,exist_ok=True)
    vw=cv2.VideoWriter(f"{TD}/render.mp4",cv2.VideoWriter_fourcc(*"mp4v"),16,(W,H))
    mw=cv2.VideoWriter(f"{TD}/render_mask.mp4",cv2.VideoWriter_fourcc(*"mp4v"),16,(W,H))
    extr,intr=[],[]; cov=[]
    for k,C in enumerate(seq):
        img,msk,M,K=render(C)
        if k==0: cv2.imwrite(f"{scene}/render_results/view0/start_frame.png",img)
        vw.write(img); mw.write(cv2.cvtColor(msk,cv2.COLOR_GRAY2BGR)); extr.append(M); intr.append(K); cov.append((msk>0).mean())
    vw.release(); mw.release()
    json.dump({"extrinsic":extr,"intrinsic":intr},open(f"{TD}/camera.json","w"))
    json.dump({"prompt":PROMPT},open(f"{TD}/traj_caption.json","w"))
    json.dump({"scene_type":"indoor"},open(f"{scene}/meta_info.json","w"))
    cv2.imwrite(f"{scene}/panorama.png",cv2.imread(PANO))
    return np.mean(cov)

manifest = {}
for g, cs in groups.items():
    if ONLY and g not in ONLY: continue
    ci = 0; manifest[g] = []
    for s in range(len(cs)-1):
        A, B = cs[s], cs[s+1]
        seglen = np.linalg.norm(B[:3,3]-A[:3,3])
        n = max(1, round(seglen/CLIP_M))
        for j in range(n):
            scene = f"{ROOT}/{PREFIX}_{g}{ci}"
            cov = build(scene, interp_t(A,B,j/n,(j+1)/n,KF))
            manifest[g].append(scene); ci += 1
    print(f"  group {g}: {len(manifest[g])} clips")
json.dump(manifest, open(f"{ROOT}/{PREFIX}_manifest.json","w"))
print(f"MANIFEST -> {ROOT}/{PREFIX}_manifest.json  total clips={sum(len(v) for v in manifest.values())}")
