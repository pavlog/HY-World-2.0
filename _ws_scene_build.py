"""Build a minimal WorldStereo scene-dir from OUR geometry (a colored cloud) — no 17B, just rendering.
Produces: render.mp4 (pcd RGB render along a short forward trajectory) + render_mask.mp4 (coverage) +
camera.json {extrinsic w2c[f,4,4], intrinsic[f,3,3]} + start_frame.png + panorama.png + meta_info.json.
Software point-splat (z-buffer painter), no GL. Feeds the stage-3 standalone test later."""
import sys, os, json, numpy as np, cv2
sys.path.insert(0, r'E:/MyGame/AIWorldStudio/docs/research/worldgen-3d-pipeline/pipeline')
import _cloud

CLOUD = sys.argv[1] if len(sys.argv) > 1 else r'D:/_pipeline_test/rf2_cloud.npz'
PANO  = sys.argv[2] if len(sys.argv) > 2 else r'E:/MyGame/Game007Trailer/NoRuddersBlueBackPanoX2NoBack.png'
SCENE = r'D:/HY-World-2.0/_ws_scene/case_cockpit'
VIEW, TRAJ = 'view0', 'traj0'
NF, W, H = 6, 832, 480     # KEYFRAMES = num_frames//4+1 = 21//4+1 = 6. WS is a keyframe pipeline:
#                            render_video/mask/camera are the 6 latent-aligned keyframes (NOT 21 dense frames).
#                            pipeline (line 219) routes 6-frame render to keyframe_vae_encode -> 6 latent -> matches
#                            the 6-frame generation latent + the 6-frame mask/camera add_inputs. Output video = 21f.
os.makedirs(f'{SCENE}/render_results/{VIEW}/{TRAJ}', exist_ok=True)

c = _cloud.load_cloud(CLOUD)
xyz = c['xyz'].astype(np.float32) * np.array([1, -1, -1], np.float32)   # un-flip -> camera frame (z fwd)
col = c['color'][:, :3].astype(np.uint8)
K0 = c['intrinsics']
if K0 is None:
    f = 0.9 * max(W, H); K0 = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]], np.float32)
else:                                                  # scale K from the cloud's native shape to (W,H)
    Hc, Wc = c['shape']; K0 = np.asarray(K0, np.float64).copy()
    K0[0] *= W / Wc; K0[1] *= H / Hc
zmed = float(np.median(xyz[:, 2][xyz[:, 2] > 0]))
print(f'cloud n={len(xyz)} median_depth={zmed:.2f}m K0 fx={K0[0,0]:.1f}')

def look_w2c(eye, fwd):                                 # build w2c (OpenCV: x right, y down, z fwd)
    fwd = fwd / np.linalg.norm(fwd)
    right = np.cross(np.array([0, 1, 0.0]), fwd); right /= np.linalg.norm(right)
    up = np.cross(fwd, right)
    R = np.stack([right, -up, fwd], 0)                  # world->cam rows; -up since y is down
    t = -R @ eye
    M = np.eye(4, dtype=np.float32); M[:3, :3] = R; M[:3, 3] = t
    return M

# trajectory: gentle forward dolly into the scene + slight yaw (camera at origin = cloud's own view at f0)
extr, intr = [], []
for i in range(NF):
    a = i / (NF - 1)
    eye = np.array([0.25 * np.sin(a * np.pi) * zmed * 0.1, 0, a * 0.35 * zmed], np.float32)  # creep forward
    yaw = (a - 0.5) * 0.15
    fwd = np.array([np.sin(yaw), 0, np.cos(yaw)], np.float32)
    extr.append(look_w2c(eye, fwd)); intr.append(K0)

def render(w2c):
    P = (xyz @ w2c[:3, :3].T) + w2c[:3, 3]              # to camera
    z = P[:, 2]; m = z > 0.05
    Pm, Cm, zm = P[m], col[m], z[m]
    u = (K0[0, 0] * Pm[:, 0] / zm + K0[0, 2]).astype(np.int32)
    v = (K0[1, 1] * Pm[:, 1] / zm + K0[1, 2]).astype(np.int32)
    ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v, Cm, zm = u[ok], v[ok], Cm[ok], zm[ok]
    order = np.argsort(-zm)                             # far first, near overwrites
    img = np.zeros((H, W, 3), np.uint8); msk = np.zeros((H, W), np.uint8); dep = np.zeros((H, W), np.float32)
    img[v[order], u[order]] = Cm[order]; msk[v[order], u[order]] = 255
    dep[v[order], u[order]] = zm[order]                 # z-buffer: near overwrites (last write = nearest)
    img = cv2.dilate(img, np.ones((2, 2), np.uint8)); msk = cv2.dilate(msk, np.ones((2, 2), np.uint8))
    return img, msk, dep                                # dep stays sparse (raw splat depth, metric m)


def sky_and_holes(msk):
    """Split the empty (coverage==0) region into true sky (connected to the frame border = open space)
    vs interior holes (gaps in sparse splat coverage, surrounded by geometry). Border-flood via CCs."""
    empty = (msk == 0).astype(np.uint8)
    n, lab = cv2.connectedComponents(empty, connectivity=4)
    border = set(lab[0, :]) | set(lab[-1, :]) | set(lab[:, 0]) | set(lab[:, -1])
    border.discard(0)                                   # label 0 = the geometry (non-empty) region
    sky = np.isin(lab, list(border)) & (empty == 1)     # empty AND reaches the border = sky/open
    holes = (empty == 1) & ~sky                         # empty but enclosed by geometry = fill-me holes
    return (sky * 255).astype(np.uint8), (holes * 255).astype(np.uint8)

TD = f'{SCENE}/render_results/{VIEW}/{TRAJ}'
_VW = lambda name: cv2.VideoWriter(f'{TD}/{name}', cv2.VideoWriter_fourcc(*'mp4v'), 16, (W, H))
vw, mw = _VW('render.mp4'), _VW('render_mask.mp4')
sw, hw, dw = _VW('sky_mask.mp4'), _VW('holes_mask.mp4'), _VW('depth_vis.mp4')   # extra per-keyframe artifacts
depths = []
for i, w2c in enumerate(extr):
    img, msk, dep = render(w2c)
    sky, holes = sky_and_holes(msk)
    depths.append(dep)
    if i == 0:
        cv2.imwrite(f'{SCENE}/render_results/{VIEW}/start_frame.png', img)
    vw.write(img); mw.write(cv2.cvtColor(msk, cv2.COLOR_GRAY2BGR))
    sw.write(cv2.cvtColor(sky, cv2.COLOR_GRAY2BGR)); hw.write(cv2.cvtColor(holes, cv2.COLOR_GRAY2BGR))
    # depth_vis: normalize valid (>0) depth to 0..255 for glanceable preview (TURBO colormap)
    dv = np.zeros_like(dep, np.uint8); valid = dep > 0
    if valid.any():
        dn = dep.copy(); dn[valid] = (dn[valid] - dn[valid].min()) / (np.ptp(dn[valid]) + 1e-6) * 255
        dv = dn.astype(np.uint8)
    dw.write(cv2.applyColorMap(dv, cv2.COLORMAP_TURBO))
for _w in (vw, mw, sw, hw, dw):
    _w.release()
np.save(f'{TD}/depth.npy', np.stack(depths).astype(np.float32))   # [NF,H,W] metric depth (m), 0=no geometry
print(f'artifacts: render/mask/sky/holes/depth_vis .mp4 + depth.npy [{len(depths)},{H},{W}]')

json.dump({'extrinsic': [e.tolist() for e in extr], 'intrinsic': [np.asarray(K0).tolist()] * NF},
          open(f'{SCENE}/render_results/{VIEW}/{TRAJ}/camera.json', 'w'))
cv2.imwrite(f'{SCENE}/panorama.png', cv2.imread(PANO))
json.dump({'scene_type': 'indoor'}, open(f'{SCENE}/meta_info.json', 'w'))
cov = (cv2.imread(f'{SCENE}/render_results/{VIEW}/start_frame.png').sum(-1) > 0).mean()
print(f'WS scene built -> {SCENE}  (start_frame coverage {cov*100:.0f}%)')
