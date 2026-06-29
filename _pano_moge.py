import sys, os, torch, numpy as np, cv2
sys.path.insert(0, r'D:/HY-World-2.0/hyworld2/worldgen')
from src.panorama_utils import pred_pano_depth, convert_rgbd2mesh_panorama
from moge.model.v2 import MoGeModel
from PIL import Image
import open3d as o3d

PANO = sys.argv[1] if len(sys.argv) > 1 else r'D:/HY-World-2.0/examples/worldgen/case000/panorama.png'
OUT  = sys.argv[2] if len(sys.argv) > 2 else r'D:/HY-World-2.0/_pano_out'
TAG  = sys.argv[3] if len(sys.argv) > 3 else 'pano_moge'
os.makedirs(OUT, exist_ok=True)

def tnp(x):
    if torch.is_tensor(x):
        return x.detach().float().cpu().numpy()
    return np.asarray(x, dtype=np.float32)

print('loading MoGe-2...')
moge = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").cuda().eval()
img = Image.open(PANO).convert('RGB')
RESIZE_TO = int(os.environ.get('PANO_RESIZE_TO', '1920'))
print(f'pred_pano_depth... (resize_to={RESIZE_TO})')
pred = pred_pano_depth(moge, img, scale=1.0, resize_to=RESIZE_TO)
dist, rays, mask = pred['distance'], pred['rays'], pred['mask']

dd = tnp(dist)                       # (H, W)
H, W = dd.shape
print('distance', dd.shape, 'rays', tuple(rays.shape), 'mask', None if mask is None else tuple(np.shape(tnp(mask))))
_v = dd[np.isfinite(dd) & (dd > 0)]
print(f'METRIC PRIOR (meters): min={_v.min():.3f}  p05={np.percentile(_v,5):.3f}  median={np.median(_v):.3f}  p95={np.percentile(_v,95):.3f}  max={_v.max():.3f}')

# equirect depth viz
finite = np.isfinite(dd)
lo, hi = np.nanmin(dd[finite]), np.nanmax(dd[finite])
dn = np.clip((np.nan_to_num(dd, nan=lo) - lo) / (hi - lo + 1e-6), 0, 1)
cv2.imwrite(os.path.join(OUT, f'{TAG}_depth_color.png'),
            cv2.applyColorMap((dn * 255).astype(np.uint8), cv2.COLORMAP_TURBO))
np.save(os.path.join(OUT, f'{TAG}_distance.npy'), dd)

# rgb resized to depth grid, [0,1]
rgb_np = np.asarray(Image.open(PANO).convert('RGB').resize((W, H)), dtype=np.float32) / 255.0
rgb_t  = torch.from_numpy(rgb_np).float()
dist_t = torch.from_numpy(dd).float()
rays_t = rays.detach().float().cpu() if torch.is_tensor(rays) else torch.from_numpy(np.asarray(rays, np.float32))
# honor a remove-bg alpha matte on the equirect (cull unwanted region from the surround)
_src = Image.open(PANO)
_alpha = np.asarray(_src.convert('RGBA'))[..., 3] if (_src.mode in ('RGBA', 'LA') or 'transparency' in _src.info) else None
m = tnp(mask).astype(bool) if mask is not None else np.ones((H, W), bool)
if _alpha is not None:
    am = np.asarray(Image.fromarray(_alpha).resize((W, H), Image.NEAREST)) > 127
    print(f'alpha matte applied (pano): {int((m & am).sum())}/{m.size} kept')
    m = m & am
excl = torch.from_numpy(~m).bool()           # convert wants True = EXCLUDE

print('convert_rgbd2mesh_panorama...')
mesh = convert_rgbd2mesh_panorama(
    rgb_t, dist_t, rays_t,
    excluded_region_mask=excl,
    connect_boundary_max_dist=0.5,
    connect_boundary_repeat_times=2,
    device='cuda',
)
glb = os.path.join(OUT, f'{TAG}_surround.glb')
o3d.io.write_triangle_mesh(glb, mesh)
print('MESH saved ->', glb, '| verts', len(mesh.vertices), 'tris', len(mesh.triangles))
