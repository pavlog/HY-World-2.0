"""Bake a sharp UV texture onto the HW2 mesh by projecting the source views, export a textured GLB.
- cameras come from HW2's own gs.opencv.Parser (normalize=True) so they sit in the SAME space as the mesh
- xatlas UV unwrap -> nvdiffrast rasterizes UV space to get per-texel 3D position+normal
- each source image is projected onto the texels with a per-camera z-buffer (occlusion) + front-facing weight
- final mesh is mapped back to ORIGINAL metric space via inv(parser.transform)  (Z-up, real size)

  trellis_py _ws_bake_texture_glb.py <mesh.ply> <gs_data_dir> <out.glb> [tex=2048]
"""
import sys, os, json
import numpy as np, torch, trimesh, cv2
import nvdiffrast.torch as dr
import xatlas
from PIL import Image
sys.path.insert(0, "D:/HY-World-2.0/hyworld2/worldgen")
from gs.opencv import Parser

MESH, GS, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
TEX = int(sys.argv[4]) if len(sys.argv) > 4 else 2048
dev = "cuda"

# ---- cameras in mesh (normalized) space, exactly as the trainer used them ----
parser = Parser(data_dir=GS, factor=1, normalize=True, test_every=999999,
                downsample_pts_num=1_000_000, downsample_mode="geometry_aware",
                detect_anchor_candidates=False)
c2ws = parser.camtoworlds.astype(np.float32)                       # [V,4,4] normalized
names = parser.image_names; ids = parser.camera_ids; paths = parser.image_paths
print(f"[bake] {len(names)} cameras, tex={TEX}", flush=True)

# ---- mesh + normals (normalized space) ----
m = trimesh.load(MESH); V = np.asarray(m.vertices, np.float32); F = np.asarray(m.faces, np.int32)
m.vertex_normals  # force compute
VN = np.asarray(m.vertex_normals, np.float32)
VC = (np.asarray(m.visual.vertex_colors)[:, :3].astype(np.float32) / 255.0
      if getattr(m.visual, "vertex_colors", None) is not None else np.full((len(V), 3), 0.5, np.float32))
print(f"[bake] mesh {len(V)} verts / {len(F)} faces", flush=True)

# ---- UV unwrap ----
vmap, idx, uv = xatlas.parametrize(V, F)             # vmap:[N2] old idx, idx:[F,3], uv:[N2,2]
V2 = V[vmap]; VN2 = VN[vmap]; F2 = idx.astype(np.int32)
print(f"[bake] xatlas -> {len(V2)} uv-verts", flush=True)

VC2 = VC[vmap]
Vt = torch.tensor(V2, device=dev); VNt = torch.tensor(VN2, device=dev); Ft = torch.tensor(F2, device=dev)
VCt = torch.tensor(VC2, device=dev)
uvt = torch.tensor(uv, device=dev, dtype=torch.float32)

# ---- rasterize UV space -> per-texel world pos + normal ----
glctx = dr.RasterizeCudaContext()
uv_clip = torch.cat([uvt * 2 - 1, torch.zeros_like(uvt[:, :1]), torch.ones_like(uvt[:, :1])], 1)[None]
rast, _ = dr.rasterize(glctx, uv_clip.contiguous(), Ft, (TEX, TEX))
Ptex, _ = dr.interpolate(Vt[None], rast, Ft)         # [1,TEX,TEX,3]
Ntex, _ = dr.interpolate(VNt[None], rast, Ft)
VCtex, _ = dr.interpolate(VCt[None], rast, Ft)        # TSDF vertex-color fallback per texel
texmask = (rast[..., 3:4] > 0).float()               # [1,TEX,TEX,1]
P = Ptex[0]; N = torch.nn.functional.normalize(Ntex[0], dim=-1); M = texmask[0, ..., 0] > 0
Pf = P.reshape(-1, 3); Nf = N.reshape(-1, 3); valid0 = M.reshape(-1)
print(f"[bake] texels filled: {int(valid0.sum())}/{TEX*TEX}", flush=True)

accum = torch.zeros(TEX * TEX, 3, device=dev); wsum = torch.zeros(TEX * TEX, 1, device=dev)
TOL = 0.01 * float(np.linalg.norm(V2.max(0) - V2.min(0)))   # occlusion tol ~1% of diag

for i in range(len(names)):
    c2w = torch.tensor(c2ws[i], device=dev); w2c = torch.linalg.inv(c2w)
    R = w2c[:3, :3]; t = w2c[:3, 3]; cam = c2w[:3, 3]
    K = torch.tensor(parser.Ks_dict[ids[i]], device=dev, dtype=torch.float32)
    W, H = parser.imsize_dict[ids[i]]
    img = cv2.cvtColor(cv2.imread(paths[i]), cv2.COLOR_BGR2RGB)
    if img.shape[1] != W or img.shape[0] != H: img = cv2.resize(img, (W, H))
    imgt = torch.tensor(img, device=dev, dtype=torch.float32) / 255.0          # [H,W,3]

    Pc = Pf @ R.T + t; z = Pc[:, 2]
    px = K[0, 0] * Pc[:, 0] / z + K[0, 2]; py = K[1, 1] * Pc[:, 1] / z + K[1, 2]
    vdir = torch.nn.functional.normalize(cam[None] - Pf, dim=-1)
    face = (Nf * vdir).sum(-1)                                                  # front-facing cosine
    inb = valid0 & (z > 1e-4) & (px >= 0) & (px < W - 1) & (py >= 0) & (py < H - 1) & (face > 0.05)
    if inb.sum() == 0: continue
    # per-camera z-buffer over texels -> occlusion
    pxl = (py.round().long().clamp(0, H - 1) * W + px.round().long().clamp(0, W - 1))
    zbuf = torch.full((H * W,), float("inf"), device=dev)
    zbuf.scatter_reduce_(0, pxl[inb], z[inb], reduce="amin", include_self=True)
    vis = inb & (z <= zbuf[pxl] + TOL)
    if vis.sum() == 0: continue
    # bilinear sample color
    gx = (px[vis] / (W - 1) * 2 - 1); gy = (py[vis] / (H - 1) * 2 - 1)
    grid = torch.stack([gx, gy], -1)[None, None]                                # [1,1,n,2]
    col = torch.nn.functional.grid_sample(imgt.permute(2, 0, 1)[None], grid, align_corners=True)[0, :, 0].T
    w = (face[vis] / (z[vis] ** 2)).clamp(min=0)[:, None]
    accum.index_add_(0, vis.nonzero(as_tuple=True)[0], col * w)
    wsum.index_add_(0, vis.nonzero(as_tuple=True)[0], w)
    if (i + 1) % 80 == 0: print(f"[bake] {i+1}/{len(names)} views", flush=True)

# seen texels -> sharp projected color; unseen mesh texels -> TSDF vertex-color fallback (no black)
vcol = VCtex[0].reshape(-1, 3)
atlas = torch.where(wsum > 0, accum / wsum.clamp(min=1e-8), vcol).reshape(TEX, TEX, 3)
img = atlas.clamp(0, 1).cpu().numpy().astype(np.float32)
valid = M.cpu().numpy()                               # all mesh texels now have color (projected or vcol)
seen_pct = 100 * (wsum.reshape(TEX, TEX) > 0).cpu().numpy().sum() / max(1, M.cpu().numpy().sum())
# island padding: grow valid colors outward (avg of valid neighbors) to kill black-edge bleed at UV seams
k = np.ones((3, 3), np.float32)
for _ in range(48):
    inv = ~valid
    if inv.sum() == 0: break
    vm = valid.astype(np.float32)
    num = cv2.filter2D(img * vm[..., None], -1, k); den = cv2.filter2D(vm, -1, k)
    grow = (den > 0) & inv
    fill = num / np.maximum(den, 1e-6)[..., None]
    img[grow] = fill[grow]; valid = valid | grow
atlas = (np.clip(img, 0, 1) * 255).astype(np.uint8)
atlas = np.flipud(atlas).copy()                       # nvdiffrast bottom-left -> image top-left
uv_img = uv.copy(); uv_img[:, 1] = 1 - uv_img[:, 1]   # match the flip
filled = valid
Image.fromarray(atlas).save(OUT.replace(".glb", "_albedo.png"))
print(f"[bake] atlas saved, directly seen {seen_pct:.1f}% of mesh texels (rest = edge-padded)", flush=True)

# ---- back to ORIGINAL metric space (Z-up, real size) via inverse parser transform ----
Tinv = np.linalg.inv(parser.transform.astype(np.float64))
Vh = np.concatenate([V2.astype(np.float64), np.ones((len(V2), 1))], 1)
V_orig = (Vh @ Tinv.T)[:, :3]
ext = V_orig.max(0) - V_orig.min(0)
print(f"[bake] metric bbox extent (m): {ext.round(2)}", flush=True)

mat = trimesh.visual.material.PBRMaterial(baseColorTexture=Image.fromarray(atlas), metallicFactor=0.0, roughnessFactor=1.0)
vis = trimesh.visual.TextureVisuals(uv=uv_img, material=mat)
out = trimesh.Trimesh(vertices=V_orig, faces=F2, visual=vis, process=False)
out.export(OUT)
print(f"[bake] DONE -> {OUT}", flush=True)
