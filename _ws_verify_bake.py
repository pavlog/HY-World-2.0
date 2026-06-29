"""Render the baked texture from a SOURCE camera via nvdiffrast and dump alongside the real frame.
If they match -> bake/UV are correct (any chaos is a GLB/Blender import convention issue)."""
import sys, numpy as np, torch, trimesh, cv2, xatlas
import nvdiffrast.torch as dr
sys.path.insert(0, "D:/HY-World-2.0/hyworld2/worldgen")
from gs.opencv import Parser
dev = "cuda"
MESH = "D:/_world_hangar/_ws_gs_native/ply/fuse_filled.ply"
GS = "D:/_world_hangar/_ws_lean/scene/gs_data"
ATLAS = "D:/_world_hangar/_ws_gs_native/hangar_filled_albedo.png"
parser = Parser(data_dir=GS, factor=1, normalize=True, test_every=999999,
                downsample_pts_num=1_000_000, downsample_mode="geometry_aware", detect_anchor_candidates=False)
m = trimesh.load(MESH, process=False); V = np.asarray(m.vertices, np.float32); F = np.asarray(m.faces, np.int32)
vmap, idx, uv = xatlas.parametrize(V, F)
V2 = V[vmap]; F2 = idx.astype(np.int32)
uv_img = uv.copy(); uv_img[:, 1] = 1 - uv_img[:, 1]      # SAME flip as the baker export
atlas = cv2.cvtColor(cv2.imread(ATLAS), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
tex = torch.tensor(atlas, device=dev)[None]              # [1,H,W,3]
Vt = torch.tensor(V2, device=dev); Ft = torch.tensor(F2, device=dev); uvt = torch.tensor(uv_img, device=dev)
glctx = dr.RasterizeCudaContext()

for i in [0, 200, 400]:
    c2w = torch.tensor(parser.camtoworlds[i], device=dev, dtype=torch.float32); w2c = torch.linalg.inv(c2w)
    R = w2c[:3, :3]; t = w2c[:3, 3]
    K = torch.tensor(parser.Ks_dict[parser.camera_ids[i]], device=dev, dtype=torch.float32)
    W, H = parser.imsize_dict[parser.camera_ids[i]]
    Pc = Vt @ R.T + t; Z = Pc[:, 2].clamp(min=1e-4)
    n, f = 0.01, 50.0
    cx = (2 * (K[0, 0] * Pc[:, 0] + K[0, 2] * Z) / W - Z)
    cy = (Z - 2 * (K[1, 1] * Pc[:, 1] + K[1, 2] * Z) / H)   # flip y for GL
    cz = (2 * (Z - n) / (f - n) - 1) * Z
    clip = torch.stack([cx, cy, cz, Z], -1)[None]
    rast, _ = dr.rasterize(glctx, clip.contiguous(), Ft, (H, W))
    uvi, _ = dr.interpolate(uvt[None], rast, Ft)
    col = dr.texture(tex, uvi)[0]                        # [H,W,3]
    col = (col.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    col[rast[0, ..., 3].cpu().numpy() == 0] = 90
    real = cv2.cvtColor(cv2.imread(parser.image_paths[i]), cv2.COLOR_BGR2RGB)
    real = cv2.resize(real, (W, H))
    cmp = np.concatenate([real, col], 1)                 # left=real photo, right=baked-on-mesh
    cv2.imwrite(f"D:/_world_hangar/_ws_verify_{i}.png", cv2.cvtColor(cmp, cv2.COLOR_RGB2BGR))
    print(f"[verify] cam {i} ({parser.image_names[i]}) -> _ws_verify_{i}.png", flush=True)
print("DONE", flush=True)
