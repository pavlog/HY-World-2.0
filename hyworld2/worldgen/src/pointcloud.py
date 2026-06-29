"""pytorch3d-FREE drop-in for HW2's point renderer.

Original used pytorch3d's PointsRasterizer/AlphaCompositor; there is no prebuilt pytorch3d wheel for our
torch 2.11+cu128 and a Windows source build is fragile, so the GPU point rasterizer is reimplemented here as a
plain torch radius-aware z-buffer splat. Public contract is byte-compatible with the original:

  point_rendering(K, w2cs, points, colors, device, h, w, background_color, render_radius, points_per_pixel,
                  return_depth) ->
     return_depth=False -> (rgbs [F,C,H,W], masks [F,1,H,W])   masks: 1 where EMPTY (zbuf==-1), 0 where covered
     return_depth=True  -> (rgbs [F,H,W,C] (un-rearranged),    depth [F,1,H,W])  depth: camera-space z, -1 empty

  multi_gpu_point_rendering(...) -> (renders [F,3,H,W], mask [F,1,H,W])  (single-GPU short-circuits dist)

Approximations vs pytorch3d: alpha-compositing of the `points_per_pixel` nearest points is replaced by the
single front-most point (z-buffer winner). render_radius is interpreted in pytorch3d's NDC units (half-extent =
min(h,w)/2 px) and splatted as an integer-pixel disk. Adequate for the conditioning/guidance reprojection the
memory bank uses (depth_alignment guided depth; traj_render memory images).

`points_padding` and `depth2pcd` are kept verbatim (no pytorch3d).
"""
import contextlib
import os
import sys

import einops
import numpy as np
import torch
import torch.distributed as dist
import torchvision.transforms as transforms

from .general_utils import split_n_into_d_parts


def points_padding(points):
    padding = torch.ones_like(points)[..., 0:1]
    points = torch.cat([points, padding], dim=-1)
    return points


@contextlib.contextmanager
def suppress_stdout_stderr():
    with open(os.devnull, 'w') as devnull:
        old_stdout_fd = os.dup(sys.stdout.fileno())
        old_stderr_fd = os.dup(sys.stderr.fileno())
        try:
            os.dup2(devnull.fileno(), sys.stdout.fileno())
            os.dup2(devnull.fileno(), sys.stderr.fileno())
            yield
        finally:
            os.dup2(old_stdout_fd, sys.stdout.fileno())
            os.dup2(old_stderr_fd, sys.stderr.fileno())
            os.close(old_stdout_fd)
            os.close(old_stderr_fd)


def _disk_offsets(rad_px, device):
    """[(dx,dy)] integer offsets inside a disk of pixel radius rad_px (>=1 -> center only)."""
    if rad_px <= 0:
        return torch.zeros((1, 2), dtype=torch.long, device=device)
    a = torch.arange(-rad_px, rad_px + 1, device=device)
    ys, xs = torch.meshgrid(a, a, indexing='ij')
    keep = (xs * xs + ys * ys) <= rad_px * rad_px
    return torch.stack([xs[keep], ys[keep]], dim=1).long()  # [D,2] (dx,dy)


def point_rendering(K, w2cs, points, colors, device, h, w, background_color=[0, 0, 0],
                    render_radius=0.008, points_per_pixel=8, return_depth=False):
    """torch radius-aware z-buffer splat. See module docstring for the exact contract.
    K:[F,3,3]  w2cs:[F,4,4] (opencv world->cam)  points:[N,3]  colors:[N,C]."""
    K = torch.as_tensor(K, dtype=torch.float32, device=device)
    w2cs = torch.as_tensor(w2cs, dtype=torch.float32, device=device)
    pts = torch.as_tensor(points, dtype=torch.float32, device=device)
    cols = torch.as_tensor(colors, dtype=torch.float32, device=device)
    if cols.ndim == 1:
        cols = cols[:, None]
    F = w2cs.shape[0]
    N = pts.shape[0]
    C = cols.shape[1]
    bg = background_color if len(background_color) == C else [0.0] * C
    bg = torch.tensor(bg, dtype=torch.float32, device=device)

    rad_px = int(round(render_radius * min(h, w) / 2.0))      # pytorch3d NDC radius -> pixels
    off = _disk_offsets(rad_px, device)                       # [D,2]
    D = off.shape[0]
    pts_h = torch.cat([pts, torch.ones((N, 1), device=device)], dim=1)  # [N,4]

    rgbs_out = bg.view(1, 1, 1, C).expand(F, h, w, C).clone()
    depth_out = torch.full((F, h, w), -1.0, dtype=torch.float32, device=device)

    for f in range(F):
        Pc = (w2cs[f] @ pts_h.T).T[:, :3]                     # [N,3] camera-space
        z = Pc[:, 2]
        u = K[f, 0, 0] * Pc[:, 0] / z + K[f, 0, 2]
        v = K[f, 1, 1] * Pc[:, 1] / z + K[f, 1, 2]
        ui = u.round().long()[:, None] + off[None, :, 0]      # [N,D]
        vi = v.round().long()[:, None] + off[None, :, 1]
        ui = ui.reshape(-1); vi = vi.reshape(-1)
        zz = z[:, None].expand(N, D).reshape(-1)
        pid = torch.arange(N, device=device)[:, None].expand(N, D).reshape(-1)
        ok = (zz > 1e-4) & (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
        ui, vi, zz, pid = ui[ok], vi[ok], zz[ok], pid[ok]
        if ui.numel() == 0:
            continue
        flat = vi * w + ui                                    # [M]
        depth_buf = torch.full((h * w,), float('inf'), device=device)
        depth_buf.scatter_reduce_(0, flat, zz, reduce='amin', include_self=True)
        won = zz <= depth_buf[flat] + 1e-6                    # nearest point(s) per pixel
        col_buf = bg.view(1, C).expand(h * w, C).clone()
        col_buf[flat[won]] = cols[pid[won]]                   # ties: arbitrary winner (fine)
        covered = torch.isfinite(depth_buf)
        rgbs_out[f] = col_buf.reshape(h, w, C)
        df = depth_buf.clone(); df[~covered] = -1.0
        depth_out[f] = df.reshape(h, w)

    if not return_depth:
        render_masks = (depth_out == -1).float()[:, None]                 # [F,1,H,W] 1=empty
        render_rgbs = rgbs_out.permute(0, 3, 1, 2).contiguous()           # [F,C,H,W]
        return render_rgbs, render_masks
    else:
        render_depth = depth_out[:, None].contiguous()                    # [F,1,H,W]
        return rgbs_out, render_depth                                     # rgbs [F,H,W,C] un-rearranged


def multi_gpu_point_rendering(image, Ks, w2cs, render_points, render_colors, image_h, image_w, device, device_num,
                              render_radius=0.008, points_per_pixel=20, slice_size=4, local_rank=0, replace_first_frame=True):
    image_tensor = (transforms.ToTensor()(image) * 2 - 1)[None].to(device)
    Ks_tensor = Ks if isinstance(Ks, torch.Tensor) else torch.tensor(Ks).float()
    w2cs_tensor = w2cs if isinstance(w2cs, torch.Tensor) else torch.tensor(w2cs).float()

    if device_num == 1:                                       # single-GPU: no dist, just render every slice
        renders, masks = [], []
        n = Ks_tensor.shape[0]
        for s in range(0, n, slice_size):
            r, m = point_rendering(K=Ks_tensor[s:s + slice_size], w2cs=w2cs_tensor[s:s + slice_size],
                                   points=render_points, colors=render_colors, h=image_h, w=image_w,
                                   render_radius=render_radius, points_per_pixel=points_per_pixel,
                                   device=device, background_color=[0, 0, 0])
            renders.append(r); masks.append(m)
        gather_pcd_renders = torch.cat(renders, dim=0).to(torch.float32)
        gather_pcd_mask = torch.cat(masks, dim=0).to(torch.float32)
        if replace_first_frame:
            gather_pcd_renders[0:1] = image_tensor
            gather_pcd_mask[0:1] = 0
        return gather_pcd_renders, gather_pcd_mask

    # multi-GPU path (unchanged from original; never hit in our single-rank runs)
    pcd_renders, pcd_mask = [], []
    n_per_gpu_list = split_n_into_d_parts(Ks_tensor.shape[0], device_num)
    cumsum_gpu_list = np.cumsum(n_per_gpu_list)
    if local_rank == 0:
        Ks_tensor = Ks_tensor[:cumsum_gpu_list[0]]
        w2cs_tensor = w2cs_tensor[:cumsum_gpu_list[0]]
    else:
        Ks_tensor = Ks_tensor[cumsum_gpu_list[local_rank - 1]:cumsum_gpu_list[local_rank]]
        w2cs_tensor = w2cs_tensor[cumsum_gpu_list[local_rank - 1]:cumsum_gpu_list[local_rank]]
    gather_pcd_renders_r = [torch.zeros((n_per_gpu_list[j], 1, image_h, image_w), dtype=torch.float32, device=device) for j in range(device_num)]
    gather_pcd_renders_g = [torch.zeros((n_per_gpu_list[j], 1, image_h, image_w), dtype=torch.float32, device=device) for j in range(device_num)]
    gather_pcd_renders_b = [torch.zeros((n_per_gpu_list[j], 1, image_h, image_w), dtype=torch.float32, device=device) for j in range(device_num)]
    gather_pcd_mask = [torch.zeros((n_per_gpu_list[j], 1, image_h, image_w), dtype=torch.float32, device=device) for j in range(device_num)]
    slice_times = w2cs_tensor.shape[0] // slice_size
    if w2cs_tensor.shape[0] % slice_size != 0:
        slice_times += 1
    for si in range(slice_times):
        pcd_renders_, pcd_mask_ = point_rendering(K=Ks_tensor[si * slice_size:(si + 1) * slice_size],
                                                  w2cs=w2cs_tensor[si * slice_size:(si + 1) * slice_size],
                                                  points=render_points, colors=render_colors,
                                                  h=image_h, w=image_w, render_radius=render_radius, points_per_pixel=points_per_pixel,
                                                  device=device, background_color=[0, 0, 0])
        pcd_renders.append(pcd_renders_)
        pcd_mask.append(pcd_mask_)
    pcd_renders = torch.cat(pcd_renders, dim=0).to(torch.float32)
    pcd_mask = torch.cat(pcd_mask, dim=0).to(torch.float32)
    dist.barrier()
    dist.all_gather(gather_pcd_renders_r, pcd_renders[:, 0:1].contiguous())
    dist.all_gather(gather_pcd_renders_g, pcd_renders[:, 1:2].contiguous())
    dist.all_gather(gather_pcd_renders_b, pcd_renders[:, 2:3].contiguous())
    dist.all_gather(gather_pcd_mask, pcd_mask)
    dist.barrier()
    gather_pcd_renders_r = torch.cat(gather_pcd_renders_r, dim=0)
    gather_pcd_renders_g = torch.cat(gather_pcd_renders_g, dim=0)
    gather_pcd_renders_b = torch.cat(gather_pcd_renders_b, dim=0)
    gather_pcd_renders = torch.cat([gather_pcd_renders_r, gather_pcd_renders_g, gather_pcd_renders_b], dim=1)
    gather_pcd_mask = torch.cat(gather_pcd_mask, dim=0)
    if replace_first_frame:
        gather_pcd_renders[0:1] = image_tensor
        gather_pcd_mask[0:1] = 0
    return gather_pcd_renders, gather_pcd_mask


def depth2pcd(w2c, K, points2d, depth, colors, mask):
    points3d = w2c.inverse() @ points_padding((K.inverse() @ points2d.T).T * depth.reshape(-1, 1)).T
    points3d = points3d.T[:, :3]
    points3d = points3d[mask.reshape(-1)]
    colors = colors[mask.reshape(-1)]

    return points3d, colors
