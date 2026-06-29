import sys, os, numpy as np, cv2
from PIL import Image
sys.path.insert(0, r'D:/Lyra2/Lyra-2/lyra_2/_src/inference/depth_anything_3/src')
PANO=r'D:/HY-World-2.0/examples/worldgen/case000/panorama.png'
OUT=r'D:/HY-World-2.0/_pano_out/cube'; os.makedirs(OUT, exist_ok=True)
pano=np.array(Image.open(PANO).convert('RGB')); H,W=pano.shape[:2]
F=768  # face resolution
# face basis: (forward, right, up)
faces={
 'front':((0,0,1),(1,0,0),(0,1,0)),
 'back' :((0,0,-1),(-1,0,0),(0,1,0)),
 'right':((1,0,0),(0,0,-1),(0,1,0)),
 'left' :((-1,0,0),(0,0,1),(0,1,0)),
 'up'   :((0,1,0),(1,0,0),(0,0,-1)),
 'down' :((0,-1,0),(1,0,0),(0,0,1)),
}
def sample_face(fwd,right,up):
    fwd=np.array(fwd,float); right=np.array(right,float); up=np.array(up,float)
    j,i=np.meshgrid(np.arange(F),np.arange(F))
    x=(2*(j+0.5)/F-1); y=(2*(i+0.5)/F-1)   # 90deg FOV -> tan45=1
    d=fwd[None,None,:]+x[...,None]*right[None,None,:]+y[...,None]*up[None,None,:]
    d=d/np.linalg.norm(d,axis=-1,keepdims=True)
    lon=np.arctan2(d[...,0],d[...,2])         # 0 at +Z
    lat=np.arcsin(np.clip(d[...,1],-1,1))
    u=((lon/(2*np.pi))%1.0)*W
    v=(0.5-lat/np.pi)*H
    face=cv2.remap(pano,(u).astype(np.float32),(v).astype(np.float32),cv2.INTER_LINEAR,borderMode=cv2.BORDER_WRAP)
    return face
# build + save faces
imgs={}
for nm,(fwd,r,u) in faces.items():
    fc=sample_face(fwd,r,u); imgs[nm]=fc
    Image.fromarray(fc).save(os.path.join(OUT,f'face_{nm}.png'))
print('6 faces saved', F,'x',F)
# DA3 multi-view on all 6 faces (consistent depth)
from depth_anything_3.api import DepthAnything3
import torch
from safetensors.torch import load_file
m=DepthAnything3(model_name='da3-large')
m.load_state_dict(load_file(r'D:/Models/models/depthanything3/da3_large.safetensors'),strict=False)
m=m.to('cuda').eval()
paths=[os.path.join(OUT,f'face_{nm}.png') for nm in faces]
pred=m.inference(paths, process_res=768, process_res_method='upper_bound_resize', align_to_input_extrinsics=False)
dep=np.asarray(pred.depth)  # (6,h,w) ?
print('DA3 multiview depth shape', dep.shape, 'range %.2f-%.2f'%(float(dep.min()),float(dep.max())))
np.save(os.path.join(OUT,'cube_depths.npy'), dep)
for i,nm in enumerate(faces):
    d=dep[i]; dn=(d-d.min())/(d.max()-d.min()+1e-6)
    cv2.imwrite(os.path.join(OUT,f'depth_{nm}.png'),(dn*255).astype(np.uint8))
print('per-face depths saved')
