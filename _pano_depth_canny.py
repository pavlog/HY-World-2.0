import sys, numpy as np, cv2
from PIL import Image
sys.path.insert(0, r'D:/Lyra2/Lyra-2/lyra_2/_src/inference/depth_anything_3/src')
PANO=r'D:/HY-World-2.0/examples/worldgen/case000/panorama.png'
OUT=r'D:/HY-World-2.0/_pano_out'
import os; os.makedirs(OUT, exist_ok=True)
img=np.array(Image.open(PANO).convert('RGB')); H,W=img.shape[:2]
print('pano size', W,'x',H)
# --- Canny (360 edge map) ---
gray=cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
canny=cv2.Canny(gray, 80, 180)
cv2.imwrite(os.path.join(OUT,'pano_canny.png'), canny)
print('canny saved')
# --- DA3 depth on equirect ---
from depth_anything_3.api import DepthAnything3
import torch
from safetensors.torch import load_file
m=DepthAnything3(model_name='da3-large')
st=load_file(r'D:/Models/models/depthanything3/da3_large.safetensors')
m.load_state_dict(st, strict=False)
m=m.to('cuda').eval()
pred=m.inference([PANO], process_res=1024, process_res_method='upper_bound_resize', align_to_input_extrinsics=False)
d=np.asarray(pred.depth[0] if np.asarray(pred.depth).ndim==3 else pred.depth, dtype=np.float32)
np.save(os.path.join(OUT,'pano_depth.npy'), d)
dn=(d-d.min())/(d.max()-d.min()+1e-6)
cv2.imwrite(os.path.join(OUT,'pano_depth.png'), (dn*255).astype(np.uint8))
cv2.imwrite(os.path.join(OUT,'pano_depth_color.png'), cv2.applyColorMap((dn*255).astype(np.uint8), cv2.COLORMAP_TURBO))
print('depth saved, shape', d.shape, 'range %.2f-%.2f'%(d.min(),d.max()))
