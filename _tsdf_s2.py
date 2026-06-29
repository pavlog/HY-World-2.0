import open3d as o3d, numpy as np, json, glob, os
from PIL import Image
D=r"D:/HY-World-2.0/_out_s2/wm_input_s2/20260623_015847"
cp=json.load(open(os.path.join(D,"camera_params.json")))
intr=cp["intrinsics"]; ex=cp["extrinsics"]; N=cp["num_cameras"]
depths=sorted(glob.glob(os.path.join(D,"depth","*.npy")))
imgs=sorted(glob.glob("E:/MyGame/Game007Trailer/wm_input_s2/*.png"))
d0=np.load(depths[0]); H,W=d0.shape
vol=o3d.pipelines.integration.ScalableTSDFVolume(voxel_length=0.006, sdf_trunc=0.024,
    color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
for i in range(N):
    K=np.array(intr[i]["matrix"]); fx,fy,cx,cy=K[0,0],K[1,1],K[0,2],K[1,2]
    it=o3d.camera.PinholeCameraIntrinsic(W,H,fx,fy,cx,cy)
    w2c=np.linalg.inv(np.array(ex[i]["matrix"]))
    depth=np.load(depths[i]).astype(np.float32)
    color=np.array(Image.open(imgs[i]).convert("RGB").resize((W,H)),dtype=np.uint8)
    rgbd=o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(np.ascontiguousarray(color)), o3d.geometry.Image(depth),
        depth_scale=1.0, depth_trunc=6.0, convert_rgb_to_intensity=False)
    vol.integrate(rgbd, it, w2c)
mesh=vol.extract_triangle_mesh(); mesh.compute_vertex_normals()
import copy
ti,counts,_=mesh.cluster_connected_triangles(); ti=np.asarray(ti); counts=np.asarray(counts)
keep=counts>=max(2000,int(0.01*counts.max())); m2=copy.deepcopy(mesh)
m2.remove_triangles_by_mask(~keep[ti]); m2.remove_unreferenced_vertices()
print("RAW",len(mesh.vertices),"v",len(mesh.triangles),"t | CLEAN",len(m2.vertices),"v",len(m2.triangles),"t | clusters",int(keep.sum()),"/",len(counts))
o3d.io.write_triangle_mesh(os.path.join(D,"_tsdf_clean.ply"), m2)
print("SAVED", os.path.join(D,"_tsdf_clean.ply"))
