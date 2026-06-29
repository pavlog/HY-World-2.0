import open3d as o3d, numpy as np, json, os
from PIL import Image
D=r"D:/HY-World-2.0/_out_cockpit_single/wm_input/20260623_020811"
cp=json.load(open(os.path.join(D,"camera_params.json")))
K=np.array(cp["intrinsics"][0]["matrix"]); fx,fy,cx,cy=K[0,0],K[1,1],K[0,2],K[1,2]
c2w=np.array(cp["extrinsics"][0]["matrix"]); w2c=np.linalg.inv(c2w)
depth=np.load(os.path.join(D,"depth","depth_0000.npy")).astype(np.float32); H,W=depth.shape
color=np.array(Image.open(r"E:/MyGame/Game007Trailer/ForDepthMesh.png").convert("RGB").resize((W,H)),dtype=np.uint8)
it=o3d.camera.PinholeCameraIntrinsic(W,H,fx,fy,cx,cy)
vol=o3d.pipelines.integration.ScalableTSDFVolume(voxel_length=0.005, sdf_trunc=0.02,
    color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
rgbd=o3d.geometry.RGBDImage.create_from_color_and_depth(
    o3d.geometry.Image(np.ascontiguousarray(color)), o3d.geometry.Image(depth),
    depth_scale=1.0, depth_trunc=5.0, convert_rgb_to_intensity=False)
vol.integrate(rgbd, it, w2c)
mesh=vol.extract_triangle_mesh(); mesh.compute_vertex_normals()
import copy
ti,counts,_=mesh.cluster_connected_triangles(); ti=np.asarray(ti); counts=np.asarray(counts)
keep=counts>=max(2000,int(0.01*counts.max())); m2=copy.deepcopy(mesh)
m2.remove_triangles_by_mask(~keep[ti]); m2.remove_unreferenced_vertices()
print("RAW",len(mesh.vertices),"v",len(mesh.triangles),"t | CLEAN",len(m2.vertices),"v",len(m2.triangles),"t")
o3d.io.write_triangle_mesh(os.path.join(D,"_tsdf_cockpit.ply"), m2)
print("SAVED", os.path.join(D,"_tsdf_cockpit.ply"))
