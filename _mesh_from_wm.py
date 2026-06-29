import open3d as o3d, numpy as np, json, os
from PIL import Image
D=r"D:/HY-World-2.0/_out_cockpit_single/wm_input/20260623_020811"
depth=np.load(os.path.join(D,"depth","depth_0000.npy")).astype(np.float32)
H,W=depth.shape
cp=json.load(open(os.path.join(D,"camera_params.json")))
K=np.array(cp["intrinsics"][0]["matrix"]); fx,fy,cx,cy=K[0,0],K[1,1],K[0,2],K[1,2]
col=np.array(Image.open(r"E:/MyGame/Game007Trailer/wm_input/cockpit.png").convert("RGB").resize((W,H)),dtype=np.float32)/255.0
print("depth",depth.shape,"K fx,fy,cx,cy",round(fx,1),round(fy,1),round(cx,1),round(cy,1))

# ---------- A: depth -> unprojected dense mesh ----------
u,v=np.meshgrid(np.arange(W),np.arange(H))
Z=depth; X=(u-cx)*Z/fx; Y=(v-cy)*Z/fy
pts=np.stack([X,Y,Z],-1).reshape(-1,3); cols=col.reshape(-1,3)
idx=np.arange(H*W).reshape(H,W)
v00=idx[:-1,:-1]; v10=idx[1:,:-1]; v01=idx[:-1,1:]; v11=idx[1:,1:]
t1=np.stack([v00,v10,v11],-1).reshape(-1,3); t2=np.stack([v00,v11,v01],-1).reshape(-1,3)
tris=np.concatenate([t1,t2])
# mild depth-discontinuity cull (drop stretched tris at occlusion edges)
zt=Z.reshape(-1)
zz=zt[tris]; jump=zz.max(1)-zz.min(1); zmed=np.median(zt)
keep=jump < 0.06*zmed   # ~6% of median depth
tris_k=tris[keep]
mA=o3d.geometry.TriangleMesh()
mA.vertices=o3d.utility.Vector3dVector(pts); mA.triangles=o3d.utility.Vector3iVector(tris_k)
mA.vertex_colors=o3d.utility.Vector3dVector(cols)
mA.remove_unreferenced_vertices(); mA.compute_vertex_normals()
o3d.io.write_triangle_mesh(os.path.join(D,"_mesh_A_depth.ply"), mA)
print("A depth-mesh: verts",len(mA.vertices),"tris",len(mA.triangles),"(culled",len(tris)-len(tris_k),"stretch tris)")

# ---------- B: points/splat -> Poisson ----------
pcd=o3d.io.read_point_cloud(os.path.join(D,"points.ply"))
if not pcd.has_normals():
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.05,max_nn=30))
    pcd.orient_normals_towards_camera_location(np.array([0,0,0]))
print("points",len(pcd.points),"has_normals",pcd.has_normals(),"has_colors",pcd.has_colors())
mB,dens=o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd,depth=10,scale=1.1)
dens=np.asarray(dens); thr=np.quantile(dens,0.04)
mB.remove_vertices_by_mask(dens<thr); mB.compute_vertex_normals()
o3d.io.write_triangle_mesh(os.path.join(D,"_mesh_B_poisson.ply"), mB)
print("B poisson-mesh: verts",len(mB.vertices),"tris",len(mB.triangles))
print("SAVED both in", D)
