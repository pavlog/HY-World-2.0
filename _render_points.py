import numpy as np, os
d=r"D:/HY-World-2.0/_out_lyra/wm_input_lyra/20260623_013832"
ply=os.path.join(d,"points.ply")
# load via open3d if available else plyfile
try:
    import open3d as o3d
    pcd=o3d.io.read_point_cloud(ply)
    pts=np.asarray(pcd.points); col=np.asarray(pcd.colors)
    print("loaded via open3d", pts.shape, "has_color", col.shape)
except Exception as e:
    print("open3d failed:",e)
    from plyfile import PlyData
    p=PlyData.read(ply); v=p['vertex']
    pts=np.stack([v['x'],v['y'],v['z']],1)
    try: col=np.stack([v['red'],v['green'],v['blue']],1)/255.0
    except: col=None
    print("loaded via plyfile", pts.shape)
print("bbox min",pts.min(0),"max",pts.max(0))
# downsample
n=pts.shape[0]; idx=np.random.RandomState(0).choice(n,min(80000,n),replace=False)
P=pts[idx]; C=col[idx] if col is not None and len(col) else None
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
# center+scale
c=P.mean(0); P=P-c; s=np.percentile(np.abs(P),99); P=np.clip(P/s,-1,1)
views=[(20,-60,"a"),(20,30,"b"),(80,-90,"top")]
fig=plt.figure(figsize=(15,5))
for i,(el,az,nm) in enumerate(views):
    ax=fig.add_subplot(1,3,i+1,projection='3d')
    ax.scatter(P[:,0],P[:,1],P[:,2],c=C if C is not None else P[:,2],s=0.5,marker='.',linewidths=0)
    ax.view_init(elev=el,azim=az); ax.set_axis_off(); ax.set_title(nm)
plt.tight_layout(); out=os.path.join(d,"_points_preview.png"); plt.savefig(out,dpi=90); print("SAVED",out)
