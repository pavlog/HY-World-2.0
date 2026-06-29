import numpy as np, json, os, glob
D=r"D:/HY-World-2.0/_out_lyra/wm_input_lyra/20260623_013832"
cp=json.load(open(os.path.join(D,"camera_params.json")))
print("top keys:", list(cp.keys()))
print("num_cameras:", cp.get("num_cameras"))
if "intrinsics" in cp:
    intr=cp["intrinsics"]
    print("intrinsics type:", type(intr).__name__, "len", len(intr) if hasattr(intr,'__len__') else '-')
    print("intr[0]:", json.dumps(intr[0] if isinstance(intr,list) else intr)[:400])
ex=cp["extrinsics"]
M0=np.array(ex[0]["matrix"]); M8=np.array(ex[8]["matrix"]); M16=np.array(ex[16]["matrix"])
print("extr0 t:", M0[:3,3].round(4), "| extr8 t:", M8[:3,3].round(4), "| extr16 t:", M16[:3,3].round(4))
ds=sorted(glob.glob(os.path.join(D,"depth","*.npy")))
d=np.load(ds[0]); print("depth",d.shape,d.dtype,"min",round(float(np.nanmin(d)),3),"max",round(float(np.nanmax(d)),3),"med",round(float(np.nanmedian(d)),3),"nan%",round(float(np.isnan(d).mean()*100),1))
imgs=sorted(glob.glob("E:/MyGame/Game007Trailer/wm_input_lyra/*.png"))
print("num depth:",len(ds),"num imgs:",len(imgs))
