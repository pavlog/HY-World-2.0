import numpy as np, cv2, os
OUT=r'D:/HY-World-2.0/_pano_out/cube'
dep=np.load(os.path.join(OUT,'cube_depths.npy')).astype(np.float32)  # (6,h,w)
N,fh,fw=dep.shape
fwd=np.array([[0,0,1],[0,0,-1],[1,0,0],[-1,0,0],[0,1,0],[0,-1,0]],float)
right=np.array([[1,0,0],[-1,0,0],[0,0,-1],[0,0,1],[1,0,0],[1,0,0]],float)
up=np.array([[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,0,-1],[0,0,1]],float)
Wd,Hd=2048,1024
uu,vv=np.meshgrid(np.arange(Wd),np.arange(Hd))
lon=(uu+0.5)/Wd*2*np.pi; lat=(0.5-(vv+0.5)/Hd)*np.pi
D=np.stack([np.cos(lat)*np.sin(lon),np.sin(lat),np.cos(lat)*np.cos(lon)],-1)
ax=np.argmax(np.abs(D),axis=-1); sg=np.sign(D[np.arange(Hd)[:,None],np.arange(Wd)[None,:],ax])
face_of=np.full((Hd,Wd),-1,int)
face_of[(ax==2)&(sg>0)]=0; face_of[(ax==2)&(sg<0)]=1
face_of[(ax==0)&(sg>0)]=2; face_of[(ax==0)&(sg<0)]=3
face_of[(ax==1)&(sg>0)]=4; face_of[(ax==1)&(sg<0)]=5
def bilin(img,x,y):
    x0=np.clip(np.floor(x).astype(int),0,img.shape[1]-1); y0=np.clip(np.floor(y).astype(int),0,img.shape[0]-1)
    x1=np.clip(x0+1,0,img.shape[1]-1); y1=np.clip(y0+1,0,img.shape[0]-1)
    fx=x-x0; fy=y-y0
    return (img[y0,x0]*(1-fx)*(1-fy)+img[y0,x1]*fx*(1-fy)+img[y1,x0]*(1-fx)*fy+img[y1,x1]*fx*fy)
radial=np.zeros((Hd,Wd),np.float32)
for f in range(6):
    msk=face_of==f
    if not msk.any(): continue
    d=D[msk]
    fz=d@fwd[f]; px=(d@right[f])/fz; py=(d@up[f])/fz
    j=((px+1)/2*fw); i=((py+1)/2*fh)
    radial[msk]=bilin(dep[f], j, i)/np.clip(fz,1e-3,None)
np.save(r'D:/HY-World-2.0/_pano_out/equirect_depth_cube.npy', radial)
dn=(radial-radial.min())/(radial.max()-radial.min()+1e-6)
cv2.imwrite(r'D:/HY-World-2.0/_pano_out/equirect_depth_cube_color.png', cv2.applyColorMap((dn*255).astype(np.uint8),cv2.COLORMAP_TURBO))
print('equirect cube-depth', radial.shape, 'range %.2f-%.2f'%(radial.min(),radial.max()))
