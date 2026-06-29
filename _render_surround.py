import bpy, math, mathutils, os
GLB=r"D:/HY-World-2.0/_pano_out/pano_surround.glb"
OUT=r"D:/HY-World-2.0/_pano_out/_surround_render"; os.makedirs(OUT, exist_ok=True)
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=GLB)
sc=bpy.context.scene; sc.render.engine='BLENDER_WORKBENCH'; sc.render.resolution_x=640; sc.render.resolution_y=480
try: sc.display.shading.light='MATCAP'; sc.display.shading.show_cavity=True
except Exception: pass
cam_d=bpy.data.cameras.new("c"); cam=bpy.data.objects.new("c",cam_d); sc.collection.objects.link(cam); sc.camera=cam
cam_d.lens=18  # wide
def look(o,t):
    d=o.location-t; o.rotation_euler=d.to_track_quat('Z','Y').to_euler()
# interior: camera at origin, look toward 4 horizontal dirs + outside overview
import mathutils as M
shots=[("in_front",(0,0,0),(0,0,3)),("in_right",(0,0,0),(3,0,0)),("in_back",(0,0,0),(0,0,-3)),
       ("in_left",(0,0,0),(-3,0,0)),("outside",(0,3,8),(0,0,0)),("top",(0,9,0.01),(0,0,0))]
for nm,loc,tgt in shots:
    cam.location=M.Vector(loc); look(cam, M.Vector(tgt))
    sc.render.filepath=f"{OUT}/{nm}.png"; bpy.ops.render.render(write_still=True)
print("rendered surround; tris:", sum(len(o.data.polygons) for o in sc.objects if o.type=='MESH'))
