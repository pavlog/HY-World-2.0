import bpy, math, mathutils, os
GLB=r"D:/HY-World-2.0/_pano_out/pano_surround_textured.glb"
OUT=r"D:/HY-World-2.0/_pano_out/_surround_tex"; os.makedirs(OUT, exist_ok=True)
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=GLB)
sc=bpy.context.scene; sc.render.engine='BLENDER_WORKBENCH'; sc.render.resolution_x=700; sc.render.resolution_y=500
try:
    sc.display.shading.light='FLAT'; sc.display.shading.color_type='TEXTURE'
except Exception: pass
cd=bpy.data.cameras.new("c"); cam=bpy.data.objects.new("c",cd); sc.collection.objects.link(cam); sc.camera=cam
cd.lens=20
import mathutils as M
def look(o,t): d=o.location-t; o.rotation_euler=d.to_track_quat('Z','Y').to_euler()
for nm,tgt in [("front",(0,0,3)),("right",(3,0,0)),("back",(0,0,-3)),("left",(-3,0,0))]:
    cam.location=M.Vector((0,0.2,0)); look(cam, M.Vector(tgt))
    sc.render.filepath=f"{OUT}/tex_{nm}.png"; bpy.ops.render.render(write_still=True)
print("rendered textured surround")
