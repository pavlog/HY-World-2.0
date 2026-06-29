import bpy, math, os, mathutils as M
GLB=r"D:/HY-World-2.0/_pano_out/pano_surround_cube_clean.glb"; OUT=r"D:/HY-World-2.0/_pano_out/_cube_render2"; os.makedirs(OUT,exist_ok=True)
bpy.ops.wm.read_factory_settings(use_empty=True); bpy.ops.import_scene.gltf(filepath=GLB)
sc=bpy.context.scene; sc.render.engine='BLENDER_WORKBENCH'; sc.render.resolution_x=720; sc.render.resolution_y=520
try: sc.display.shading.light='FLAT'; sc.display.shading.color_type='TEXTURE'
except Exception: pass
cd=bpy.data.cameras.new("c"); cam=bpy.data.objects.new("c",cd); sc.collection.objects.link(cam); sc.camera=cam; cd.lens=14
def look(loc,tgt):
    cam.location=M.Vector(loc); cam.rotation_euler=(M.Vector(tgt)-M.Vector(loc)).to_track_quat('-Z','Y').to_euler()
for nm,tgt in [("front",(0,0,3)),("right",(3,0,0)),("back",(0,0,-3)),("left",(-3,0,0))]:
    look((0,0,0),tgt); sc.render.filepath=f"{OUT}/{nm}.png"; bpy.ops.render.render(write_still=True)
look((0,4,9),(0,0,0)); sc.render.filepath=f"{OUT}/out.png"; bpy.ops.render.render(write_still=True)
print("done")
