import bpy, math, os, mathutils as M
GLB=r"D:/HY-World-2.0/_pano_out/pano_surround_cube.glb"; OUT=r"D:/HY-World-2.0/_pano_out/_cube_render"; os.makedirs(OUT,exist_ok=True)
bpy.ops.wm.read_factory_settings(use_empty=True); bpy.ops.import_scene.gltf(filepath=GLB)
sc=bpy.context.scene; sc.render.engine='BLENDER_WORKBENCH'; sc.render.resolution_x=720; sc.render.resolution_y=520
try: sc.display.shading.light='FLAT'; sc.display.shading.color_type='TEXTURE'
except Exception: pass
cd=bpy.data.cameras.new("c"); cam=bpy.data.objects.new("c",cd); sc.collection.objects.link(cam); sc.camera=cam; cd.lens=16
def look(o,t): d=o.location-t; o.rotation_euler=d.to_track_quat('Z','Y').to_euler()
for nm,tgt in [("front",(0,0.1,3)),("right",(3,0.1,0)),("back",(0,0.1,-3)),("left",(-3,0.1,0)),("out",(0,4,9))]:
    cam.location=M.Vector((0,0.1,0)) if nm!="out" else M.Vector((0,4,9)); look(cam,M.Vector((0,0,0)) if nm=="out" else M.Vector(tgt))
    sc.render.filepath=f"{OUT}/{nm}.png"; bpy.ops.render.render(write_still=True)
print("done")
