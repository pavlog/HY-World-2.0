import bpy, math, os
GLB=r"D:/HY-World-2.0/_pano_out/pano_surround_cube_clean.glb"; OUT=r"D:/HY-World-2.0/_pano_out/_cube_interior"; os.makedirs(OUT,exist_ok=True)
bpy.ops.wm.read_factory_settings(use_empty=True); bpy.ops.import_scene.gltf(filepath=GLB)
sc=bpy.context.scene; sc.render.engine='BLENDER_WORKBENCH'; sc.render.resolution_x=760; sc.render.resolution_y=480
try: sc.display.shading.light='FLAT'; sc.display.shading.color_type='TEXTURE'
except Exception: pass
cd=bpy.data.cameras.new("c"); cam=bpy.data.objects.new("c",cd); sc.collection.objects.link(cam); sc.camera=cam; cd.lens=16
cam.location=(0,0,0)
# after gltf Y-up->Z-up: horizontal view = rotation_euler (90deg, 0, azimuth)
for nm,az in [("a",0),("b",90),("c",180),("d",270)]:
    cam.rotation_euler=(math.radians(90),0,math.radians(az))
    sc.render.filepath=f"{OUT}/view_{nm}.png"; bpy.ops.render.render(write_still=True)
print("done")
