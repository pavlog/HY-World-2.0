import bpy, math, mathutils, os
PLY=r"D:/HY-World-2.0/_out_lyra/wm_input_lyra/20260623_013832/_tsdf_mesh_clean.ply"
OUT=r"D:/HY-World-2.0/_out_lyra/wm_input_lyra/20260623_013832/_tsdf_render"
os.makedirs(OUT, exist_ok=True)
bpy.ops.wm.read_factory_settings(use_empty=True)
try: bpy.ops.wm.ply_import(filepath=PLY)
except Exception: bpy.ops.import_mesh.ply(filepath=PLY)
mins=mathutils.Vector((1e9,1e9,1e9)); maxs=mathutils.Vector((-1e9,-1e9,-1e9))
for o in bpy.context.scene.objects:
    if o.type=='MESH':
        for c in o.bound_box:
            w=o.matrix_world@mathutils.Vector(c)
            for i in range(3): mins[i]=min(mins[i],w[i]); maxs[i]=max(maxs[i],w[i])
center=(mins+maxs)/2.0; radius=max((maxs-mins).length/2.0,1e-3); dist=radius*2.4
sc=bpy.context.scene; sc.render.engine='BLENDER_WORKBENCH'
sc.render.resolution_x=640; sc.render.resolution_y=480
try:
    sc.display.shading.light='FLAT'; sc.display.shading.color_type='VERTEX'
except Exception: pass
cam_d=bpy.data.cameras.new("c"); cam=bpy.data.objects.new("c",cam_d)
sc.collection.objects.link(cam); sc.camera=cam
def look(o,t):
    d=o.location-t; o.rotation_euler=d.to_track_quat('Z','Y').to_euler()
views=[("front",0,10),("l45",315,15),("r45",45,15),("side",90,10),("back",180,15),("top",0,75)]
for nm,az,el in views:
    a=math.radians(az); e=math.radians(el)
    cam.location=center+mathutils.Vector((dist*math.cos(e)*math.sin(a),-dist*math.cos(e)*math.cos(a),dist*math.sin(e)))
    look(cam,center); sc.render.filepath=f"{OUT}/t_{nm}.png"; bpy.ops.render.render(write_still=True)
print("tris:", sum(len(o.data.polygons) for o in sc.objects if o.type=='MESH'))
