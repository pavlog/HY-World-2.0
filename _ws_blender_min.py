import bpy, sys, os, math
from mathutils import Vector
argv=sys.argv[sys.argv.index("--")+1:]; GLB,OUTDIR=argv[0],argv[1]; os.makedirs(OUTDIR,exist_ok=True)
bpy.ops.wm.read_factory_settings(use_empty=True); bpy.ops.import_scene.gltf(filepath=GLB)
obj=[o for o in bpy.context.scene.objects if o.type=="MESH"][0]
# NO material override at all - use importer's PBR as-is
bb=[obj.matrix_world@Vector(c) for c in obj.bound_box]
mn=Vector((min(v.x for v in bb),min(v.y for v in bb),min(v.z for v in bb)))
mx=Vector((max(v.x for v in bb),max(v.y for v in bb),max(v.z for v in bb))); ctr=(mn+mx)/2; size=mx-mn
sc=bpy.context.scene
for e in ("BLENDER_EEVEE_NEXT","BLENDER_EEVEE"):
    try: sc.render.engine=e; break
    except: pass
w=bpy.data.worlds.new("w"); sc.world=w; w.use_nodes=True
w.node_tree.nodes["Background"].inputs[0].default_value=(0.5,0.5,0.5,1); w.node_tree.nodes["Background"].inputs[1].default_value=1.0
sc.render.resolution_x=1100; sc.render.resolution_y=730
for rot,en in [((0.5,0.1,0.3),3),((-0.6,0.2,2.2),2)]:
    s=bpy.data.objects.new("s",bpy.data.lights.new("s","SUN")); sc.collection.objects.link(s); s.rotation_euler=rot; s.data.energy=en
cam=bpy.data.objects.new("c",bpy.data.cameras.new("c")); sc.collection.objects.link(cam); sc.camera=cam; cam.data.lens=20
eye=ctr+Vector((-size.x*0.35,-size.y*0.35,size.z*0.05)); tgt=ctr+Vector((size.x*0.45,size.y*0.45,0))
cam.location=eye; cam.rotation_euler=(cam.location-tgt).to_track_quat('Z','Y').to_euler()
sc.render.filepath=os.path.join(OUTDIR,"min_across.png"); bpy.ops.render.render(write_still=True); print("DONE",flush=True)
