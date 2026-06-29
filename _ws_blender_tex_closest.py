import bpy, sys, os, math
from mathutils import Vector
argv=sys.argv[sys.argv.index("--")+1:]; GLB,OUTDIR=argv[0],argv[1]; os.makedirs(OUTDIR,exist_ok=True)
bpy.ops.wm.read_factory_settings(use_empty=True); bpy.ops.import_scene.gltf(filepath=GLB)
obj=[o for o in bpy.context.scene.objects if o.type=="MESH"][0]
uvname = obj.data.uv_layers[0].name if obj.data.uv_layers else None
print("UV layers:", [l.name for l in obj.data.uv_layers], "| materials:", [m.name for m in obj.data.materials])
for m in obj.data.materials:
    if not m or not m.use_nodes: continue
    nt=m.node_tree
    bsdf=next((n for n in nt.nodes if n.type=="BSDF_PRINCIPLED"),None)
    tex =next((n for n in nt.nodes if n.type=="TEX_IMAGE"),None)
    if bsdf:
        bsdf.inputs["Roughness"].default_value=1.0
        for k in ("Specular IOR Level","Specular"):
            if k in bsdf.inputs: bsdf.inputs[k].default_value=0.0; break
    if tex and uvname:
        uvn=nt.nodes.new("ShaderNodeUVMap"); uvn.uv_map=uvname; tex.interpolation="Closest"
        nt.links.new(uvn.outputs["UV"], tex.inputs["Vector"])   # <-- the missing link
        if bsdf: nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
        print("  wired UV ->", tex.image.name if tex.image else "noimg")
bb=[obj.matrix_world@Vector(c) for c in obj.bound_box]
mn=Vector((min(v.x for v in bb),min(v.y for v in bb),min(v.z for v in bb)))
mx=Vector((max(v.x for v in bb),max(v.y for v in bb),max(v.z for v in bb))); ctr=(mn+mx)/2; size=mx-mn
sc=bpy.context.scene
for e in ("BLENDER_EEVEE_NEXT","BLENDER_EEVEE"):
    try: sc.render.engine=e; break
    except: pass
w=bpy.data.worlds.new("w"); sc.world=w; w.use_nodes=True
w.node_tree.nodes["Background"].inputs[0].default_value=(0.5,0.5,0.5,1); w.node_tree.nodes["Background"].inputs[1].default_value=0.7
sc.render.resolution_x=1100; sc.render.resolution_y=730
for rot,en in [((0.5,0.1,0.3),3),((-0.6,0.2,2.2),2)]:
    s=bpy.data.objects.new("s",bpy.data.lights.new("s","SUN")); sc.collection.objects.link(s); s.rotation_euler=rot; s.data.energy=en
cam=bpy.data.objects.new("c",bpy.data.cameras.new("c")); sc.collection.objects.link(cam); sc.camera=cam; cam.data.lens=20
views={"across":(ctr+Vector((-size.x*0.35,-size.y*0.35,size.z*0.05)),ctr+Vector((size.x*0.45,size.y*0.45,0))),
       "floor":(ctr+Vector((0,0,size.z*0.35)),ctr+Vector((0,size.y*0.2,-size.z*0.5)))}
for nm,(eye,tgt) in views.items():
    cam.location=eye; cam.rotation_euler=(cam.location-tgt).to_track_quat('Z','Y').to_euler()
    sc.render.filepath=os.path.join(OUTDIR,f"fix_{nm}.png"); bpy.ops.render.render(write_still=True); print("rendered",nm,flush=True)
print("DONE",flush=True)
