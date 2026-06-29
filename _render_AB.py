import bpy, math, mathutils, os
D=r"D:/HY-World-2.0/_out_cockpit_single/wm_input/20260623_020811"
def render(ply, outdir, tag):
    os.makedirs(outdir, exist_ok=True)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    try: bpy.ops.wm.ply_import(filepath=ply)
    except Exception: bpy.ops.import_mesh.ply(filepath=ply)
    mn=mathutils.Vector((1e9,)*3); mx=mathutils.Vector((-1e9,)*3)
    for o in bpy.context.scene.objects:
        if o.type=='MESH':
            for c in o.bound_box:
                w=o.matrix_world@mathutils.Vector(c)
                for i in range(3): mn[i]=min(mn[i],w[i]); mx[i]=max(mx[i],w[i])
    ctr=(mn+mx)/2; rad=max((mx-mn).length/2,1e-3); dist=rad*2.1
    sc=bpy.context.scene; sc.render.engine='BLENDER_WORKBENCH'; sc.render.resolution_x=640; sc.render.resolution_y=420
    try: sc.display.shading.light='FLAT'; sc.display.shading.color_type='VERTEX'
    except Exception: pass
    cd=bpy.data.cameras.new("c"); cam=bpy.data.objects.new("c",cd); sc.collection.objects.link(cam); sc.camera=cam
    def look(o,t): d=o.location-t; o.rotation_euler=d.to_track_quat('Z','Y').to_euler()
    for nm,az,el in [("front",0,6),("l35",325,16),("side",80,12)]:
        a=math.radians(az); e=math.radians(el)
        cam.location=ctr+mathutils.Vector((dist*math.cos(e)*math.sin(a),-dist*math.cos(e)*math.cos(a),dist*math.sin(e)))
        look(cam,ctr); sc.render.filepath=f"{outdir}/{tag}_{nm}.png"; bpy.ops.render.render(write_still=True)
render(D+"/_mesh_A_depth.ply", D+"/_render_AB", "A")
render(D+"/_mesh_B_poisson.ply", D+"/_render_AB", "B")
print("done")
