import bpy, math, mathutils, os
OUT=r"E:/MyGame/Game007Trailer/_cmp_wm_da3"; os.makedirs(OUT, exist_ok=True)
jobs=[("DA3", r"E:/MyGame/Game007Trailer/ForDepthMesh_da3_dense.glb"),
      ("WM",  r"E:/MyGame/Game007Trailer/ForDepthMesh_WMdepth_dense_1024.glb")]
for tag,glb in jobs:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=glb)
    mn=mathutils.Vector((1e9,)*3); mx=mathutils.Vector((-1e9,)*3)
    for o in bpy.context.scene.objects:
        if o.type=='MESH':
            for c in o.bound_box:
                w=o.matrix_world@mathutils.Vector(c)
                for i in range(3): mn[i]=min(mn[i],w[i]); mx[i]=max(mx[i],w[i])
    ctr=(mn+mx)/2; rad=max((mx-mn).length/2,1e-3); dist=rad*2.0
    sc=bpy.context.scene; sc.render.engine='BLENDER_WORKBENCH'; sc.render.resolution_x=600; sc.render.resolution_y=400
    try: sc.display.shading.light='STUDIO'; sc.display.shading.color_type='TEXTURE'
    except Exception: pass
    cd=bpy.data.cameras.new("c"); cam=bpy.data.objects.new("c",cd); sc.collection.objects.link(cam); sc.camera=cam
    def look(o,t): d=o.location-t; o.rotation_euler=d.to_track_quat('Z','Y').to_euler()
    for nm,az,el in [("front",0,6),("l30",330,15)]:
        a=math.radians(az); e=math.radians(el)
        cam.location=ctr+mathutils.Vector((dist*math.cos(e)*math.sin(a),-dist*math.cos(e)*math.cos(a),dist*math.sin(e)))
        look(cam,ctr); sc.render.filepath=f"{OUT}/{tag}_{nm}.png"; bpy.ops.render.render(write_still=True)
    print(tag,"tris",sum(len(o.data.polygons) for o in sc.objects if o.type=='MESH'))
