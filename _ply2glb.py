import bpy, os
M=r"E:/MyGame/AIWorldStudio/docs/research/worldgen-3d-pipeline/meshes"
jobs=[("wm_cockpit_A_depth-unproject.ply","wm_cockpit_A_depth-unproject.glb"),
 ("wm_cockpit_B_poisson.ply","wm_cockpit_B_poisson.glb"),
 ("wm_harbor_tsdf.ply","wm_harbor_tsdf.glb"),
 ("wm_s2_palace_tsdf.ply","wm_s2_palace_tsdf.glb")]
for src,dst in jobs:
    sp=os.path.join(M,src); dp=os.path.join(M,dst)
    if not os.path.exists(sp): print("SKIP",src); continue
    bpy.ops.wm.read_factory_settings(use_empty=True)
    try: bpy.ops.wm.ply_import(filepath=sp)
    except Exception: bpy.ops.import_mesh.ply(filepath=sp)
    bpy.ops.export_scene.gltf(filepath=dp, export_format='GLB')
    print("OK", dst)
print("done")
