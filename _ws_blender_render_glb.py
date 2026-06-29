import bpy, sys, math, os
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:]
GLB, OUTDIR = argv[0], argv[1]
os.makedirs(OUTDIR, exist_ok=True)

# clean scene
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=GLB)
meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
obj = meshes[0]

# world bbox center + size
bb = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
mn = Vector((min(v.x for v in bb), min(v.y for v in bb), min(v.z for v in bb)))
mx = Vector((max(v.x for v in bb), max(v.y for v in bb), max(v.z for v in bb)))
ctr = (mn + mx) / 2; size = (mx - mn)

# keep the baseColor texture under Principled BSDF; make it matte (roughness=1, no spec) so baked
# lighting reads naturally. Add sky+sun lighting below. (pure-emission made dark panel grooves look like holes)
for m in obj.data.materials:
    if not m or not m.use_nodes: continue
    bsdf = next((n for n in m.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
    if bsdf:
        bsdf.inputs["Roughness"].default_value = 1.0
        if "Specular IOR Level" in bsdf.inputs: bsdf.inputs["Specular IOR Level"].default_value = 0.0
        elif "Specular" in bsdf.inputs: bsdf.inputs["Specular"].default_value = 0.0

scene = bpy.context.scene
for eng in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"):
    try: scene.render.engine = eng; break
    except Exception: pass
# mid-grey world so holes/gaps read grey (not alarming black) and we can judge the surface
world = bpy.data.worlds.new("w"); scene.world = world; world.use_nodes = True
world.node_tree.nodes["Background"].inputs[0].default_value = (0.5, 0.5, 0.5, 1)
world.node_tree.nodes["Background"].inputs[1].default_value = 1.0
scene.render.resolution_x = 960; scene.render.resolution_y = 640
scene.render.film_transparent = False
scene.view_settings.view_transform = "Standard"

# even lighting: brighter ambient world + two suns
world.node_tree.nodes["Background"].inputs[1].default_value = 0.8
for rot, e in [((0.5, 0.1, 0.3), 2.5), ((-0.6, 0.2, 2.2), 1.5)]:
    s = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN")); scene.collection.objects.link(s)
    s.rotation_euler = rot; s.data.energy = e

cam_data = bpy.data.cameras.new("cam"); cam = bpy.data.objects.new("cam", cam_data)
scene.collection.objects.link(cam); scene.camera = cam
cam_data.lens = 24

def look_at(camobj, target):
    d = (camobj.location - target); camobj.rotation_euler = d.to_track_quat('Z', 'Y').to_euler()

R = max(size.x, size.y) * 0.42
views = {
    "interior_a": (ctr + Vector((0, 0, size.z * 0.1)), ctr + Vector(( R,  R*0.6, 0))),
    "interior_b": (ctr + Vector((0, 0, size.z * 0.1)), ctr + Vector((-R,  R*0.6, 0))),
    "interior_c": (ctr + Vector((0, 0, size.z * 0.15)), ctr + Vector((0, -R, -size.z*0.2))),
    "exterior":   (ctr + Vector((size.x*1.1, size.y*1.1, size.z*0.9)), ctr),
}
for name, (eye, tgt) in views.items():
    cam.location = eye; look_at(cam, tgt)
    scene.render.filepath = os.path.join(OUTDIR, f"glb_{name}.png")
    bpy.ops.render.render(write_still=True)
    print("rendered", name, flush=True)
print("DONE", flush=True)
