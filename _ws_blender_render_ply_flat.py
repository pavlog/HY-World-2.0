import bpy, sys, os
from mathutils import Vector
argv = sys.argv[sys.argv.index("--") + 1:]
PLY, OUTDIR = argv[0], argv[1]
os.makedirs(OUTDIR, exist_ok=True)
bpy.ops.wm.read_factory_settings(use_empty=True)
try: bpy.ops.wm.ply_import(filepath=PLY)
except Exception: bpy.ops.import_mesh.ply(filepath=PLY)
obj = [o for o in bpy.context.scene.objects if o.type == "MESH"][0]
bb = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
mn = Vector((min(v.x for v in bb), min(v.y for v in bb), min(v.z for v in bb)))
mx = Vector((max(v.x for v in bb), max(v.y for v in bb), max(v.z for v in bb)))
ctr = (mn + mx) / 2; size = mx - mn
mat = bpy.data.materials.new("flat"); mat.use_nodes = True
bsdf = mat.node_tree.nodes.get("Principled BSDF")
if bsdf: bsdf.inputs["Base Color"].default_value = (0.7, 0.7, 0.72, 1)
obj.data.materials.clear(); obj.data.materials.append(mat)
sc = bpy.context.scene
for eng in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"):
    try: sc.render.engine = eng; break
    except Exception: pass
w = bpy.data.worlds.new("w"); sc.world = w; w.use_nodes = True
w.node_tree.nodes["Background"].inputs[0].default_value = (0.5, 0.5, 0.5, 1)
sc.render.resolution_x = 960; sc.render.resolution_y = 640
sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN")); sc.collection.objects.link(sun)
sun.rotation_euler = (0.6, 0.2, 0.3); sun.data.energy = 3
cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam")); sc.collection.objects.link(cam); sc.camera = cam
cam.data.lens = 35
import math
for name, ang in [("exterior", 30), ("side", 110)]:
    a = math.radians(ang); R = max(size.x, size.y) * 1.3
    cam.location = ctr + Vector((math.cos(a) * R, math.sin(a) * R, size.z * 0.4))
    d = cam.location - ctr; cam.rotation_euler = d.to_track_quat('Z', 'Y').to_euler()
    sc.render.filepath = os.path.join(OUTDIR, f"flat_{name}.png"); bpy.ops.render.render(write_still=True)
    print("rendered", name, flush=True)
print("DONE", flush=True)
