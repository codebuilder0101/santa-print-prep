"""Blender background renderer for colored meshes.
Usage: blender -b --factory-startup -P render_npz.py -- <mesh.npz> <views.json>
mesh.npz: vertices (N,3) float, faces (M,3) int, and either face_colors (M,3|4) or vertex_colors (N,3|4) uint8.
views.json: {"out_dir":..., "prefix":..., "res":[w,h], "samples":int,
             "views":[{"name":"front","dir":[0,-1,0],"center":null|[x,y,z],"scale":null|float}]}
"dir" points from the model to the camera.
"""
import sys, os, json, math
import numpy as np
import bpy
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:]
data = np.load(argv[0])
cfg = json.load(open(argv[1]))
os.makedirs(cfg["out_dir"], exist_ok=True)

bpy.ops.wm.read_factory_settings(use_empty=True)
V = data["vertices"].astype(np.float32); F = data["faces"].astype(np.int32)
me = bpy.data.meshes.new("m")
me.vertices.add(len(V)); me.vertices.foreach_set("co", V.ravel())
me.loops.add(F.size); me.loops.foreach_set("vertex_index", F.ravel())
me.polygons.add(len(F))
me.polygons.foreach_set("loop_start", np.arange(0, F.size, 3, dtype=np.int32))
me.polygons.foreach_set("loop_total", np.full(len(F), 3, np.int32))
me.update(); me.validate(clean_customdata=False)
me.polygons.foreach_set("use_smooth", np.ones(len(F), bool))

def to_rgba(c):
    c = c.astype(np.float32) / 255.0
    if c.shape[1] == 3: c = np.hstack([c, np.ones((len(c), 1), np.float32)])
    return c
if "face_colors" in data.files:
    ca = me.color_attributes.new("Col", "FLOAT_COLOR", "CORNER")
    fc = to_rgba(data["face_colors"])
    ca.data.foreach_set("color", np.repeat(fc, 3, axis=0).ravel())
else:
    ca = me.color_attributes.new("Col", "FLOAT_COLOR", "POINT")
    ca.data.foreach_set("color", to_rgba(data["vertex_colors"]).ravel())

obj = bpy.data.objects.new("model", me)
bpy.context.scene.collection.objects.link(obj)
mat = bpy.data.materials.new("m"); mat.use_nodes = True
nt = mat.node_tree; bsdf = nt.nodes["Principled BSDF"]
attr = nt.nodes.new("ShaderNodeAttribute"); attr.attribute_name = "Col"
# colours are authored in sRGB -> convert to linear for correct display
gamma = nt.nodes.new("ShaderNodeGamma"); gamma.inputs[1].default_value = 2.2
nt.links.new(attr.outputs["Color"], gamma.inputs[0]); nt.links.new(gamma.outputs[0], bsdf.inputs["Base Color"])
bsdf.inputs["Roughness"].default_value = 0.6
me.materials.append(mat)

sc = bpy.context.scene
sc.render.engine = "CYCLES"; sc.cycles.device = "CPU"; sc.cycles.samples = cfg.get("samples", 48)
sc.cycles.use_denoising = True
sc.render.resolution_x, sc.render.resolution_y = cfg.get("res", [900, 1200])
sc.view_settings.view_transform = "Standard"
w = bpy.data.worlds.new("w"); sc.world = w; w.use_nodes = True
w.node_tree.nodes["Background"].inputs[0].default_value = (0.93, 0.93, 0.94, 1)
w.node_tree.nodes["Background"].inputs[1].default_value = 0.7

bmin, bmax = V.min(0), V.max(0)
ctr0 = Vector(((bmin + bmax) / 2).tolist()); H = float(bmax[2] - bmin[2])
cam_d = bpy.data.cameras.new("c"); cam_d.type = "ORTHO"
cam = bpy.data.objects.new("c", cam_d); sc.collection.objects.link(cam); sc.camera = cam
sun = bpy.data.objects.new("s", bpy.data.lights.new("s", "SUN")); sc.collection.objects.link(sun)
sun.data.energy = 2.6; sun.data.angle = math.radians(10)

for v in cfg["views"]:
    d = Vector(v["dir"]).normalized()
    ctr = Vector(v["center"]) if v.get("center") else ctr0
    cam_d.ortho_scale = v.get("scale") or H * 1.12
    cam.location = ctr + d * H * 3
    cam.rotation_euler = d.to_track_quat("Z", "Y").to_euler()
    sun.rotation_euler = (d.lerp(Vector((0, 0, 1)), 0.5) + d.cross(Vector((0, 0, 1))) * 0.5).normalized().to_track_quat("Z", "Y").to_euler()
    sc.render.filepath = os.path.join(cfg["out_dir"], f"{cfg['prefix']}_{v['name']}.png")
    bpy.ops.render.render(write_still=True)
    print("RENDERED", sc.render.filepath)
