"""Blender (background): import, remove debris, check self-intersections, export repaired STL, render 4 views.
Usage: blender -b -P repair_render.py -- <in.stl> <out_dir>
"""
import sys, os, math, time
import bpy, bmesh
from mathutils import Vector
from mathutils.bvhtree import BVHTree

argv = sys.argv[sys.argv.index("--") + 1:]
src, out = argv[0], argv[1]
os.makedirs(out, exist_ok=True)
log = open(os.path.join(out, "repair_log.txt"), "w")
def say(*a):
    s = " ".join(str(x) for x in a); print(s); log.write(s + "\n"); log.flush()

bpy.ops.wm.read_factory_settings(use_empty=True)
t0 = time.time()
bpy.ops.wm.stl_import(filepath=src)
obj = bpy.context.selected_objects[0]
bpy.context.view_layer.objects.active = obj
say("imported", obj.name, "faces", len(obj.data.polygons), f"{time.time()-t0:.1f}s")

# --- repair: merge coincident verts, drop loose debris shells, recalc normals outward
bm = bmesh.new(); bm.from_mesh(obj.data)
n_before = len(bm.faces)
r = bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-7)
# connected components
bm.faces.ensure_lookup_table()
seen, comps = set(), []
for f in bm.faces:
    if f.index in seen: continue
    stack, comp = [f], []
    seen.add(f.index)
    while stack:
        cur = stack.pop(); comp.append(cur)
        for e in cur.edges:
            for nf in e.link_faces:
                if nf.index not in seen:
                    seen.add(nf.index); stack.append(nf)
    comps.append(comp)
comps.sort(key=len, reverse=True)
say("components:", [len(c) for c in comps[:10]])
debris = [f for c in comps[1:] for f in c]
bmesh.ops.delete(bm, geom=debris, context="FACES")
loose_v = [v for v in bm.verts if not v.link_faces]
bmesh.ops.delete(bm, geom=loose_v, context="VERTS")
bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
bm.edges.ensure_lookup_table()
boundary = sum(1 for e in bm.edges if e.is_boundary)
nonman = sum(1 for e in bm.edges if not e.is_manifold)
say(f"removed debris faces: {len(debris)}  faces now {len(bm.faces)} (was {n_before})")
say(f"after repair: boundary edges {boundary}, non-manifold edges {nonman}")

# --- self intersections (same approach as Blender's 3D Print Toolbox)
bm.faces.ensure_lookup_table()
t1 = time.time()
tree = BVHTree.FromBMesh(bm, epsilon=0.0)
pairs = tree.overlap(tree)
si_faces = {i for p in pairs for i in p}
say(f"self-intersecting faces: {len(si_faces)}  pairs {len(pairs)}  ({time.time()-t1:.1f}s)")
if si_faces:
    import collections
    cs = [bm.faces[i].calc_center_median() for i in si_faces]
    zs = sorted(c.z for c in cs)
    say("  self-intersection Z range:", round(zs[0], 4), "-", round(zs[-1], 4))
    # rough clusters by 2cm-equivalent grid (2% of height)
    h = max(v.co.z for v in bm.verts) - min(v.co.z for v in bm.verts)
    g = h * 0.03
    cl = collections.Counter((round(c.x / g), round(c.y / g), round(c.z / g)) for c in cs)
    say("  clusters (grid cell -> faces), top 15:", [(tuple(round(x * g, 3) for x in k), n) for k, n in cl.most_common(15)])
bm.to_mesh(obj.data); obj.data.update(); bm.free()

bpy.ops.object.select_all(action="DESELECT"); obj.select_set(True)
bpy.ops.wm.stl_export(filepath=os.path.join(out, "work_repaired.stl"), export_selected_objects=True, ascii_format=False)
say("exported repaired STL")

# --- render setup
bpy.ops.object.shade_smooth()
mat = bpy.data.materials.new("clay"); mat.use_nodes = True
bsdf = mat.node_tree.nodes["Principled BSDF"]
bsdf.inputs["Base Color"].default_value = (0.72, 0.72, 0.70, 1); bsdf.inputs["Roughness"].default_value = 0.55
obj.data.materials.append(mat)

sc = bpy.context.scene
sc.render.engine = "CYCLES"; sc.cycles.device = "CPU"; sc.cycles.samples = 48
sc.cycles.use_denoising = True
sc.render.resolution_x, sc.render.resolution_y = 900, 1200
sc.render.film_transparent = False
world = bpy.data.worlds.new("w"); sc.world = world; world.use_nodes = True
world.node_tree.nodes["Background"].inputs[0].default_value = (0.95, 0.95, 0.96, 1)
world.node_tree.nodes["Background"].inputs[1].default_value = 0.6
sc.view_settings.view_transform = "Standard"

bb = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
ctr = sum(bb, Vector()) / 8
H = max(c.z for c in bb) - min(c.z for c in bb)

cam_data = bpy.data.cameras.new("cam"); cam_data.type = "ORTHO"; cam_data.ortho_scale = H * 1.12
cam = bpy.data.objects.new("cam", cam_data); sc.collection.objects.link(cam); sc.camera = cam
key = bpy.data.objects.new("key", bpy.data.lights.new("key", "SUN")); sc.collection.objects.link(key)
key.data.energy = 3.0; key.data.angle = math.radians(8)

views = {"front": (0, -1), "back": (0, 1), "left": (-1, 0), "right": (1, 0)}
for name, (dx, dy) in views.items():
    d = Vector((dx, dy, 0))
    cam.location = ctr + d * H * 3
    cam.rotation_euler = (math.radians(90), 0, math.atan2(dx, -dy))
    # key light from camera side, upper-left
    key.rotation_euler = (math.radians(50), 0, math.atan2(dx, -dy) + math.radians(-35))
    sc.render.filepath = os.path.join(out, f"render_{name}.png")
    t2 = time.time(); bpy.ops.render.render(write_still=True)
    say(f"rendered {name} {time.time()-t2:.1f}s")
say(f"total {time.time()-t0:.1f}s")
