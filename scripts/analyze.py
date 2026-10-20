"""Read-only analysis of an STL: size, topology, shells, holes, normals, self-intersections."""
import sys, time, json
import numpy as np
import trimesh

path = sys.argv[1]
t0 = time.time()
raw = trimesh.load(path, process=False)
print(f"raw triangles: {len(raw.faces):,}  raw vertices (soup): {len(raw.vertices):,}")

m = trimesh.load(path, process=True)  # merges duplicate vertices
print(f"merged vertices: {len(m.vertices):,}  faces: {len(m.faces):,}  load {time.time()-t0:.1f}s")
ext = m.extents
print("bounds min", m.bounds[0].round(3).tolist(), "max", m.bounds[1].round(3).tolist())
print("extents X/Y/Z", ext.round(3).tolist(), "(units as stored, presumably mm)")

# degenerate / duplicate
areas = m.area_faces
degen = int((areas < 1e-12).sum())
sorted_f = np.sort(m.faces, axis=1)
_, cnt = np.unique(sorted_f, axis=0, return_counts=True)
dups = int((cnt > 1).sum())
print(f"degenerate (zero-area) faces: {degen}  duplicate faces: {dups}")

# edge topology
edges = np.sort(m.edges, axis=1)
uniq, ecount = np.unique(edges, axis=0, return_counts=True)
boundary = uniq[ecount == 1]
nonmanifold = int((ecount > 2).sum())
print(f"unique edges: {len(uniq):,}  boundary edges (holes): {len(boundary):,}  non-manifold edges (>2 faces): {nonmanifold:,}")

# hole loops
if len(boundary):
    import networkx as nx
    g = nx.Graph(); g.add_edges_from(boundary.tolist())
    loops = [len(c) for c in nx.connected_components(g)]
    loops.sort(reverse=True)
    print(f"hole loops: {len(loops)}  largest loop sizes (edges): {loops[:10]}")

print("watertight:", m.is_watertight, " winding consistent:", m.is_winding_consistent, " is_volume:", m.is_volume)

# shells
shells = m.split(only_watertight=False)
print(f"shells (connected components): {len(shells)}")
info = []
for i, s in enumerate(shells):
    vol = s.volume if s.is_watertight else float('nan')
    info.append((len(s.faces), s.extents.round(2).tolist(), s.centroid.round(1).tolist(), s.is_watertight, round(float(vol), 2)))
info.sort(key=lambda x: -x[0])
for row in info[:25]:
    print("  faces %8d  extents %s  centroid %s  watertight %s  volume %s" % row)
if len(info) > 25:
    print(f"  ... {len(info)-25} more shells, faces: {[r[0] for r in info[25:]][:40]}")
neg = sum(1 for r in info if r[3] and r[4] < 0)
print("watertight shells with NEGATIVE volume (inside-out / flipped):", neg)

# self intersections via pymeshlab
try:
    import pymeshlab
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(m.vertices, m.faces))
    t1 = time.time()
    ms.apply_filter("compute_selection_by_self_intersections_per_face")
    n_si = ms.current_mesh().selected_face_number()
    print(f"self-intersecting faces: {n_si:,}  ({time.time()-t1:.1f}s)")
    topo = ms.get_topological_measures()
    print("meshlab topology:", json.dumps({k: topo[k] for k in topo}, default=str))
except Exception as e:
    print("self-intersection check failed:", e)
print(f"total {time.time()-t0:.1f}s")
