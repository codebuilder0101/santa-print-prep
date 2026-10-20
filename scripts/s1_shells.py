"""Step 1: shell analysis of the repaired master + crease map for step 2 planning.
Outputs (in 01_repaired/): shells_report.txt, shells_colored.ply, crease_map.ply
"""
import os, sys
import numpy as np
import trimesh
import scipy.sparse as sp

ROOT = r"C:\github_storage\pro-device"
src = os.path.join(ROOT, "01_repaired", "work_repaired.stl")
out = os.path.join(ROOT, "01_repaired")
rep = open(os.path.join(out, "shells_report.txt"), "w", encoding="utf-8")
def say(*a):
    s = " ".join(str(x) for x in a); print(s); rep.write(s + "\n")

m = trimesh.load(src)
H = m.extents[2]
mm = 250.0 / H  # report sizes at 25 cm scale for intuition
say(f"master: {len(m.faces):,} faces, {len(m.vertices):,} verts, height {H:.4f} units")

# --- connected components, by shared vertex (loosest definition) and by manifold edge
labels_v = trimesh.graph.connected_component_labels(m.face_adjacency, node_count=len(m.faces))
n_edge = labels_v.max() + 1
# vertex-sharing components
adj = sp.coo_matrix((np.ones(len(m.faces) * 3), (np.repeat(np.arange(len(m.faces)), 3), m.faces.ravel())),
                    shape=(len(m.faces), len(m.vertices))).tocsr()
ff = (adj @ adj.T).tocoo()
n_vert, lab_vert = sp.csgraph.connected_components(sp.coo_matrix((np.ones_like(ff.row), (ff.row, ff.col))), directed=False)
say(f"shells by shared edge: {n_edge}   shells by shared vertex: {n_vert}")

rng = np.random.default_rng(7)
for sid in range(n_edge):
    idx = np.where(labels_v == sid)[0]
    sub = m.submesh([idx], append=True)
    b = sub.bounds
    say(f"  shell {sid}: {len(idx):,} faces  bbox min {b[0].round(3).tolist()} max {b[1].round(3).tolist()}"
        f"  size@25cm {np.round(sub.extents*mm,1).tolist()} mm  watertight {sub.is_watertight}")

# --- edge length distribution (uniform lengths => voxel/remeshed, i.e. accessories fused)
el = m.edges_unique_length * mm
say(f"edge length @25cm (mm): median {np.median(el):.3f}  p5 {np.percentile(el,5):.3f}  p95 {np.percentile(el,95):.3f}  "
    f"max {el.max():.3f}  p95/p5 ratio {np.percentile(el,95)/np.percentile(el,5):.1f}")

# --- shell-colored PLY (random colour per shell)
cols = (rng.random((n_edge, 3)) * 200 + 40).astype(np.uint8)
fc = np.hstack([cols[labels_v], np.full((len(m.faces), 1), 255, np.uint8)])
cm = m.copy(); cm.visual = trimesh.visual.ColorVisuals(cm, face_colors=fc)
cm.export(os.path.join(out, "shells_colored.ply"))

# --- crease map: signed concavity from umbrella Laplacian vs vertex normal
e = m.edges_unique
n = len(m.vertices)
A = sp.coo_matrix((np.ones(len(e) * 2), (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(n, n)).tocsr()
deg = np.asarray(A.sum(1)).ravel()
lap = (A @ m.vertices) / deg[:, None] - m.vertices
conc = np.einsum("ij,ij->i", lap, m.vertex_normals)  # >0 = concave (crease), <0 = convex (ridge)
# smooth twice and normalise by local edge length
for _ in range(2):
    conc = (A @ conc) / deg
loc = np.median(m.edges_unique_length)
k = conc / loc
lo, hi = np.percentile(k, [2, 98])
t = np.clip((k - lo) / (hi - lo), 0, 1)
# blue = convex ridges, white = flat, dark red = deep creases
c = np.zeros((n, 4), np.uint8); c[:, 3] = 255
mid = 0.5 + 0 * t
w = np.clip((t - lo / (lo - hi) * -1), 0, 1)
crease = np.clip((t - 0.55) / 0.45, 0, 1)
ridge = np.clip((0.45 - t) / 0.45, 0, 1)
c[:, 0] = (235 - 120 * ridge - 20 * crease).astype(np.uint8)
c[:, 1] = (235 - 60 * ridge - 200 * crease).astype(np.uint8)
c[:, 2] = (235 - 0 * ridge - 200 * crease).astype(np.uint8)
km = m.copy(); km.visual = trimesh.visual.ColorVisuals(km, vertex_colors=c)
km.export(os.path.join(out, "crease_map.ply"))
say(f"crease map: {(crease > 0.5).mean()*100:.1f}% of vertices strongly concave")
rep.close()
