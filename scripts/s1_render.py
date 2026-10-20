"""Step 1 renders: shell-coloured views and crease map views."""
import json, os, subprocess
import numpy as np
import trimesh

ROOT = r"C:\github_storage\pro-device"
BLENDER = r"C:\Users\ADMINI~1\AppData\Local\Temp\1\claude\c--github-storage-pro-device\c0ddccea-bb77-4948-b51b-3398cc943d2b\scratchpad\blender_dl\blender-4.5.14-windows-x64\blender.exe"
TMP = os.path.join(ROOT, "01_repaired", "_render_tmp"); os.makedirs(TMP, exist_ok=True)
VIEWS = [{"name": "front", "dir": [0, -1, 0]}, {"name": "back", "dir": [0, 1, 0]},
         {"name": "left", "dir": [-1, 0, 0]}, {"name": "right", "dir": [1, 0, 0]}]

def render(ply, prefix, kind):
    m = trimesh.load(ply, process=False)
    npz = os.path.join(TMP, prefix + ".npz")
    if kind == "face":
        np.savez(npz, vertices=m.vertices, faces=m.faces, face_colors=m.visual.face_colors[:, :3])
    else:
        np.savez(npz, vertices=m.vertices, faces=m.faces, vertex_colors=m.visual.vertex_colors[:, :3])
    js = os.path.join(TMP, prefix + ".json")
    json.dump({"out_dir": os.path.join(ROOT, "renders"), "prefix": prefix, "views": VIEWS, "samples": 40}, open(js, "w"))
    r = subprocess.run([BLENDER, "-b", "--factory-startup", "-P", os.path.join(ROOT, "scripts", "render_npz.py"), "--", npz, js],
                       capture_output=True, text=True)
    print("\n".join(l for l in r.stdout.splitlines() if "RENDERED" in l or "Error" in l)); print(r.stderr[-1500:])

render(os.path.join(ROOT, "01_repaired", "shells_colored.ply"), "s1_shells", "face")
render(os.path.join(ROOT, "01_repaired", "crease_map.ply"), "s1_creases", "vertex")
