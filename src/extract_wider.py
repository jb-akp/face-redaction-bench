"""Pick (max 6 per WIDER event category for scene diversity) a small clean subset of WIDER FACE val: 1-3 faces, all >=80px tall, no blur/occlusion/invalid."""
import json, sys, pyarrow.parquet as pq
from pathlib import Path
src, out = Path(sys.argv[1]), Path(sys.argv[2])
(out / "img").mkdir(parents=True, exist_ok=True)
f = pq.ParquetFile(src); ann = []; MAX = 160; per_event = {}; CAP = 6
for g in range(f.metadata.num_row_groups):
    for r in f.read_row_group(g, columns=["image", "faces"]).to_pylist():
        fc = r["faces"]; n = len(fc["bbox"])
        if not 1 <= n <= 3: continue
        if any(b[3] < 80 or b[2] < 50 for b in fc["bbox"]): continue
        if any(fc["blur"]) or any(fc["occlusion"]) or any(fc["invalid"]) or any(fc["pose"]): continue
        name = Path(r["image"]["path"]).name
        ev = name.split("_")[0]
        if per_event.get(ev, 0) >= CAP: continue
        per_event[ev] = per_event.get(ev, 0) + 1
        (out / "img" / name).write_bytes(r["image"]["bytes"])
        ann.append({"file": name, "boxes": [list(map(float, b)) for b in fc["bbox"]]})
        if len(ann) >= MAX: break
    if len(ann) >= MAX: break
json.dump(ann, open(out / "annotations.json", "w"), indent=0)
print(len(ann), "images", sum(len(a["boxes"]) for a in ann), "faces")
