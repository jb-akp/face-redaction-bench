"""Run every YuNet pass once per sample at a low threshold (0.3) and cache detections + timings.
Configs (baseline / tuned / ablations) are then composed offline in evaluate.py."""
import json, sys, time, cv2, numpy as np
from common import tiled, yunet, clahe, stretch, sharpen, ROOT
from stress import sample, n_samples, CONDS

PRE = {"orig": lambda x: x, "clahe": clahe, "stretch": stretch, "sharpen": sharpen}
PASSES = [("orig", 1, 0), ("clahe", 1, 0), ("stretch", 1, 0), ("sharpen", 1, 0), ("orig", 1, 1),
          ("orig", 2, 0), ("clahe", 2, 0), ("stretch", 2, 0)]

def run(img, pre, scale, flip):
    t = time.perf_counter()
    x = PRE[pre](img)
    if scale != 1: x = cv2.resize(x, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    if flip: x = cv2.flip(x, 1)
    d = tiled(x) if x.shape[1] > 2000 else yunet(x, 0.3).copy()
    if flip: d[:, 0] = x.shape[1] - d[:, 0] - d[:, 2]
    d[:, :4] /= scale
    return d.round(2).tolist(), time.perf_counter() - t

for cond in (sys.argv[1:] or CONDS):
    out = []
    for k in range(n_samples()):
        img, gt, neg = sample(cond, k)
        t = time.perf_counter(); base = yunet(img, 0.9).round(2).tolist(); tb = time.perf_counter() - t
        rec = {"k": k, "neg": neg, "gt": gt, "shape": img.shape[:2], "baseline": base, "t_baseline": tb, "passes": {}}
        for p in PASSES:
            d, dt = run(img, *p); rec["passes"]["%s@%dx%s" % (p[0], p[1], "_flip" if p[2] else "")] = {"d": d, "t": dt}
        out.append(rec)
    json.dump(out, open(ROOT / f"results/passes_{cond}.json", "w"))
    print(cond, "done", flush=True)
