"""Seconds per frame on this machine, run alone (nothing else running). Median of 7 runs after warm-up."""
import json, sys, time, cv2, numpy as np
from common import ROOT, yunet, tuned, tiled
from stress import sample

def med(fn, x, n=7):
    fn(x); ts = []
    for _ in range(n):
        t = time.perf_counter(); fn(x); ts.append(time.perf_counter() - t)
    return float(np.median(ts))

img, _, _ = sample("small_mirror", 7)                       # 1920x1080 frame
frames = {"1920x1080 (4K downscaled 2x)": img, "3840x2160 (native 4K)": cv2.resize(img, (3840, 2160))}
cfgs = {"yunet_0.9 single pass": lambda x: (tiled(x, thr=0.9) if x.shape[1] > 2000 else yunet(x, 0.9)),
        "tuned (orig+clahe+stretch, 1x+2x tiled)": lambda x: tuned(x, 0.3),
        "tuned 1x only (orig+clahe+stretch, no upscale)": lambda x: tuned(x, 0.3, upscale_small=False)}
out = {}
for threads in (8, 4):
    cv2.setNumThreads(threads)
    for fname, x in frames.items():
        for cname, fn in cfgs.items():
            if "native" in fname and "2x" in cname: continue      # 2x of 4K = 8K: not a sensible config
            out[f"{threads} threads | {fname} | {cname}"] = round(med(fn, x), 4); print(list(out.items())[-1], flush=True)
json.dump(out, open(ROOT / "results/timing.json", "w"), indent=1)
