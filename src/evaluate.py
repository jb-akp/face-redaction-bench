"""Compose configs from cached passes, pick the tuned threshold on the DEV split (even k), report on TEST (odd k)."""
import json, numpy as np
from common import nms, match, ROOT
from stress import CONDS

R = {c: json.load(open(ROOT / f"results/passes_{c}.json")) for c in CONDS}
REFL = [c for c in CONDS if c != "clean"]
CONFIGS = {
    "yunet_0.9 (their setting)": None,
    "yunet_low_thr": ["orig@1x"],
    "+clahe+stretch": ["orig@1x", "clahe@1x", "stretch@1x"],
    "+clahe+stretch+2x": ["orig@1x", "clahe@1x", "stretch@1x", "orig@2x", "clahe@2x", "stretch@2x"],
    "+clahe+stretch+2x(small boxes only)": ["orig@1x", "clahe@1x", "stretch@1x", "orig@2x<", "clahe@2x<", "stretch@2x<"],
    "+clahe+stretch+2x+flip+sharpen": ["orig@1x", "clahe@1x", "stretch@1x", "orig@2x", "clahe@2x", "stretch@2x", "orig@1x_flip", "sharpen@1x"],
}

def dets(rec, passes):
    if passes is None: return np.array(rec["baseline"], np.float32).reshape(-1, 5)
    parts = []
    for p in passes:
        a = np.array(rec["passes"][p.rstrip("<")]["d"], np.float32).reshape(-1, 5)
        if p.endswith("<"): a = a[a[:, 3] < 48]     # upscaled pass only contributes faces < 48 px tall
        parts.append(a)
    a = np.concatenate(parts)
    return nms(a, 0.3)

def split(c, s): return [r for r in R[c] if r["k"] % 2 == s]

# cache NMS-merged dets once per config to make the threshold sweep fast
for name, passes in CONFIGS.items():
    for c in CONDS:
        for r in R[c]: r.setdefault("_d", {})[name] = dets(r, passes)
def stats_c(recs, name, thr):
    tp = fn = fp = fpn = 0; nneg = 0
    for r in recs:
        d = r["_d"][name]; d = d[d[:, 4] >= thr]
        a, b, c = match(r["gt"], d); tp += a; fn += b; fp += c
        if r["neg"]: fpn += c; nneg += 1
    return dict(recall=tp / max(tp + fn, 1), precision=tp / max(tp + fp, 1), tp=tp, fn=fn, fp=fp, fp_per_noface=fpn / max(nneg, 1))

out = {"thresholds": {}, "test": {}, "pr_dev": {}}
for name, passes in CONFIGS.items():
    if passes is None: thr = 0.9
    else:
        dev = [r for c in CONDS for r in split(c, 0)]; best = None; curve = []
        for thr in np.round(np.arange(0.30, 0.951, 0.025), 3):
            s = stats_c(dev, name, thr); curve.append((float(thr), s["recall"], s["precision"]))
            if s["precision"] >= 0.95 and (best is None or s["recall"] > best[1]): best = (float(thr), s["recall"])
        thr = best[0] if best else 0.9
        out["pr_dev"][name] = curve
    out["thresholds"][name] = thr
    out["test"][name] = {c: stats_c(split(c, 1), name, thr) for c in CONDS}
    out["test"][name]["ALL_REFLECTION"] = stats_c([r for c in REFL for r in split(c, 1)], name, thr)
    out["test"][name]["ALL"] = stats_c([r for c in CONDS for r in split(c, 1)], name, thr)
# test-split operating points for the main configs (pooled reflection conditions)
out["test_pr"] = {n: [(t, *[round(v, 4) for v in (lambda s: (s["recall"], s["precision"], s["fp_per_noface"]))(stats_c([r for c in REFL for r in split(c, 1)], n, t))])
                      for t in (0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9)] for n in ("yunet_low_thr", "+clahe+stretch", "+clahe+stretch+2x(small boxes only)")}
json.dump(out, open(ROOT / "results/eval.json", "w"), indent=1)

def fmt(s): return f"{100*s['recall']:.1f}% / {100*s['precision']:.1f}% / {s['fp_per_noface']:.3f}"
names = list(CONFIGS)
print("TEST split. cell = recall / precision / false alarms per no-face image")
print("| condition | " + " | ".join(f"{n} (thr {out['thresholds'][n]})" for n in names) + " |")
print("|---" * (len(names) + 1) + "|")
for c in CONDS + ["ALL_REFLECTION", "ALL"]:
    print(f"| {c} | " + " | ".join(fmt(out["test"][n][c]) for n in names) + " |")
print("\nTEST, pooled reflection conditions: thr -> recall, precision, false alarms per no-face image")
for n_, pts in out["test_pr"].items(): print(n_, pts)
n = len([r for c in CONDS for r in split(c, 1)]); print("test samples:", n, "faces:", sum(len(r['gt']) for c in CONDS for r in split(c, 1)))
