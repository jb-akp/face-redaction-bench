"""Synthetic head-cam reflection clip: a mirrored face drifts/jitters across a 1280x720 frame for 120 frames;
~35% of frames get a burst degradation (frosted glass, glare, or motion blur). Compare per-frame coverage
of the face with raw detections vs detections + offline IoU tracker."""
import json, cv2, numpy as np
from common import ROOT, yunet, tuned, iou
from stress import ANN, load_pos, mirror, glass, glare, room_bg
from tracker import track
from redact import pad_box

ev = json.load(open(ROOT / "results/eval.json")); THR = ev["thresholds"]["+clahe+stretch+2x(small boxes only)"]

def covered(gt, boxes, need=0.9):
    """True if the face is >=90% inside one of the padded redaction boxes (what actually gets pixelated)."""
    gx2, gy2 = gt[0] + gt[2], gt[1] + gt[3]
    for b in boxes:
        x0, y0, x1, y1 = pad_box(b, (720, 1280)); b = [x0, y0, x1 - x0, y1 - y0]
        iw = max(0, min(gx2, b[0] + b[2]) - max(gt[0], b[0])); ih = max(0, min(gy2, b[1] + b[3]) - max(gt[1], b[1]))
        if iw * ih >= need * gt[2] * gt[3]: return True
    return False

def clip(seed, n=120):
    rng = np.random.default_rng(seed)
    single = [i for i, a in enumerate(ANN) if len(a["boxes"]) == 1]   # one face only, so no unlabelled neighbours leak into the crop
    img, boxes = load_pos(int(rng.choice(single)))
    b = max(boxes, key=lambda r: r[3])
    s = rng.uniform(55, 75) / b[3]
    x0, y0 = int(max(0, b[0] - b[2])), int(max(0, b[1] - b[3] * 0.8))
    crop = img[y0:int(b[1] + 2 * b[3]), x0:int(b[0] + 2 * b[2])]
    crop = cv2.resize(crop, None, fx=s, fy=s); fb = [(b[0] - x0) * s, (b[1] - y0) * s, b[2] * s, b[3] * s]
    crop, (fb,) = mirror(crop, [fb], rng)
    bg = room_bg(rng, 1280, 720); ch, cw = crop.shape[:2]
    frames, gts, bad = [], [], []
    for f in range(n):
        ox = int(100 + (1280 - cw - 200) * f / n + rng.normal(0, 6)); oy = int(300 + 80 * np.sin(f / 15) + rng.normal(0, 6))
        oy = min(max(oy, 0), 720 - ch); fr = bg.copy(); fr[oy:oy + ch, ox:ox + cw] = crop
        g = [fb[0] + ox, fb[1] + oy, fb[2], fb[3]]; deg = rng.random() < 0.35
        if deg:
            kind = rng.integers(3)
            if kind == 0: fr = glass(fr, rng, 5.0, 8)
            elif kind == 1: fr = glare(fr, rng)
            else:
                k = np.zeros((25, 25), np.float32); k[12] = 1 / 25
                M = cv2.getRotationMatrix2D((12, 12), rng.uniform(0, 180), 1); fr = cv2.filter2D(fr, -1, cv2.warpAffine(k, M, (25, 25)))
        frames.append(fr); gts.append(g); bad.append(deg)
    return frames, gts, bad

rows = []
for seed in range(5):
    frames, gts, bad = clip(seed)
    raw = {"yunet_0.9": [yunet(f, 0.9).tolist() for f in frames],
           "tuned": [(lambda d: d[d[:, 4] >= THR].tolist())(tuned(f, 0.3, variants=("orig", "clahe", "stretch"), upscale_small=True)) for f in frames]}
    for name, per in raw.items():
        tr = track(per)
        stray_raw = sum(sum(iou(g, b) == 0 for b in d) for g, d in zip(gts, per))
        cov_raw = sum(covered(g, d) for g, d in zip(gts, per)); cov_tr = sum(covered(g, d) for g, d in zip(gts, tr))
        fa = sum(sum(iou(g, b) == 0 for b in d) for g, d in zip(gts, tr))   # boxes not touching the face at all
        rows.append(dict(clip=seed, detector=name, frames=len(frames), degraded=sum(bad), covered_raw=cov_raw, covered_tracked=cov_tr, stray_boxes_raw=stray_raw, stray_boxes_after_tracking=fa))
        print(rows[-1], flush=True)
json.dump(rows, open(ROOT / "results/track_demo.json", "w"), indent=1)
for name in ("yunet_0.9", "tuned"):
    r = [x for x in rows if x["detector"] == name]; n = sum(x["frames"] for x in r)
    print(f"{name}: covered raw {sum(x['covered_raw'] for x in r)}/{n}  tracked {sum(x['covered_tracked'] for x in r)}/{n}  stray raw {sum(x['stray_boxes_raw'] for x in r)}  extra boxes {sum(x['stray_boxes_after_tracking'] for x in r)}")
