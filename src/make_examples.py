"""Before/after redaction examples from the synthetic set, using the tuned detector at the dev-chosen threshold."""
import json, cv2, numpy as np
from common import ROOT, tuned, yunet
from stress import sample
from redact import redact, pad_box

THR = json.load(open(ROOT / "results/eval.json"))["thresholds"]["+clahe+stretch+2x(small boxes only)"]
for cond, k in [("mirror", 11), ("appliance_door", 24), ("shower_glass_light", 37), ("small_mirror", 59), ("shower_glass_heavy", 37)]:
    img, gt, _ = sample(cond, k)
    d = tuned(img, 0.3); d = d[d[:, 4] >= THR]
    base = yunet(img, 0.9)
    out = redact(img, d.tolist(), seed=k)
    if cond.startswith("small"):   # crop around the faces so the 20-40 px faces are visible
        x0 = int(max(0, min(g[0] for g in gt) - 200)); y0 = int(max(0, min(g[1] for g in gt) - 150))
        img, out = img[y0:y0 + 300, x0:x0 + 400], out[y0:y0 + 300, x0:x0 + 400]
        img, out = [cv2.resize(v, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST) for v in (img, out)]
    h = 360; a, b = [cv2.resize(v, (int(v.shape[1] * h / v.shape[0]), h)) for v in (img, out)]
    lab = f"{cond}: YuNet@0.9 found {len(base)}/{len(gt)} | tuned found {len(d)} box(es) for {len(gt)} face(s)"
    canvas = np.hstack([a, np.full((h, 8, 3), 255, np.uint8), b])
    canvas = np.vstack([np.full((28, canvas.shape[1], 3), 255, np.uint8), canvas])
    cv2.putText(canvas, lab, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(str(ROOT / f"examples/before_after_{cond}.jpg"), canvas, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(lab)
