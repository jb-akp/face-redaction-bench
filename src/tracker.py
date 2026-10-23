"""Offline IoU tracker: link detections frame to frame, drop short tracks, interpolate gaps, pad track ends.
Offline (two-pass) is fine for redaction: the whole clip is available before export."""
import numpy as np
from common import iou

def track(per_frame, link_iou=0.2, max_gap=15, min_hits=3, extend=5):
    tracks = []  # each: dict frame -> box(x,y,w,h)
    for f, dets in enumerate(per_frame):
        for d in sorted(dets, key=lambda r: -r[4]):
            best, bt = link_iou, None
            for t in tracks:
                last = max(t)
                if t.get(f) is not None or f - last > max_gap: continue
                v = iou(t[last], d)
                # allow linking across a gap if the box is near where it was (centre within one box size)
                if v < link_iou and f - last > 1:
                    lb = t[last]; c1 = np.array([lb[0] + lb[2] / 2, lb[1] + lb[3] / 2]); c2 = np.array([d[0] + d[2] / 2, d[1] + d[3] / 2])
                    v = link_iou + 1e-3 if np.linalg.norm(c1 - c2) < max(lb[2], lb[3]) * (1 + 0.1 * (f - last)) else 0
                if v > best: best, bt = v, t
            if bt is None: tracks.append({f: list(d[:4])})
            else: bt[f] = list(d[:4])
    out = [[] for _ in per_frame]
    for t in tracks:
        if len(t) < min_hits: continue
        fs = sorted(t)
        for a, b in zip(fs, fs[1:]):
            for f in range(a, b):
                w = (f - a) / (b - a); out[f].append([t[a][i] * (1 - w) + t[b][i] * w for i in range(4)])
        out[fs[-1]].append(t[fs[-1]])
        for f in range(max(0, fs[0] - extend), fs[0]): out[f].append(t[fs[0]])
        for f in range(fs[-1] + 1, min(len(per_frame), fs[-1] + extend + 1)): out[f].append(t[fs[-1]])
    return out
