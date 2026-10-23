import cv2, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
YUNET = str(ROOT / "models/face_detection_yunet_2023mar.onnx")
_det = {}

def yunet(img, thr):
    """Raw YuNet at native resolution. Returns array Nx5 (x,y,w,h,score)."""
    d = _det.get(thr)
    if d is None:
        d = _det[thr] = cv2.FaceDetectorYN.create(YUNET, "", (320, 320), thr, 0.3, 5000)
    h, w = img.shape[:2]
    d.setInputSize((w, h))
    _, f = d.detect(img)
    return np.zeros((0, 5), np.float32) if f is None else f[:, [0, 1, 2, 3, 14]].astype(np.float32)

def tiled(x, ov=96, thr=0.3):
    """2x2 overlapping tiles: same pixels as one big pass, ~4x lower peak memory (matters on an 8 GB machine)."""
    H, W = x.shape[:2]; out = []
    for y0, y1 in ((0, H // 2 + ov), (H // 2 - ov, H)):
        for x0, x1 in ((0, W // 2 + ov), (W // 2 - ov, W)):
            d = yunet(np.ascontiguousarray(x[y0:y1, x0:x1]), thr).copy(); d[:, 0] += x0; d[:, 1] += y0; out.append(d)
    return np.concatenate(out)

def clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[..., 0] = cv2.createCLAHE(3.0, (8, 8)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def stretch(img):
    """Per-image 1-99 percentile contrast stretch (undoes mirror veil / dark door tint)."""
    lo, hi = np.percentile(img, (1, 99))
    return np.clip((img.astype(np.float32) - lo) * 255.0 / max(hi - lo, 1), 0, 255).astype(np.uint8)

def sharpen(img):
    b = cv2.GaussianBlur(img, (0, 0), 3)
    return cv2.addWeighted(img, 2.0, b, -1.0, 0)

def nms(boxes, iou=0.3):
    if len(boxes) == 0: return boxes
    keep = cv2.dnn.NMSBoxes(boxes[:, :4].tolist(), boxes[:, 4].tolist(), 0.0, iou)
    return boxes[np.array(keep).reshape(-1)]

def tuned(img, thr=0.5, variants=("orig", "clahe", "stretch"), upscale_small=True, hflip=False):
    """Test-time augmentation: run YuNet on several photometric variants (+2x upscale, +hflip), merge with NMS."""
    fns = {"orig": lambda x: x, "clahe": clahe, "stretch": stretch, "sharpen": sharpen}
    dets = []
    for v in variants:
        x = fns[v](img)
        dets.append(yunet(x, thr))
        if hflip:
            W = x.shape[1]; f = yunet(cv2.flip(x, 1), thr).copy()
            f[:, 0] = W - f[:, 0] - f[:, 2]; dets.append(f)
        if upscale_small:
            x2 = cv2.resize(x, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            u = tiled(x2, thr=thr) if x2.shape[1] > 2000 else yunet(x2, thr).copy()
            u[:, :4] /= 2; dets.append(u[u[:, 3] < 48])   # upscaled pass only contributes small faces
    return nms(np.concatenate(dets))

def iou(a, b):
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0, min(ax2, bx2) - max(a[0], b[0])); ih = max(0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    return inter / (a[2] * a[3] + b[2] * b[3] - inter + 1e-9)

def match(gt, dets, thr=0.4):
    """Greedy match by score. Returns (tp, fn, fp)."""
    used = set(); tp = 0
    for d in sorted(dets.tolist(), key=lambda r: -r[4]):
        best, bi = 0, -1
        for i, g in enumerate(gt):
            if i in used: continue
            v = iou(d, g)
            if v > best: best, bi = v, i
        if best >= thr: used.add(bi); tp += 1
    return tp, len(gt) - tp, len(dets) - tp
