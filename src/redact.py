import cv2, numpy as np

def pad_box(b, shape, pad=0.2):
    x, y, w, h = b[:4]; H, W = shape[:2]
    x0, y0 = int(max(0, np.floor(x - pad * w))), int(max(0, np.floor(y - pad * h)))
    x1, y1 = int(min(W, np.ceil(x + w * (1 + pad)))), int(min(H, np.ceil(y + h * (1 + pad))))
    return x0, y0, x1, y1

def redact(img, boxes, blocks=5, noise=48, seed=None):
    """Irreversible: each padded face box becomes a ~5x5 grid of block means plus per-block random noise.
    Only pixels inside the padded boxes are written; everything else is untouched."""
    out = img.copy(); rng = np.random.default_rng(seed)
    for b in boxes:
        x0, y0, x1, y1 = pad_box(b, img.shape)
        if x1 <= x0 or y1 <= y0: continue
        r = out[y0:y1, x0:x1]
        g = cv2.resize(r, (blocks, blocks), interpolation=cv2.INTER_AREA).astype(np.int16)
        g = np.clip(g + rng.integers(-noise, noise + 1, g.shape), 0, 255).astype(np.uint8)
        out[y0:y1, x0:x1] = cv2.resize(g, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST)
    return out

def light_blur(img, boxes):
    """The common 'light gaussian' redaction, used here only as a control."""
    out = img.copy()
    for b in boxes:
        x0, y0, x1, y1 = pad_box(b, img.shape)
        if x1 <= x0 or y1 <= y0: continue
        s = max(1.0, 0.04 * (y1 - y0))
        out[y0:y1, x0:x1] = cv2.GaussianBlur(out[y0:y1, x0:x1], (0, 0), s)
    return out
