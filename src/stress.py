"""Synthetic reflection stress set. Positives: WIDER FACE val subset. Negatives: face-free synthetic + background crops.
Every sample is generated deterministically from (condition, index) so nothing large is stored on disk."""
import json, cv2, numpy as np
from common import ROOT

ANN = json.load(open(ROOT / "data/wider_subset/annotations.json"))
CONDS = ["clean", "mirror", "appliance_door", "shower_glass_light", "shower_glass_heavy",
         "small_scale", "small_mirror", "mirror_occluded"]
N_NEG = 200

def load_pos(i):
    a = ANN[i]; img = cv2.imread(str(ROOT / "data/wider_subset/img" / a["file"]))
    s = 640 / max(img.shape[:2])
    return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), [[v * s for v in b] for b in a["boxes"]]

def hflip(img, boxes):
    W = img.shape[1]
    return cv2.flip(img, 1), [[W - b[0] - b[2], b[1], b[2], b[3]] for b in boxes]

def veil(img, alpha, level, rng):
    h, w = img.shape[:2]
    g = np.linspace(level - 25, level + 25, w)[None, :, None] * np.ones((h, 1, 3))
    return np.clip(img * alpha + g * (1 - alpha), 0, 255).astype(np.uint8)

def mirror(img, boxes, rng):
    img, boxes = hflip(img, boxes)
    img = cv2.GaussianBlur(img, (0, 0), 1.0)                        # camera focused past the mirror plane
    return veil(img, rng.uniform(0.5, 0.65), 185, rng), boxes        # silvering / smudge haze

def glare(img, rng):
    h, w = img.shape[:2]; m = np.zeros((h, w), np.float32)
    c = (int(rng.uniform(0.2, 0.8) * w), int(rng.uniform(0.2, 0.8) * h))
    cv2.ellipse(m, c, (int(w * rng.uniform(0.08, 0.2)), int(h * rng.uniform(0.4, 0.8))), rng.uniform(20, 70), 0, 360, 1.0, -1)
    m = cv2.GaussianBlur(m, (0, 0), w * 0.04)[..., None]
    return np.clip(img + m * rng.uniform(150, 230), 0, 255).astype(np.uint8)

def appliance_door(img, boxes, rng):
    img, boxes = hflip(img, boxes)
    tint = np.array([1.05, 1.0, 0.9]) * rng.uniform(0.3, 0.45)   # dark steel / black glass
    img = cv2.GaussianBlur(img, (0, 0), 1.5).astype(np.float32) * tint + 25
    return glare(np.clip(img, 0, 255).astype(np.uint8), rng), boxes

def glass(img, rng, sigma, amp):
    h, w = img.shape[:2]
    n = cv2.GaussianBlur(rng.standard_normal((h, w, 2)).astype(np.float32), (0, 0), 4)
    n *= amp / (n.std() + 1e-6)
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    img = cv2.remap(img, gx + n[..., 0], gy + n[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    img = cv2.GaussianBlur(img, (0, 0), sigma)
    return veil(img, 0.5, 200, rng)                                  # frosted / fogged

def shower(img, boxes, rng, heavy):
    img, boxes = hflip(img, boxes)
    return glass(img, rng, 5.0 if heavy else 2.5, 8 if heavy else 4), boxes

def occlude(img, boxes, rng):
    img = img.copy()
    for b in boxes:
        x, y, w, h = map(int, b); frac = rng.uniform(0.3, 0.45); side = rng.integers(4)
        col = tuple(int(c) for c in rng.integers(40, 220, 3))
        if side == 0: cv2.rectangle(img, (x, y + int(h * (1 - frac))), (x + w, y + h), col, -1)   # towel / hand below
        elif side == 1: cv2.rectangle(img, (x, y), (x + w, y + int(h * frac)), col, -1)           # cap / door frame above
        elif side == 2: cv2.rectangle(img, (x, y), (x + int(w * frac), y + h), col, -1)           # mirror edge left
        else: cv2.rectangle(img, (x + int(w * (1 - frac)), y), (x + w, y + h), col, -1)          # mirror edge right
    return img, boxes

def room_bg(rng, w=1920, h=1080):
    bg = np.zeros((h, w, 3), np.float32) + rng.uniform(60, 200, 3)
    for _ in range(12):
        p = (int(rng.uniform(0, w)), int(rng.uniform(0, h)))
        cv2.rectangle(bg, p, (p[0] + int(rng.uniform(100, 700)), p[1] + int(rng.uniform(100, 500))), rng.uniform(20, 235, 3).tolist(), -1)
    bg = cv2.GaussianBlur(bg, (0, 0), 6) + rng.normal(0, 4, (h, w, 3))
    return np.clip(bg, 0, 255).astype(np.uint8)

def small(img, boxes, rng, deg=None):
    """Paste the whole image scaled so the largest face is 20-40 px tall into a 1920x1080 frame."""
    target = rng.uniform(20, 40); mh = max(b[3] for b in boxes) if boxes else img.shape[0] * 0.3
    s = target / mh
    p = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    bx = [[v * s for v in b] for b in boxes]
    if deg: p, bx = deg(p, bx, rng)
    bg = room_bg(rng); ph, pw = p.shape[:2]
    ox, oy = int(rng.uniform(0, 1920 - pw)), int(rng.uniform(0, 1080 - ph))
    bg[oy:oy + ph, ox:ox + pw] = p
    return bg, [[b[0] + ox, b[1] + oy, b[2], b[3]] for b in bx]

# ---------- negatives ----------
def cartoon(rng, w=640, h=480):
    img = room_bg(rng, w, h)
    for _ in range(rng.integers(1, 4)):
        r = int(rng.uniform(30, 110)); c = (int(rng.uniform(r, w - r)), int(rng.uniform(r, h - r)))
        col = rng.uniform(0, 255, 3).tolist(); kind = rng.integers(3)
        if kind == 0:   # smiley print
            cv2.circle(img, c, r, col, -1)
            for dx in (-0.35, 0.35): cv2.circle(img, (int(c[0] + dx * r), int(c[1] - 0.25 * r)), max(2, r // 8), (20, 20, 20), -1)
            cv2.ellipse(img, (c[0], c[1] + r // 6), (r // 2, r // 3), 0, 10, 170, (20, 20, 20), max(2, r // 12))
        elif kind == 1:  # teddy / toy head with ears and button eyes
            for dx in (-0.75, 0.75): cv2.circle(img, (int(c[0] + dx * r), int(c[1] - 0.75 * r)), r // 3, col, -1)
            cv2.circle(img, c, r, col, -1); cv2.circle(img, (c[0], c[1] + r // 4), r // 3, (200, 220, 235), -1)
            for dx in (-0.4, 0.4): cv2.circle(img, (int(c[0] + dx * r), int(c[1] - 0.2 * r)), max(3, r // 6), (10, 10, 10), -1)
            cv2.circle(img, (c[0], c[1] + r // 6), max(2, r // 9), (10, 10, 10), -1)
        else:            # anime / kawaii mascot: oversized eyes with highlights
            cv2.ellipse(img, c, (r, int(r * 1.1)), 0, 0, 360, (190, 210, 250), -1)
            for dx in (-0.4, 0.4):
                e = (int(c[0] + dx * r), int(c[1]))
                cv2.ellipse(img, e, (r // 5, r // 3), 0, 0, 360, (60, 30, 20), -1); cv2.circle(img, (e[0] - r // 15, e[1] - r // 8), max(2, r // 14), (255, 255, 255), -1)
            cv2.line(img, (c[0] - r // 8, c[1] + r // 2), (c[0] + r // 8, c[1] + r // 2), (60, 60, 160), 2)
    return img

def pattern(rng, w=640, h=480):
    """Packaging / fabric print: tiled smileys."""
    img = np.full((h, w, 3), rng.uniform(120, 255, 3), np.uint8); r = int(rng.uniform(10, 30))
    for y in range(r, h, 3 * r):
        for x in range(r, w, 3 * r):
            cv2.circle(img, (x, y), r, (0, 200, 255), -1)
            cv2.circle(img, (x - r // 3, y - r // 4), max(1, r // 7), (0, 0, 0), -1); cv2.circle(img, (x + r // 3, y - r // 4), max(1, r // 7), (0, 0, 0), -1)
            cv2.ellipse(img, (x, y + r // 6), (r // 2, r // 3), 0, 10, 170, (0, 0, 0), 2)
    return img

def bg_crop(rng):
    """Real-world clutter: crop from a WIDER image that avoids every annotated face (1.6x margin)."""
    for _ in range(200):
        img, boxes = load_pos(int(rng.integers(len(ANN))))
        h, w = img.shape[:2]; cw, ch = int(rng.uniform(160, 320)), int(rng.uniform(160, 320))
        if cw >= w or ch >= h: continue
        x, y = int(rng.uniform(0, w - cw)), int(rng.uniform(0, h - ch))
        ok = all(x + cw < b[0] - 0.3 * b[2] or x > b[0] + 1.3 * b[2] or y + ch < b[1] - 0.3 * b[3] or y > b[1] + 1.6 * b[3] for b in boxes)
        if ok: return cv2.resize(img[y:y + ch, x:x + cw], (480, 480))
    raise RuntimeError

def load_neg(j):
    rng = np.random.default_rng(10_000 + j)
    return cartoon(rng) if j < 80 else pattern(rng) if j < 110 else bg_crop(rng)

DEG = {"clean": lambda i, b, r: (i, b), "mirror": mirror, "appliance_door": appliance_door,
       "shower_glass_light": lambda i, b, r: shower(i, b, r, False), "shower_glass_heavy": lambda i, b, r: shower(i, b, r, True),
       "mirror_occluded": lambda i, b, r: occlude(*mirror(i, b, r), r)}

def sample(cond, k):
    """k < len(ANN): positive k; otherwise negative k-len(ANN). Returns (img, gt_boxes, is_negative)."""
    neg = k >= len(ANN)
    img, boxes = (load_neg(k - len(ANN)), []) if neg else load_pos(k)
    rng = np.random.default_rng((CONDS.index(cond) * 100_003 + k))
    if cond == "small_scale": img, boxes = small(img, boxes, rng)
    elif cond == "small_mirror": img, boxes = small(img, boxes, rng, mirror)
    else: img, boxes = DEG[cond](img, boxes, rng)
    return img, boxes, neg

def n_samples(): return len(ANN) + N_NEG
