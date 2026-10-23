"""(a) Recovery attack on redacted faces: smooth + upscale + unsharp. Measure face re-detection, SSIM and
SFace identity match vs the original. (b) Byte-identical check outside the padded boxes (in memory and after PNG round-trip)."""
import json, cv2, numpy as np, tempfile, os
from common import ROOT, yunet, iou
from stress import ANN, load_pos
from redact import redact, light_blur, pad_box

rec = cv2.FaceRecognizerSF.create(str(ROOT / "models/face_recognition_sface_2021dec.onnx"), "")
YN = cv2.FaceDetectorYN.create(str(ROOT / "models/face_detection_yunet_2023mar.onnx"), "", (320, 320), 0.5)
SAME = 0.363  # OpenCV SFace cosine threshold for "same identity"

def attack(img, box):
    x0, y0, x1, y1 = pad_box(box, img.shape); r = img[y0:y1, x0:x1]
    sm = cv2.GaussianBlur(r, (0, 0), max(1.0, (y1 - y0) / 12))            # remove block edges
    up = cv2.resize(cv2.resize(sm, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA), (x1 - x0, y1 - y0), interpolation=cv2.INTER_CUBIC)
    sh = cv2.addWeighted(up, 2.5, cv2.GaussianBlur(up, (0, 0), 2), -1.5, 0)  # unsharp "deblur"
    out = img.copy(); out[y0:y1, x0:x1] = sh; return out

def ssim(a, b):
    a, b = [cv2.cvtColor(x, cv2.COLOR_BGR2GRAY).astype(np.float64) for x in (a, b)]
    C1, C2 = 6.5025, 58.5225; mu = lambda x: cv2.GaussianBlur(x, (11, 11), 1.5)
    ma, mb = mu(a), mu(b); va, vb, cov = mu(a * a) - ma ** 2, mu(b * b) - mb ** 2, mu(a * b) - ma * mb
    return float((((2 * ma * mb + C1) * (2 * cov + C2)) / ((ma ** 2 + mb ** 2 + C1) * (va + vb + C2))).mean())

emb_orig, emb_red = [], []   # for the chance-level baseline below
res = {m: dict(faces=0, redetected=0, ssim=[], id_faces=0, id_match=0) for m in ("light_gaussian_control", "pixelate_noise (ours)", "pixelate_noise + attack")}
outside = dict(images=0, identical_in_memory=0, identical_after_png=0)
for i in range(len(ANN)):
    img, gt = load_pos(i)
    YN.setInputSize((img.shape[1], img.shape[0])); det15 = YN.detect(img)[1]
    det15 = [] if det15 is None else list(det15)
    variants = {"light_gaussian_control": light_blur(img, gt), "pixelate_noise (ours)": redact(img, gt, seed=i)}
    variants["pixelate_noise + attack"] = variants["pixelate_noise (ours)"]
    for g in gt: variants["pixelate_noise + attack"] = attack(variants["pixelate_noise + attack"], g)
    for m, out in variants.items():
        d2 = yunet(out, 0.5)
        for g in gt:
            R = res[m]; R["faces"] += 1
            if any(iou(g, d) >= 0.3 for d in d2): R["redetected"] += 1
            x0, y0, x1, y1 = pad_box(g, img.shape); R["ssim"].append(ssim(img[y0:y1, x0:x1], out[y0:y1, x0:x1]))
            m0 = [d for d in det15 if iou(g, d) >= 0.4]
            if m0:   # identity: align both crops with the ORIGINAL face's landmarks (best case for the attacker)
                fr = m0[0]
                e0 = rec.feature(rec.alignCrop(img, fr)); e1 = rec.feature(rec.alignCrop(out, fr))
                if m == "pixelate_noise (ours)": emb_orig.append(e0); emb_red.append(e1)
                R["id_faces"] += 1; R["id_match"] += int(rec.match(e0, e1, cv2.FaceRecognizerSF_FR_COSINE) >= SAME)
    # (b) outside the padded boxes must be byte-identical
    out = variants["pixelate_noise (ours)"]; mask = np.zeros(img.shape[:2], bool)
    for g in gt: x0, y0, x1, y1 = pad_box(g, img.shape); mask[y0:y1, x0:x1] = True
    outside["images"] += 1; outside["identical_in_memory"] += int(np.array_equal(img[~mask], out[~mask]))
    fd, p = tempfile.mkstemp(suffix=".png"); os.close(fd); cv2.imwrite(p, out); back = cv2.imread(p); os.remove(p)
    outside["identical_after_png"] += int(np.array_equal(img[~mask], back[~mask]))
summary = {m: dict(faces=R["faces"], redetected_by_yunet_0_5=R["redetected"], mean_ssim_vs_original=round(float(np.mean(R["ssim"])), 3),
                   identity_checked=R["id_faces"], sface_same_identity=R["id_match"]) for m, R in res.items()}
summary["outside_box_check"] = outside
# chance level: how often does a redacted face 'match' a DIFFERENT person's original at the same threshold?
n = len(emb_orig); diff = sum(rec.match(emb_orig[j], emb_red[i], cv2.FaceRecognizerSF_FR_COSINE) >= SAME for i in range(n) for j in range(n) if i != j)
summary["sface_chance_match_rate_vs_other_people"] = round(diff / (n * (n - 1)), 4)
json.dump(summary, open(ROOT / "results/redaction_verify.json", "w"), indent=1); print(json.dumps(summary, indent=1))
