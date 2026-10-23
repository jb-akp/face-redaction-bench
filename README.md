# Face redaction benchmark: reflections, small faces, irreversible blur

A small, reproducible CPU benchmark of OpenCV YuNet for redacting faces in household head-cam video, focused on the failure you described: faces in mirrors, appliance doors and shower glass.

**Read this first: every number here is measured on synthetic reflections made from public photos, not on your footage.** It shows the method and the size of the gaps. The real numbers have to come from your answer-key frames.

## Results (test split, 1,440 images, 840 faces)

Each cell: **recall / precision / false alarms per no-face image**. Match = IoU >= 0.4 with the ground-truth box. Thresholds for the tuned configs were picked on a separate dev split (best recall with precision >= 95%), then frozen and applied to the test split.

| condition | YuNet @ 0.9 (your setting) | YuNet @ 0.75 | + CLAHE + contrast stretch TTA @ 0.825 | + 2x upscale for small faces @ 0.85 (**tuned**) |
|---|---|---|---|---|
| clean (no reflection) | 81.0% / 100% / 0.000 | 99.0% / 96.3% / 0.000 | 98.1% / 98.1% / 0.000 | 97.1% / 95.3% / 0.010 |
| mirror | 70.5% / 100% / 0.000 | 97.1% / 97.1% / 0.000 | 96.2% / 99.0% / 0.000 | 94.3% / 99.0% / 0.000 |
| appliance door (dark tint + glare) | 32.4% / 100% / 0.000 | 69.5% / 97.3% / 0.010 | 76.2% / 100% / 0.000 | 69.5% / 100% / 0.000 |
| shower glass, light frost | 1.9% / 100% / 0.000 | 54.3% / 91.9% / 0.050 | 60.0% / 92.6% / 0.050 | 52.4% / 93.2% / 0.040 |
| shower glass, heavy frost | 0.0% / n/a / 0.000 | 8.6% / 75.0% / 0.030 | 12.4% / 68.4% / 0.040 | 7.6% / 80.0% / 0.010 |
| small face (20-40 px in 1080p) | 22.9% / 100% / 0.000 | 74.3% / 98.7% / 0.000 | 61.0% / 100% / 0.000 | 88.6% / 98.9% / 0.000 |
| small face in a mirror | 6.7% / 100% / 0.000 | 67.6% / 100% / 0.000 | 61.0% / 100% / 0.000 | 74.3% / 100% / 0.000 |
| mirror + partial occlusion | 26.7% / 100% / 0.000 | 81.0% / 97.7% / 0.000 | 81.9% / 98.9% / 0.000 | 78.1% / 98.8% / 0.000 |
| **all reflection conditions** | **23.0% / 100% / 0.000** | 64.6% / 96.7% / 0.013 | 64.1% / 97.3% / 0.013 | **66.4% / 98.2% / 0.007** |
| all conditions | 30.2% / 100% / 0.000 | 68.9% / 96.7% / 0.011 | 68.3% / 97.5% / 0.011 | 70.2% / 97.7% / 0.007 |

What this says, plainly:

1. **The 0.9 threshold is most of the problem.** On these images YuNet scores many real faces 0.86-0.89 even with no reflection (median 0.92). Dropping to 0.75 takes reflection recall from 23% to 65% for zero extra compute.
2. **Contrast TTA and upscaling help specific cases, not everything.** Upscaling only for small boxes lifts small faces from 74% to 89%. CLAHE/stretch help dark appliance doors. Adding flip + sharpen TTA on top moved reflection recall by only +0.5 points (`results/eval.json`).
3. **Shower glass is not solved by tuning.** Heavy frost: under 13% recall for every config. That matches what you saw. It needs training data, not thresholds.
4. **None of this reaches 99% recall.** It moves the operating point. Getting to 99% / 95% needs the plan at the bottom.

Recall vs threshold on the test split (pooled reflection conditions) is in `results/eval.json` under `test_pr`. Example for the tuned config: thr 0.70 gives 83.0% recall at 80.7% precision; thr 0.80 gives 76.1% at 94.3%; thr 0.85 gives 66.4% at 98.2%.

## Temporal gap filling (tracker)

5 synthetic clips, 120 frames each, a mirrored face drifting and jittering across a 1280x720 frame. About 35% of frames get a burst of heavy frost, glare or motion blur. "Covered" = the face is at least 90% inside a padded redaction box.

| detector | frames covered, detector only | frames covered, + offline IoU tracker | stray boxes added by tracker |
|---|---|---|---|
| YuNet @ 0.9 | 347 / 600 (57.8%) | 464 / 600 (77.3%) | 6 |
| tuned @ 0.85 | 356 / 600 (59.3%) | 453 / 600 (75.5%) | 8 |

One of the 5 clips (clip 1) was never detected in any frame by either detector, so the tracker had nothing to extend. On the other 4 clips the tracker took YuNet @ 0.9 from 347/480 to 464/480 frames (96.7%). The tracker links detections by IoU or centre distance, drops tracks shorter than 3 hits, interpolates gaps up to 15 frames, and holds the box 5 frames past each end. It runs offline (two passes), which is fine for redaction because the whole clip exists before export.

## Redaction: irreversible, and nothing else touched

Each face box is padded 20%, reduced to a 5x5 grid of block means, given random noise of +/-48 per block, and written back with nearest-neighbour. Verified on all 213 faces (160 images):

| method | YuNet re-detects the face (thr 0.5) | SSIM vs original face | SFace says "same person" (cosine >= 0.363) |
|---|---|---|---|
| light Gaussian blur (common approach, control) | 189 / 213 | 0.568 | 61 / 211 |
| pixelate + noise (this repo) | 0 / 213 | 0.370 | 2 / 211 |
| pixelate + noise, then a deblur/upscale attack | 8 / 213 | 0.423 | 1 / 211 |

The attack smooths away the block edges, re-upscales with bicubic and applies a strong unsharp mask. It gets YuNet to see a face-shaped blob in 8 cases, but identity does not come back: 1 / 211 = 0.5% "same person", versus a 0.44% rate when the redacted faces are matched against **different** people's photos. That is chance level. SFace was given the original face's landmarks for alignment, which is the best case for the attacker. If you want zero ambiguity, `redact()` can be a solid fill instead.

**Outside the boxes:** 160 / 160 images byte-identical to the input outside the padded boxes, checked on the array and again after a PNG write/read. Note: the final video encode (H.264/H.265) is lossy over the whole frame, so this guarantee holds at the frame level before encoding, or with a lossless codec.

Before/after examples (synthetic, illustrative; the tables are the evidence): `examples/before_after_*.jpg`. `shower_glass_heavy` is included as a failure example: tuned finds 0 of 1.

## Speed and cost

Apple M1 (8 GB MacBook Air), CPU only, OpenCV 5.0 DNN, median of 7 runs, `src/timing.py`. 4-thread numbers are shown to compare with your 4-core figure.

| config | 1920x1080 (4K downscaled 2x) | 3840x2160 native |
|---|---|---|
| YuNet @ 0.9, single pass | 0.051 s/frame | 0.249 s/frame |
| CLAHE + stretch TTA, 1x only | 0.262 s/frame | 1.173 s/frame |
| tuned (TTA + 2x tiled upscale) | 1.135 s/frame | not run (would be 8K) |

Cost per video-hour. Assumptions: 30 fps = 108,000 frames/hour; detect every Nth frame and let the tracker fill the gaps; a 4-vCPU cloud instance at **$0.1785/hour** (AWS c7i.xlarge on-demand list price, please check current pricing); M1 speed taken as cloud speed. A cloud vCPU is probably slower than an M1 core, so treat these as lower bounds. GPU was not measured.

| config | N=1 | N=3 | N=5 |
|---|---|---|---|
| YuNet @ 0.9 (1080p) | 1.5 h, $0.27 | 0.5 h, $0.09 | 0.3 h, $0.05 |
| CLAHE + stretch TTA (1080p) | 7.9 h, $1.40 | 2.6 h, $0.47 | 1.6 h, $0.28 |
| tuned with 2x upscale (1080p) | 34.0 h, $6.08 | 11.3 h, $2.03 | 6.8 h, $1.22 |
| your RetinaFace figure, 1.55 s/frame | 46.5 h, $8.30 | 15.5 h, $2.77 | 9.3 h, $1.66 |

The cheapest real gain is the threshold: YuNet @ 0.75 costs the same as your current setup.

## How the test set was built

- **Positives:** 160 photos / 213 faces from the WIDER FACE validation set (max 6 images per event category, 30 categories). Only images with 1-3 faces, every face at least 80 px tall, and no blur, occlusion, odd pose or invalid flag. Resized to 640 px on the long side.
- **Reflection conditions, generated in code with boxes carried through every transform** (`src/stress.py`):
  - **mirror:** horizontal flip, slight defocus, a 35-50% grey haze
  - **appliance door:** flip, dark steel tint (30-45% brightness), a soft diagonal glare streak
  - **shower glass light / heavy:** flip, a random refraction warp (4 / 8 px), Gaussian blur (sigma 2.5 / 5) and a 50% fog layer
  - **small:** the whole photo scaled so faces are 20-40 px tall, pasted into a 1920x1080 synthetic room frame (plain and mirrored)
  - **mirror + occlusion:** 30-45% of each face covered from one side, like a towel, hand or mirror edge
- **Negatives (200 images, no faces):** 80 drawn cartoon faces (smileys, teddy/toy heads, big-eyed mascots), 30 smiley-print patterns (packaging/fabric), 90 real background crops from WIDER images, kept well away from every annotated face. Each negative gets the same reflection treatment as the positives.
- **Split:** even indices = dev (used only to pick thresholds), odd = test (reported).
- **Not covered:** TV screens and printed photos of people (these should count as faces), real glass physics, GoPro fisheye distortion, real motion blur at 4K.

## Weights and licences

| file | used for | licence |
|---|---|---|
| `face_detection_yunet_2023mar.onnx` (OpenCV Zoo) | detection, the pipeline itself | MIT, commercial use OK |
| `face_recognition_sface_2021dec.onnx` (OpenCV Zoo) | **verification only** (identity leak test), not in the pipeline | Apache 2.0, commercial use OK |
| WIDER FACE val images | test images only | research/non-commercial; images are not committed, `fetch.sh` downloads them. The 5 example JPEGs are derived from WIDER and are here for illustration only |

**RetinaFace was not run.** The usual CPU route is PyTorch (~700 MB install, over this machine's disk budget), and the common pretrained weights are trained on WIDER FACE (non-commercial terms) or come from insightface (non-commercial). No commercial-safe RetinaFace weight that installs quickly was found, so it is skipped rather than reported with a licence caveat.

## How I'd get to 99% recall / 95% precision on your footage

1. **Measure first.** Run this harness on your answer-key frames: 8 reflection spans, your no-face episodes as negatives. Same tables, real numbers.
2. **Fine-tune on your hard cases.** Fine-tune YuNet or a small commercially licensed detector, using answer-key reflection frames (plus the synthetic reflection augmentations here) as positives and false-alarm frames from no-face episodes (toys, prints, packaging, TV) as hard negatives. Shower glass will need real examples.
3. **Pick the threshold from a PR curve on held-out episodes** at the 95% precision line. Never use a fixed 0.9.
4. **Add the tracker.** Detect every 2-3 frames and use the tracker to fill gaps. Recall is then measured per span and per frame, not per detection.
5. **Price GPU vs CPU** on your real resolution. If you stay on CPU, the 1x TTA config is about $0.47 per video-hour at N=3 (assumptions above).

## Reproduce

```
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
./fetch.sh                                  # weights + 160-image WIDER subset (~18 MB kept)
cd src
../.venv/bin/python run_passes.py           # caches all detector passes (about 25 min on M1)
../.venv/bin/python evaluate.py             # results tables -> results/eval.json
../.venv/bin/python track_demo.py           # tracker table -> results/track_demo.json
../.venv/bin/python verify_redaction.py     # redaction checks -> results/redaction_verify.json
../.venv/bin/python timing.py               # speed -> results/timing.json
../.venv/bin/python make_examples.py
```

Code: `src/common.py` (YuNet, TTA, NMS, matching), `src/stress.py` (test set), `src/tracker.py`, `src/redact.py`.
