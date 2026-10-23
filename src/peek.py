import sys, cv2
from stress import sample, CONDS, ANN
out = sys.argv[1]
for c in CONDS:
    for k in (3, len(ANN) + 5, len(ANN) + 90, len(ANN) + 150):
        img, b, n = sample(c, k)
        for x in b: cv2.rectangle(img, (int(x[0]), int(x[1])), (int(x[0] + x[2]), int(x[1] + x[3])), (0, 255, 0), 1)
        cv2.imwrite(f"{out}/{c}_{k}.jpg", img)
