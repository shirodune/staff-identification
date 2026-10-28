"""Gridded crop sheets for manual badge labelling. Grid lines every 10 px in
ORIGINAL-frame coordinates of the crop (crop-local), labels on axes."""
import argparse, csv, json
from collections import defaultdict
from pathlib import Path
import cv2, numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--video", default="data/sample.mp4")
ap.add_argument("--tracks", default="results/sample/person_dets.csv")
ap.add_argument("--frames", required=True, help="comma list of frame:track")
ap.add_argument("--out", default="data/labels/sheets")
ap.add_argument("--scale", type=float, default=4.0)
ap.add_argument("--pad", type=int, default=8)
ap.add_argument("--per_sheet", type=int, default=4)
ap.add_argument("--prefix", default="g")
args = ap.parse_args()
S = args.scale
out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
want = [(int(a), int(b)) for a, b in (p.split(":") for p in args.frames.split(","))]
boxes = {}
with open(args.tracks) as f:
    for r in csv.DictReader(f):
        boxes[(int(r["frame"]), int(r["track_id"]))] = tuple(int(r[k]) for k in ("x1", "y1", "x2", "y2"))
cap = cv2.VideoCapture(args.video)
tiles = []; meta = {}
for fr, tid in want:
    if (fr, tid) not in boxes: print("missing", fr, tid); continue
    x1, y1, x2, y2 = boxes[(fr, tid)]
    cap.set(cv2.CAP_PROP_POS_FRAMES, fr); ok, img = cap.read()
    X1, Y1 = max(0, x1 - args.pad), max(0, y1 - args.pad)
    X2, Y2 = min(img.shape[1], x2 + args.pad), min(img.shape[0], y2 + args.pad)
    crop = img[Y1:Y2, X1:X2]
    up = cv2.resize(crop, None, fx=S, fy=S, interpolation=cv2.INTER_CUBIC)
    h, w = up.shape[:2]
    for i in range(0, X2 - X1, 10):
        cv2.line(up, (int(i * S), 0), (int(i * S), h), (0, 255, 255), 1)
        cv2.putText(up, str(i), (int(i * S) + 1, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0, 255, 255), 1)
    for j in range(0, Y2 - Y1, 10):
        cv2.line(up, (0, int(j * S)), (w, int(j * S)), (0, 255, 255), 1)
        cv2.putText(up, str(j), (1, int(j * S) + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0, 255, 255), 1)
    name = f"f{fr:04d}_t{tid:03d}"
    cv2.putText(up, name, (w // 2 - 40, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
    tiles.append(up); meta[name] = dict(frame=fr, track=tid, X1=X1, Y1=Y1, X2=X2, Y2=Y2)
H = max(t.shape[0] for t in tiles)
k = 0
for i in range(0, len(tiles), args.per_sheet):
    chunk = [cv2.copyMakeBorder(t, 0, H - t.shape[0], 0, 12, cv2.BORDER_CONSTANT, value=(40, 40, 40)) for t in tiles[i:i + args.per_sheet]]
    cv2.imwrite(str(out / f"{args.prefix}_{k:02d}.jpg"), np.hstack(chunk)); k += 1
json.dump(meta, open(out / f"{args.prefix}_meta.json", "w"), indent=1)
print("sheets", k, "crops", len(tiles))
