"""Build a YOLO dataset of person crops (native resolution) with badge boxes.

Inputs:
  --labels   CSV name,x1,y1,x2,y2 (crop-local px; empty box = negative crop)
  --meta     JSON from grid_sheets.py (crop offsets X1,Y1,X2,Y2 per name)
  --neg      extra negative crops: comma list of track ids (non-staff people),
             sampled every --neg_every frames
Output: data/yolo/{images,labels}/{train,val}, data/yolo/badge.yaml
"""
import argparse, csv, json, random
from collections import defaultdict
from pathlib import Path
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("--video", default="data/sample.mp4")
ap.add_argument("--tracks", default="results/sample/person_dets.csv")
ap.add_argument("--labels", nargs="+", required=True)
ap.add_argument("--meta", nargs="+", required=True)
ap.add_argument("--neg", default="")
ap.add_argument("--neg_every", type=int, default=12)
ap.add_argument("--pad", type=int, default=8)
ap.add_argument("--out", default="data/badge")
ap.add_argument("--val_frac", type=float, default=0.2)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
random.seed(args.seed)
out = Path(args.out)
for s in ("train", "val"):
    (out / "images" / s).mkdir(parents=True, exist_ok=True); (out / "labels" / s).mkdir(parents=True, exist_ok=True)

meta = {}
for m in args.meta: meta.update(json.load(open(m)))
samples = {}  # name -> (frame, X1,Y1,X2,Y2, [boxes])
for lf in args.labels:
    for r in csv.DictReader(open(lf)):
        nm = r["name"]; m = meta[nm]
        key = nm
        if key not in samples: samples[key] = [m["frame"], m["X1"], m["Y1"], m["X2"], m["Y2"], []]
        if r["x1"] != "":
            samples[key][5].append(tuple(int(r[k]) for k in ("x1", "y1", "x2", "y2")))

if args.neg:
    ids = {int(i) for i in args.neg.split(",")}
    with open(args.tracks) as f:
        for r in csv.DictReader(f):
            tid, fr = int(r["track_id"]), int(r["frame"])
            if tid in ids and fr % args.neg_every == 0:
                x1, y1, x2, y2 = (int(r[k]) for k in ("x1", "y1", "x2", "y2"))
                nm = f"neg_f{fr:04d}_t{tid:03d}"
                samples[nm] = [fr, max(0, x1 - args.pad), max(0, y1 - args.pad), min(960, x2 + args.pad), min(720, y2 + args.pad), []]

cap = cv2.VideoCapture(args.video)
by_frame = defaultdict(list)
for nm, s in samples.items(): by_frame[s[0]].append(nm)
names = list(samples); random.shuffle(names)
val = set(names[: int(len(names) * args.val_frac)])
npos = nneg = 0
for fr in sorted(by_frame):
    cap.set(cv2.CAP_PROP_POS_FRAMES, fr); ok, img = cap.read()
    for nm in by_frame[fr]:
        _, X1, Y1, X2, Y2, boxes = samples[nm]
        crop = img[Y1:Y2, X1:X2]; H, W = crop.shape[:2]
        split = "val" if nm in val else "train"
        cv2.imwrite(str(out / "images" / split / f"{nm}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        with open(out / "labels" / split / f"{nm}.txt", "w") as f:
            for x1, y1, x2, y2 in boxes:
                cx, cy, w, h = (x1 + x2) / 2 / W, (y1 + y2) / 2 / H, (x2 - x1) / W, (y2 - y1) / H
                f.write(f"0 {cx:.5f} {cy:.5f} {w:.5f} {h:.5f}\n")
        if boxes: npos += 1
        else: nneg += 1
(out / "badge.yaml").write_text(f"path: {out.resolve()}\ntrain: images/train\nval: images/val\nnames:\n  0: badge\n")
print(f"positives={npos} negatives={nneg} val={len(val)}")
