"""Round-2 labelling: run the current badge model on sampled crops of given tracks,
write predictions as candidate labels + review sheets (box + conf drawn)."""
import argparse, csv, json
from collections import defaultdict
from pathlib import Path
import cv2, numpy as np
from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("--video", default="data/sample.mp4")
ap.add_argument("--tracks", default="results/sample/person_dets.csv")
ap.add_argument("--ids", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--every", type=int, default=3)
ap.add_argument("--pad", type=int, default=8)
ap.add_argument("--imgsz", type=int, default=416)
ap.add_argument("--conf", type=float, default=0.10)
ap.add_argument("--skip", default="", help="comma list of names already labelled (csv files) to skip")
ap.add_argument("--tile", type=int, default=300)
args = ap.parse_args()
out = Path(args.out); (out / "review").mkdir(parents=True, exist_ok=True)
ids = {int(i) for i in args.ids.split(",")}
skip = set()
for p in args.skip.split(","):
    if p: skip |= {r["name"] for r in csv.DictReader(open(p))}
by_frame = defaultdict(list)
with open(args.tracks) as f:
    for r in csv.DictReader(f):
        if int(r["track_id"]) in ids and int(r["frame"]) % args.every == 0:
            by_frame[int(r["frame"])].append((int(r["track_id"]), *[int(r[k]) for k in ("x1", "y1", "x2", "y2")]))
model = YOLO(args.model)
cap = cv2.VideoCapture(args.video); n = 0; rows = []; tiles = []; meta = {}
while True:
    ok, img = cap.read()
    if not ok: break
    for tid, x1, y1, x2, y2 in by_frame.get(n, []):
        name = f"f{n:04d}_t{tid:03d}"
        if name in skip: continue
        X1, Y1 = max(0, x1 - args.pad), max(0, y1 - args.pad); X2, Y2 = min(img.shape[1], x2 + args.pad), min(img.shape[0], y2 + args.pad)
        crop = img[Y1:Y2, X1:X2]
        if crop.size == 0: continue
        r = model.predict(crop, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
        meta[name] = dict(frame=n, track=tid, X1=X1, Y1=Y1, X2=X2, Y2=Y2)
        vis = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        if r.boxes is not None and len(r.boxes):
            for k, (b, c) in enumerate(zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist())):
                bx1, by1, bx2, by2 = map(int, b)
                rows.append([name, n, tid, k, bx1, by1, bx2, by2, round(c, 3)])
                col = (0, 255, 0) if k == 0 else (0, 200, 255)
                cv2.rectangle(vis, (bx1 * 3, by1 * 3), (bx2 * 3, by2 * 3), col, 1)
                cv2.putText(vis, f"{k}:{c:.2f}", (bx1 * 3, max(10, by1 * 3 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)
        else:
            rows.append([name, n, tid, -1, "", "", "", "", 0])
        cv2.putText(vis, name, (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        tiles.append(vis)
    n += 1
with open(out / "pred_candidates.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["name", "frame", "track", "k", "x1", "y1", "x2", "y2", "conf"]); w.writerows(rows)
json.dump(meta, open(out / "pred_meta.json", "w"), indent=1)
T = args.tile
for i in range(0, len(tiles), 20):
    chunk = []
    for v in tiles[i:i + 20]:
        h, w = v.shape[:2]; s = T / max(h, w); v = cv2.resize(v, (int(w * s), int(h * s)))
        canvas = np.full((T, T, 3), 30, np.uint8); canvas[:v.shape[0], :v.shape[1]] = v; chunk.append(canvas)
    while len(chunk) < 20: chunk.append(np.zeros((T, T, 3), np.uint8))
    cv2.imwrite(str(out / "review" / f"sheet_{i // 20:03d}.jpg"), np.vstack([np.hstack(chunk[r * 5:(r + 1) * 5]) for r in range(4)]))
print(f"crops={len(tiles)} with_pred={sum(1 for r in rows if r[3]==0)} sheets={(len(tiles)+19)//20}")
