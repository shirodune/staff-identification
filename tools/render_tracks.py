"""Render raw tracker output: every box coloured by its track ID (ID changes = fragments), badge sightings in yellow.
  python tools/render_tracks.py --dets results/trackers/ocsort/person_dets.csv --out results/trackers/ocsort/tracks_only.mp4"""
import argparse, csv, colorsys
from collections import defaultdict
import cv2, numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("--video", default="data/sample.mp4"); ap.add_argument("--dets", required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
by = defaultdict(list)
for r in csv.DictReader(open(a.dets)): by[int(r["frame"])].append((int(r["track_id"]), int(r["x1"]), int(r["y1"]), int(r["x2"]), int(r["y2"]), float(r["badge"])))
def col(t): h = (t * 0.618) % 1.0; r, g, b = colorsys.hsv_to_rgb(h, 0.85, 0.95); return int(b * 255), int(g * 255), int(r * 255)
cap = cv2.VideoCapture(a.video); fps = cap.get(cv2.CAP_PROP_FPS); W, H = int(cap.get(3)), int(cap.get(4))
vw = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H)); n = 0
trail = defaultdict(list)   # per-ID history, as in the docs "Plotting Tracks Over Time" example
while True:
    ok, fr = cap.read()
    if not ok: break
    for t, x1, y1, x2, y2, b in by.get(n, []):
        cv2.rectangle(fr, (x1, y1), (x2, y2), col(t), 2); cv2.putText(fr, f"id {t}", (x1, y2 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col(t), 2)
        pts = trail[t]; pts.append(((x1 + x2) // 2, (y1 + y2) // 2)); del pts[:-60]
        if len(pts) > 1: cv2.polylines(fr, [np.array(pts, np.int32).reshape(-1, 1, 2)], False, col(t), 3)
        if b >= 0.3: cv2.putText(fr, f"badge {b:.2f}", (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    cv2.rectangle(fr, (0, 0), (420, 28), (0, 0, 0), -1); cv2.putText(fr, f"f{n}  tracker only (OC-SORT): colour = track ID, line = its last 60 positions", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    vw.write(fr); n += 1
vw.release(); print("wrote", a.out)
