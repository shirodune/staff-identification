#!/usr/bin/env python
"""Find the badge-wearing staff member in a top-down sensor video.

  1. track people          model.track(frame, persist=True)            (Ultralytics track mode, ByteTrack)
  2. look for the badge    badge.predict(person crop)                  (small YOLO trained on this badge)
  3. pick the staff path   highest-scoring path through person boxes   (one dynamic programme)
  4. write staff_frames.csv (frame, present, x, y) and an annotated video

  python detect_staff.py --video sample.mp4 --out results/
"""
import argparse, csv, json, time
from collections import defaultdict
from pathlib import Path
import cv2, numpy as np
from ultralytics import YOLO

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--video", required=True)
ap.add_argument("--out", default="results")
ap.add_argument("--person_conf", type=float, default=0.05, help="person score threshold (top-down people score low)")
ap.add_argument("--person_imgsz", type=int, default=960, help="person detector input size; 640 is ~2x faster, slightly lower recall")
ap.add_argument("--badge_conf", type=float, default=0.30, help="badge score needed to count as evidence")
ap.add_argument("--speed", type=float, default=30, help="px/frame the staff may move for free (running speed in this view)")
ap.add_argument("--gap", type=int, default=40, help="longest run of frames the path may skip (badge hidden / tracker lost him)")
ap.add_argument("--min_hits", type=int, default=5, help="badge sightings a path needs to count as a staff member")
ap.add_argument("--from_dets", default="", help="reuse a person_dets.csv from an earlier run (skips stages 1-2)")
ap.add_argument("--no_video", action="store_true")
args = ap.parse_args()
out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
t0 = time.time()

# ------------------------------------------------------------------ 1 + 2: people, then badge inside each person
cap = cv2.VideoCapture(args.video)
fps = cap.get(cv2.CAP_PROP_FPS) or 25; W, H = int(cap.get(3)), int(cap.get(4))
dets = []  # frame, track_id, x1, y1, x2, y2, badge_score
if args.from_dets:
    dets = [[int(r["frame"]), int(r["track_id"]), int(r["x1"]), int(r["y1"]), int(r["x2"]), int(r["y2"]), float(r["badge"])]
            for r in csv.DictReader(open(args.from_dets))]
    n = max(d[0] for d in dets) + 1
else:
    people, badge = YOLO("weights/yolo26n.pt"), YOLO("weights/badge.pt")
    n = 0
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok: break
        res = people.track(frame, persist=True, tracker="configs/tracker.yaml", classes=[0],
                           conf=args.person_conf, imgsz=args.person_imgsz, verbose=False)[0]
        if res.boxes and res.boxes.is_track:
            for (x1, y1, x2, y2), tid in zip(res.boxes.xyxy.int().cpu().tolist(), res.boxes.id.int().cpu().tolist()):
                crop = frame[max(0, y1 - 8):y2 + 8, max(0, x1 - 8):x2 + 8]      # pad a little, badge sits at the collar
                if crop.shape[0] < 16 or crop.shape[1] < 16: continue
                b = badge.predict(crop, imgsz=416, conf=0.05, verbose=False)[0]
                dets.append([n, tid, x1, y1, x2, y2, round(float(b.boxes.conf.max()), 3) if len(b.boxes) else 0.0])
        n += 1
        if n % 100 == 0: print(f"  frame {n}  {time.time()-t0:.0f}s", flush=True)
cap.release()
N = n
print(f"stages 1-2: {N} frames, {len(dets)} person boxes, {time.time()-t0:.0f}s")

# ------------------------------------------------------------------ 3: the staff is the best path through the boxes
# Nodes = person boxes. An edge joins a box at frame f to a box at frame f+k (k <= gap).
# Reward: badge score. Cost: movement beyond `speed` px/frame, 0.02 per skipped frame, and 1.0 for
# switching person (leaving a track that is still alive, or entering one that already existed;
# continuing onto a track that *starts* after ours ended is free - that is the tracker re-finding him).
GAP_COST, SWITCH_COST = 0.02, 1.0
center = lambda d: ((d[2] + d[4]) / 2, (d[3] + d[5]) / 2)

def best_path(nodes):
    by_frame, alive, first = defaultdict(list), set(), {}
    for i, d in enumerate(nodes):
        by_frame[d[0]].append(i); alive.add((d[1], d[0])); first.setdefault(d[1], d[0])
    reward = [d[6] if d[6] >= args.badge_conf else 0.0 for d in nodes]
    best, prev = np.array(reward), np.full(len(nodes), -1)
    for i, d in enumerate(nodes):
        f, (cx, cy) = d[0], center(d)
        for k in range(1, args.gap + 1):
            for j in by_frame.get(f - k, []):
                e = nodes[j]; px, py = center(e)
                cost = max(0.0, np.hypot(cx - px, cy - py) - args.speed * k) / args.speed + GAP_COST * (k - 1)
                if e[1] != d[1] and ((e[1], f) in alive or first[d[1]] < f - 3): cost += SWITCH_COST
                if best[j] - cost + reward[i] > best[i]: best[i], prev[i] = best[j] - cost + reward[i], j
    i = int(np.argmax(best)); path = []
    while i >= 0: path.append(i); i = prev[i]
    return path[::-1], reward

staff, nodes = [], sorted(dets)                      # one path per staff member; repeat while evidence remains
while nodes:
    path, reward = best_path(nodes)
    hits = sum(1 for i in path if reward[i] > 0)
    if hits < args.min_hits: break
    pts = {nodes[i][0]: (center(nodes[i]), nodes[i][1], reward[i] > 0) for i in path}
    fs = sorted(pts)
    for a, b in zip(fs, fs[1:]):                       # interpolate the frames the path skipped
        for f in range(a + 1, b):
            t = (f - a) / (b - a); (ax, ay), (bx, by) = pts[a][0], pts[b][0]
            pts[f] = ((ax + t * (bx - ax), ay + t * (by - ay)), -1, False)
    ids = {nodes[i][1] for i in path}
    staff.append(dict(frames=pts, tracks=sorted(ids), hits=hits))
    print(f"staff #{len(staff)}: frames {fs[0]}-{fs[-1]}, {len(ids)} tracklets, {hits} badge sightings")
    nodes = [d for d in nodes if d[1] not in ids]

# ------------------------------------------------------------------ 4: outputs
with open(out / "staff_frames.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["frame", "time_s", "staff_present", "x", "y", "track_id", "badge_visible"])
    for fr in range(N):
        hit = [(s["frames"][fr], k) for k, s in enumerate(staff) if fr in s["frames"]]
        if hit: ((x, y), tid, vis), k = hit[0]; w.writerow([fr, round(fr / fps, 3), 1, int(x), int(y), tid, int(vis)])
        else: w.writerow([fr, round(fr / fps, 3), 0, "", "", "", 0])
with open(out / "person_dets.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["frame", "track_id", "x1", "y1", "x2", "y2", "badge"]); w.writerows(dets)
json.dump(dict(args=vars(args), frames=N, staff=[dict(tracks=s["tracks"], hits=s["hits"], first=min(s["frames"]), last=max(s["frames"]))
                                                 for s in staff], runtime_s=round(time.time() - t0)), open(out / "summary.json", "w"), indent=1)

if not args.no_video:
    staff_ids = {t for s in staff for t in s["tracks"]}
    cap = cv2.VideoCapture(args.video); vw = cv2.VideoWriter(str(out / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    by_frame = defaultdict(list)
    for d in dets: by_frame[d[0]].append(d)
    trail = []
    for fr in range(N):
        ok, frame = cap.read()
        if not ok: break
        for d in by_frame[fr]:
            red = d[1] in staff_ids
            cv2.rectangle(frame, (d[2], d[3]), (d[4], d[5]), (0, 0, 255) if red else (0, 200, 0), 2 if red else 1)
            if d[6] >= args.badge_conf: cv2.putText(frame, f"badge {d[6]:.2f}", (d[2], d[3] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        for s in staff:
            if fr in s["frames"]:
                (x, y) = s["frames"][fr][0]; trail.append((int(x), int(y))); trail = trail[-60:]
                cv2.polylines(frame, [np.array(trail, np.int32).reshape(-1, 1, 2)], False, (0, 0, 255), 2)
                cv2.putText(frame, "STAFF", (int(x) - 25, int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        present = any(fr in s["frames"] for s in staff)
        cv2.putText(frame, f"f{fr}  {'STAFF PRESENT' if present else 'no staff'}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255) if present else (200, 200, 200), 2)
        vw.write(frame)
    vw.release()
print(f"done in {time.time()-t0:.0f}s -> {out}")
