"""Frame-level evaluation + visualisations.

  python scripts/evaluate.py --pred results/staff_frames.csv --gt data/ground_truth.csv --out results/
Produces metrics.json, timeline.png (GT vs prediction), trajectory.png (staff xy over a background frame).
"""
import argparse, csv, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("--pred", required=True)
ap.add_argument("--gt", default="data/ground_truth.csv")
ap.add_argument("--video", default="data/sample.mp4")
ap.add_argument("--out", default="results")
ap.add_argument("--label", default="")
ap.add_argument("--extra", nargs="*", default=[], help="extra prediction csvs for the timeline: label=path")
args = ap.parse_args()
out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

gt = {int(r["frame"]): (int(r["staff_present"]), r["flag"]) for r in csv.DictReader(open(args.gt))}
def load_pred(p):
    d = {}
    for r in csv.DictReader(open(p)):
        d[int(r["frame"])] = (int(r["staff_present"]), r["x"], r["y"], r.get("badge_visible", "0"))
    return d
pred = load_pred(args.pred)
N = max(gt) + 1

def metrics(frames):
    tp = sum(1 for f in frames if gt[f][0] == 1 and pred.get(f, (0,))[0] == 1)
    fp = sum(1 for f in frames if gt[f][0] == 0 and pred.get(f, (0,))[0] == 1)
    fn = sum(1 for f in frames if gt[f][0] == 1 and pred.get(f, (0,))[0] == 0)
    tn = sum(1 for f in frames if gt[f][0] == 0 and pred.get(f, (0,))[0] == 0)
    P = tp / (tp + fp) if tp + fp else 0.0; R = tp / (tp + fn) if tp + fn else 0.0
    F1 = 2 * P * R / (P + R) if P + R else 0.0
    return dict(frames=len(frames), tp=tp, fp=fp, fn=fn, tn=tn, precision=round(P, 4), recall=round(R, 4), f1=round(F1, 4), accuracy=round((tp + tn) / len(frames), 4))
res = dict(all_frames=metrics(list(range(N))),
           clear_frames=metrics([f for f in range(N) if gt[f][1] == "clear"]),
           badge_visible_frames=int(sum(1 for f in pred if pred[f][3] == "1")))
json.dump(res, open(out / "metrics.json", "w"), indent=1)
print(json.dumps(res, indent=1))

# ---- timeline
series = [("Ground truth", np.array([gt[f][0] for f in range(N)]), np.array([gt[f][1] for f in range(N)]))]
series.append((args.label or "Prediction", np.array([pred.get(f, (0,))[0] for f in range(N)]), None))
for e in args.extra:
    lab, p = e.split("=", 1); d = load_pred(p); series.append((lab, np.array([d.get(f, (0,))[0] for f in range(N)]), None))
bv = np.array([1 if pred.get(f, (0, 0, 0, "0"))[3] == "1" else 0 for f in range(N)])
series.insert(1, ("Badge detected (raw)", bv, None))
fig, ax = plt.subplots(figsize=(14, 0.9 * len(series) + 1.2))
for i, (lab, v, flag) in enumerate(series):
    y = len(series) - 1 - i
    if flag is not None:
        for f in range(N):
            if v[f]:
                c = {"clear": "#2a9d8f", "edge": "#b7d9d3", "glitch": "#999999"}[flag[f]]
                ax.add_patch(plt.Rectangle((f, y - 0.35), 1, 0.7, color=c, lw=0))
    else:
        ax.fill_between(np.arange(N), y - 0.35, y + 0.35, where=v > 0, step="post", color="#e76f51" if i == 1 else "#264653", lw=0)
    ax.text(-15, y, lab, ha="right", va="center", fontsize=10)
ax.set_xlim(0, N); ax.set_ylim(-0.6, len(series) - 0.4); ax.set_yticks([])
ax.set_xlabel("frame (25 fps)"); ax.set_title("Staff presence: ground truth vs prediction" + (f"  [{args.label}]" if args.label else ""))
ax.text(N, len(series) - 0.45, "GT shading: dark = clearly visible, light = at image edge (identity by continuity), grey = corrupted frame", ha="right", fontsize=7, color="#555")
for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
plt.tight_layout(); plt.savefig(out / "timeline.png", dpi=150); plt.close()

# ---- trajectory
pts = [(f, float(pred[f][1]), float(pred[f][2])) for f in sorted(pred) if pred[f][0] == 1 and pred[f][1] != ""]
fig, ax = plt.subplots(figsize=(9, 6.75))
cap = cv2.VideoCapture(args.video); cap.set(cv2.CAP_PROP_POS_FRAMES, 500); ok, bg = cap.read()
if ok: ax.imshow(cv2.cvtColor(bg, cv2.COLOR_BGR2RGB), alpha=0.75)   # background frame only if the video is available
else: ax.set_xlim(0, 960); ax.set_ylim(720, 0); ax.set_aspect("equal")
if pts:
    fr, xs, ys = zip(*pts)
    sc = ax.scatter(xs, ys, c=fr, cmap="plasma", s=10, lw=0)
    ax.plot(xs, ys, color="white", lw=0.5, alpha=0.5)
    cb = plt.colorbar(sc, ax=ax, fraction=0.03); cb.set_label("frame")
ax.set_title("Staff xy trajectory (bonus task)"); ax.axis("off")
plt.tight_layout(); plt.savefig(out / "trajectory.png", dpi=150); plt.close()
print("wrote", out / "timeline.png", out / "trajectory.png")

# ---- localisation (bonus task): predicted xy vs verified staff position
import os
gtxy_path = "data/ground_truth_xy.csv"
if os.path.exists(gtxy_path):
    gxy = {int(r["frame"]): (float(r["x"]), float(r["y"])) for r in csv.DictReader(open(gtxy_path))}
    tol = 80
    n = hit = miss = absent = 0; dists = []
    for f, (gx, gy) in gxy.items():
        if gt[f][1] != "clear": continue
        n += 1; p = pred.get(f)
        if not p or p[0] != 1 or p[1] == "": absent += 1; continue
        d = float(np.hypot(float(p[1]) - gx, float(p[2]) - gy)); dists.append(d)
        if d <= tol: hit += 1
        else: miss += 1
    loc = dict(frames_with_xy_gt=n, within_tol=hit, wrong_position=miss, no_prediction=absent, tol_px=tol,
               localisation_acc=round(hit / n, 4) if n else None, median_err_px=round(float(np.median(dists)), 1) if dists else None)
    res["localisation"] = loc
    json.dump(res, open(out / "metrics.json", "w"), indent=1)
    print("localisation:", json.dumps(loc))
