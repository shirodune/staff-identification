"""What does the path step add? Three variants of detect_staff.py on the same person_dets.csv:
  A. badge only     a frame is 'staff present' iff a badge is seen in it (no path)
  B. short gaps only the path may skip at most 5 frames (--gap 5): no bridging of lost tracks
  C. full           default (--gap 40)
  python tools/ablation.py --dets results/sample/person_dets.csv
"""
import argparse, csv, json, subprocess, sys
from pathlib import Path

ap = argparse.ArgumentParser(); ap.add_argument("--dets", required=True); ap.add_argument("--out", default="results/ablation"); a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
rows = list(csv.DictReader(open(a.dets))); N = max(int(r["frame"]) for r in rows) + 1
seen = {}
for r in rows:
    f, s = int(r["frame"]), float(r["badge"])
    if s >= 0.3 and s > seen.get(f, (0,))[0]: seen[f] = (s, (int(r["x1"]) + int(r["x2"])) // 2, (int(r["y1"]) + int(r["y2"])) // 2, r["track_id"])
(out / "A").mkdir(exist_ok=True)
with open(out / "A" / "staff_frames.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["frame", "time_s", "staff_present", "x", "y", "track_id", "badge_visible"])
    for fr in range(N): w.writerow([fr, round(fr / 25, 3), 1, seen[fr][1], seen[fr][2], seen[fr][3], 1] if fr in seen else [fr, round(fr / 25, 3), 0, "", "", "", 0])
base = [sys.executable, "detect_staff.py", "--video", "data/sample.mp4", "--from_dets", a.dets, "--no_video"]
subprocess.run(base + ["--out", str(out / "B"), "--gap", "5"], check=True, stdout=subprocess.DEVNULL)
subprocess.run(base + ["--out", str(out / "C")], check=True, stdout=subprocess.DEVNULL)
labels = {"A": "A: badge only", "B": "B: path, gaps <= 5 frames", "C": "C: full"}
print(f"{'variant':32s} {'recall':>8s} {'loc.acc':>8s} {'wrong-person':>13s}")
for k, lab in labels.items():
    subprocess.run([sys.executable, "tools/evaluate.py", "--pred", str(out / k / "staff_frames.csv"), "--out", str(out / k), "--label", lab], check=True, stdout=subprocess.DEVNULL)
    m = json.load(open(out / k / "metrics.json")); c, l = m["clear_frames"], m["localisation"]
    print(f"{lab:32s} {c['recall']:8.3f} {l['localisation_acc']:8.3f} {l['wrong_position']:13d}")
subprocess.run([sys.executable, "tools/evaluate.py", "--pred", str(out / "C" / "staff_frames.csv"), "--out", str(out), "--label", "C: full",
                "--extra", f"B: no gap bridging={out/'B'/'staff_frames.csv'}", f"A: badge only={out/'A'/'staff_frames.csv'}"], check=True, stdout=subprocess.DEVNULL)
