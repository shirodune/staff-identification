"""Write per-frame ground truth for sample.mp4 (manually established by frame inspection).

present: 1 = staff in field of view, 0 = not
flag:    clear  = staff clearly visible (corridor / desk, identity unambiguous)
         edge   = staff at the far-left image edge behind the decorative screen, badge never
                  visible, identity inferred from temporal continuity (walks in from / out to there)
         glitch = corrupted (near-black) frame
Metrics are reported on all frames and on 'clear' frames only.
"""
import csv
N = 1341
rows = []
for f in range(N):
    if f == 1300: rows.append((f, 1, "glitch"))
    elif f < 330: rows.append((f, 1, "edge"))
    elif f <= 1261: rows.append((f, 1, "clear"))
    else: rows.append((f, 1, "edge"))
with open("data/ground_truth.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["frame", "staff_present", "flag"]); w.writerows(rows)
print("wrote", len(rows))
