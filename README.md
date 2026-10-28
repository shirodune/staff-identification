# Staff identification from a top-down sensor video

Which frames contain the badge-wearing staff member, and where is he?

```
video ─► track people ─► badge in each person box ─► best path through boxes ─► present / (x, y)
         YOLO26n + OC-SORT   one-class YOLO26n (trained)   one dynamic programme (30 lines)
```

The badge is ~15×6 px and visible in 15 % of frames, but identity belongs to the person: badge
sightings are rewards, and the staff trajectory is the highest-scoring path through the person
boxes (penalised for implausible movement and for switching person). Nothing depends on clothing —
the staff member changes from a white shirt to a black jacket half-way through the clip.

## Run

```bash
pip install -r requirements.txt
python detect_staff.py --video sample.mp4 --out results/   # CPU, ~8 min for the 54 s clip
python app.py                                             # same thing as a web UI (Gradio)
```

`results/staff_frames.csv` — `frame, time_s, staff_present, x, y` per frame · `annotated.mp4` — red = staff

| flag | default | |
|---|---|---|
| `--speed` | 30 | px/frame the person may move for free |
| `--gap` | 40 | longest run of frames the path may skip |
| `--min_hits` | 5 | badge sightings needed to count as staff |
| `--badge_conf` | 0.30 | badge score that counts as a sighting |

## Result on `sample.mp4` (932 clearly-visible frames)

| | recall | right person (x,y within 80 px) |
|---|---|---|
| badge seen in the frame only | 19.2 % | 28.7 % |
| full pipeline | **89.4 %** | **94.9 %** |

`python tools/evaluate.py --pred results/staff_frames.csv` reproduces this; `tools/ablation.py` the table.

## Files

`detect_staff.py` pipeline · `train_badge.py` badge detector (training crops not published: they are frames of a private video) ·
`configs/tracker.yaml` OC-SORT with lower thresholds · `weights/` · `tools/` labelling, GT, evaluation ·
`docs/solution.md` one-page write-up
