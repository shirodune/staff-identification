"""Train the badge detector (Ultralytics train mode: https://docs.ultralytics.com/modes/train).

Dataset: person crops with the badge boxed, YOLO format, in data/badge/ (built with tools/build_dataset.py).
The badge can appear at any angle under a top-down camera, hence full rotation + flips.

  python train_badge.py                    # ~35 min on a 2-core CPU, GPU: ~2 min
"""
from ultralytics import YOLO

model = YOLO("yolo26n.pt")
model.train(data="data/badge/badge.yaml", epochs=80, imgsz=416, batch=8,     # dataset not in the public repo (private video)
            degrees=180, flipud=0.5, fliplr=0.5, scale=0.3, erasing=0.0,
            project="runs", name="badge", exist_ok=True)
# -> runs/badge/weights/best.pt ; copy to weights/badge.pt
