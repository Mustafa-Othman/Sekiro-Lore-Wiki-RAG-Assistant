"""Fine-tune YOLOv8n on the full 40-class improved Sekiro boss dataset.

Tuned for the Roboflow v2 export (1280x720 native aspect, severe class
imbalance: ~1 to ~153 train instances per class).
"""

from __future__ import annotations

from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
DATA_YAML = ROOT / "data" / "images" / "dataset" / "data.yaml"
PROJECT = ROOT / "runs"
RUN_NAME = "boss40_v1"


def main() -> None:
    if not DATA_YAML.exists():
        raise SystemExit(f"Missing dataset config: {DATA_YAML}")

    model = YOLO("yolov8n.pt")
    model.train(
        data=str(DATA_YAML),
        epochs=150,
        patience=40,
        imgsz=640,
        batch=8,
        device=0,
        workers=2,
        project=str(PROJECT),
        name=RUN_NAME,
        exist_ok=True,
        pretrained=True,
        seed=0,
        deterministic=True,
        # Prefer classification quality across 40 imbalanced classes
        cls=1.0,
        box=7.5,
        dfl=1.5,
        # Schedule / regularisation
        cos_lr=True,
        close_mosaic=20,
        amp=True,
        # Mild mix helps rare classes without wrecking localisation
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.05,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        translate=0.1,
        scale=0.5,
        # Native 16:9 frames -- letterbox (Ultralytics default), do NOT stretch
        plots=True,
        save=True,
        val=True,
    )

    best = PROJECT / RUN_NAME / "weights" / "best.pt"
    print(f"\nBest weights: {best}")
    print("Validate on held-out test split...")
    YOLO(str(best)).val(data=str(DATA_YAML), split="test", imgsz=640, device=0)


if __name__ == "__main__":
    main()
