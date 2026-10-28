"""Fine-tune YOLO26n for single-class bird detection."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "bird.yaml",
                        help="YOLO dataset YAML file.")
    parser.add_argument("--model", default="yolo26n.pt",
                        help="Pretrained checkpoint or model YAML (default: yolo26n.pt).")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8,
                        help="Batch size; use -1 to let Ultralytics choose it.")
    parser.add_argument("--device", default=None,
                        help="CUDA device (e.g. 0), cpu, or omit for automatic selection.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--project", type=Path, default=ROOT / "runs")
    parser.add_argument("--name", default="bird_yolo26n")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise SystemExit("Missing dependencies. Run: python -m pip install -r requirements.txt") from error
    data_file = args.data.resolve()
    if not data_file.is_file():
        raise FileNotFoundError(f"Dataset config was not found: {data_file}")

    model = YOLO(args.model)
    results = model.train(
        data=str(data_file),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(args.project.resolve()),
        name=args.name,
        seed=args.seed,
        deterministic=True,
        exist_ok=True,
        pretrained=True,
        plots=True,
    )

    best = Path(results.save_dir) / "weights" / "best.pt"
    destination = ROOT / "weights" / "best.pt"
    if not best.is_file():
        raise RuntimeError(f"Training finished but best checkpoint is missing: {best}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, destination)
    print(f"Best checkpoint copied to: {destination}")
    print(f"Training artifacts: {results.save_dir}")


if __name__ == "__main__":
    main()

