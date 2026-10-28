"""Run a YOLO bird detector frame-by-frame and save an annotated video."""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Source video path.")
    parser.add_argument("--output", required=True, type=Path, help="Annotated MP4 path.")
    parser.add_argument("--weights", type=Path, default=ROOT / "weights" / "best.pt")
    parser.add_argument("--conf", type=float, default=0.25, help="Minimum detection confidence.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None, help="CUDA device, cpu, or automatic selection.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import cv2
        from ultralytics import YOLO
    except ImportError as error:
        raise SystemExit("Missing dependencies. Run: python -m pip install -r requirements.txt") from error
    if not args.input.is_file():
        raise FileNotFoundError(f"Input video was not found: {args.input}")
    if not args.weights.is_file():
        raise FileNotFoundError(
            f"Weights were not found: {args.weights}. Run train.py or pass --weights."
        )
    if not 0.0 <= args.conf <= 1.0:
        raise ValueError("--conf must be between 0 and 1")

    capture = cv2.VideoCapture(str(args.input))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open: {args.input}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        raise RuntimeError("Video has invalid frame dimensions")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"OpenCV could not create: {args.output}")

    model = YOLO(str(args.weights))
    frames = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            result = model.predict(
                frame, conf=args.conf, imgsz=args.imgsz, device=args.device, verbose=False
            )[0]
            # Ultralytics' renderer includes box, class name and confidence.
            writer.write(result.plot(labels=True, conf=True))
            frames += 1
    finally:
        capture.release()
        writer.release()

    if frames == 0:
        raise RuntimeError("No frames were read from the input video")
    print(f"Processed {frames} frames. Saved annotated video to: {args.output}")


if __name__ == "__main__":
    main()

