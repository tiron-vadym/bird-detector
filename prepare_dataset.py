"""Download a small, deterministic YOLO-format bird-detection dataset subset."""

from __future__ import annotations

import argparse
import json
import shutil
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPOSITORY = "zhizhen777/bird-detection-yolo-dataset"
API_URL = f"https://api.github.com/repos/{REPOSITORY}/git/trees/main?recursive=1"
RAW_URL = f"https://raw.githubusercontent.com/{REPOSITORY}/main"


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "bird-detector-dataset-preparer"})
    with urllib.request.urlopen(request) as response, destination.open("wb") as output:
        shutil.copyfileobj(response, output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=int, default=100, help="Number of training images (default: 100).")
    parser.add_argument("--val", type=int, default=20, help="Number of validation images (default: 20).")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "birds")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.train < 1 or args.val < 1:
        raise ValueError("--train and --val must both be positive")
    request = urllib.request.Request(API_URL, headers={"User-Agent": "bird-detector-dataset-preparer"})
    with urllib.request.urlopen(request) as response:
        tree = json.load(response)["tree"]

    files = {entry["path"] for entry in tree if entry["type"] == "blob"}
    downloads: list[tuple[str, Path]] = []
    for split, amount in (("train", args.train), ("val", args.val)):
        image_prefix = f"images/{split}/"
        images = sorted(path for path in files if path.startswith(image_prefix) and path.lower().endswith(".jpg"))
        selected = [path for path in images if path.replace("images/", "labels/", 1).replace(".jpg", ".txt") in files][:amount]
        if len(selected) < amount:
            raise RuntimeError(f"Only found {len(selected)} paired {split} samples, expected {amount}")
        for image_path in selected:
            label_path = image_path.replace("images/", "labels/", 1).replace(".jpg", ".txt")
            for source_path in (image_path, label_path):
                relative = Path(source_path)
                destination = args.output / relative
                if not destination.is_file():
                    downloads.append((f"{RAW_URL}/{source_path}", destination))
        print(f"Downloaded {len(selected)} {split} image/label pairs")

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(lambda item: download(*item), downloads))

    print(f"Dataset ready at: {args.output.resolve()}")


if __name__ == "__main__":
    main()
