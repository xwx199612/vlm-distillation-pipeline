from pathlib import Path
import argparse
import json


def build_manifest(
    image_dir: Path,
    output_path: Path,
    task: str,
    query: str,
):
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    images = sorted(
        p for p in image_dir.iterdir()
        if p.suffix.lower() in image_exts
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for idx, image_path in enumerate(images, start=1):
            row = {
                "id": f"{task}-{idx:06d}",
                "image": str(image_path).replace("\\", "/"),
                "task": task,
                "query": query,
            }

            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(f"Created manifest:")
    print(output_path)
    print(f"Samples: {len(images)}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image-dir",
        required=True,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    parser.add_argument(
        "--task",
        required=True,
    )

    parser.add_argument(
        "--query",
        required=True,
    )

    args = parser.parse_args()

    build_manifest(
        image_dir=Path(args.image_dir),
        output_path=Path(args.output),
        task=args.task,
        query=args.query,
    )


if __name__ == "__main__":
    main()