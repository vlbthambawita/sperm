import argparse
import sys
from pathlib import Path
from typing import List, Tuple, Optional

try:
    import cv2  # type: ignore
except ModuleNotFoundError as exc:  # pragma: no cover - environment-specific
    raise SystemExit(
        "This script requires OpenCV (cv2). Please install it, for example:\n"
        "  pip install opencv-python\n"
    ) from exc

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    yaml = None  # type: ignore


DEFAULT_DATASET_ROOT = Path(
    "/work/vajira/DATA/SPERM_data_2025/manually_annotated_100_frames_v0.1"
)


def load_class_names(classes_path: Path) -> List[str]:
    """
    Load class names from classes.txt.

    Supports both:
      - `sperm`
      - `0: sperm`
    style formats.
    """
    names: List[str] = []

    if not classes_path.is_file():
        return names

    with classes_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if ":" in line:
                # Format like `0: sperm`
                idx_str, name = line.split(":", 1)
                try:
                    idx = int(idx_str.strip())
                except ValueError:
                    continue
                name = name.strip()
                while len(names) <= idx:
                    names.append(f"class_{len(names)}")
                names[idx] = name
            else:
                # Plain name per line
                names.append(line)

    return names


def resolve_dataset_root(dataset_root_arg: Optional[str], script_dir: Path) -> Path:
    """
    Resolve the dataset root directory.

    Priority:
      1. --dataset-root argument if provided
      2. path from dataset.yaml (if readable)
      3. DEFAULT_DATASET_ROOT constant
    """
    if dataset_root_arg:
        return Path(dataset_root_arg).expanduser().resolve()

    # Try dataset.yaml beside the known default root
    dataset_yaml_candidates = [
        DEFAULT_DATASET_ROOT / "dataset.yaml",
        script_dir.parent.parent / "DATA" / "SPERM_data_2025" / "manually_annotated_100_frames_v0.1" / "dataset.yaml",
    ]

    for yaml_path in dataset_yaml_candidates:
        if not yaml_path.is_file():
            continue

        if yaml is None:
            break

        try:
            with yaml_path.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            path_value = data.get("path")
            if path_value:
                return Path(path_value).expanduser().resolve()
        except Exception as exc:
            print(f"Warning: failed to read dataset.yaml at {yaml_path}: {exc}", file=sys.stderr)
            break

    return DEFAULT_DATASET_ROOT


def load_yolo_labels(
    label_path: Path, img_width: int, img_height: int
) -> List[Tuple[int, int, int, int, int]]:
    """
    Load YOLO-format labels from a .txt file and convert to pixel coordinates.

    Returns a list of tuples: (class_id, x1, y1, x2, y2)
    """
    boxes: List[Tuple[int, int, int, int, int]] = []

    if not label_path.is_file():
        return boxes

    with label_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                # Malformed line; skip
                continue
            try:
                class_id = int(parts[0])
                x_center = float(parts[1]) * img_width
                y_center = float(parts[2]) * img_height
                w = float(parts[3]) * img_width
                h = float(parts[4]) * img_height
            except ValueError:
                continue

            x1 = int(round(x_center - w / 2))
            y1 = int(round(y_center - h / 2))
            x2 = int(round(x_center + w / 2))
            y2 = int(round(y_center + h / 2))

            # Clamp to image bounds
            x1 = max(0, min(x1, img_width - 1))
            y1 = max(0, min(y1, img_height - 1))
            x2 = max(0, min(x2, img_width - 1))
            y2 = max(0, min(y2, img_height - 1))

            if x2 <= x1 or y2 <= y1:
                continue

            boxes.append((class_id, x1, y1, x2, y2))

    return boxes


def draw_boxes_on_image(
    image,
    boxes: List[Tuple[int, int, int, int, int]],
    class_names: List[str],
    line_thickness: int = 2,
) -> None:
    """
    Draw bounding boxes (in-place) on an OpenCV image array.
    """
    for class_id, x1, y1, x2, y2 in boxes:
        color = (0, 255, 0)  # green
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=line_thickness)

        label = None
        if 0 <= class_id < len(class_names):
            label = class_names[class_id]
        elif class_names:
            label = f"class_{class_id}"

        if label:
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_thickness = 1
            (tw, th), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
            # Ensure the label box stays within the image
            text_x1 = max(0, x1)
            text_y1 = max(th + baseline, y1)
            text_x2 = min(image.shape[1] - 1, text_x1 + tw)
            text_y2 = text_y1 + th + baseline

            cv2.rectangle(image, (text_x1, text_y1 - th - baseline), (text_x2, text_y2), color, thickness=-1)
            cv2.putText(
                image,
                label,
                (text_x1, text_y1 - baseline),
                font,
                font_scale,
                (0, 0, 0),
                thickness=font_thickness,
                lineType=cv2.LINE_AA,
            )


def collect_images(images_dir: Path, max_images: Optional[int], random_order: bool) -> List[Path]:
    image_paths = sorted(
        [p for p in images_dir.glob("**/*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}]
    )

    if not image_paths:
        return []

    if random_order:
        import random

        random.shuffle(image_paths)

    if max_images is not None and max_images > 0:
        image_paths = image_paths[: max_images]

    return image_paths


def visualize_dataset(
    dataset_root_arg: Optional[str],
    split_subdir: str,
    output_dir: Optional[str],
    max_images: Optional[int],
    random_order: bool,
) -> None:
    script_dir = Path(__file__).resolve().parent
    dataset_root = resolve_dataset_root(dataset_root_arg, script_dir)

    images_dir = dataset_root / split_subdir / "images"
    labels_dir = dataset_root / split_subdir / "labels"
    classes_path = dataset_root / "classes.txt"

    if not images_dir.is_dir():
        raise SystemExit(f"Images directory not found: {images_dir}")
    if not labels_dir.is_dir():
        print(f"Warning: labels directory not found: {labels_dir}", file=sys.stderr)

    class_names = load_class_names(classes_path)
    if not class_names:
        class_names = ["sperm"]

    image_paths = collect_images(images_dir, max_images, random_order)
    if not image_paths:
        raise SystemExit(f"No images found in {images_dir}")

    save_dir: Optional[Path] = None
    if output_dir:
        save_dir = Path(output_dir).expanduser().resolve()
        save_dir.mkdir(parents=True, exist_ok=True)

    for idx, img_path in enumerate(image_paths, start=1):
        label_path = labels_dir / (img_path.stem + ".txt")

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Warning: failed to read image: {img_path}", file=sys.stderr)
            continue

        h, w = img.shape[:2]
        boxes = load_yolo_labels(label_path, w, h)

        if not boxes:
            print(f"[{idx}/{len(image_paths)}] {img_path.name}: no boxes")
        else:
            print(f"[{idx}/{len(image_paths)}] {img_path.name}: {len(boxes)} boxes")

        draw_boxes_on_image(img, boxes, class_names)

        if save_dir is not None:
            out_path = save_dir / img_path.name
            ok = cv2.imwrite(str(out_path), img)
            if not ok:
                print(f"Warning: failed to write {out_path}", file=sys.stderr)
        else:
            window_title = f"{img_path.name} ({len(boxes)} boxes)"
            cv2.imshow(window_title, img)
            key = cv2.waitKey(0) & 0xFF
            cv2.destroyWindow(window_title)
            if key in (27, ord("q")):
                break

    if save_dir is None:
        cv2.destroyAllWindows()


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize YOLO bounding boxes for the manually_annotated_100_frames_v0.1 sperm dataset."
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=None,
        help=(
            "Path to dataset root directory containing dataset.yaml. "
            "If omitted, tries dataset.yaml, then falls back to a built-in default."
        ),
    )
    parser.add_argument(
        "--split-subdir",
        type=str,
        default="all",
        help="Subdirectory under dataset root with 'images' and 'labels' folders (default: 'all').",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="If set, saves annotated images to this directory instead of displaying them.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=50,
        help="Maximum number of images to visualize (default: 50). Use <=0 to process all.",
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="If set, randomly sample images instead of iterating in sorted order.",
    )

    args = parser.parse_args(argv)
    if args.max_images is not None and args.max_images <= 0:
        args.max_images = None
    return args


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    visualize_dataset(
        dataset_root_arg=args.dataset_root,
        split_subdir=args.split_subdir,
        output_dir=args.output_dir,
        max_images=args.max_images,
        random_order=args.random,
    )


if __name__ == "__main__":
    main()

