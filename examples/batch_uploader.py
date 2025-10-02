# pip install labelbox pillow

import os
import uuid
from pathlib import Path
from PIL import Image
import labelbox as lb
import labelbox.types as lb_types

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# ---------------- USER CONFIG ----------------
# Configuration - Use environment variables for sensitive data
API_KEY = os.getenv("LABELBOX_API_KEY")  # Set via environment variable
PROJECT_ID = os.getenv("PROJECT_ID", "your_project_id_here")  # Set via environment variable
FOLDER = os.getenv("DATA_FOLDER", "/work/vajira/DATA/SPERM_data_2025/testing_set")  # Set via environment variable

# Validate required settings
if not API_KEY:
    raise ValueError("LABELBOX_API_KEY environment variable must be set")
if PROJECT_ID == "your_project_id_here":
    raise ValueError("PROJECT_ID environment variable must be set")
YOLO_TO_LB_NAME = {0: "sperm"}         # YOLO class 0 -> ontology tool "sperm"
IMPORT_MODE = "MAL"                    # "MAL" or "GT"
SKIP_UNLABELED = False                 # True = don't import images without a .txt
# Suffixes to try stripping from image stem when looking for .txt
STRIP_SUFFIXES = ["_clean"]            # extend if needed (e.g., "_aug", "_denoise")
# ------------------------------------------------


def yolo_line_to_xyxy(line: str, W: int, H: int):
    cls, x_c, y_c, w, h = line.strip().split()
    cls = int(cls)
    x_c, y_c, w, h = float(x_c) * W, float(y_c) * H, float(w) * W, float(h) * H
    left, top = x_c - w/2, y_c - h/2
    right, bottom = x_c + w/2, y_c + h/2
    left, top = max(0, left), max(0, top)
    right, bottom = min(W - 1, right), min(H - 1, bottom)
    return cls, left, top, right, bottom


def load_yolo(txt_path: Path, img_path: Path):
    with Image.open(img_path) as im:
        W, H = im.size
    boxes = []
    if txt_path and txt_path.exists():
        with open(txt_path, "r") as f:
            for line in f:
                if line.strip():
                    boxes.append(yolo_line_to_xyxy(line, W, H))
    return boxes


def rect_annotation(tool_name, l, t, r, b):
    return lb_types.ObjectAnnotation(
        name=tool_name,
        value=lb_types.Rectangle(
            start=lb_types.Point(x=l, y=t),
            end=lb_types.Point(x=r, y=b)
        )
    )


def make_label(global_key, boxes):
    anns = []
    for cls, l, t, r, b in boxes:
        tool = YOLO_TO_LB_NAME.get(cls)
        if tool:
            anns.append(rect_annotation(tool, l, t, r, b))
    return lb_types.Label(
        data={"global_key": global_key},   # version-agnostic
        annotations=anns
    )


def find_txt_for_image(img_path: Path) -> Path | None:
    """
    First try <stem>.txt (e.g., foo_clean.jpg -> foo_clean.txt)
    Then try stripping known suffixes (e.g., foo_clean.jpg -> foo.txt)
    """
    direct = img_path.with_suffix(".txt")
    if direct.exists():
        return direct

    stem = img_path.stem
    for suf in STRIP_SUFFIXES:
        if stem.endswith(suf):
            candidate = img_path.with_name(stem[: -len(suf)] + ".txt")
            if candidate.exists():
                return candidate
    return None


def main():
    client = lb.Client(API_KEY)

    # 1) Create dataset and upload all images
    dataset = client.create_dataset(name="yolo-batch-" + uuid.uuid4().hex[:6])

    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    image_paths = [p for p in sorted(Path(FOLDER).iterdir())
                   if p.is_file() and p.suffix.lower() in image_exts]

    if not image_paths:
        raise SystemExit("No images found in folder.")

    rows = [{
        "row_data": str(p),
        "global_key": p.name,
        "external_id": p.stem
    } for p in image_paths]

    task = dataset.create_data_rows(rows)
    task.wait_till_done()
    if task.errors:
        raise RuntimeError(f"DataRow upload errors: {task.errors}")
    print(f"✅ Uploaded {len(rows)} images")

    # 2) Attach to project
    project = client.get_project(PROJECT_ID)
    batch = project.create_batch(
        name="batch-" + uuid.uuid4().hex[:6],
        global_keys=[r["global_key"] for r in rows],
        priority=5
    )
    print("✅ Batch created:", batch.uid)

    # 3) Build labels from YOLO
    labels = []
    skipped_unlabeled = 0
    for img_path in image_paths:
        txt = find_txt_for_image(img_path)
        if txt is None and SKIP_UNLABELED:
            skipped_unlabeled += 1
            continue

        boxes = load_yolo(txt, img_path) if txt else []
        labels.append(make_label(img_path.name, boxes))

    if SKIP_UNLABELED:
        print(f"ℹ️ Skipped unlabeled images: {skipped_unlabeled}")

    # 4) Import as MAL or GT
    if IMPORT_MODE.upper() == "MAL":
        job = lb.MALPredictionImport.create_from_objects(
            client=client,
            project_id=project.uid,
            name="mal-batch-" + uuid.uuid4().hex[:6],
            predictions=labels
        )
    else:
        job = lb.LabelImport.create_from_objects(
            client=client,
            project_id=project.uid,
            name="gt-batch-" + uuid.uuid4().hex[:6],
            labels=labels
        )

    print("⏳ Uploading annotations…")
    job.wait_till_done()
    print("✅ Import finished")
    print("Statuses:", job.statuses)
    print("Errors:", job.errors)


if __name__ == "__main__":
    main()