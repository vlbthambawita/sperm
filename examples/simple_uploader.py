import os
import uuid
from pathlib import Path
from typing import List, Tuple

from PIL import Image
import labelbox as lb
import labelbox.types as lb_types

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ---------------- USER SETTINGS ----------------
# Configuration - Use environment variables for sensitive data
API_KEY = os.getenv("LABELBOX_API_KEY")  # Set via environment variable
PROJECT_ID = os.getenv("PROJECT_ID", "your_project_id_here")  # Set via environment variable
FOLDER = os.getenv("DATA_FOLDER", "/path/to/your/data/folder")  # Set via environment variable
IMPORT_MODE = os.getenv("IMPORT_MODE", "MAL")  # "MAL" for prelabels, "GT" for ground-truth

# Validate required settings
if not API_KEY:
    raise ValueError("LABELBOX_API_KEY environment variable must be set")
if PROJECT_ID == "your_project_id_here":
    raise ValueError("PROJECT_ID environment variable must be set")

# Ontology mapping: YOLO class index -> Labelbox tool name (must match project ontology)
YOLO_TO_LB_NAME = {0: "sperm"}
# ------------------------------------------------


def yolo_to_xyxy(yolo_line: str, W: int, H: int) -> Tuple[float, float, float, float, int]:
    """
    YOLO (class x_c y_c w h) normalized -> pixel (left, top, right, bottom), class_id
    """
    parts = yolo_line.strip().split()
    if len(parts) != 5:
        raise ValueError(f"Bad YOLO line: {yolo_line!r}")
    cls = int(parts[0])
    x_c, y_c, w, h = map(float, parts[1:])

    x_c *= W
    y_c *= H
    w   *= W
    h   *= H

    left   = max(0.0, x_c - w/2)
    top    = max(0.0, y_c - h/2)
    right  = min(W - 1.0, x_c + w/2)
    bottom = min(H - 1.0, y_c + h/2)
    return left, top, right, bottom, cls


def load_boxes(txt_path: Path, img_path: Path) -> List[Tuple[int, float, float, float, float]]:
    """Read all boxes for one image; returns [(cls, left, top, right, bottom), ...]."""
    if not txt_path.exists():
        return []

    with Image.open(img_path) as im:
        W, H = im.size

    out = []
    with open(txt_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            l, t, r, b, cls = yolo_to_xyxy(line, W, H)
            out.append((cls, l, t, r, b))
    return out


def rect_annotation(tool_name: str, l: float, t: float, r: float, b: float) -> lb_types.ObjectAnnotation:
    return lb_types.ObjectAnnotation(
        name=tool_name,  # must match your bounding-box tool name in ontology
        value=lb_types.Rectangle(
            start=lb_types.Point(x=l, y=t),
            end=lb_types.Point(x=r, y=b)
        )
    )


def make_label(global_key: str, boxes: List[Tuple[int, float, float, float, float]]) -> lb_types.Label:
    anns = []
    for cls, l, t, r, b in boxes:
        tool = YOLO_TO_LB_NAME.get(cls)
        if tool is None:
            continue
        anns.append(rect_annotation(tool, l, t, r, b))

    return lb_types.Label(
        data=lb_types.DataRow(global_key=global_key),
        annotations=anns
    )


def main():
    client = lb.Client(API_KEY)

    # 1) Create a dataset and upload all images (we use global_key = filename)
    dataset = client.create_dataset(name=f"yolo-upload-{uuid.uuid4().hex[:8]}")
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

    rows = []
    for p in sorted(Path(FOLDER).iterdir()):
        if p.is_file() and p.suffix.lower() in image_exts:
            rows.append({
                "row_data": str(p),            # local path; SDK uploads it
                "global_key": p.name,          # we’ll reference by filename later
                "external_id": p.stem
            })

    if not rows:
        raise SystemExit("No images found in FOLDER.")

    task = dataset.create_data_rows(rows)
    task.wait_till_done()
    if task.errors:
        raise RuntimeError(f"DataRow upload errors: {task.errors}")
    print(f"✅ Uploaded {len(rows)} images to dataset: {dataset.uid}")

    # 2) Add rows to the project queue (via batch using global_keys)
    project = client.get_project(PROJECT_ID)
    batch = project.create_batch(
        name=f"batch-{uuid.uuid4().hex[:8]}",
        global_keys=[r["global_key"] for r in rows],
        priority=5,
    )
    print(f"✅ Created batch for project: {project.uid} -> {batch.uid}")

    # 3) Build labels from YOLO files and import
    labels = []
    missing_classes = set()

    for p in sorted(Path(FOLDER).iterdir()):
        if not (p.is_file() and p.suffix.lower() in image_exts):
            continue
        txt = p.with_suffix(".txt")
        boxes = load_boxes(txt, p)

        # check class mapping
        for cls, *_ in boxes:
            if cls not in YOLO_TO_LB_NAME:
                missing_classes.add(cls)

        labels.append(make_label(global_key=p.name, boxes=boxes))

    if missing_classes:
        print("⚠️  Unmapped classes were skipped:", sorted(missing_classes))

    if IMPORT_MODE.upper() == "MAL":
        job = lb.MALPredictionImport.create_from_objects(
            client=client,
            project_id=project.uid,
            name=f"mal-{uuid.uuid4().hex[:8]}",
            predictions=labels
        )
    elif IMPORT_MODE.upper() == "GT":
        job = lb.LabelImport.create_from_objects(
            client=client,
            project_id=project.uid,
            name=f"gt-{uuid.uuid4().hex[:8]}",
            labels=labels
        )
    else:
        raise ValueError("IMPORT_MODE must be 'MAL' or 'GT'")

    print("⏳ Uploading annotations…")
    job.wait_till_done()
    print("✅ Import finished.")
    print("Statuses:", job.statuses)
    print("Errors:", job.errors)


if __name__ == "__main__":
    main()