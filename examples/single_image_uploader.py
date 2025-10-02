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

# Validate required settings
if not API_KEY:
    raise ValueError("LABELBOX_API_KEY environment variable must be set")
if PROJECT_ID == "your_project_id_here":
    raise ValueError("PROJECT_ID environment variable must be set")
IMAGE_PATH = "/work/vajira/DATA/SPERM_data_2025/testing_set/1.1_1sec-frame0_clean.jpg"
YOLO_PATH = "/work/vajira/DATA/SPERM_data_2025/testing_set/1.1_1sec-frame0.txt"

YOLO_TO_LB_NAME = {0: "sperm"}        # YOLO class 0 -> ontology tool "sperm"
IMPORT_MODE = "MAL"                   # "MAL" or "GT"
# ------------------------------------------------


def yolo_line_to_xyxy(line: str, W: int, H: int):
    cls, x_c, y_c, w, h = line.strip().split()
    cls = int(cls)
    x_c, y_c, w, h = float(x_c) * W, float(y_c) * H, float(w) * W, float(h) * H
    left, top = x_c - w/2, y_c - h/2
    right, bottom = x_c + w/2, y_c + h/2
    # clamp to image
    left, top = max(0,left), max(0,top)
    right, bottom = min(W-1,right), min(H-1,bottom)
    return cls, left, top, right, bottom


def load_yolo(txt_path: Path, img_path: Path):
    with Image.open(img_path) as im:
        W, H = im.size
    boxes = []
    if txt_path.exists():
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
    # *** KEY CHANGE: pass a dict with the global_key ***
    return lb_types.Label(
        data={"global_key": global_key},
        annotations=anns
    )


def main():
    client = lb.Client(API_KEY)

    # 1) Create dataset and upload the image
    dataset = client.create_dataset(name="yolo-test-" + uuid.uuid4().hex[:6])
    global_key = Path(IMAGE_PATH).name
    task = dataset.create_data_rows([{
        "row_data": IMAGE_PATH,          # local path -> SDK uploads
        "global_key": global_key,
        "external_id": Path(IMAGE_PATH).stem
    }])
    task.wait_till_done()
    print("✅ Image uploaded")

    # 2) Attach to project via batch (using the global_key)
    project = client.get_project(PROJECT_ID)
    batch = project.create_batch(
        name="batch-" + uuid.uuid4().hex[:6],
        global_keys=[global_key],
        priority=5
    )
    print("✅ Batch created:", batch.uid)

    # 3) Build label from YOLO
    boxes = load_yolo(Path(YOLO_PATH), Path(IMAGE_PATH))
    label = make_label(global_key, boxes)

    # 4) Import as MAL (prelabels) or GT (final)
    if IMPORT_MODE.upper() == "MAL":
        job = lb.MALPredictionImport.create_from_objects(
            client=client,
            project_id=project.uid,
            name="mal-test-" + uuid.uuid4().hex[:6],
            predictions=[label]
        )
    else:
        job = lb.LabelImport.create_from_objects(
            client=client,
            project_id=project.uid,
            name="gt-test-" + uuid.uuid4().hex[:6],
            labels=[label]
        )

    print("⏳ Uploading annotations…")
    job.wait_till_done()
    print("✅ Import finished")
    print("Statuses:", job.statuses)
    print("Errors:", job.errors)


if __name__ == "__main__":
    main()