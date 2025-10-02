import json
import time
import uuid
from pathlib import Path
from typing import Iterable, List, Tuple, Dict, Optional
import os

from PIL import Image
import labelbox as lb
import labelbox.types as lb_types

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ---------------- USER CONFIG ----------------
# Configuration - Use environment variables for sensitive data
API_KEY = os.getenv("LABELBOX_API_KEY")  # Set via environment variable
PROJECT_ID = os.getenv("PROJECT_ID", "your_project_id_here")  # Set via environment variable
FOLDER = os.getenv("DATA_FOLDER", "/path/to/your/data/folder")  # Set via environment variable
YOLO_TO_LB_NAME = {0: "sperm"}              # class 0 -> "sperm"
IMPORT_MODE = os.getenv("IMPORT_MODE", "MAL")  # "MAL" (prelabels) or "GT" (ground truth)
SKIP_UNLABELED = os.getenv("SKIP_UNLABELED", "False").lower() == "true"  # True: skip images with no .txt
STRIP_SUFFIXES = ["_clean"]                 # e.g., ["_clean", "_aug"]
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "10"))  # rows per upload/import job (reduced for stability)
CHECKPOINT_PATH = "labelbox_upload_checkpoint.json"

# Validate required settings
if not API_KEY:
    raise ValueError("LABELBOX_API_KEY environment variable must be set")
if PROJECT_ID == "your_project_id_here":
    raise ValueError("PROJECT_ID environment variable must be set")

# Allowed image extensions
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
# ------------------------------------------------


# ---------- Utilities ----------
def chunked(iterable: Iterable, n: int):
    buf = []
    for x in iterable:
        buf.append(x)
        if len(buf) >= n:
            yield buf
            buf = []
    if buf:
        yield buf


def save_checkpoint(done_keys: set):
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump(sorted(list(done_keys)), f)


def load_checkpoint() -> set:
    p = Path(CHECKPOINT_PATH)
    if p.exists():
        return set(json.loads(p.read_text()))
    return set()


def backoff_call(fn, *args, max_attempts=5, base_delay=1.5, **kwargs):
    attempt = 0
    while True:
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            attempt += 1
            if attempt >= max_attempts:
                raise
            time.sleep(base_delay ** attempt)


# ---------- YOLO helpers ----------
def yolo_line_to_xyxy(line: str, W: int, H: int) -> Tuple[int, float, float, float, float]:
    cls, x_c, y_c, w, h = line.strip().split()
    cls = int(cls)
    x_c, y_c, w, h = float(x_c) * W, float(y_c) * H, float(w) * W, float(h) * H
    left, top = x_c - w/2, y_c - h/2
    right, bottom = x_c + w/2, y_c + h/2
    left, top = max(0, left), max(0, top)
    right, bottom = min(W - 1, right), min(H - 1, bottom)
    return cls, left, top, right, bottom


def load_yolo(txt_path: Optional[Path], img_path: Path) -> List[Tuple[int, float, float, float, float]]:
    with Image.open(img_path) as im:
        W, H = im.size
    boxes = []
    if txt_path and txt_path.exists():
        with open(txt_path, "r") as f:
            for line in f:
                if line.strip():
                    boxes.append(yolo_line_to_xyxy(line, W, H))
    return boxes


def find_txt_for_image(img_path: Path) -> Optional[Path]:
    # try <stem>.txt first
    direct = img_path.with_suffix(".txt")
    if direct.exists():
        return direct
    # then try stripping known suffixes (e.g., _clean)
    stem = img_path.stem
    for suf in STRIP_SUFFIXES:
        if stem.endswith(suf):
            cand = img_path.with_name(stem[: -len(suf)] + ".txt")
            if cand.exists():
                return cand
    return None


# ---------- Labelbox helpers ----------
def rect_annotation(tool_name: str, l: float, t: float, r: float, b: float) -> lb_types.ObjectAnnotation:
    return lb_types.ObjectAnnotation(
        name=tool_name,
        value=lb_types.Rectangle(
            start=lb_types.Point(x=l, y=t),
            end=lb_types.Point(x=r, y=b)
        )
    )


def make_label(global_key: str, boxes: List[Tuple[int, float, float, float, float]]) -> lb_types.Label:
    anns = []
    for cls, l, t, r, b in boxes:
        tool = YOLO_TO_LB_NAME.get(cls)
        if tool:
            anns.append(rect_annotation(tool, l, t, r, b))
    # Version-agnostic reference to the data row by global_key
    return lb_types.Label(
        data={"global_key": global_key},
        annotations=anns
    )


def main():
    client = lb.Client(API_KEY)

    # Collect images
    all_images = [p for p in sorted(Path(FOLDER).iterdir())
                  if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    if not all_images:
        raise SystemExit("No images found.")

    # Create dataset once
    dataset_name = os.getenv("DATASET_NAME", "yolo-bulk-" + uuid.uuid4().hex[:6])
    dataset = backoff_call(client.create_dataset, name=dataset_name)
    print(f"✅ Created dataset: {dataset.uid}")

    # Resume from checkpoint
    done_keys = load_checkpoint()
    print(f"↻ Resuming. {len(done_keys)} rows already done (from checkpoint).")

    # Set consistent prefix for this run
    global_key_prefix = os.getenv("GLOBAL_KEY_PREFIX", f"{int(time.time())}")
    print(f"🔑 Using global key prefix: {global_key_prefix}")

    # Stream in chunks
    for img_chunk in chunked(all_images, CHUNK_SIZE):
        # Skip already processed rows in this chunk
        rows_payload = []
        chunk_keys = []

        for img in img_chunk:
            # Add prefix to make global keys unique
            gk = f"{global_key_prefix}_{img.name}"
            if gk in done_keys:
                continue
            rows_payload.append({
                "row_data": str(img),   # local path; SDK uploads
                "global_key": gk,
                "external_id": img.stem
            })
            chunk_keys.append(gk)

        if not rows_payload:
            continue

        # 1) Create data rows for the chunk
        print(f"⬆️  Uploading {len(rows_payload)} rows…")
        task = backoff_call(dataset.create_data_rows, rows_payload)
        task.wait_till_done()
        if task.errors:
            # If some rows failed, print and continue with the rest
            print("⚠️ DataRow upload errors:", task.errors)

        # Only include successfully created rows (filter out duplicates)
        successfully_created = set()
        if task.errors:
            # Extract failed global keys from errors
            failed_keys = set()
            for error in task.errors:
                if 'failedDataRows' in error:
                    for failed_row in error['failedDataRows']:
                        failed_keys.add(failed_row.get('globalKey'))
            # Only include rows that weren't failed
            successfully_created = {r["global_key"] for r in rows_payload if r["global_key"] not in failed_keys}
        else:
            successfully_created = {r["global_key"] for r in rows_payload}
        
        # 2) Attach chunk rows to project (only if we have valid rows)
        if successfully_created:
            project = client.get_project(PROJECT_ID)
            batch = backoff_call(
                project.create_batch,
                name="batch-" + uuid.uuid4().hex[:6],
                global_keys=list(successfully_created),
                priority=5
            )
            print("✅ Batch created:", batch.uid)
        else:
            print("ℹ️ No new rows to batch (all were duplicates)")
            successfully_created = set()  # Ensure it's empty for later logic

        # 3) Build labels for the chunk
        labels = []
        skipped_unlabeled = 0
        for img in img_chunk:
            # Use the same prefixed global key as used for upload
            gk = f"{global_key_prefix}_{img.name}"
            if gk not in successfully_created:
                continue  # not uploaded in this pass

            txt = find_txt_for_image(img)
            if txt is None and SKIP_UNLABELED:
                skipped_unlabeled += 1
                continue

            boxes = load_yolo(txt, img) if txt else []
            if boxes:
                print(f"📦 Found {len(boxes)} boxes for {img.name}")
                labels.append(make_label(gk, boxes))
            else:
                print(f"⚠️ No boxes found for {img.name}")

        if SKIP_UNLABELED and skipped_unlabeled:
            print(f"ℹ️ Skipped unlabeled in this chunk: {skipped_unlabeled}")

        # 4) Import annotations for the chunk
        if not labels:
            print("ℹ️ No labels to import in this chunk.")
            # still record the rows as done, so we don't re-upload
            done_keys.update(successfully_created)
            save_checkpoint(done_keys)
            continue

        if IMPORT_MODE.upper() == "MAL":
            job = backoff_call(
                lb.MALPredictionImport.create_from_objects,
                client=client,
                project_id=project.uid,
                name="mal-" + uuid.uuid4().hex[:8],
                predictions=labels
            )
        else:
            job = backoff_call(
                lb.LabelImport.create_from_objects,
                client=client,
                project_id=project.uid,
                name="gt-" + uuid.uuid4().hex[:8],
                labels=labels
            )

        print(f"⏳ Uploading {len(labels)} annotations…")
        job.wait_till_done()
        print("✅ Import finished")
        print("Statuses:", job.statuses)
        print("Errors:", job.errors)

        # 5) Update checkpoint so reruns skip this chunk next time
        done_keys.update(successfully_created)
        save_checkpoint(done_keys)

    print("🎉 All done.")


if __name__ == "__main__":
    main()