"""
Upload JPEG images + YOLO bounding boxes (single class 'sperm') to a Labelbox project.

Images and YOLO txt files are in the SAME directory.

YOLO txt (no header):
0 x_center_norm y_center_norm width_norm height_norm

Key updates (aligned with image.py reference notebook):
- Ensure ontology tool name -> 'sperm'
- After uploading data rows, create a Batch for the project (needed for MAL / labeling workflow)
- Option to import as ground truth (LabelImport) or as pre-labels (MAL) via --import_mode
- Use global keys consistently (Labelbox best practice)
- Use schemaId (feature schema id) for AnnotationImport NDJSON
- Removed unnecessary 'unit' field (not required)
- Added validation & clearer logging
- Added --dry_run option

Prerequisites:
  export LB_API_KEY=YOUR_KEY
  Ontology: Bounding Box tool named exactly 'sperm'

Run example:
  python upload_to_labelbox.py \
      --data_dir /path/data \
      --project_id <PROJECT_ID> \
      --dataset_name "Sperm Dataset" \
      --import_mode ground_truth

Possible import_mode values: ground_truth | mal
"""

import os
import json
import argparse
import uuid
from typing import List, Dict
from PIL import Image
from tqdm import tqdm

from labelbox import Client
from labelbox import MALPredictionImport
from labelbox.schema.annotation_import import AnnotationImport

# -------------- Argument Parsing -------------- #
def parse_args():
    p = argparse.ArgumentParser(description="Upload images + YOLO boxes (same dir) to Labelbox.")
    p.add_argument("--data_dir", required=True, help="Directory containing .jpg and .txt files")
    p.add_argument("--project_id", required=True, help="Labelbox project ID")
    p.add_argument("--dataset_id", help="Existing dataset ID (if omitted a new one is created)")
    p.add_argument("--dataset_name", default="Sperm YOLO Dataset", help="Name if creating new dataset")
    p.add_argument("--batch_size", type=int, default=50, help="Batch size for data row upload")
    p.add_argument("--api_key_env", default="LB_API_KEY", help="Env var with API key")
    p.add_argument("--global_key_prefix", default="", help="Optional prefix for global keys")
    p.add_argument("--skip_unlabeled", action="store_true", help="Skip images without txt label file")
    p.add_argument("--import_mode", choices=["ground_truth", "mal"], default="ground_truth",
                   help="Import as ground truth annotations or model-assisted (pre-labels)")
    p.add_argument("--batch_name", help="Optional batch name (auto-generated if omitted)")
    p.add_argument("--dry_run", action="store_true", help="Parse & build payload but do not upload annotations")
    return p.parse_args()


# -------------- Helpers -------------- #
def get_client(api_key_env: str) -> Client:
    key = os.getenv(api_key_env)
    if not key:
        raise ValueError(f"Environment variable {api_key_env} not set")
    return Client(api_key=key)


def ensure_dataset(client: Client, project_id: str, dataset_id: str = None, dataset_name: str = None):
    """
    Return an existing dataset or create a new one.
    Some newer Labelbox SDK versions removed project.attach_dataset().
    We attempt to call it if present; otherwise we log and continue.
    Creating a batch with global keys will still associate the DataRows
    to the project for labeling.
    """
    project = client.get_project(project_id)
    if dataset_id:
        ds = client.get_dataset(dataset_id)
        print(f"Using existing dataset: {ds.uid}")
        return ds

    ds = client.create_dataset(name=dataset_name or "YOLO Dataset")
    print(f"Created dataset: {ds.uid}")

    # Try to attach (older SDKs). Safe fallback if not available.
    try:
        if hasattr(project, "attach_dataset"):
            project.attach_dataset(ds)
            print(f"Attached dataset {ds.uid} to project {project_id}")
        else:
            print("project.attach_dataset not available in this SDK version. Skipping (likely automatic association handled via batches).")
    except Exception as e:
        print(f"Warning: Failed to attach dataset to project: {e}")

    return ds


def load_tool_schema_id(project, tool_name: str = "sperm") -> str:
    ontology = project.ontology()
    for tool in ontology.tools():
        if tool.name == tool_name:
            return tool.feature_schema_id
    raise ValueError(f"Bounding Box tool '{tool_name}' not found in ontology.")


def collect_images(data_dir: str) -> List[str]:
    return sorted([os.path.join(data_dir, f)
                   for f in os.listdir(data_dir)
                   if f.lower().endswith(".jpg")])


def yolo_boxes(txt_path: str, img_w: int, img_h: int) -> List[Dict]:
    if not os.path.exists(txt_path):
        return []
    boxes = []
    with open(txt_path, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                print(f"Warning: {txt_path} line {line_num} malformed -> '{line}'")
                continue
            cls, xc, yc, w, h = parts
            if cls != "0":
                # Only one class in ontology; ignore other ids
                continue
            try:
                xc = float(xc) * img_w
                yc = float(yc) * img_h
                bw = float(w) * img_w
                bh = float(h) * img_h
            except ValueError:
                print(f"Warning: Non-numeric values in {txt_path} line {line_num}")
                continue
            left = xc - bw / 2
            top = yc - bh / 2
            left = max(0, min(left, img_w - 1))
            top = max(0, min(top, img_h - 1))
            bw = max(1, min(bw, img_w - left))
            bh = max(1, min(bh, img_h - top))
            boxes.append({
                "left": int(round(left)),
                "top": int(round(top)),
                "width": int(round(bw)),
                "height": int(round(bh))
            })
    return boxes


def upload_data_rows(dataset, image_paths: List[str], batch_size: int, prefix: str) -> Dict[str, str]:
    payload = [{
        "row_data": p,
        "global_key": f"{prefix}{os.path.splitext(os.path.basename(p))[0]}"
    } for p in image_paths]

    want_keys = {item["global_key"] for item in payload}

    for i in tqdm(range(0, len(payload), batch_size), desc="Uploading images"):
        batch = payload[i:i + batch_size]
        task = dataset.create_data_rows(batch)
        task.wait_till_done()
        if task.errors:
            print("Data row upload errors:", task.errors)

    # Map global key -> data row id
    mapping = {}
    for dr in dataset.data_rows():
        if dr.global_key in want_keys:
            mapping[dr.global_key] = dr.uid
    missing = want_keys - set(mapping.keys())
    if missing:
        print(f"Warning: {len(missing)} data rows not found after upload.")
    return mapping


def ensure_batch(project, global_keys: List[str], batch_name: str = None):
    batch_name = batch_name or f"batch-{uuid.uuid4()}"
    project.create_batch(
        name=batch_name,
        global_keys=global_keys,
        priority=1
    )
    print(f"Created batch '{batch_name}' with {len(global_keys)} items.")


# -------------- Annotation Payload Builders -------------- #
def build_ndjson_ground_truth(image_paths: List[str],
                              data_dir: str,
                              prefix: str,
                              tool_schema_id: str,
                              skip_unlabeled: bool) -> List[dict]:
    ndjson = []
    for img_path in tqdm(image_paths, desc="Preparing GT annotations"):
        stem = os.path.splitext(os.path.basename(img_path))[0]
        txt_path = os.path.join(data_dir, stem + ".txt")
        if not os.path.exists(txt_path) and skip_unlabeled:
            continue
        with Image.open(img_path) as im:
            w, h = im.size
        boxes = yolo_boxes(txt_path, w, h)
        if not boxes:
            continue
        for b in boxes:
            ndjson.append({
                "uuid": str(uuid.uuid4()),
                "schemaId": tool_schema_id,
                "dataRow": {"globalKey": f"{prefix}{stem}"},
                "bbox": {
                    "top": b["top"],
                    "left": b["left"],
                    "height": b["height"],
                    "width": b["width"]
                }
            })
    return ndjson


def build_python_mal_labels(image_paths: List[str],
                            data_dir: str,
                            prefix: str,
                            project,
                            tool_name: str,
                            skip_unlabeled: bool):
    """
    Build Python Annotation Type labels for MAL import.
    """
    import labelbox.types as lb_types
    labels = []
    for img_path in tqdm(image_paths, desc="Preparing MAL predictions"):
        stem = os.path.splitext(os.path.basename(img_path))[0]
        txt_path = os.path.join(data_dir, stem + ".txt")
        if not os.path.exists(txt_path) and skip_unlabeled:
            continue
        with Image.open(img_path) as im:
            w, h = im.size
        boxes = yolo_boxes(txt_path, w, h)
        if not boxes:
            continue
        object_annotations = []
        for b in boxes:
            object_annotations.append(
                lb_types.ObjectAnnotation(
                    name=tool_name,
                    value=lb_types.Rectangle(
                        start=lb_types.Point(x=b["left"], y=b["top"]),
                        end=lb_types.Point(x=b["left"] + b["width"],
                                           y=b["top"] + b["height"])
                    )
                )
            )
        if object_annotations:
            labels.append(
                lb_types.Label(
                    data={"global_key": f"{prefix}{stem}"},
                    annotations=object_annotations
                )
            )
    return labels


# -------------- Import Functions -------------- #
def import_ground_truth(client: Client, project_id: str, ndjson_payload: List[dict], dry_run: bool):
    if not ndjson_payload:
        print("No ground truth annotations to import.")
        return
    print(f"Ground truth payload size: {len(ndjson_payload)}")
    if dry_run:
        print("Dry run enabled. Skipping ground truth import.")
        return
    ann_imp = AnnotationImport.create_from_ndjson(
        client=client,
        project_id=project_id,
        name="YOLO GT Import",
        annotations=ndjson_payload
    )
    ann_imp.wait_until_done()
    print("Import state:", ann_imp.state)
    if ann_imp.errors:
        print("Errors (truncated):", json.dumps(ann_imp.errors, indent=2)[:2000])
    else:
        print("Ground truth import successful.")


def import_mal_predictions(client: Client, project_id: str, labels, dry_run: bool):
    if not labels:
        print("No MAL prediction labels to import.")
        return
    print(f"MAL prediction label count: {len(labels)}")
    if dry_run:
        print("Dry run enabled. Skipping MAL import.")
        return
    job = MALPredictionImport.create_from_objects(
        client=client,
        project_id=project_id,
        name="YOLO MAL Import " + str(uuid.uuid4()),
        predictions=labels
    )
    job.wait_until_done()
    print("MAL import states:", job.statuses)
    if job.errors:
        print("Errors (truncated):", json.dumps(job.errors, indent=2)[:2000])
    else:
        print("MAL prediction import successful.")


# -------------- Main -------------- #
def main():
    args = parse_args()
    client = get_client(args.api_key_env)
    project = client.get_project(args.project_id)
    dataset = ensure_dataset(client, args.project_id, args.dataset_id, args.dataset_name)

    tool_schema_id = load_tool_schema_id(project, tool_name="sperm")
    tool_name = "sperm"

    images = collect_images(args.data_dir)
    if not images:
        raise ValueError("No .jpg images found.")
    print(f"Found {len(images)} images.")

    # Upload data rows
    dr_map = upload_data_rows(dataset, images, args.batch_size, args.global_key_prefix)

    # Create batch in project (required for labeling/MAL workflows)
    global_keys = [f"{args.global_key_prefix}{os.path.splitext(os.path.basename(p))[0]}" for p in images]
    ensure_batch(project, global_keys, args.batch_name)

    # Build and import annotations
    if args.import_mode == "ground_truth":
        ndjson_payload = build_ndjson_ground_truth(
            image_paths=images,
            data_dir=args.data_dir,
            prefix=args.global_key_prefix,
            tool_schema_id=tool_schema_id,
            skip_unlabeled=args.skip_unlabeled
        )
        import_ground_truth(client, project.uid, ndjson_payload, args.dry_run)
    else:  # MAL
        labels = build_python_mal_labels(
            image_paths=images,
            data_dir=args.data_dir,
            prefix=args.global_key_prefix,
            project=project,
            tool_name=tool_name,
            skip_unlabeled=args.skip_unlabeled
        )
        import_mal_predictions(client, project.uid, labels, args.dry_run)

    print("Done.")


if __name__ == "__main__":
    main()