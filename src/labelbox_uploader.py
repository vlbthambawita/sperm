"""
Upload images + YOLO bounding boxes to a Labelbox project using Python Annotation Types.
Based on: https://docs.labelbox.com/reference/import-image-annotations

Assumptions:
- Images and matching YOLO .txt files are in the SAME directory (--data_dir).
- YOLO txt (no header): class_id x_center y_center width height (normalized 0..1)
- Only class_id == 0 is used (single class); tool name must match your ontology feature.

Environment Variables:
  LABELBOX_API_KEY: Your Labelbox API key
  PROJECT_ID: Your Labelbox project ID (optional, can be passed as argument)
  DATA_FOLDER: Default data directory path (optional)

Example:
  python labelbox_uploader.py \
    --data_dir /path/data \
    --project_id <PROJECT_ID> \
    --dataset_name "Sperm YOLO Dataset" \
    --tool_name sperm \
    --import_mode ground_truth
"""

import os
import json
import uuid
import argparse
from typing import List, Dict
from PIL import Image
from tqdm import tqdm

from labelbox import Client, LabelImport, MALPredictionImport
import labelbox.types as lb_types
from lbox.exceptions import MalformedQueryException


# ----------- Args ----------- #
def parse_args():
    p = argparse.ArgumentParser("Upload images + YOLO boxes via Python Annotation Types.")
    p.add_argument("--data_dir", help="Directory with images (.jpg/.jpeg/.png) and YOLO .txt files")
    p.add_argument("--project_id", help="Labelbox project ID")
    p.add_argument("--dataset_id", help="Existing dataset ID (optional)")
    p.add_argument("--dataset_name", default="Sperm YOLO Dataset", help="New dataset name if dataset_id not provided")
    p.add_argument("--api_key_env", default="LABELBOX_API_KEY", help="Env var containing Labelbox API key")
    p.add_argument("--batch_size", type=int, default=100, help="Data row upload batch size")
    p.add_argument("--global_key_prefix", default="", help="Optional prefix for global keys")
    p.add_argument("--tool_name", default="sperm", help="Ontology feature name (e.g., 'sperm')")
    p.add_argument("--skip_unlabeled", action="store_true", help="Skip images without .txt")
    p.add_argument("--import_mode", choices=["ground_truth", "mal"], default="ground_truth",
                   help="Import as ground truth labels or as MAL predictions")
    p.add_argument("--batch_name", help="Optional batch name (auto-generated if omitted)")
    p.add_argument("--dry_run", action="store_true", help="Build labels but do not submit to Labelbox")
    return p.parse_args()


# ----------- Helpers ----------- #
def get_client(api_key_env: str) -> Client:
    key = os.getenv(api_key_env)
    if not key:
        raise ValueError(f"Environment variable {api_key_env} not set")
    return Client(api_key=key)


def ensure_dataset(client: Client, project_id: str, dataset_id: str | None, dataset_name: str):
    if dataset_id:
        ds = client.get_dataset(dataset_id)
        print(f"Using existing dataset: {ds.uid}")
        return ds
    ds = client.create_dataset(name=dataset_name)
    print(f"Created dataset: {ds.uid}")
    # Note: Some SDK versions do not expose project.attach_dataset. Batches associate rows to the project.
    return ds


def collect_images(data_dir: str) -> List[str]:
    exts = {".jpg", ".jpeg", ".png"}
    return sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir)
                   if os.path.splitext(f)[1].lower() in exts])


# NEW: derive global keys from file paths
def global_keys_for_images(image_paths: List[str], prefix: str) -> List[str]:
    return [f"{prefix}{os.path.splitext(os.path.basename(p))[0]}" for p in image_paths]


# NEW: upsert datarows (skip duplicates, only create what’s missing)
def upsert_data_rows(client: Client,
                     dataset,
                     image_paths: List[str],
                     batch_size: int,
                     prefix: str):
    """
    Returns (global_keys, mapping global_key -> data_row_id)
    """
    global_keys = global_keys_for_images(image_paths, prefix)

    # Which keys already exist in the org?
    existing = client.get_data_row_ids_for_global_keys(global_keys)  # {global_key: data_row_id}
    to_create = []
    for img_path, gk in zip(image_paths, global_keys):
        if gk not in existing:
            to_create.append({"row_data": img_path, "global_key": gk})

    if to_create:
        for i in tqdm(range(0, len(to_create), batch_size), desc="Creating new data rows"):
            batch = to_create[i:i + batch_size]
            task = dataset.create_data_rows(batch)
            task.wait_till_done()
            if task.errors:
                print("Data row create errors:", task.errors)
        # refresh after creation
        existing = client.get_data_row_ids_for_global_keys(global_keys)
    else:
        print("All data rows already exist. Skipping upload.")

    return global_keys, existing


def ensure_batch(project, global_keys: List[str], batch_name: str | None):
    name = batch_name or f"batch-{uuid.uuid4()}"
    try:
        project.create_batch(name=name, global_keys=global_keys, priority=1)
        print(f"Created batch '{name}' with {len(global_keys)} items.")
    except MalformedQueryException as e:
        print(f"Could not create batch: {e}")
        print("Likely all data rows are already in the project and/or already batched. Continuing.")


# ----------- Build Python Annotation Type labels ----------- #
def yolo_to_pixel_boxes(txt_path: str, w: int, h: int) -> List[Dict]:
    """Convert YOLO format annotations to pixel coordinates."""
    if not os.path.exists(txt_path):
        return []
    
    boxes = []
    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                continue
            cls, x_c, y_c, width, height = parts
            if cls != "0":  # Only process class 0 (sperm)
                continue
            
            # Convert normalized coordinates to pixels
            x_c = float(x_c) * w
            y_c = float(y_c) * h
            width = float(width) * w
            height = float(height) * h
            
            # Calculate bounding box coordinates
            left = max(0, x_c - width/2)
            top = max(0, y_c - height/2)
            width = min(width, w - left)
            height = min(height, h - top)
            
            boxes.append({
                "left": int(left),
                "top": int(top),
                "width": int(width),
                "height": int(height)
            })
    
    return boxes


def build_python_labels(image_paths: List[str],
                        data_dir: str,
                        prefix: str,
                        tool_name: str,
                        skip_unlabeled: bool):
    """
    Returns a list of lb_types.Label objects:
      lb_types.Label(
        data={"global_key": "..."},
        annotations=[lb_types.ObjectAnnotation(
            name=tool_name,
            value=lb_types.Rectangle(start=Point(x=left,y=top), end=Point(x=left+width,y=top+height))
        )]
      )
    """
    labels = []
    for img_path in tqdm(image_paths, desc="Preparing Python labels"):
        stem = os.path.splitext(os.path.basename(img_path))[0]
        txt = os.path.join(data_dir, stem + ".txt")
        if not os.path.exists(txt) and skip_unlabeled:
            continue

        with Image.open(img_path) as im:
            w, h = im.size

        boxes = yolo_to_pixel_boxes(txt, w, h) if os.path.exists(txt) else []
        if not boxes:
            continue

        anns = []
        for b in boxes:
            # Python Annotation Types: Rectangle takes two Points (top-left = start, bottom-right = end)
            ann = lb_types.ObjectAnnotation(
                name=tool_name,  # must match your ontology feature's name
                value=lb_types.Rectangle(
                    start=lb_types.Point(x=b["left"], y=b["top"]),  # x=left, y=top
                    end=lb_types.Point(x=b["left"] + b["width"], y=b["top"] + b["height"])  # x=left+width, y=top+height
                )
            )
            anns.append(ann)

        if anns:
            labels.append(lb_types.Label(data=lb_types.DataRow(global_key=f"{prefix}{stem}"), annotations=anns))

    return labels


# ----------- Importers ----------- #
def import_ground_truth(client: Client, project_id: str, labels, dry_run: bool):
    if not labels:
        print("No ground truth labels to import.")
        return
    print(f"Importing {len(labels)} ground truth labels (Python objects)...")
    if dry_run:
        print("Dry run enabled. Skipping LabelImport.")
        return
    job = LabelImport.create_from_objects(
        client=client,
        project_id=project_id,
        name="YOLO GT (Python Types) Import",
        labels=labels
    )
    job.wait_until_done()
    print("LabelImport state:", job.state)
    if job.errors:
        print("Errors (truncated):", json.dumps(job.errors, indent=2)[:2000])
    else:
        print("Ground truth import successful.")


def import_mal(client: Client, project_id: str, labels, dry_run: bool):
    if not labels:
        print("No MAL predictions to import.")
        return
    print(f"Importing {len(labels)} MAL predictions (Python objects)...")
    if dry_run:
        print("Dry run enabled. Skipping MALPredictionImport.")
        return
    job = MALPredictionImport.create_from_objects(
        client=client,
        project_id=project_id,
        name="YOLO MAL (Python Types) Import",
        predictions=labels
    )
    job.wait_until_done()
    print("MALPredictionImport states:", job.statuses)
    if job.errors:
        print("Errors (truncated):", json.dumps(job.errors, indent=2)[:2000])
    else:
        print("MAL import successful.")


# ----------- Main ----------- #
def main():
    args = parse_args()
    
    # Use environment variables as defaults if arguments not provided
    data_dir = args.data_dir or os.getenv('DATA_FOLDER')
    project_id = args.project_id or os.getenv('PROJECT_ID')
    
    if not data_dir:
        raise ValueError("Data directory must be provided via --data_dir or DATA_FOLDER environment variable")
    if not project_id:
        raise ValueError("Project ID must be provided via --project_id or PROJECT_ID environment variable")
    
    client = get_client(args.api_key_env)
    project = client.get_project(project_id)
    dataset = ensure_dataset(client, project_id, args.dataset_id, args.dataset_name)

    images = collect_images(data_dir)
    if not images:
        raise ValueError("No images found (.jpg/.jpeg/.png)")
    print(f"Found {len(images)} images.")

    # REPLACE upload + batch steps with upsert + safe batch
    global_keys, _ = upsert_data_rows(client, dataset, images, args.batch_size, args.global_key_prefix)
    ensure_batch(project, global_keys, args.batch_name)

    # Build Python Annotation Type labels from YOLO files
    labels = build_python_labels(
        image_paths=images,
        data_dir=data_dir,
        prefix=args.global_key_prefix,
        tool_name=args.tool_name,
        skip_unlabeled=args.skip_unlabeled
    )

    if args.import_mode == "ground_truth":
        import_ground_truth(client, project.uid, labels, args.dry_run)
    else:
        import_mal(client, project.uid, labels, args.dry_run)

    print("Done.")


if __name__ == "__main__":
    main()