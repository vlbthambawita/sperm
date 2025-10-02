````python
# filepath: /work/vajira/DL2025/sperm/upload_to_labelbox.py
"""
Upload JPEG images and their YOLO bounding boxes (single class 'sperm') to a Labelbox project.
Images and YOLO txt files live in the SAME directory.

YOLO txt format (no header):
0 x_center_norm y_center_norm width_norm height_norm

Prerequisites:
  export LB_API_KEY=YOUR_KEY
  Ontology must contain ONE Bounding Box tool named exactly: sperm

Run:
  python upload_to_labelbox.py \
      --data_dir /path/to/data \
      --project_id <PROJECT_ID> \
      --dataset_name "Sperm Dataset"

Optional:
  --dataset_id <existing_dataset_id>
  --batch_size 50
  --skip_unlabeled (do not upload images without a txt)
  --global_key_prefix prefix_
"""

import os
import json
import argparse
from uuid import uuid4
from typing import List, Dict
from PIL import Image
from tqdm import tqdm

from labelbox import Client
from labelbox import AnnotationImport


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
    return p.parse_args()


def get_client(api_key_env: str) -> Client:
    key = os.getenv(api_key_env)
    if not key:
        raise ValueError(f"Environment variable {api_key_env} not set")
    return Client(api_key=key)


def ensure_dataset(client: Client, project_id: str, dataset_id: str = None, dataset_name: str = None):
    if dataset_id:
        return client.get_dataset(dataset_id)
    project = client.get_project(project_id)
    ds = client.create_dataset(name=dataset_name or "YOLO Dataset")
    project.attach_dataset(ds)
    return ds


def load_tool_schema_id(project, tool_name: str = "sperm") -> str:
    ontology = project.ontology()
    for tool in ontology.tools():
        if tool.name == tool_name:
            return tool.feature_schema_id
    raise ValueError(f"Bounding Box tool '{tool_name}' not found in ontology.")


def collect_images(data_dir: str) -> List[str]:
    return sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.lower().endswith(".jpg")])


def yolo_boxes(txt_path: str, img_w: int, img_h: int) -> List[Dict]:
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
            cls, xc, yc, w, h = parts
            if cls != "0":
                continue
            xc = float(xc) * img_w
            yc = float(yc) * img_h
            bw = float(w) * img_w
            bh = float(h) * img_h
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
    payload = [{"row_data": p, "global_key": f"{prefix}{os.path.splitext(os.path.basename(p))[0]}"} for p in image_paths]
    for i in tqdm(range(0, len(payload), batch_size), desc="Uploading images"):
        batch = payload[i:i + batch_size]
        task = dataset.create_data_rows(batch)
        task.wait_till_done()
        if task.errors:
            print("Data row upload errors:", task.errors)
    # Map global_key -> id
    lookup = {}
    want = {item["global_key"] for item in payload}
    for dr in dataset.data_rows():
        if dr.global_key in want:
            lookup[dr.global_key] = dr.uid
    return lookup


def build_ndjson(image_paths: List[str],
                 data_dir: str,
                 dr_map: Dict[str, str],
                 tool_schema_id: str,
                 prefix: str,
                 skip_unlabeled: bool) -> List[dict]:
    ndjson = []
    for img_path in tqdm(image_paths, desc="Preparing annotations"):
        stem = os.path.splitext(os.path.basename(img_path))[0]
        txt_path = os.path.join(data_dir, stem + ".txt")
        if not os.path.exists(txt_path) and skip_unlabeled:
            continue
        with Image.open(img_path) as im:
            w, h = im.size
        boxes = yolo_boxes(txt_path, w, h)
        if not boxes:
            continue
        dr_id = dr_map.get(f"{prefix}{stem}")
        if not dr_id:
            print(f"Missing data row id for {stem}, skipping.")
            continue
        for b in boxes:
            ndjson.append({
                "uuid": str(uuid4()),
                "schemaId": tool_schema_id,
                "dataRow": {"id": dr_id},
                "bbox": {
                    "top": b["top"],
                    "left": b["left"],
                    "height": b["height"],
                    "width": b["width"]
                },
                "unit": "PIXEL"
            })
    return ndjson


def import_annotations(client: Client, project_id: str, ndjson_payload: List[dict]):
    if not ndjson_payload:
        print("No annotations to import.")
        return
    print(f"Importing {len(ndjson_payload)} boxes...")
    ann_imp = AnnotationImport.create_from_ndjson(
        client=client,
        project_id=project_id,
        name="YOLO Import",
        annotations=ndjson_payload
    )
    ann_imp.wait_until_done()
    print("State:", ann_imp.state)
    if ann_imp.errors:
        print("Errors (truncated):", json.dumps(ann_imp.errors, indent=2)[:2000])
    else:
        print("Import successful.")


def main():
    args = parse_args()
    client = get_client(args.api_key_env)
    project = client.get_project(args.project_id)
    dataset = ensure_dataset(client, args.project_id, args.dataset_id, args.dataset_name)
    tool_schema_id = load_tool_schema_id(project, "sperm")

    images = collect_images(args.data_dir)
    if not images:
        raise ValueError("No .jpg images found.")
    print(f"Found {len(images)} images.")

    dr_map = upload_data_rows(dataset, images, args.batch_size, args.global_key_prefix)
    ndjson_payload = build_ndjson(images,
                                  args.data_dir,
                                  dr_map,
                                  tool_schema_id,
                                  args.global_key_prefix,
                                  args.skip_unlabeled)
    import_annotations(client, args.project_id, ndjson_payload)
    print("Done.")


if __name__ == "__main__":
    main()
````