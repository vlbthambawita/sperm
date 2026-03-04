#!/usr/bin/env python3
"""
Labelbox Project Image & YOLO Bounding Box Downloader
Downloads images and exports bounding box annotations in YOLO format.

Setup:
    pip install labelbox python-dotenv requests Pillow

.env file example:
    LABELBOX_API_KEY=your_api_key_here
    LABELBOX_PROJECT_ID=your_project_id_here
    OUTPUT_DIR=./labelbox_export
    SPLIT_RATIOS=0.8,0.1,0.1        # train/val/test (optional, omit to skip splitting)
    IMAGE_SUBDIR=images              # subdirectory name for images
    LABEL_SUBDIR=labels              # subdirectory name for YOLO label files
    DOWNLOAD_WORKERS=4               # parallel download threads
    SKIP_UNLABELED=true              # skip images with no annotations
"""

import os
import json
import time
import shutil
import random
import logging
import argparse
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, cast

import requests
from dotenv import load_dotenv
import labelbox as lb

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config() -> dict:
    load_dotenv()

    def _bool(val: str) -> bool:
        return val.strip().lower() in ("1", "true", "yes")

    api_key = os.getenv("LABELBOX_API_KEY", "").strip()
    project_id = os.getenv("LABELBOX_PROJECT_ID", "").strip()

    if not api_key:
        raise ValueError("LABELBOX_API_KEY is not set in .env or environment.")
    if not project_id:
        raise ValueError("LABELBOX_PROJECT_ID is not set in .env or environment.")

    split_env = os.getenv("SPLIT_RATIOS", "").strip()
    split_ratios = None
    if split_env:
        parts = [float(x) for x in split_env.split(",")]
        assert len(parts) == 3, "SPLIT_RATIOS must have exactly 3 values: train,val,test"
        assert abs(sum(parts) - 1.0) < 1e-6, "SPLIT_RATIOS must sum to 1.0"
        split_ratios = parts

    return {
        "api_key": api_key,
        "project_id": project_id,
        "output_dir": Path(os.getenv("OUTPUT_DIR", "./labelbox_export")),
        "image_subdir": os.getenv("IMAGE_SUBDIR", "images"),
        "label_subdir": os.getenv("LABEL_SUBDIR", "labels"),
        "workers": int(os.getenv("DOWNLOAD_WORKERS", "4")),
        "skip_unlabeled": _bool(os.getenv("SKIP_UNLABELED", "true")),
        "split_ratios": split_ratios,
    }

# ---------------------------------------------------------------------------
# Labelbox GraphQL helpers
# ---------------------------------------------------------------------------

GQL_ENDPOINT = "https://api.labelbox.com/graphql"


def get_project_info(api_key: str, project_id: str) -> dict:
    """
    Fetch project info and ontology using the Labelbox Python SDK.
    This replaces the previous GraphQL-based implementation.
    """
    client = lb.Client(api_key)
    project = client.get_project(project_id)
    if project is None:
        raise ValueError(f"Project '{project_id}' not found or not accessible.")

    # Ontology is fetched separately via the project.
    # Handle both attribute and callable forms for compatibility across SDK versions.
    ontology_attr = getattr(project, "ontology")
    if callable(ontology_attr):
        ontology_attr = ontology_attr()
    ontology = getattr(ontology_attr, "normalized", {})

    return {
        "id": project.uid,
        "name": project.name,
        "ontology": {
            "normalized": ontology,
        },
    }


def export_labels_v2(api_key: str, project_id: str) -> list[dict]:
    """
    Use Export V2 via the Python SDK to export labels for a single project.
    Returns a list of data row records in Export V2 JSON format.
    """
    client = lb.Client(api_key)
    project = client.get_project(project_id)

    log.info("Triggering Export V2 task …")

    # Match the official Export V2 pattern from docs/export_data.py
    export_params: Dict[str, bool] = {
        "attachments": False,
        "metadata_fields": False,
        "data_row_details": True,
        "project_details": True,
        "label_details": True,
        "performance_details": False,
        "interpolated_frames": False,
        "embeddings": False,
    }

    # No filters → export all data rows in the project
    filters: Dict[str, Any] = {}

    export_task = project.export_v2(params=export_params, filters=filters)  # type: ignore[arg-type]
    export_task.wait_till_done()

    if export_task.errors:
        raise RuntimeError(f"Export V2 task failed: {export_task.errors}")

    export_json_raw = export_task.result
    return cast(List[Dict[str, Any]], export_json_raw)

# ---------------------------------------------------------------------------
# Ontology / class-map helpers
# ---------------------------------------------------------------------------

def build_class_map(ontology_normalized: dict) -> dict[str, int]:
    """Return {class_name: class_index} from the project ontology."""
    class_map: dict[str, int] = {}
    tools = ontology_normalized.get("tools", [])
    for tool in tools:
        if tool.get("tool") in ("rectangle", "bounding-box"):
            name = tool["name"]
            if name not in class_map:
                class_map[name] = len(class_map)
    # Fall back: collect all tool names regardless of type
    if not class_map:
        for tool in tools:
            name = tool.get("name", "")
            if name and name not in class_map:
                class_map[name] = len(class_map)
    log.info(f"Class map ({len(class_map)} classes): {class_map}")
    return class_map

# ---------------------------------------------------------------------------
# YOLO conversion
# ---------------------------------------------------------------------------

def bbox_to_yolo(bbox: dict, img_w: int, img_h: int):
    """
    Convert Labelbox bounding-box annotation to YOLO format.
    Labelbox bbox keys: top, left, height, width  (all in pixels).
    YOLO format: cx cy w h  (all normalised 0..1, relative to image size).
    """
    left = bbox["left"]
    top = bbox["top"]
    w = bbox["width"]
    h = bbox["height"]

    cx = (left + w / 2) / img_w
    cy = (top + h / 2) / img_h
    nw = w / img_w
    nh = h / img_h

    # Clamp to [0, 1]
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    nw = max(0.0, min(1.0, nw))
    nh = max(0.0, min(1.0, nh))
    return cx, cy, nw, nh


def extract_annotations(label_record: dict, class_map: dict[str, int]) -> list[str]:
    """
    Return a list of YOLO annotation lines from a single Labelbox label record.
    Each line: "<class_id> <cx> <cy> <w> <h>"
    """
    lines = []
    media_attrs = label_record.get("media_attributes", {})
    img_w = media_attrs.get("width", 0)
    img_h = media_attrs.get("height", 0)

    # In Export V2, project labels live under:
    #   record["projects"][<project_id>]["labels"][...]["annotations"]["objects"]
    # We also keep compatibility with the older formats this script handled.
    objects: list[dict] = []

    # Export V2 structure
    projects = label_record.get("projects") or {}
    for _proj_id, proj_payload in projects.items():
        for label in proj_payload.get("labels", []):
            annotations = label.get("annotations", {})
            objects.extend(annotations.get("objects") or [])

    # Legacy fallbacks (if any)
    if not objects:
        annotations = label_record.get("labels") or {}
        objects = (
            annotations.get("objects")
            or label_record.get("Label", {}).get("objects")
            or []
        )

    for obj in objects:
        tool_type = obj.get("value", "") or obj.get("title", "")
        class_name = obj.get("title") or obj.get("value") or obj.get("featureSchemaId", "")
        bbox = obj.get("bounding_box") or obj.get("bbox")

        if not bbox:
            continue

        if not img_w or not img_h:
            # Try to get from bbox if image dims missing
            log.warning(f"  No image dimensions found for {label_record.get('data_row', {}).get('id')}, skipping bbox.")
            continue

        class_id = class_map.get(class_name)
        if class_id is None:
            # Try to add it dynamically
            class_id = len(class_map)
            class_map[class_name] = class_id
            log.warning(f"  Unknown class '{class_name}' added as index {class_id}.")

        cx, cy, w, h = bbox_to_yolo(bbox, img_w, img_h)
        lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    return lines

# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

_print_lock = threading.Lock()


def download_image(url: str, dest: Path, api_key: str) -> bool:
    try:
        # Export V2 provides pre-signed asset URLs that already include
        # authorization in the query parameters, so we should NOT send our
        # Labelbox API key as an Authorization header here.
        resp = requests.get(url, timeout=60, stream=True)
        resp.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=65536):
                f.write(chunk)
        return True
    except Exception as exc:
        with _print_lock:
            log.error(f"  Failed to download {url}: {exc}")
        return False

# ---------------------------------------------------------------------------
# Dataset splitting
# ---------------------------------------------------------------------------

def split_dataset(records: list, ratios: list[float]) -> dict[str, list]:
    random.shuffle(records)
    n = len(records)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    return {
        "train": records[:n_train],
        "val": records[n_train: n_train + n_val],
        "test": records[n_train + n_val:],
    }

# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def process_records(records: list, split_name: str, cfg: dict, class_map: dict):
    img_base = cfg["output_dir"] / split_name / cfg["image_subdir"]
    lbl_base = cfg["output_dir"] / split_name / cfg["label_subdir"]
    img_base.mkdir(parents=True, exist_ok=True)
    lbl_base.mkdir(parents=True, exist_ok=True)

    downloaded = skipped = failed = 0

    def _process_one(record):
        nonlocal downloaded, skipped, failed

        data_row = record.get("data_row", {})
        row_id = data_row.get("id", "unknown")
        global_key = data_row.get("global_key") or data_row.get("external_id")
        basename = global_key or row_id

        img_url = data_row.get("row_data") or record.get("Labeled Data", "")
        ext = Path(img_url.split("?")[0]).suffix or ".jpg"
        img_name = f"{basename}{ext}"
        img_path = img_base / img_name
        lbl_path = lbl_base / f"{basename}.txt"

        anno_lines = extract_annotations(record, class_map)

        if cfg["skip_unlabeled"] and not anno_lines:
            with _print_lock:
                skipped += 1
            return

        ok = download_image(img_url, img_path, cfg["api_key"])
        if not ok:
            with _print_lock:
                failed += 1
            return

        lbl_path.write_text("\n".join(anno_lines))
        with _print_lock:
            downloaded += 1
            if downloaded % 50 == 0:
                log.info(f"  [{split_name}] downloaded {downloaded} so far …")

    with ThreadPoolExecutor(max_workers=cfg["workers"]) as pool:
        futures = {pool.submit(_process_one, r): r for r in records}
        for fut in as_completed(futures):
            exc = fut.exception()
            if exc:
                log.error(f"Unexpected error: {exc}")

    log.info(
        f"[{split_name}] done — downloaded: {downloaded}, "
        f"skipped (no labels): {skipped}, failed: {failed}"
    )
    return downloaded


def write_yaml(cfg: dict, class_map: dict, splits: list[str]):
    """Write a dataset YAML for use with Ultralytics YOLO."""
    lines = [f"path: {cfg['output_dir'].resolve()}", ""]
    for split in splits:
        lines.append(f"{split}: {split}/{cfg['image_subdir']}")
    lines += [
        "",
        f"nc: {len(class_map)}",
        f"names: {list(class_map.keys())}",
    ]
    yaml_path = cfg["output_dir"] / "dataset.yaml"
    yaml_path.write_text("\n".join(lines))
    log.info(f"Wrote dataset YAML → {yaml_path}")


def main():
    parser = argparse.ArgumentParser(description="Labelbox → YOLO dataset exporter")
    parser.add_argument("--env", default=".env", help="Path to .env file (default: .env)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting")
    args = parser.parse_args()

    if args.env != ".env":
        load_dotenv(args.env)
    random.seed(args.seed)

    cfg = load_config()
    cfg["output_dir"].mkdir(parents=True, exist_ok=True)
    log.info(f"Output directory: {cfg['output_dir'].resolve()}")

    # 1. Fetch project & ontology
    log.info(f"Fetching project info for ID: {cfg['project_id']}")
    project = get_project_info(cfg["api_key"], cfg["project_id"])
    log.info(f"Project name: {project['name']}")

    ontology = project.get("ontology", {}).get("normalized", {})
    class_map = build_class_map(ontology)

    # 2. Export labels using Export V2
    records = export_labels_v2(cfg["api_key"], cfg["project_id"])
    log.info(f"Total records in Export V2: {len(records)}")

    # 3. Split or use single directory
    if cfg["split_ratios"]:
        splits_data = split_dataset(records, cfg["split_ratios"])
        log.info(
            f"Split: train={len(splits_data['train'])}, "
            f"val={len(splits_data['val'])}, test={len(splits_data['test'])}"
        )
        for split_name, split_records in splits_data.items():
            log.info(f"Processing split: {split_name} ({len(split_records)} records)")
            process_records(split_records, split_name, cfg, class_map)
        write_yaml(cfg, class_map, list(splits_data.keys()))
    else:
        log.info("No split ratios set — saving all records to 'all/' directory.")
        process_records(records, "all", cfg, class_map)
        write_yaml(cfg, class_map, ["all"])

    # 4. Write classes.txt
    classes_path = cfg["output_dir"] / "classes.txt"
    classes_path.write_text(
        "\n".join(f"{i}: {name}" for name, i in sorted(class_map.items(), key=lambda x: x[1]))
    )
    log.info(f"Wrote class list → {classes_path}")
    log.info("✅ Export complete.")


if __name__ == "__main__":
    main()