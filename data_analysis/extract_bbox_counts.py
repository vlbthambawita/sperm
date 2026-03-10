"""
Extract per-image bounding box counts from YOLO annotation files.

Outputs a CSV with:
    image_name, n_ai, n_manual

It reuses the same folder structure as `compare_yolo_annotations.py`:
    - AI_DIR:     folder with AI-generated .txt files
    - MANUAL_DIR: folder with manually corrected .txt files
"""

import os
import glob
import pandas as pd
import numpy as np

# You can edit these paths if needed
AI_DIR     = "/work/vajira/DATA/SPERM_data_2025/clean_100_frames_yolo_ready/all/labels"
MANUAL_DIR = "/work/vajira/DATA/SPERM_data_2025/manually_annotated_100_frames_v1.0/all/labels"
OUTPUT_DIR = "comparison_results"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_yolo_file(path: str) -> np.ndarray:
    """
    Load a YOLO annotation file and return an array of boxes.
    Each line is expected as: <class> <cx> <cy> <w> <h>.
    """
    if not os.path.exists(path):
        return np.zeros((0, 5), dtype=float)

    boxes = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                vals = list(map(float, parts[:5]))
                boxes.append(vals)
            except ValueError:
                continue

    if not boxes:
        return np.zeros((0, 5), dtype=float)
    return np.array(boxes, dtype=float)


def extract_counts() -> pd.DataFrame:
    """
    Build a per-image table with:
        image_name, n_ai, n_manual
    """
    ai_files = {
        os.path.splitext(os.path.basename(p))[0]: p
        for p in glob.glob(os.path.join(AI_DIR, "*.txt"))
    }
    man_files = {
        os.path.splitext(os.path.basename(p))[0]: p
        for p in glob.glob(os.path.join(MANUAL_DIR, "*.txt"))
    }

    all_frames = sorted(set(ai_files) | set(man_files))
    records = []

    for name in all_frames:
        ai_raw = load_yolo_file(ai_files.get(name, ""))
        man_raw = load_yolo_file(man_files.get(name, ""))

        n_ai = len(ai_raw)
        n_man = len(man_raw)

        records.append(
            {
                "image_name": name,
                "n_ai": n_ai,
                "n_manual": n_man,
            }
        )

    df = pd.DataFrame(records)
    return df


def main() -> None:
    df = extract_counts()
    out_path = os.path.join(OUTPUT_DIR, "bbox_counts_per_image.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved bounding-box counts → {out_path}")


if __name__ == "__main__":
    main()

