"""
YOLO Bounding Box Annotation Comparison Tool
Compares AI-automated vs manually corrected sperm annotations.

Expected folder structure:
    ai_annotations/       <- folder with AI-generated .txt files
    manual_annotations/   <- folder with manually corrected .txt files

YOLO format per line: <class> <cx> <cy> <w> <h>  (all normalized 0-1)
"""

import os
import glob
import numpy as np
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from scipy.optimize import linear_sum_assignment
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG — edit these paths
# ─────────────────────────────────────────────
AI_DIR     = "/work/vajira/DATA/SPERM_data_2025/manually_annotated_100_frames_v1.0/all/labels"        # folder with AI .txt files
MANUAL_DIR = "/work/vajira/DATA/SPERM_data_2025/clean_100_frames_yolo_ready/all/labels"    # folder with manual .txt files
IMAGES_DIR = "/work/vajira/DATA/SPERM_data_2025/clean_100_frames_yolo_ready/all/images"                    # folder with images (optional; if None, uses IMG_W/IMG_H fallback)
IOU_THRESHOLD = 0.5                  # match threshold
IMG_W, IMG_H  = 1920, 1080          # fallback pixel dimensions (used when IMAGES_DIR is None)
OUTPUT_DIR    = "comparison_results"
# ─────────────────────────────────────────────

# Image extensions to search for
IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Helpers ──────────────────────────────────

def get_image_dimensions(images_dir):
    """
    Get (width, height) from the first image found in images_dir.
    Searches recursively. Returns (IMG_W, IMG_H) fallback if no images found.
    """
    if not images_dir or not os.path.isdir(images_dir):
        return IMG_W, IMG_H
    for root, _, files in os.walk(images_dir):
        for f in files:
            if f.lower().endswith(IMG_EXTENSIONS):
                try:
                    with Image.open(os.path.join(root, f)) as img:
                        w, h = img.size
                        return w, h
                except Exception:
                    continue
    return IMG_W, IMG_H


def get_image_dimensions_for_frame(images_dir, frame_name):
    """
    Get (width, height) for a specific frame. Tries frame_name as-is,
    then with common extensions. Returns (IMG_W, IMG_H) fallback if not found.
    """
    if not images_dir or not os.path.isdir(images_dir):
        return IMG_W, IMG_H
    candidates = [frame_name]
    if not any(frame_name.lower().endswith(ext) for ext in IMG_EXTENSIONS):
        candidates.extend(frame_name + ext for ext in IMG_EXTENSIONS)
    for root, _, files in os.walk(images_dir):
        for f in files:
            base = os.path.splitext(f)[0]
            if base == os.path.splitext(frame_name)[0] or f in candidates:
                try:
                    with Image.open(os.path.join(root, f)) as img:
                        return img.size
                except Exception:
                    pass
    return IMG_W, IMG_H


def load_yolo_file(path):
    """Return Nx5 array [class, cx, cy, w, h] or empty array."""
    boxes = []
    if not os.path.exists(path):
        return np.zeros((0, 5))
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                parts = list(map(float, line.split()))
                if len(parts) >= 5:
                    boxes.append(parts[:5])
    return np.array(boxes) if boxes else np.zeros((0, 5))


def yolo_to_xyxy(boxes):
    """Convert [cls, cx, cy, w, h] → [cls, x1, y1, x2, y2]."""
    if len(boxes) == 0:
        return boxes
    b = boxes.copy()
    b[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # x1
    b[:, 2] = boxes[:, 2] - boxes[:, 4] / 2  # y1
    b[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # x2
    b[:, 4] = boxes[:, 2] + boxes[:, 4] / 2  # y2
    return b


def compute_iou(b1, b2):
    """IoU between two boxes [x1,y1,x2,y2]."""
    xi1, yi1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    xi2, yi2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def match_boxes(ai_xyxy, man_xyxy, threshold):
    """
    Hungarian matching; returns lists of (ai_idx, man_idx, iou),
    unmatched_ai indices, unmatched_man indices.
    """
    if len(ai_xyxy) == 0 or len(man_xyxy) == 0:
        return [], list(range(len(ai_xyxy))), list(range(len(man_xyxy)))

    iou_mat = np.zeros((len(ai_xyxy), len(man_xyxy)))
    for i, a in enumerate(ai_xyxy):
        for j, m in enumerate(man_xyxy):
            iou_mat[i, j] = compute_iou(a[1:], m[1:])

    row_ind, col_ind = linear_sum_assignment(-iou_mat)
    matches, unmatched_ai, unmatched_man = [], [], []
    matched_ai, matched_man = set(), set()

    for r, c in zip(row_ind, col_ind):
        iou = iou_mat[r, c]
        if iou >= threshold:
            matches.append((r, c, iou))
            matched_ai.add(r); matched_man.add(c)

    unmatched_ai  = [i for i in range(len(ai_xyxy))  if i not in matched_ai]
    unmatched_man = [j for j in range(len(man_xyxy)) if j not in matched_man]
    return matches, unmatched_ai, unmatched_man


# ── Main analysis ─────────────────────────────

def run_analysis():
    # Resolve image dimensions from images folder if configured
    img_w, img_h = get_image_dimensions(IMAGES_DIR)
    if IMAGES_DIR:
        print(f"Image dimensions from {IMAGES_DIR}: {img_w}×{img_h} px")
    else:
        print(f"Image dimensions (fallback): {img_w}×{img_h} px")

    ai_files  = {os.path.splitext(os.path.basename(p))[0]: p
                 for p in glob.glob(os.path.join(AI_DIR,     "*.txt"))}
    man_files = {os.path.splitext(os.path.basename(p))[0]: p
                 for p in glob.glob(os.path.join(MANUAL_DIR, "*.txt"))}

    all_frames = sorted(set(ai_files) | set(man_files))
    print(f"Frames found — AI: {len(ai_files)}, Manual: {len(man_files)}, Union: {len(all_frames)}")

    records = []           # per-frame summary
    matched_ious = []      # IoU of every matched pair
    centroid_dists = []    # centroid distance (normalised) per matched pair
    w_diffs, h_diffs = [], []
    area_ai_all, area_man_all = [], []

    for name in all_frames:
        ai_raw  = load_yolo_file(ai_files.get(name,  ""))
        man_raw = load_yolo_file(man_files.get(name, ""))

        ai_xyxy  = yolo_to_xyxy(ai_raw)  if len(ai_raw)  else ai_raw
        man_xyxy = yolo_to_xyxy(man_raw) if len(man_raw) else man_raw

        n_ai, n_man = len(ai_raw), len(man_raw)
        matches, fp_idx, fn_idx = match_boxes(ai_xyxy, man_xyxy, IOU_THRESHOLD)

        tp = len(matches)
        fp = len(fp_idx)
        fn = len(fn_idx)
        precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
        recall    = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        f1        = 2*precision*recall/(precision+recall) if (precision+recall) > 0 else np.nan
        mean_iou  = np.mean([m[2] for m in matches]) if matches else np.nan

        records.append(dict(frame=name, n_ai=n_ai, n_man=n_man,
                            count_diff=n_ai - n_man,
                            tp=tp, fp=fp, fn=fn,
                            precision=precision, recall=recall,
                            f1=f1, mean_iou=mean_iou))

        for ai_i, man_j, iou in matches:
            matched_ious.append(iou)
            # centroid distance (normalised units, multiply by img dims for px)
            cx_diff = (ai_raw[ai_i, 1] - man_raw[man_j, 1]) * img_w
            cy_diff = (ai_raw[ai_i, 2] - man_raw[man_j, 2]) * img_h
            centroid_dists.append(np.sqrt(cx_diff**2 + cy_diff**2))
            w_diffs.append((ai_raw[ai_i, 3] - man_raw[man_j, 3]))
            h_diffs.append((ai_raw[ai_i, 4] - man_raw[man_j, 4]))

        area_ai_all.extend( ai_raw[:, 3] * ai_raw[:, 4] if len(ai_raw)  else [])
        area_man_all.extend(man_raw[:, 3]* man_raw[:, 4] if len(man_raw) else [])

    df = pd.DataFrame(records)

    # ── Statistical tests ─────────────────────
    count_stat  = stats.wilcoxon(df["n_ai"], df["n_man"], zero_method="wilcox") \
                  if len(df) >= 10 else None
    iou_arr     = np.array(matched_ious)
    cdist_arr   = np.array(centroid_dists)
    w_arr       = np.array(w_diffs)
    h_arr       = np.array(h_diffs)

    # ICC(2,1) approximation via one-way ANOVA
    def icc_oneway(y1, y2):
        n = len(y1)
        data = np.column_stack([y1, y2])
        grand = data.mean()
        ss_b = n * np.sum((data.mean(axis=1) - grand)**2)
        ss_w = np.sum((data - data.mean(axis=1, keepdims=True))**2)
        ms_b = ss_b / (n - 1)
        ms_w = ss_w / n
        icc  = (ms_b - ms_w) / (ms_b + ms_w)
        return icc

    icc_count = icc_oneway(df["n_ai"].values, df["n_man"].values)

    # ── Print summary ─────────────────────────
    print("\n" + "="*55)
    print("  ANNOTATION COMPARISON SUMMARY")
    print("="*55)
    print(f"\n{'Frames analysed':<35} {len(df)}")
    print(f"{'Total matched pairs (TP)':<35} {int(df['tp'].sum())}")
    print(f"{'Total false positives (AI only)':<35} {int(df['fp'].sum())}")
    print(f"{'Total false negatives (missed)':<35} {int(df['fn'].sum())}")
    print(f"\n--- Count agreement ---")
    print(f"  Mean AI count/frame:              {df['n_ai'].mean():.2f} ± {df['n_ai'].std():.2f}")
    print(f"  Mean manual count/frame:          {df['n_man'].mean():.2f} ± {df['n_man'].std():.2f}")
    print(f"  Mean count difference (AI-manual):{df['count_diff'].mean():.2f} ± {df['count_diff'].std():.2f}")
    print(f"  ICC (count):                      {icc_count:.3f}")
    if count_stat:
        print(f"  Wilcoxon p-value (counts):        {count_stat.pvalue:.4f}")
    print(f"\n--- Detection quality (IoU ≥ {IOU_THRESHOLD}) ---")
    print(f"  Mean Precision:  {df['precision'].mean():.3f} ± {df['precision'].std():.3f}")
    print(f"  Mean Recall:     {df['recall'].mean():.3f} ± {df['recall'].std():.3f}")
    print(f"  Mean F1:         {df['f1'].mean():.3f} ± {df['f1'].std():.3f}")
    if len(iou_arr):
        print(f"\n--- IoU of matched pairs ---")
        print(f"  Mean IoU:   {iou_arr.mean():.3f}")
        print(f"  Median IoU: {np.median(iou_arr):.3f}")
        print(f"  Std IoU:    {iou_arr.std():.3f}")
        print(f"  % IoU≥0.75: {100*(iou_arr>=0.75).mean():.1f}%")
    if len(cdist_arr):
        print(f"\n--- Centroid distance (pixels) ---")
        print(f"  Mean:   {cdist_arr.mean():.2f} px")
        print(f"  Median: {np.median(cdist_arr):.2f} px")
    if len(w_arr):
        w_p = stats.wilcoxon(w_arr) if (w_arr != 0).any() else None
        h_p = stats.wilcoxon(h_arr) if (h_arr != 0).any() else None
        print(f"\n--- Box size differences (AI - manual, normalised) ---")
        print(f"  Width  bias: {w_arr.mean():.4f}  p={w_p.pvalue:.4f}" if w_p else f"  Width  bias: {w_arr.mean():.4f}")
        print(f"  Height bias: {h_arr.mean():.4f}  p={h_p.pvalue:.4f}" if h_p else f"  Height bias: {h_arr.mean():.4f}")
    else:
        w_p, h_p = None, None
    print("="*55)

    # ── Save summary to Markdown ──────────────
    md_lines = [
        "# YOLO Annotation Comparison Summary",
        "",
        "## Overview",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Frames analysed | {len(df)} |",
        f"| Total matched pairs (TP) | {int(df['tp'].sum())} |",
        f"| Total false positives (AI only) | {int(df['fp'].sum())} |",
        f"| Total false negatives (missed) | {int(df['fn'].sum())} |",
        "",
        "## Count agreement",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Mean AI count/frame | {df['n_ai'].mean():.2f} ± {df['n_ai'].std():.2f} |",
        f"| Mean manual count/frame | {df['n_man'].mean():.2f} ± {df['n_man'].std():.2f} |",
        f"| Mean count difference (AI - manual) | {df['count_diff'].mean():.2f} ± {df['count_diff'].std():.2f} |",
        f"| ICC (count) | {icc_count:.3f} |",
    ]
    if count_stat:
        md_lines.append(f"| Wilcoxon p-value (counts) | {count_stat.pvalue:.4f} |")
    md_lines.extend([
        "",
        f"## Detection quality (IoU ≥ {IOU_THRESHOLD})",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Mean Precision | {df['precision'].mean():.3f} ± {df['precision'].std():.3f} |",
        f"| Mean Recall | {df['recall'].mean():.3f} ± {df['recall'].std():.3f} |",
        f"| Mean F1 | {df['f1'].mean():.3f} ± {df['f1'].std():.3f} |",
        "",
    ])
    if len(iou_arr):
        md_lines.extend([
            "## IoU of matched pairs",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Mean IoU | {iou_arr.mean():.3f} |",
            f"| Median IoU | {np.median(iou_arr):.3f} |",
            f"| Std IoU | {iou_arr.std():.3f} |",
            f"| % IoU ≥ 0.75 | {100*(iou_arr>=0.75).mean():.1f}% |",
            "",
        ])
    if len(cdist_arr):
        md_lines.extend([
            "## Centroid distance (pixels)",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Mean | {cdist_arr.mean():.2f} px |",
            f"| Median | {np.median(cdist_arr):.2f} px |",
            "",
        ])
    if len(w_arr):
        w_str = f"{w_arr.mean():.4f} (p={w_p.pvalue:.4f})" if w_p else f"{w_arr.mean():.4f}"
        h_str = f"{h_arr.mean():.4f} (p={h_p.pvalue:.4f})" if h_p else f"{h_arr.mean():.4f}"
        md_lines.extend([
            "## Box size differences (AI - manual, normalised)",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Width bias | {w_str} |",
            f"| Height bias | {h_str} |",
            "",
        ])
    md_lines.extend([
        "## Metrics description",
        "",
        "- **Frames analysed**: Number of image frames where AI and manual annotations were compared.",
        "- **Matched pairs (TP)**: Number of AI–manual box pairs matched using the IoU threshold (IoU ≥ "
        f"{IOU_THRESHOLD}).",
        "- **False positives (FP)**: AI detections that could not be matched to any manual annotation (extra boxes).",
        "- **False negatives (FN)**: Manual annotations that did not have a corresponding AI detection (missed boxes).",
        "- **Precision**: Proportion of AI detections that are correct, TP / (TP + FP).",
        "- **Recall**: Proportion of manual annotations that are recovered by the AI, TP / (TP + FN).",
        "- **F1 score**: Harmonic mean of precision and recall, summarising detection performance in a single number.",
        "- **Mean IoU / IoU distribution**: Overlap between matched AI and manual boxes; higher values indicate closer"
        " spatial agreement.",
        "- **Centroid distance (pixels)**: Euclidean distance between the centres of matched boxes, measured in image"
        " pixels.",
        "- **Box size differences (AI - manual)**: Bias in normalised width and height; positive values mean the AI"
        " boxes tend to be larger than manual boxes.",
        "- **ICC (count)**: Intraclass correlation coefficient for sperm counts per frame, quantifying overall count"
        " agreement between AI and manual annotations.",
        "- **Wilcoxon p-values**: Non-parametric tests for systematic differences in counts or box sizes; small"
        " p-values indicate statistically significant bias.",
        "",
        "## Visual summary",
        "",
        "![AI vs manual annotation comparison plots](comparison_plots.png)",
        "",
        "## Outputs",
        "",
        "- [per_frame_stats.csv](per_frame_stats.csv)",
        "- [comparison_plots.png](comparison_plots.png)",
        "",
    ])
    summary_md_path = os.path.join(OUTPUT_DIR, "summary.md")
    with open(summary_md_path, "w") as f:
        f.write("\n".join(md_lines))
    print(f"\nSummary saved → {summary_md_path}")

    # ── Save CSV ──────────────────────────────
    csv_path = os.path.join(OUTPUT_DIR, "per_frame_stats.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nPer-frame CSV saved → {csv_path}")

    # ── Plots ─────────────────────────────────
    fig = plt.figure(figsize=(18, 14))
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    # 1. Count scatter
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(df["n_man"], df["n_ai"], alpha=0.5, s=20, color="#2196F3")
    lim = max(df[["n_ai","n_man"]].max()) * 1.05
    ax1.plot([0, lim], [0, lim], "r--", lw=1)
    ax1.set_xlabel("Manual count"); ax1.set_ylabel("AI count")
    ax1.set_title("Count agreement (per frame)")

    # 2. Bland-Altman: counts
    ax2 = fig.add_subplot(gs[0, 1])
    mean_c = (df["n_ai"] + df["n_man"]) / 2
    diff_c = df["n_ai"] - df["n_man"]
    md, sd = diff_c.mean(), diff_c.std()
    ax2.scatter(mean_c, diff_c, alpha=0.5, s=20, color="#4CAF50")
    ax2.axhline(md,          color="red",  lw=1.5, label=f"Bias={md:.2f}")
    ax2.axhline(md + 1.96*sd, color="gray", lw=1, ls="--", label=f"±1.96 SD")
    ax2.axhline(md - 1.96*sd, color="gray", lw=1, ls="--")
    ax2.set_xlabel("Mean count"); ax2.set_ylabel("Difference (AI – manual)")
    ax2.set_title("Bland-Altman: counts"); ax2.legend(fontsize=8)

    # 3. Count difference histogram
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.hist(diff_c, bins=20, color="#FF9800", edgecolor="white")
    ax3.axvline(0, color="red", lw=1.5, ls="--")
    ax3.set_xlabel("AI count – Manual count"); ax3.set_ylabel("Frames")
    ax3.set_title("Count difference distribution")

    # 4. IoU distribution
    ax4 = fig.add_subplot(gs[1, 0])
    if len(iou_arr):
        ax4.hist(iou_arr, bins=30, color="#9C27B0", edgecolor="white")
        ax4.axvline(iou_arr.mean(), color="red", lw=1.5, ls="--",
                    label=f"Mean={iou_arr.mean():.3f}")
        ax4.axvline(0.5,  color="orange", lw=1, ls=":", label="IoU=0.5")
        ax4.axvline(0.75, color="green",  lw=1, ls=":", label="IoU=0.75")
        ax4.set_xlabel("IoU"); ax4.set_ylabel("Matched pairs")
        ax4.set_title("IoU distribution (matched boxes)"); ax4.legend(fontsize=8)

    # 5. Centroid distance histogram
    ax5 = fig.add_subplot(gs[1, 1])
    if len(cdist_arr):
        ax5.hist(cdist_arr, bins=30, color="#00BCD4", edgecolor="white")
        ax5.axvline(cdist_arr.mean(), color="red", lw=1.5, ls="--",
                    label=f"Mean={cdist_arr.mean():.1f} px")
        ax5.set_xlabel("Centroid distance (px)"); ax5.set_ylabel("Matched pairs")
        ax5.set_title("Centroid distance distribution"); ax5.legend(fontsize=8)

    # 6. Precision / Recall / F1 per frame
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.boxplot([df["precision"].dropna(), df["recall"].dropna(), df["f1"].dropna()],
                labels=["Precision", "Recall", "F1"], patch_artist=True,
                boxprops=dict(facecolor="#E3F2FD"))
    ax6.set_ylabel("Score"); ax6.set_title("Per-frame P / R / F1")
    ax6.set_ylim(0, 1.05)

    # 7. Per-frame mean IoU boxplot / violin
    ax7 = fig.add_subplot(gs[2, 0])
    frame_ious = df["mean_iou"].dropna()
    ax7.violinplot(frame_ious, positions=[1], showmedians=True)
    ax7.set_xticks([1]); ax7.set_xticklabels(["Mean IoU/frame"])
    ax7.set_ylabel("IoU"); ax7.set_title("Per-frame mean IoU")

    # 8. Width / Height bias scatter
    ax8 = fig.add_subplot(gs[2, 1])
    if len(w_arr):
        ax8.scatter(w_arr, h_arr, alpha=0.3, s=10, color="#FF5722")
        ax8.axhline(0, color="k", lw=0.8); ax8.axvline(0, color="k", lw=0.8)
        ax8.set_xlabel("Width diff (AI–manual)"); ax8.set_ylabel("Height diff (AI–manual)")
        ax8.set_title("Box size bias per matched pair")

    # 9. TP / FP / FN stacked bar (sampled frames or totals)
    ax9 = fig.add_subplot(gs[2, 2])
    totals = [df["tp"].sum(), df["fp"].sum(), df["fn"].sum()]
    colors = ["#4CAF50", "#F44336", "#FF9800"]
    bars = ax9.bar(["TP", "FP\n(AI extra)", "FN\n(AI missed)"],
                   totals, color=colors, edgecolor="white")
    for bar, val in zip(bars, totals):
        ax9.text(bar.get_x() + bar.get_width()/2, bar.get_height() + totals[0]*0.01,
                 str(int(val)), ha="center", fontsize=9)
    ax9.set_ylabel("Total boxes"); ax9.set_title(f"TP / FP / FN  (IoU≥{IOU_THRESHOLD})")

    fig.suptitle("AI vs Manual Annotation Comparison — Sperm Detection",
                 fontsize=14, fontweight="bold")
    plot_path = os.path.join(OUTPUT_DIR, "comparison_plots.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plots saved → {plot_path}")

    return df


if __name__ == "__main__":
    run_analysis()