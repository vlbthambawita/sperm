"""
╔══════════════════════════════════════════════════════════════════════╗
║  5-Fold Cross-Validation Comparison: AI vs Manual YOLO Annotations  ║
║                                                                      ║
║  Strategy A: Train on 4 folds AI-annotated  → Validate on 1 fold    ║
║              human-annotated  (same frame IDs)                       ║
║                                                                      ║
║  Strategy B: Train on 4 folds human-annotated → Validate on 1 fold  ║
║              human-annotated  (standard manual CV)                   ║
║                                                                      ║
║  Both strategies share the SAME image folds so results are           ║
║  directly comparable (folds use frames with BOTH AI & manual labels).║
╚══════════════════════════════════════════════════════════════════════╝

Folder layout expected:
    dataset/
    ├── images/          ← all .jpg/.png frames (shared)
    ├── ai_labels/       ← AI-generated YOLO .txt files
    └── manual_labels/   ← manually corrected YOLO .txt files

Output:
    cv_results/
    ├── fold_X/
    │   ├── strategy_A/  ← YOLO run dirs
    │   └── strategy_B/
    Script folder (main):
    ├── all_fold_metrics.csv
    ├── comparison_plots.png
    ├── results_report.md
    └── statistical_comparison.csv
"""

# ─────────────────────── USER CONFIG ────────────────────────────────
IMAGES_DIR    = "/work/vajira/DATA/SPERM_data_2025/clean_100_frames_yolo_ready/all/images"                # directory with image files
AI_LABELS_DIR = "/work/vajira/DATA/SPERM_data_2025/clean_100_frames_yolo_ready/all/labels"                # AI annotation .txt files
MAN_LABELS_DIR= "/work/vajira/DATA/SPERM_data_2025/manually_annotated_100_frames_v1.0/all/labels"         # manual annotation .txt files

N_FOLDS       = 5
NUM_CLASSES   = 1                        # number of object classes
CLASS_NAMES   = ["sperm"]               # list of class names
IMG_SIZE      = 640                      # YOLO input resolution
EPOCHS        = 50                       # training epochs per fold
BATCH_SIZE    = 16
YOLO_MODEL    = "yolo26n.pt"            # base weights (downloads auto)
                                         # change to yolov8n.pt / yolo11s.pt etc.
DEVICE        = "0"                      # "0" for GPU 0, "cpu" for CPU
WORKERS       = 4
OUTPUT_DIR    = "cv_results"
RANDOM_SEED   = 42
# ─────────────────────────────────────────────────────────────────────

import os, sys, shutil, yaml, random, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from sklearn.model_selection import KFold

try:
    from ultralytics import YOLO
except ImportError:
    sys.exit("❌  ultralytics not found. Run:  pip install ultralytics")


# ──────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def get_image_stems(images_dir):
    stems = [Path(p).stem for p in Path(images_dir).iterdir()
             if Path(p).suffix.lower() in IMAGE_EXTS]
    return sorted(stems)


def write_txt_list(path, items):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(items) + "\n")


def build_yaml(yaml_path, train_txt, val_txt, nc, names):
    """Use paths relative to the YAML file so Ultralytics resolves correctly."""
    base = Path(yaml_path).parent
    cfg = dict(
        train = str(Path(train_txt).relative_to(base)),
        val   = str(Path(val_txt).relative_to(base)),
        nc    = nc,
        names = names,
    )
    with open(yaml_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    return yaml_path


def symlink_labels(stems, src_label_dir, dst_label_dir, images_dir):
    """
    YOLO finds labels by replacing the images path segment with 'labels'.
    Easiest approach: create a flat labels folder symlinked/copied per split.
    We copy (small txt files) to avoid symlink permission issues.
    """
    dst = Path(dst_label_dir)
    dst.mkdir(parents=True, exist_ok=True)
    for stem in stems:
        src = Path(src_label_dir) / f"{stem}.txt"
        dst_file = dst / f"{stem}.txt"
        if src.exists():
            shutil.copy2(src, dst_file)
        else:
            # write empty file so YOLO doesn't complain about missing labels
            dst_file.touch()


def image_paths_for_stems(stems, images_dir):
    paths = []
    for stem in stems:
        for ext in IMAGE_EXTS:
            p = Path(images_dir) / f"{stem}{ext}"
            if p.exists():
                paths.append(str(p.resolve()))
                break
    return paths


def extract_metrics(results_dir):
    """Parse best metrics from ultralytics results.csv"""
    csv_path = Path(results_dir) / "results.csv"
    if not csv_path.exists():
        return {}
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    # take the best epoch row (highest mAP50)
    col_map = {c: c for c in df.columns}
    map50_col   = next((c for c in df.columns if "mAP50"  in c and "95" not in c), None)
    map5095_col = next((c for c in df.columns if "mAP50-95" in c or "mAP50.95" in c), None)
    prec_col    = next((c for c in df.columns if c.strip().lower() in ("metrics/precision(b)", "precision")), None)
    rec_col     = next((c for c in df.columns if c.strip().lower() in ("metrics/recall(b)", "recall")), None)

    if map50_col is None:
        return {}

    best_row = df.loc[df[map50_col].idxmax()]
    out = {
        "best_epoch"  : int(best_row.get("epoch", -1)),
        "precision"   : float(best_row[prec_col])    if prec_col    else np.nan,
        "recall"      : float(best_row[rec_col])     if rec_col     else np.nan,
        "mAP50"       : float(best_row[map50_col]),
        "mAP50_95"    : float(best_row[map5095_col]) if map5095_col else np.nan,
    }
    return out


# ──────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────

def run():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    out_root = Path(OUTPUT_DIR)
    out_root.mkdir(parents=True, exist_ok=True)
    summary_dir = Path(__file__).parent  # Summary outputs go in script folder

    # ── 1. Collect frame stems present in BOTH label sets ──
    all_stems = get_image_stems(IMAGES_DIR)
    ai_stems  = {Path(p).stem for p in Path(AI_LABELS_DIR).glob("*.txt")}
    man_stems = {Path(p).stem for p in Path(MAN_LABELS_DIR).glob("*.txt")}
    common    = sorted(set(all_stems) & ai_stems & man_stems)

    if len(common) < N_FOLDS:
        sys.exit(f"❌  Only {len(common)} frames found — need at least {N_FOLDS}.")

    print(f"✔  {len(common)} frames with both AI & manual annotations found.")
    print(f"   Running {N_FOLDS}-fold CV  ×  2 strategies  =  {N_FOLDS*2} YOLO runs\n")

    stems_arr = np.array(common)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    all_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(stems_arr)):
        fold_num    = fold_idx + 1
        train_stems = stems_arr[train_idx].tolist()
        val_stems   = stems_arr[val_idx].tolist()

        print(f"{'═'*60}")
        print(f"  FOLD {fold_num}/{N_FOLDS}  |  train={len(train_stems)}  val={len(val_stems)}")
        print(f"{'═'*60}")

        fold_dir = out_root / f"fold_{fold_num}"

        for strategy in ["A", "B"]:
            # Strategy A: train=AI labels,     val=manual labels
            # Strategy B: train=manual labels,  val=manual labels
            train_label_src = AI_LABELS_DIR  if strategy == "A" else MAN_LABELS_DIR
            val_label_src   = MAN_LABELS_DIR  # always manual for val

            strat_dir  = fold_dir / f"strategy_{strategy}"
            labels_dir = strat_dir / "labels"

            # Copy labels into split-specific folders
            train_lbl_dir = labels_dir / "train"
            val_lbl_dir   = labels_dir / "val"
            symlink_labels(train_stems, train_label_src, train_lbl_dir, IMAGES_DIR)
            symlink_labels(val_stems,   val_label_src,   val_lbl_dir,   IMAGES_DIR)

            # Write image path list files
            train_img_paths = image_paths_for_stems(train_stems, IMAGES_DIR)
            val_img_paths   = image_paths_for_stems(val_stems,   IMAGES_DIR)
            train_txt = strat_dir / "train.txt"
            val_txt   = strat_dir / "val.txt"
            write_txt_list(train_txt, train_img_paths)
            write_txt_list(val_txt,   val_img_paths)

            # Patch image paths so YOLO finds the right labels sub-folder
            # YOLO replaces "images" with "labels" in path — so we point images
            # to a mirrored structure under strat_dir/images/
            train_img_link_dir = strat_dir / "images" / "train"
            val_img_link_dir   = strat_dir / "images" / "val"
            train_img_link_dir.mkdir(parents=True, exist_ok=True)
            val_img_link_dir.mkdir(parents=True, exist_ok=True)

            def link_images(stems, dst_dir):
                img_list = []
                for stem in stems:
                    for ext in IMAGE_EXTS:
                        src = Path(IMAGES_DIR) / f"{stem}{ext}"
                        if src.exists():
                            dst = dst_dir / f"{stem}{ext}"
                            if not dst.exists():
                                shutil.copy2(src, dst)   # copy (use symlink if space is concern)
                            img_list.append(str(dst.resolve()))
                            break
                return img_list

            tr_imgs = link_images(train_stems, train_img_link_dir)
            vl_imgs = link_images(val_stems,   val_img_link_dir)

            write_txt_list(train_txt, tr_imgs)
            write_txt_list(val_txt,   vl_imgs)

            # Build dataset YAML
            yaml_path = strat_dir / "dataset.yaml"
            build_yaml(yaml_path, train_txt, val_txt, NUM_CLASSES, CLASS_NAMES)

            # ── Train ──
            run_name = f"fold{fold_num}_strategy{strategy}"
            print(f"\n  ▶  Strategy {strategy}  ({'AI train → Manual val' if strategy=='A' else 'Manual train → Manual val'})")
            t0 = time.time()

            model = YOLO(YOLO_MODEL)
            model.train(
                data    = str(yaml_path),
                epochs  = EPOCHS,
                imgsz   = IMG_SIZE,
                batch   = BATCH_SIZE,
                device  = DEVICE,
                workers = WORKERS,
                project = str(strat_dir / "runs"),
                name    = run_name,
                exist_ok= True,
                verbose = False,
                seed    = RANDOM_SEED,
            )
            elapsed = time.time() - t0

            # ── Extract metrics ──
            # With project=strat_dir/'runs', Ultralytics saves under strat_dir/'runs'/run_name
            run_dir = strat_dir / "runs" / run_name
            metrics = extract_metrics(run_dir)
            default_metrics = {"precision": np.nan, "recall": np.nan, "mAP50": np.nan, "mAP50_95": np.nan, "best_epoch": -1}

            row = dict(
                fold       = fold_num,
                strategy   = strategy,
                train_src  = "AI"     if strategy == "A" else "Manual",
                val_src    = "Manual",
                n_train    = len(train_stems),
                n_val      = len(val_stems),
                train_time_s = round(elapsed, 1),
                **{**default_metrics, **metrics},
            )
            all_results.append(row)
            _f = lambda v: f"{v:.3f}" if isinstance(v, (int, float)) else "N/A"
            print(f"     mAP@50={_f(metrics.get('mAP50'))}  "
                  f"P={_f(metrics.get('precision'))}  "
                  f"R={_f(metrics.get('recall'))}  "
                  f"({elapsed/60:.1f} min)")

    # ── Aggregate & save ──────────────────────
    df = pd.DataFrame(all_results)
    csv_path = summary_dir / "all_fold_metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n✔  All metrics saved → {csv_path}")

    # ── Summary table ──
    summary = df.groupby("strategy")[["precision","recall","mAP50","mAP50_95"]].agg(["mean","std"])
    print("\n" + "="*65)
    print("  CROSS-VALIDATION SUMMARY")
    print("="*65)
    print(summary.round(4).to_string())
    print("\nStrategy A = Train on AI annotations, validate on manual")
    print("Strategy B = Train on manual annotations, validate on manual")

    # ── Statistical comparison: B vs A (paired across folds) ──
    from scipy.stats import wilcoxon
    metrics_list = ["mAP50", "mAP50_95", "precision", "recall"]
    metric_labels = {"mAP50": "mAP@50", "mAP50_95": "mAP@50:95", "precision": "Precision", "recall": "Recall"}

    stats_rows = []
    for metric in metrics_list:
        a_vals = df[df.strategy == "A"][metric].values
        b_vals = df[df.strategy == "B"][metric].values
        valid = ~(np.isnan(a_vals) | np.isnan(b_vals))
        a_vals, b_vals = a_vals[valid], b_vals[valid]

        mean_a = float(np.mean(a_vals)) if len(a_vals) else np.nan
        mean_b = float(np.mean(b_vals)) if len(b_vals) else np.nan
        mean_diff = float(np.mean(b_vals - a_vals)) if len(a_vals) else np.nan
        std_diff = float(np.std(b_vals - a_vals, ddof=1)) if len(a_vals) >= 2 else np.nan

        if len(a_vals) >= 2:
            t_stat, t_p = stats_ttest_paired(a_vals, b_vals)
            ci_lo, ci_hi = paired_ci(a_vals, b_vals)
            cohen_d = cohens_d_paired(a_vals, b_vals)
            try:
                w_stat, w_p = wilcoxon(a_vals, b_vals)
            except Exception:
                w_stat, w_p = np.nan, np.nan
        else:
            t_stat, t_p = np.nan, np.nan
            ci_lo, ci_hi = np.nan, np.nan
            cohen_d = np.nan
            w_stat, w_p = np.nan, np.nan

        stats_rows.append(dict(
            metric=metric,
            mean_A=mean_a,
            mean_B=mean_b,
            mean_diff_B_minus_A=mean_diff,
            std_diff=std_diff,
            CI_lower=ci_lo,
            CI_upper=ci_hi,
            t_stat=t_stat,
            t_p_value=t_p,
            cohen_d=cohen_d,
            wilcoxon_stat=w_stat,
            wilcoxon_p=w_p,
        ))

    stats_df = pd.DataFrame(stats_rows)
    stats_csv = summary_dir / "statistical_comparison.csv"
    stats_df.to_csv(stats_csv, index=False)
    print(f"\n✔  Statistical comparison saved → {stats_csv}")

    # Console output
    print("\n" + "=" * 75)
    print(f"  STATISTICAL COMPARISON: Strategy B vs A ({N_FOLDS}-fold paired)")
    print("=" * 75)
    for _, r in stats_df.iterrows():
        ci_str = f"[{r['CI_lower']:.3f}, {r['CI_upper']:.3f}]" if not np.isnan(r['CI_lower']) else "N/A"
        sig = "*" if r["t_p_value"] < 0.05 else ""
        w_p_str = f"{r['wilcoxon_p']:.4f}{'*' if r['wilcoxon_p'] < 0.05 else ''}" if not np.isnan(r['wilcoxon_p']) else "N/A"
        mlabel = metric_labels.get(str(r["metric"]), str(r["metric"]))
        print(f"  {mlabel:12} | A={r['mean_A']:.3f} | B={r['mean_B']:.3f} | "
              f"B-A={r['mean_diff_B_minus_A']:+.3f} {ci_str} | t p={r['t_p_value']:.4f}{sig} | "
              f"d={r['cohen_d']:.2f} | Wilcoxon p={w_p_str}")
    print("  * p < 0.05")

    # ── Markdown report ──
    write_markdown_report(df, stats_df, summary_dir, N_FOLDS)

    # ── Plots ─────────────────────────────────
    plot_results(df, summary_dir)


def stats_ttest_paired(a, b):
    from scipy import stats
    result = stats.ttest_rel(a, b)
    return result.statistic, result.pvalue


def cohens_d_paired(a, b):
    """Cohen's d for paired samples: d = mean(diff) / std(diff)."""
    from scipy import stats
    diff = np.asarray(b) - np.asarray(a)
    if len(diff) < 2 or diff.std() == 0:
        return np.nan
    return float(np.mean(diff) / diff.std(ddof=1))


def paired_ci(a, b, alpha=0.05):
    """95% confidence interval for mean(B - A). Returns (lower, upper)."""
    from scipy import stats
    diff = np.asarray(b) - np.asarray(a)
    n = len(diff)
    if n < 2:
        return np.nan, np.nan
    mean_diff = np.mean(diff)
    sem = diff.std(ddof=1) / np.sqrt(n)
    t_crit = stats.t.ppf(1 - alpha / 2, df=n - 1)
    margin = t_crit * sem
    return float(mean_diff - margin), float(mean_diff + margin)


def write_markdown_report(df, stats_df, summary_dir, n_folds):
    """Write a Markdown report to results_report.md in the given directory (script folder)."""
    md_path = summary_dir / "results_report.md"
    metric_labels = {"mAP50": "mAP@50", "mAP50_95": "mAP@50:95", "precision": "Precision", "recall": "Recall"}

    lines = [
        "# 5-Fold Cross-Validation: Strategy A vs Strategy B",
        "",
        "**Strategy A**: Train on AI annotations, validate on manual",
        "**Strategy B**: Train on manual annotations, validate on manual",
        "",
        "Both strategies share the same fold splits (paired comparison).",
        "",
        "---",
        "",
        "## Cross-Validation Summary",
        "",
    ]

    # Summary table (mean ± std per strategy)
    summary = df.groupby("strategy")[["precision", "recall", "mAP50", "mAP50_95"]].agg(["mean", "std"])
    cols = ["precision", "recall", "mAP50", "mAP50_95"]
    header = "| Metric | Strategy A | Strategy B |"
    sep = "|--------|------------|------------|"
    lines.append(header)
    lines.append(sep)
    for c in cols:
        a_mean = summary.loc["A", (c, "mean")] if "A" in summary.index else np.nan
        a_std = summary.loc["A", (c, "std")] if "A" in summary.index else np.nan
        b_mean = summary.loc["B", (c, "mean")] if "B" in summary.index else np.nan
        b_std = summary.loc["B", (c, "std")] if "B" in summary.index else np.nan
        lbl = metric_labels.get(c, c)
        a_str = f"{a_mean:.4f} ± {a_std:.4f}" if not np.isnan(a_mean) else "N/A"
        b_str = f"{b_mean:.4f} ± {b_std:.4f}" if not np.isnan(b_mean) else "N/A"
        lines.append(f"| {lbl} | {a_str} | {b_str} |")
    lines.extend(["", "---", "", "## Statistical Comparison: B vs A", ""])

    # Statistical comparison table
    lines.append("| Metric | mean_A | mean_B | B-A (95% CI) | t-test p | Cohen's d | Wilcoxon p |")
    lines.append("|--------|--------|--------|--------------|----------|-----------|-------------|")
    for _, r in stats_df.iterrows():
        lbl = metric_labels.get(r["metric"], r["metric"])
        ci_str = f"[{r['CI_lower']:.3f}, {r['CI_upper']:.3f}]" if not np.isnan(r["CI_lower"]) else "N/A"
        t_p = f"{r['t_p_value']:.4f}" + ("*" if r["t_p_value"] < 0.05 else "")
        d_str = f"{r['cohen_d']:.3f}" if not np.isnan(r["cohen_d"]) else "N/A"
        w_p = f"{r['wilcoxon_p']:.4f}" + ("*" if r["wilcoxon_p"] < 0.05 else "") if not np.isnan(r["wilcoxon_p"]) else "N/A"
        lines.append(f"| {lbl} | {r['mean_A']:.4f} | {r['mean_B']:.4f} | {r['mean_diff_B_minus_A']:+.4f} ({ci_str}) | {t_p} | {d_str} | {w_p} |")
    lines.extend(["", "* p < 0.05", "", "---", "", "## Interpretation", ""])

    for _, r in stats_df.iterrows():
        lbl = metric_labels.get(r["metric"], r["metric"])
        sig = "significant" if r["t_p_value"] < 0.05 else "not significant"
        direction = "improves" if r["mean_diff_B_minus_A"] > 0 else "decreases"
        ci_str = f"[{r['CI_lower']:.3f}, {r['CI_upper']:.3f}]" if not np.isnan(r["CI_lower"]) else "N/A"
        diff_str = f"{r['mean_diff_B_minus_A']:+.4f}" if not np.isnan(r["mean_diff_B_minus_A"]) else "N/A"
        p_str = f"{r['t_p_value']:.4f}" if not np.isnan(r["t_p_value"]) else "N/A"
        lines.append(f"- **{lbl}**: Strategy B {direction} over A by {diff_str} (95% CI {ci_str}) — {sig} (t-test p={p_str})")

    lines.extend(["", "---", "", "## Per-Fold Metrics", ""])
    sub = df[["fold", "strategy", "mAP50", "mAP50_95", "precision", "recall", "train_time_s"]]
    header = "| fold | strategy | mAP50 | mAP50_95 | precision | recall | train_time_s |"
    sep = "|------|----------|-------|----------|-----------|--------|--------------|"
    lines.append(header)
    lines.append(sep)
    def _fmt(v):
        return f"{v:.4f}" if isinstance(v, (int, float)) and not np.isnan(v) else "N/A"

    for _, row in sub.iterrows():
        lines.append(f"| {int(row['fold'])} | {row['strategy']} | {_fmt(row['mAP50'])} | {_fmt(row['mAP50_95'])} | {_fmt(row['precision'])} | {_fmt(row['recall'])} | {row['train_time_s']:.1f} |")

    with open(md_path, "w") as f:
        f.write("\n".join(lines))
    print(f"✔  Markdown report saved → {md_path}")


def plot_results(df, summary_dir):
    from scipy import stats as sp_stats

    metrics     = ["mAP50", "mAP50_95", "precision", "recall"]
    metric_lbls = ["mAP@50", "mAP@50:95", "Precision", "Recall"]
    colors      = {"A": "#2196F3", "B": "#FF5722"}
    labels      = {"A": "AI train → Manual val", "B": "Manual train → Manual val"}

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    # ── Per-metric: fold-by-fold line plot ──
    for ax_i, (met, lbl) in enumerate(zip(metrics[:4], metric_lbls[:4])):
        ax = axes[ax_i]
        for strat in ["A", "B"]:
            sub = df[df.strategy == strat].sort_values("fold")
            ax.plot(sub["fold"], sub[met], marker="o",
                    color=colors[strat], label=labels[strat], lw=2)
            ax.fill_between(sub["fold"],
                            sub[met] - sub[met].std(),
                            sub[met] + sub[met].std(),
                            alpha=0.12, color=colors[strat])
        ax.set_xlabel("Fold"); ax.set_ylabel(lbl)
        ax.set_title(f"{lbl} per fold")
        ax.set_xticks(df["fold"].unique())
        ax.legend(fontsize=7)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)

    # ── Box-plot comparison (all folds) ──
    ax5 = axes[4]
    data_a = [df[df.strategy=="A"][m].values for m in metrics]
    data_b = [df[df.strategy=="B"][m].values for m in metrics]
    x = np.arange(len(metrics))
    width = 0.3
    for i, (da, db, lbl) in enumerate(zip(data_a, data_b, metric_lbls)):
        bp_a = ax5.boxplot(da, positions=[x[i]-width/2], widths=width*0.9,
                           patch_artist=True,
                           boxprops=dict(facecolor=colors["A"], alpha=0.7),
                           medianprops=dict(color="white", lw=2),
                           whiskerprops=dict(color=colors["A"]),
                           capprops=dict(color=colors["A"]),
                           flierprops=dict(marker="o", markerfacecolor=colors["A"], markersize=4))
        bp_b = ax5.boxplot(db, positions=[x[i]+width/2], widths=width*0.9,
                           patch_artist=True,
                           boxprops=dict(facecolor=colors["B"], alpha=0.7),
                           medianprops=dict(color="white", lw=2),
                           whiskerprops=dict(color=colors["B"]),
                           capprops=dict(color=colors["B"]),
                           flierprops=dict(marker="o", markerfacecolor=colors["B"], markersize=4))
    ax5.set_xticks(x); ax5.set_xticklabels(metric_lbls, fontsize=9)
    ax5.set_ylabel("Score"); ax5.set_title("Strategy comparison (all folds)")
    ax5.set_ylim(0, 1.05); ax5.grid(axis="y", alpha=0.3)
    from matplotlib.patches import Patch
    ax5.legend(handles=[Patch(facecolor=colors["A"], label=labels["A"]),
                         Patch(facecolor=colors["B"], label=labels["B"])], fontsize=7)

    # ── Radar / spider chart ──
    ax6 = axes[5]
    ax6.remove()
    ax6 = fig.add_subplot(2, 3, 6, polar=True)
    met_radar = ["mAP50", "mAP50_95", "precision", "recall"]
    lbl_radar = ["mAP@50", "mAP@50:95", "Precision", "Recall"]
    N = len(met_radar)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    for strat in ["A", "B"]:
        vals = [df[df.strategy==strat][m].mean() for m in met_radar]
        vals += vals[:1]
        ax6.plot(angles, vals, color=colors[strat], lw=2, label=labels[strat])
        ax6.fill(angles, vals, color=colors[strat], alpha=0.15)
    ax6.set_xticks(angles[:-1]); ax6.set_xticklabels(lbl_radar, fontsize=8)
    ax6.set_ylim(0, 1); ax6.set_title("Mean metric radar", pad=15)
    ax6.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=7)

    fig.suptitle(
        f"5-Fold CV: AI-annotated training vs Manual-annotated training\n"
        f"(Validation always on manual annotations)",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    plot_path = summary_dir / "comparison_plots.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✔  Plots saved → {plot_path}")


if __name__ == "__main__":
    run()