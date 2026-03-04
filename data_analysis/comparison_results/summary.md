# YOLO Annotation Comparison Summary

## Overview

| Metric | Value |
|--------|-------|
| Frames analysed | 100 |
| Total matched pairs (TP) | 34508 |
| Total false positives (AI only) | 5918 |
| Total false negatives (missed) | 6564 |

## Count agreement

| Metric | Value |
|--------|-------|
| Mean AI count/frame | 404.26 ± 281.64 |
| Mean manual count/frame | 410.72 ± 264.82 |
| Mean count difference (AI - manual) | -6.46 ± 37.35 |
| ICC (count) | 1.000 |
| Wilcoxon p-value (counts) | 0.0048 |

## Detection quality (IoU ≥ 0.5)

| Metric | Value |
|--------|-------|
| Mean Precision | 0.864 ± 0.057 |
| Mean Recall | 0.817 ± 0.086 |
| Mean F1 | 0.837 ± 0.061 |

## IoU of matched pairs

| Metric | Value |
|--------|-------|
| Mean IoU | 0.974 |
| Median IoU | 1.000 |
| Std IoU | 0.093 |
| % IoU ≥ 0.75 | 94.4% |

## Centroid distance (pixels)

| Metric | Value |
|--------|-------|
| Mean | 0.17 px |
| Median | 0.00 px |

## Box size differences (AI - manual, normalised)

| Metric | Value |
|--------|-------|
| Width bias | -0.0001 (p=0.0000) |
| Height bias | -0.0001 (p=0.0000) |

## Outputs

- [per_frame_stats.csv](per_frame_stats.csv)
- [comparison_plots.png](comparison_plots.png)
