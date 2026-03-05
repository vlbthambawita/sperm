# 5-Fold Cross-Validation: Strategy A vs Strategy B

**Strategy A**: Train on AI annotations, validate on manual
**Strategy B**: Train on manual annotations, validate on manual

Both strategies share the same fold splits (paired comparison).

---

## Cross-Validation Summary

| Metric | Strategy A | Strategy B |
|--------|------------|------------|
| Precision | 0.7981 ± 0.0128 | 0.7941 ± 0.0126 |
| Recall | 0.5043 ± 0.0430 | 0.5049 ± 0.0432 |
| mAP@50 | 0.6243 ± 0.0220 | 0.6281 ± 0.0244 |
| mAP@50:95 | 0.2990 ± 0.0088 | 0.2860 ± 0.0145 |

---

## Statistical Comparison: B vs A

| Metric | mean_A | mean_B | B-A (95% CI) | t-test p | Cohen's d | Wilcoxon p |
|--------|--------|--------|--------------|----------|-----------|-------------|
| mAP@50 | 0.6243 | 0.6281 | +0.0038 ([-0.001, 0.009]) | 0.0989 | 0.9579 | 0.1875 |
| mAP@50:95 | 0.2990 | 0.2860 | -0.0130 ([-0.023, -0.003]) | 0.0233* | -1.5988 | 0.0625 |
| Precision | 0.7981 | 0.7941 | -0.0040 ([-0.010, 0.002]) | 0.1620 | -0.7658 | 0.1875 |
| Recall | 0.5043 | 0.5049 | +0.0006 ([-0.000, 0.002]) | 0.1246 | 0.8670 | 0.2500 |

* p < 0.05

---

## Interpretation

- **mAP@50**: Strategy B improves over A by +0.0038 (95% CI [-0.001, 0.009]) — not significant (t-test p=0.0989)
- **mAP@50:95**: Strategy B decreases over A by -0.0130 (95% CI [-0.023, -0.003]) — significant (t-test p=0.0233)
- **Precision**: Strategy B decreases over A by -0.0040 (95% CI [-0.010, 0.002]) — not significant (t-test p=0.1620)
- **Recall**: Strategy B improves over A by +0.0006 (95% CI [-0.000, 0.002]) — not significant (t-test p=0.1246)

---

## Per-Fold Metrics

| fold | strategy | mAP50 | mAP50_95 | precision | recall | train_time_s |
|------|----------|-------|----------|-----------|--------|--------------|
| 1 | A | 0.6238 | 0.3069 | 0.7859 | 0.5309 | 140.5 |
| 1 | B | 0.6322 | 0.2999 | 0.7883 | 0.5309 | 130.4 |
| 2 | A | 0.6331 | 0.2955 | 0.7945 | 0.5210 | 127.4 |
| 2 | B | 0.6392 | 0.2888 | 0.7829 | 0.5224 | 133.0 |
| 3 | A | 0.6112 | 0.2874 | 0.7892 | 0.4765 | 129.4 |
| 3 | B | 0.6101 | 0.2710 | 0.7873 | 0.4778 | 131.1 |
| 4 | A | 0.6557 | 0.3088 | 0.8034 | 0.5491 | 128.1 |
| 4 | B | 0.6604 | 0.2995 | 0.7975 | 0.5498 | 131.1 |
| 5 | A | 0.5978 | 0.2966 | 0.8176 | 0.4441 | 126.4 |
| 5 | B | 0.5984 | 0.2708 | 0.8145 | 0.4439 | 126.8 |