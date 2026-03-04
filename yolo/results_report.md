# 5-Fold Cross-Validation: Strategy A vs Strategy B

**Strategy A**: Train on AI annotations, validate on manual
**Strategy B**: Train on manual annotations, validate on manual

Both strategies share the same fold splits (paired comparison).

---

## Cross-Validation Summary

| Metric | Strategy A | Strategy B |
|--------|------------|------------|
| Precision | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| Recall | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| mAP@50 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| mAP@50:95 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |

---

## Statistical Comparison: B vs A

| Metric | mean_A | mean_B | B-A (95% CI) | t-test p | Cohen's d | Wilcoxon p |
|--------|--------|--------|--------------|----------|-----------|-------------|
| mAP@50 | 0.0000 | 0.0000 | +0.0000 ([0.000, 0.000]) | nan | N/A | 1.0000 |
| mAP@50:95 | 0.0000 | 0.0000 | +0.0000 ([0.000, 0.000]) | nan | N/A | 1.0000 |
| Precision | 0.0000 | 0.0000 | +0.0000 ([0.000, 0.000]) | nan | N/A | 1.0000 |
| Recall | 0.0000 | 0.0000 | +0.0000 ([0.000, 0.000]) | nan | N/A | 1.0000 |

* p < 0.05

---

## Interpretation

- **mAP@50**: Strategy B decreases over A by +0.0000 (95% CI [0.000, 0.000]) — not significant (t-test p=N/A)
- **mAP@50:95**: Strategy B decreases over A by +0.0000 (95% CI [0.000, 0.000]) — not significant (t-test p=N/A)
- **Precision**: Strategy B decreases over A by +0.0000 (95% CI [0.000, 0.000]) — not significant (t-test p=N/A)
- **Recall**: Strategy B decreases over A by +0.0000 (95% CI [0.000, 0.000]) — not significant (t-test p=N/A)

---

## Per-Fold Metrics

| fold | strategy | mAP50 | mAP50_95 | precision | recall | train_time_s |
|------|----------|-------|----------|-----------|--------|--------------|
| 1 | A | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 30.0 |
| 1 | B | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 39.8 |
| 2 | A | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 27.0 |
| 2 | B | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 85.4 |
| 3 | A | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 51.7 |
| 3 | B | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 65.5 |
| 4 | A | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 65.9 |
| 4 | B | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 34.0 |
| 5 | A | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 54.3 |
| 5 | B | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 68.4 |