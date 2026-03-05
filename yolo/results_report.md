# 5-Fold Cross-Validation: Strategy A vs Strategy B

**Strategy A**: Train on AI annotations, validate on manual
**Strategy B**: Train on manual annotations, validate on manual

Both strategies share the same fold splits (paired comparison).

---

## Cross-Validation Summary

| Metric | Strategy A | Strategy B |
|--------|------------|------------|
| Precision | N/A | N/A |
| Recall | N/A | N/A |
| mAP@50 | N/A | N/A |
| mAP@50:95 | N/A | N/A |

---

## Statistical Comparison: B vs A

| Metric | mean_A | mean_B | B-A (95% CI) | t-test p | Cohen's d | Wilcoxon p |
|--------|--------|--------|--------------|----------|-----------|-------------|
| mAP@50 | nan | nan | +nan (N/A) | nan | N/A | N/A |
| mAP@50:95 | nan | nan | +nan (N/A) | nan | N/A | N/A |
| Precision | nan | nan | +nan (N/A) | nan | N/A | N/A |
| Recall | nan | nan | +nan (N/A) | nan | N/A | N/A |

* p < 0.05

---

## Interpretation

- **mAP@50**: Strategy B decreases over A by N/A (95% CI N/A) — not significant (t-test p=N/A)
- **mAP@50:95**: Strategy B decreases over A by N/A (95% CI N/A) — not significant (t-test p=N/A)
- **Precision**: Strategy B decreases over A by N/A (95% CI N/A) — not significant (t-test p=N/A)
- **Recall**: Strategy B decreases over A by N/A (95% CI N/A) — not significant (t-test p=N/A)

---

## Per-Fold Metrics

| fold | strategy | mAP50 | mAP50_95 | precision | recall | train_time_s |
|------|----------|-------|----------|-----------|--------|--------------|
| 1 | A | N/A | N/A | N/A | N/A | 143.4 |
| 1 | B | N/A | N/A | N/A | N/A | 146.4 |
| 2 | A | N/A | N/A | N/A | N/A | 189.8 |
| 2 | B | N/A | N/A | N/A | N/A | 136.8 |
| 3 | A | N/A | N/A | N/A | N/A | 181.3 |
| 3 | B | N/A | N/A | N/A | N/A | 165.0 |
| 4 | A | N/A | N/A | N/A | N/A | 211.0 |
| 4 | B | N/A | N/A | N/A | N/A | 200.4 |
| 5 | A | N/A | N/A | N/A | N/A | 157.0 |
| 5 | B | N/A | N/A | N/A | N/A | 162.5 |