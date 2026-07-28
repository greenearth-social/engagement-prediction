# Old-data manuscript mysteries — snapshot 2026-07-27

This is a read-only preservation of the open questions in the manuscript
replication package at the time the frozen-Stage-1 MLP matrix was completed.
It is deliberately stored with the old-data analysis tooling so later
manuscript or Reveal edits cannot erase the questions this rerun is meant to
close. It is not a source for live manuscript or presentation edits.

## Evidence snapshot

- Manuscript provenance: `power_likers_paper/replication/PROVENANCE.md`
  §8–10 (read 2026-07-27).
- Paper result wording: `power_likers_paper/paper/sections/part2.tex`
  R1, R3, and F1 subsections (read 2026-07-27).
- Matrix outputs:
  `~/power-likers/full_matrix/harvest_defer_f1_20260727T170542Z/reports/`.

## Mystery register

| ID | Historical manuscript issue | Old-data closure target | Current status |
|---|---|---|---|
| M1 | Cross-trait correspondence was ρ=0.46 in draft history versus QMD ρ=0.442. | This matrix does not re-estimate correspondence; retain as a frozen-artifact provenance discrepancy. | Not addressed by this matrix. |
| M2 | Negative-emotion prose said 2–6 pp while the QMD range was 0.1–0.6 pp. | Repaired D1 can assess the old-data typical-user axis, but it cannot retroactively validate rhetoric based on different estimands. | Awaiting all-cell finite D1 rebuild. |
| M3 | Representation-gap R²=0.849 is arithmetically reproduced, but its causal/diagnostic validity was questioned. | Keep as an interpretation question, not an estimator bug. | Not addressed by remedy rerun. |
| M4 | R1 cap=5 ΔAUC had three incompatible values: −0.003 placeholder, −0.0833 QMD/native result, and the later paired value. | Establish a single fixed-cohort utility estimate. | Resolved for old data: −0.02749, 95% seed-pair CI [−0.02793, −0.02696]. Earlier values are not interchangeable estimands. |
| M5 | IPW was framed as ineffective / a falsification of gradient mass, despite IPW(1/n) reducing NegSent more than R2 in one historical table. | Compare every remedy's bias on one fixed cohort and pool, alongside paired utility. | Utility resolved; fixed-cohort bias matrix queued. Do not call IPW ineffective yet. |
| M6 | F1 was called inert and used to locate bias in Stage-3 histories, but the two-tower result was less clearly inert. | Re-run F1 and two-tower only if needed after primary MLP fixed-cohort frontier. | Explicitly deferred; no old-data closure yet. |
| M7 | Appendix D1 (typical-user-only over-serving) was a stub; D2 negative controls had never run. | Produce finite D1/D2 per cell and then pair user-level outcomes across baseline/remedy. | D1 NaN source fixed; 50-cell native rebuild in progress. D2 was finite before and is being rebuilt. |
| M8 | Methods needed a raw Lorenz/concentration figure and uncapped all-user counts. | Determine whether frozen Stage 1 can support it. | Cannot be recovered honestly: old Stage 1 lacks the pre-cap population vector. Fresh-Q3 is authoritative. |
| M9 | P1–P9 attrition facts were absent from the old portable substrate. | Locate immutable Stage-1 summary before emitting ledger. | Cannot be recovered honestly: old directory has only core data files, no summary. Harvest now fails loudly rather than emitting a phantom ledger. |
| M10 | Abstract/frontier wording claimed R2-drop30 +0.0005 AUC / “no cost.” | Use matched fixed-cohort AUC with 0.005 non-inferiority margin. | Resolved for old data: −0.00253, CI [−0.00296, −0.00215], formally flat but not an improvement. |

## Old-data utility facts safe to retain

These are MLP-only, five matched seeds, and typical-user outcomes scored on the
baseline-defined holdout. They are useful for explaining the old-data
investigation, not for replacing fresh-data figures.

| Remedy | Mean ΔAUC | Verdict (margin 0.005) |
|---|---:|---|
| R2 drop 10% | +0.00066 | flat |
| R2 drop 20% | +0.00014 | flat |
| R2 drop 30% | −0.00253 | flat |
| R2 drop 40% | −0.00671 | utility cost exceeds margin |
| R1 cap 5 | −0.02749 | utility cost exceeds margin |
| R1 cap 10 | −0.01361 | utility cost exceeds margin |
| R3 inverse-log | +0.00004 | flat |
| R3 inverse-sqrt | −0.00117 | flat |
| R3 inverse | −0.00970 | utility cost exceeds margin |

## Guardrails

- Do not write these values to the manuscript, `values.tex`, paper figures, or
  Reveal slides.
- Do not treat native-holdout D1 means as a cross-remedy frontier. The fixed
  cohort bias runner must finish first.
- Do not use F1 or two-tower mechanism language as though it was rerun here.
- Do not use old-data concentration or attrition gaps as a reason to block the
  fresh-Q3 analysis; those artifacts are structurally unavailable in this
  frozen substrate.
