# Details

The full technical write-up behind the README: what was measured, how, what
each figure shows, and — just as importantly — what is *not* established.

---

## 1. The dataset and the split

`ogbg-molhiv`, from the Open Graph Benchmark. Originally the NCI AIDS Antiviral
Screen, routed through MoleculeNet and repackaged by OGB.

```
41,127 molecules total
32,901 train / 4,113 validation / 4,113 test   (official scaffold split)
 1,443 actives overall (3.509%)
   130 actives in the test split (3.161%)
 2 / 25.5 / 222 atoms per molecule (min / mean / max), max node degree 9
```

The split ships with the dataset and is read with `dataset.get_idx_split()`.
No splitting logic in this repo touches it.

**Why the scaffold split matters.** Molecules are grouped by their Bemis–Murcko
scaffold — the core ring skeleton with side chains stripped — and whole
scaffold groups are assigned to a single split. A test molecule therefore has a
skeleton no training molecule shares. A random split would scatter close
analogues across both sides, and every score in this repo would be several
points higher and mean considerably less.

Note that the actives are **not** spread evenly: 3.509% overall but 3.161% in
test. That is a consequence of splitting by scaffold rather than at random, and
it is why the test positive count (130) has to be read off the data rather than
estimated from the headline rate.

---

## 2. The three arms, technically

**Arm 1 — Morgan fingerprints + Random Forest.** Radius-2 Morgan fingerprints,
2048 bits, via `rdFingerprintGenerator.GetMorganGenerator`. Downstream: 500
trees, `class_weight="balanced"`. No learned representation — the features are a
fixed, deterministic function of the molecule.

**Arm 2 — WEGL.** Reimplemented from Kolouri et al. (ICLR 2021), not copied
from the authors' code. Each atom's one-hot features are diffused over the bond
graph with a symmetrically normalised adjacency for 3 steps, and the
concatenation `[X, AX, A²X, A³X]` (174 × 4 = 696 dims) describes each atom. The
molecule is then a *point cloud* of atom vectors, embedded into a fixed-length
vector by linear optimal transport against one shared reference cloud.

Two deliberate deviations from the paper, both for tractability:
- PCA to 32 dimensions **before** the transport step. Without it, a reference of
  size R gives a 696·R-dimensional embedding, which does not fit for 41k
  molecules.
- Exact EMD (`ot.emd`) with uniform weights rather than the paper's sliced
  approximation. Molecules average 25.5 atoms, so exact is affordable.

Final embedding is `reference_size × pca_dim` = 30 × 32 = **960** dims.

**Arm 3 — PNA.** `PNAConv` from PyTorch Geometric with OGB's atom/bond
encoders, 4 layers, hidden 80, dropout 0.3, residual connections, mean pooling,
aggregators `mean/max/min/std`, scalers `identity/amplification/attenuation` —
the PNA authors' published molhiv configuration. 715,441 parameters. Adam,
lr 1e-3, weight decay 3e-6, batch 128, 50 epochs. About 700 s per seed on the
NVIDIA L4 the Colab session allocated.

---

## 3. Design choices that keep the comparison honest

Each of these is a place where a benchmark can quietly flatter itself.

**Test score is read at the best validation epoch, never the best test epoch.**
Following the OGB reference implementation. Choosing the epoch by test score
would be reporting the best of 50 draws.

**PNA's degree histogram comes from the training split only.** PNA's degree
scalers need the degree distribution. Computing it over the full dataset would
leak test-set structure into the model's normalisation — a subtle leak that
would never show up as an error.

**WEGL's PCA and reference cloud are fitted on training molecules only**, then
applied unchanged to validation and test.

**Every arm is scored on exactly the same 4,113 molecules.** RDKit cannot parse
7 of molhiv's SMILES (indices 137, 987, 12882, 18293, 30784, 30785, 35728 — 3 in
train, 2 in validation, 2 in test, all inactive). The fingerprint arm drops them
from *training* only, where 2048 zero bits against a real label is pure noise;
the 2 test molecules are **kept**, with an all-zero fingerprint. Dropping them
from test instead would have scored arm 1 on 4,111 molecules against the others'
4,113. `src/report.py` checks each arm's test size and positive count and
refuses to present the table as comparable if they differ.

**Model selection uses the same molecules wherever there is anything to
select.** PNA takes its best validation epoch on the official scaffold
validation split. The random-forest arms have nothing to select — a forest does
not overfit as trees are added, so 500 trees is a fixed setting, not a tuned
one. The boosted-tree variant *does* need an iteration count, and it picks one
on that same official validation split. Notably it does **not** use
scikit-learn's default `early_stopping="auto"`, which above 10k rows holds out a
random 10% of training rows — a slice that shares scaffolds with the rows it was
cut from, and which at 3% positives may contain almost no actives. Measured, that
default stops after about 35 of 300 iterations.

**A seed does not mean the same thing in each arm.** Morgan fingerprints are
deterministic, so arm 1's seed varies only the forest. Arm 2's seed varies the
PCA subsample, the random reference cloud *and* the classifier, because the
embedding is rebuilt per seed rather than frozen. Arm 3's seed varies the whole
network initialisation and the batch order. Arm 3's spread therefore covers the
most sources of randomness, which is part of why it is the largest.

---

## 4. Two uncertainties, and why both are reported

These answer different questions and are not interchangeable.

**Seed-to-seed standard deviation** — rerun the same training with a different
random seed. Measures how stable the procedure is. This is the error bar almost
every paper reports.

**Bootstrap confidence interval over test molecules** — resample the 4,113 test
molecules with replacement 1,000 times and rescore. Measures how much the result
depends on which molecules happened to land in the test set.

On this benchmark the second is roughly **twenty times larger** than the first.
A result reported with only the seed spread looks about twenty times more
precise than it is.

The CI is a property of one set of predictions, not of a mean over seeds, so
`src/report.py` quotes it for one named seed rather than averaging CIs.

---

## 5. The paired test

Marginal confidence intervals overlapping does **not** mean two models are
indistinguishable. That inference is wrong and it is worth being precise about
why.

All arms are scored on the *same* molecules. A resample containing unusually
hard molecules pushes every arm down together. That shared movement is common to
both arms and cancels out of their *difference*. So the correct test resamples
the molecules **once** per draw, scores both arms on that identical resample, and
records the difference:

```
Var(A − B) = Var(A) + Var(B) − 2·Cov(A, B)
```

`src/compare.py` implements this and also reports the fraction of resamples in
which A beats B — a direct, assumption-light answer to "would this ordering
survive a different test set?". It is **not** a p-value; no null hypothesis is
being tested.

On this data pairing narrowed the interval only slightly (±0.043 paired against
±0.042 marginal), which is itself informative: the arms' errors are only
moderately correlated.

---

## 6. Full results

All from one Colab session (NVIDIA L4, torch 2.14.1+cu130, scikit-learn 1.6.1).
Mean ± standard deviation over 3 seeds.

| | ROC-AUC | Average precision | Brier |
|---|---|---|---|
| Morgan FP + RF | 0.7984 ± 0.0017 | 0.3726 ± 0.0033 | 0.0243 |
| WEGL + boosted trees | 0.7749 ± 0.0047 | 0.3369 ± 0.0094 | 0.0355 |
| PNA | 0.7722 ± 0.0113 | 0.2051 ± 0.0339 | 0.0279 |
| WEGL + Random Forest | 0.7673 ± 0.0057 | — | — |

Per-seed ROC-AUC: arm 1 = 0.7978 / 0.7967 / 0.8008; arm 2 = 0.7801 / 0.7686 /
0.7758; arm 3 = 0.7826 / 0.7565 / 0.7776.

**Retrieval** (mean over 3 seeds; 130 actives among 4,113 molecules, base rate
3.161%):

| | actives in top 41 (1%) | EF@1% | actives in top 206 (5%) | EF@5% | precision@100 |
|---|---|---|---|---|---|
| Morgan FP + RF | 29.7 | 22.9× | 62.7 | 9.6× | 0.510 |
| WEGL | 27.3 | 21.1× | 57.7 | 8.9× | 0.440 |
| PNA | 18.0 | 13.9× | 49.3 | 7.6× | 0.383 |

**Calibration.** Predicting the base rate 0.0316 for every molecule scores
Brier 0.0306. Arm 1 (0.0243) and PNA (0.0279) beat that; arm 2's mean (0.0355)
does not, driven by seed 0 at 0.0485 where its probabilities are inflated.
Brier rewards calibration, not ranking — a well-ranked model with badly scaled
probabilities scores poorly, which is exactly arm 2's situation.

**Paired test under both metrics** — seeds (of 3) whose 95% CI excludes zero:

| Comparison | ROC-AUC gap | resolved | AP gap | resolved |
|---|---|---|---|---|
| Morgan FP vs PNA | +0.0262 | 1 of 3 | +0.1675 | **3 of 3** |
| Morgan FP vs WEGL | +0.0236 | 0 of 3 | +0.0357 | 1 of 3 |
| WEGL vs PNA | +0.0026 | 0 of 3 | +0.1318 | **3 of 3** |

PNA's instability is far worse under average precision than under ROC-AUC: its
ROC-AUC spread is ±0.0113 (1.5% relative) but its AP spread is ±0.0339 (16.5%
relative). Seed 1 drives it — ROC-AUC 0.7565 is only 0.026 below its best seed,
but AP 0.1574 is 0.076 below, and its top-1% actives drop from 20 to 14.

---

## 7. Figure by figure

**`fig1_arms.png` — the two uncertainties, to scale.**
Horizontal dot plot, one row per model. Thick bar = ±1 seed standard deviation.
Thin bar = 95% bootstrap CI over test molecules (seed 0, 1,000 resamples). Open
circles are individual seeds. *Point to make:* the thin bars are ~20× the thick
ones, and only the thick one is normally published. Also note PNA's open circles
spread much wider than any other row.

**`fig2_paired.png` — does any gap survive the paired test?**
One histogram per head-to-head comparison, showing 1,000 bootstrapped
*differences* for seed 0, with the 95% interval shaded and zero marked by a
black line. The annotation gives the fraction of resamples the leader wins and
how many of the three seeds resolve the gap. *Point to make:* zero sits inside
almost every interval.

**`fig6_metric_choice.png` — the headline.**
Two panels, same three models, same predictions — ROC-AUC on the left, average
precision on the right. Each carries the paired verdict for WEGL vs PNA under
that metric, coloured red for unresolved and green for resolved. *Point to
make:* 0 of 3 becomes 3 of 3 purely by changing the metric.

**`fig3_screening.png` — what ROC-AUC conceals.**
Left: precision-recall curves with average precision in the legend and a dashed
line at the 3.16% random-guessing rate. Right: enrichment factor against the
fraction of the library screened, best-ranked first. *Point to make:* at the top
1% the fingerprint arm is at ~23× enrichment and PNA at ~14×.

**`fig4_agreement.png` — do the models find the same molecules?**
Three scatter panels, one per pair. Each dot is a molecule at its percentile
rank under two models; actives are the larger red dots. Annotated with Spearman
correlation and top-100 overlap. *Point to make:* if the models agreed, the red
dots would lie on the diagonal; they scatter, yet the counting (section 8) shows
they still recover largely the same actives.

**`fig5_posweight.png` — validation up, test down.**
Two lines, validation and test ROC-AUC, between the unweighted run and
`--pos-weight 10`. *Point to make:* they move in opposite directions, so
selecting on validation picks the worse model.

**`results/false_negatives.png`, `false_positives.png`** — RDKit grids of the
six most confidently wrong molecules in each direction, with predicted
probabilities as captions. Produced by `src/error_analysis.py`, alongside
`error_analysis.csv` with molecular weight, ring count and logP per molecule.

---

## 8. Agreement between the arms

Spearman rank correlation over all 4,113 molecules, seed 0:

| pair | ρ | top-100 shared |
|---|---|---|
| Morgan FP vs WEGL | 0.313 | 64 / 100 |
| Morgan FP vs PNA | 0.377 | 57 / 100 |
| WEGL vs PNA | 0.371 | 56 / 100 |

The low global correlation makes it tempting to conclude the arms succeed on
different molecules. **Counting directly contradicts that.** Taking each arm's
own top 5% (206 molecules), of the 130 actives:

| found by | count |
|---|---|
| all three arms | 45 |
| exactly two | 10 |
| exactly one | 16 |
| **none** | **59** |

Per arm: 65 / 53 / 53 actives recovered. Union across all three: **71** — only
six more than the best single arm. The disagreement lives in the ordering of the
mass of inactives, not at the head of the ranking. Nearly half the actives are
missed by everything.

---

## 9. The class-weighting experiment

`--pos-weight 10`, PNA seed 0, written to `results_posweight/` so it cannot
contaminate the headline table.

| | plain | weighted | change |
|---|---|---|---|
| Validation ROC-AUC | 0.8154 | 0.8308 | **+0.015** |
| Test ROC-AUC | 0.7826 | 0.7584 | **−0.024** |
| Average precision | 0.2335 | 0.1646 | −0.069 |
| Actives in top 1% | 20 | 12 | −8 |
| Brier | 0.0268 | 0.0522 | +0.025 |
| Mean predicted probability | 0.031 | 0.096 | +0.065 |

The obvious reading — "weighting just inflates probabilities, so Brier gets
worse while the ranking is untouched" — is **wrong**. Rank correlation between
the weighted and unweighted predictions is only **0.518**, and every
scale-invariant measure fell: ROC-AUC, average precision, and enrichment. A pure
rescaling would have left all three identical. The weighted loss changed what
the model learned, not merely how confidently it stated it.

Oversampling was deliberately not the comparison here: on a scaffold split,
duplicating actives can place the same scaffold on both sides, inflating
validation directly.

---

## 10. Against the official OGB leaderboard

From the [OGB leaderboard](https://ogb.stanford.edu/docs/leader_graphprop/):

| Rank | Method | Test ROC-AUC | #Params |
|---|---|---|---|
| 1 | Multi-RF Fusion + Multi-GNN | 0.8476 ± 0.0002 | 993,331,107 |
| 2 | HyperFusion | 0.8475 ± 0.0003 | 5,908,027 |
| 3 | PAS+FPs | 0.8420 ± 0.0015 | 26,706,953 |
| 9 | Molecular FP + Random Forest | 0.8208 ± 0.0037 | **5,782** |
| 12 | MorganFP + Rand. Forest | 0.8060 ± 0.0010 | 230,000 |
| 22 | PNA | 0.7905 ± 0.0132 | 326,081 |

**The result here matches the published picture**: fingerprints (rank 12) beat
PNA (rank 22) on OGB's own leaderboard. A trained GNN failing to beat
substructure counting on scaffold-split molhiv is documented, not an artefact of
this implementation.

**The arms land near published versions of themselves.** Arm 1 measures 0.7984
against the leaderboard's 0.8060 — 0.0076 below, and the error bars do **not**
overlap, so this is "close enough to confirm the split handling and evaluator",
not a reproduction. Arm 2 measures 0.7673 against the WEGL paper's
0.7757 ± 0.0111 for the no-virtual-node configuration, inside one standard
deviation.

**On efficiency**, rank 9 reaches 0.8208 with 5,782 parameters, beating every
GNN below it including PNA's 326,081. Rank 1 spends 993 million parameters for
about +0.04 over that.

One caution on reading the top of the table: nearly everything above 0.82 is a
fingerprint/GNN hybrid, which invites the conclusion that the representations
are complementary. Section 8's counting is more guarded — and the spacing
between many adjacent leaderboard rows is inside the range this benchmark could
not resolve.

---

## 11. Reproducibility across machines

The same code, same seeds and same data gave different scores on two machines:

| Arm | Colab, scikit-learn 1.6.1 | Local CPU, scikit-learn 1.9.0 | Shift |
|---|---|---|---|
| Morgan FP + RF | 0.7984 | 0.8069 | 0.0085 |
| WEGL + Random Forest | 0.7673 | 0.7922 | **0.0249** |

For scale, the largest gap between any two *methods* here is 0.0262. Arm 2 is
the more exposed because scikit-learn enters it twice — the PCA that builds the
embedding, and the forest that classifies it.

**This is an observation, not a diagnosis.** The most likely candidate is
`PCA(svd_solver="auto")` selecting a different solver under a revised heuristic,
which would change the embedding itself. Confirming it means pinning the version
and bisecting; that has not been done. The practical consequence stands
regardless: a one-or-two-hundredths difference between two results produced on
different software stacks is not yet evidence about the methods.

---

## 12. Caveats — what is *not* established

Worth volunteering before anyone asks.

**The bootstrap assumes independent test molecules.** They are not. Scaffold-
split test molecules cluster by scaffold, so a scaffold-level block bootstrap
would give **wider** intervals than those reported. This makes the null results
stronger, not weaker, but the reported intervals are optimistic.

**The scikit-learn version effect is unexplained** (section 11).

**Arm 1's probabilities are coarse.** The random forest produces only 245–260
distinct probability values across 4,113 molecules, so cutoff-based metrics can
be tie-sensitive. All reported cutoffs were checked and are tie-independent
except arm 1 seed 1 at top-41, which moves between 29 and 30 actives depending
on tie order.

**The WEGL + Random Forest predictions were not saved.** That row is a point
estimate from its run log; it has no interval anywhere and takes no part in any
paired test. The boosted-tree variant is what `results/` contains and what every
statistic is computed from.

**Three seeds is few.** The leaderboard convention is ten. Three is enough to
show PNA's instability but not to characterise it precisely.

**Average precision is a better metric here, not the only right one.** It
weights the top of the ranking, which suits a screening framing. If the use case
were different, the metric should be too — which is the general version of the
finding in section 6.
