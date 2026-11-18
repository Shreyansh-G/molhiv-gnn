# Does a trained graph neural network actually beat counting substructures?

A study project about **how to tell whether one model is really better than
another.**

The setting is chemistry: given a molecule's 2D structure, predict whether it
blocks HIV from replicating. The dataset is `ogbg-molhiv` — 41,127 real
molecules, of which about 3.5% are active. But chemistry is only the setting.
The thing I actually wanted to study is the question underneath every
leaderboard: **when two models sit a few places apart in a ranking, is that a
real difference, or is it noise that nobody measured?**

I did not try to climb the leaderboard. People with far more compute have
already done that, and squeezing out another tenth of a point would not have
taught me anything. So I built the experiment that answers the other question
instead.

---

## What I compared

Three ways of turning a molecule into something a computer can score. They are
deliberately chosen to go from "no learning at all" to "learned from scratch":

| | Method | In plain words | Does it learn? |
|---|---|---|---|
| **Arm 1** | Morgan fingerprints + Random Forest | Make a checklist of which small chemical fragments are present, then let a forest of decision trees vote. | **No** |
| **Arm 2** | WEGL + Random Forest | Let each atom's description spread along its bonds a few times, then summarise the whole molecule with one geometric trick (optimal transport). | **No** |
| **Arm 3** | PNA graph neural network | A real neural network, 715,441 parameters, trained end to end on a GPU. | **Yes** |

All three see **exactly the same molecules**, are scored by **the same official
evaluator**, and are split the same way.

### One thing worth understanding: the split

The molecules are split by **scaffold** — their core skeleton. Every molecule in
the test set has a skeleton the model has never seen during training.

This matters. If you split randomly, close chemical cousins end up on both
sides, the model recognises them, and every score looks better than it should. A
scaffold split asks the harder and more honest question: *does this work on
chemistry you have never seen?*

---

## What I found

**1. The trained neural network did not win.** It came second, behind a simple
fragment checklist, and it was by far the least consistent of the three.

**2. Most of the differences are too small to call real.** They look convincing
until you actually test them.

**3. Changing the scoring metric reverses the conclusion.** This was the
surprise, and it is the most important thing in the project.

**4. The model's mistakes are not random** — they are a specific, explainable
chemical blind spot.

---

## The results

Scored on the same 4,113 test molecules, of which 130 are active. The score is
**ROC-AUC**: 0.5 means random guessing, 1.0 means perfect. Each model was run
with 3 different random seeds.

| Model | ROC-AUC | Learned parameters | Hardware |
|---|---|---|---|
| Morgan fingerprints + Random Forest | **0.7984** ± 0.0017 | none | CPU, minutes |
| WEGL + boosted trees | **0.7749** ± 0.0047 | none | CPU, minutes |
| PNA (trained GNN) | **0.7722** ± 0.0113 | 715,441 | GPU, ~12 min/seed |
| WEGL + Random Forest | **0.7673** ± 0.0057 | none | CPU, ~9 min total |

![Three arms with both kinds of uncertainty](results/figures/fig1_arms.png)

**How to read this picture.** Each row is one model. The dot is its average
score. There are two bars behind it, and they measure two completely different
things:

- The **thick short bar** is how much the score moved when I changed the random
  seed. This is the only error bar most papers report.
- The **thin long bar** is how much the score moves if the test set had been a
  slightly different sample of molecules. (I re-drew the test set 1,000 times to
  measure this.)

The thin bars are about **twenty times longer** than the thick ones. Which
molecules happened to land in the test set matters far more than which seed you
used — and that is the uncertainty nobody prints.

Notice also that the two WEGL rows sit either side of PNA. Those are the *same
method*; the only difference is which standard classifier is attached at the
end. That one incidental choice decides whether a model with **zero learned
parameters** finishes above or below a 715,441-parameter neural network.

---

## Are the differences real?

Fingerprints beat the GNN by 0.026, which is about 15 times the seed-to-seed
wobble. That sounds decisive. It isn't.

Because all three models are scored on the *same* molecules, a run of unusually
hard molecules drags all of them down together. To remove that shared movement,
I re-draw the test set, score **both** models on that same re-drawn set, and
record the difference. Repeat 1,000 times.

![Paired bootstrap on the difference](results/figures/fig2_paired.png)

**How to read this picture.** Each panel is one head-to-head comparison. The
histogram shows the 1,000 differences. The black line is zero — "no difference
at all". If a good chunk of the histogram sits on both sides of that line, the
gap is not established.

| Comparison | Gap | Seeds where the gap is real | Winner ahead in |
|---|---|---|---|
| Fingerprints vs WEGL | +0.0236 | **0 of 3** | 81–88% of re-draws |
| Fingerprints vs PNA | +0.0262 | 1 of 3 | 79–98% of re-draws |
| WEGL vs PNA | +0.0026 | **0 of 3** | 44–70% of re-draws |

With only 130 active molecules in the test set, this benchmark **cannot resolve
differences this small.** Fingerprints beat the GNN in 79–98% of re-drawn test
sets — likely, yes, but "wins more often than not" is not the same as "is
better", and the distance between those two sentences is the whole project.

---

## The surprise: the metric decides the answer

Everything above uses ROC-AUC, because that is the official score for this
benchmark. But ROC-AUC asks: *"if I pick one active molecule and one inactive
molecule at random, does the model rank the active one higher?"*

With 3,983 inactive molecules and only 130 active ones, that question is mostly
about easy inactives that every model already gets right. It flatters everybody.

**Average precision** asks a different question: *"of the molecules you put at
the very top of your list, how many were actually active?"* That is what you'd
care about if you were deciding which compounds to test in a lab.

Same predictions. Same molecules. Same statistical test. Only the metric changes:

![Same predictions, two metrics, opposite conclusions](results/figures/fig6_metric_choice.png)

| | Fingerprints | WEGL | PNA |
|---|---|---|---|
| ROC-AUC | 0.7984 | 0.7749 | 0.7722 |
| Average precision | 0.3726 | 0.3369 | **0.2051** |

| WEGL vs PNA | Gap | Seeds where it's real |
|---|---|---|
| under ROC-AUC | +0.0026 | **0 of 3** — a coin flip |
| under average precision | +0.1318 | **3 of 3** — ahead in 99.5%, 100%, 100% |

The three models are 6.4× further apart under average precision than under
ROC-AUC. The comparison that was "indistinguishable from a coin flip" becomes
"separated in every single seed and every single re-draw."

You can see the same thing as a practical number. If you could only afford to
test the top 1% of the library in a lab:

![Precision-recall and enrichment](results/figures/fig3_screening.png)

Fingerprints find about **30** of the 130 actives in that top 1%. The trained
neural network finds about **18**. That is a large, practical difference hiding
underneath a 0.026 gap in the headline score.

**So the honest answer has two halves.** Under the metric this benchmark is
officially scored on, almost nothing separates the three models. Under the
metric that actually suits the problem, the trained neural network is clearly
the worst of the three. Reporting only the first half — which is what the
leaderboard does — hides the second.

---

## Do the three models find the same molecules?

Mostly yes — and I got this wrong before I checked it.

The three models rank molecules quite differently overall (rank correlation only
0.31–0.38), so I assumed they must be succeeding on *different* molecules, and
that combining them would help. Counting directly proved me wrong.

![Rank agreement between models](results/figures/fig4_agreement.png)

**How to read this picture.** Each dot is a molecule. Its position is its rank
according to two different models. Red dots are the actives — the molecules we
care about. If the two models agreed perfectly, the red dots would form a
straight diagonal line.

Taking each model's own top 5%, out of the 130 actives:

- **45** are found by all three models
- **59** are found by none of them
- All three models combined find **71**; the single best model alone finds **65**

Three completely different approaches, and together they rescue only **six**
extra molecules. They disagree about how to order thousands of boring inactives,
and they agree about which molecules are worth a second look. Nearly half the
actives are missed by everything.

---

## What the mistakes look like

Numbers hide the thing a chemist would actually want to see, so the project
draws the molecules the model was most confidently wrong about.

![The six most confidently missed actives](results/false_negatives.png)

Every one of these is genuinely active, and the model gave all of them a
probability of **0.00** — not just wrong, but confidently wrong.

Two of them explain the pattern. Top right is a **cobalt coordination complex** —
a molecule built around a metal atom. Bottom right is a **bis-pyridinium
dication** — a molecule carrying two permanent positive charges.

Neither a fragment checklist nor a 2D graph network handles those well, because
neither was designed with metal bonds or permanent charges in mind. The missed
actives are also bigger and more ring-heavy on average (380 Da and 3.5 rings,
versus 277 Da and 2.8 rings for the molecules it wrongly flags).

This is a systematic blind spot, not bad luck — exactly the kind of thing a
scaffold split is designed to expose.

---

## A bonus experiment that went wrong, usefully

Only 3.5% of these molecules are active, so the obvious fix is to tell the model
to pay more attention to the rare positives. I tried it.

![Validation up, test down](results/figures/fig5_posweight.png)

The validation score went **up** by 0.015. The test score went **down** by
0.024.

If I had done the normal thing — pick the model that scores best on validation —
**I would have shipped the worse model, with a number that looked like evidence
for it.** On a scaffold split, the validation molecules and the test molecules
are different chemistry, so an improvement on one need not carry over.

---

## Running it yourself

```bash
pip install -r requirements.txt

python -m src.baseline_fp --seeds 0 1 2    # arm 1  (CPU)
python -m src.run_wegl    --seeds 0 1 2    # arm 2  (CPU)
python -m src.train       --seeds 0 1 2    # arm 3  (GPU)

python -m src.report       # the results table
python -m src.compare      # are the differences real?
python -m src.figures      # regenerate every picture above
```

Or open `notebooks/colab_run.ipynb` and run the whole thing on a free Colab GPU.
The dataset is bundled in this repo, so nothing is downloaded at run time.

Every figure above is regenerated from the saved predictions in `results/` —
you do not have to retrain anything to reproduce them.

---

## What's in here

| File | What it does |
|---|---|
| `DETAILS.md` | The full technical write-up: methods, statistics, caveats |
| `REPRODUCE.md` | Exact software versions and dataset provenance |
| `src/baseline_fp.py` | Arm 1 — fingerprints + Random Forest |
| `src/wegl.py`, `src/run_wegl.py` | Arm 2 — the non-learned graph embedding |
| `src/models.py`, `src/train.py` | Arm 3 — the PNA graph neural network |
| `src/report.py` | Builds the results table, and checks every model saw the same molecules |
| `src/compare.py` | The paired test: is a gap bigger than the noise? |
| `src/figures.py` | Regenerates every figure in this README |
| `src/error_analysis.py` | Draws the molecules the model got most wrong |
| `app/main.py` | A small web API: send a molecule, get a prediction back |
| `notebooks/colab_run.ipynb` | Runs everything end to end on a Colab GPU |

---

## Credits

Built on open-source work:

- **[OGB](https://github.com/snap-stanford/ogb)** (MIT) — the dataset, the
  official scaffold split, the atom/bond encoders and the official evaluator.
- **[PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric)** (MIT) —
  the `PNAConv` layer used by arm 3.
- **[PNA](https://github.com/lukecavabarrett/pna)** (Corso et al., NeurIPS 2020)
  — the architecture and the published hyperparameters used as a starting point.
- **WEGL** (Kolouri et al., ICLR 2021) — arm 2 is **reimplemented from the
  paper**, not copied from the authors' code, with two deviations documented in
  `DETAILS.md`.

The comparison itself, the WEGL reimplementation, the evaluation (bootstrap
intervals, the paired test, calibration), the error analysis, the figures and
the serving layer were written for this project.
