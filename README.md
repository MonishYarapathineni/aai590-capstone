# Detecting Misconception Patterns in Sequential Student Interaction Data

**AAI-590 Capstone** · Monish Yarapathineni
Master of Science in Applied Artificial Intelligence · University of San Diego

A two-stage system that infers **what kind** of misconception is behind a
student's wrong answer, rather than only that the answer was wrong.

---

## Headline result

A sequence model classifies misconception type at **0.920 macro AUC**, against
**0.848** for an otherwise identical model that sees only the current attempt.

Controlled ablations establish what that difference is and is not:

| Model | Params | Macro AUC |
|---|---|---|
| Sequence model (LSTM) | 224,104 | **0.920** |
| Single interaction, capacity-matched | 225,448 | 0.848 |
| Single interaction, original | 27,496 | 0.847 |
| Full history, **order randomised** | 224,104 | 0.846 |

- **Capacity contributes +0.0005.** An eightfold parameter increase without
  sequential access buys essentially nothing.
- **Sequence contributes +0.072.**
- **Shuffling destroys 100% of it.** A student's complete history, as an
  unordered set, is worth no more than their current attempt alone.

### The signal is short-range

| History available | Share of the sequence advantage |
|---|---|
| 5 problems | 58% |
| 10 problems | 76% |
| 25 problems | 89% |
| 350 problems | 98% |

The most recent handful of attempts are roughly **800× more informative per
problem** than the long tail.

**Interpretation.** The hypothesis — that sequential context beats an isolated
answer — holds. The assumed mechanism does not. The model reads a student's
*recent state within a topic*, not a durable personal characteristic. For
deployment this is favourable: a rolling buffer of ~25 attempts suffices, with no
persistent learner model to store or maintain.

---

## Pipeline

**Stage 1 — offline labeling.** A large language model derives an eight-category
misconception taxonomy by clustering 2,587 expert-authored descriptions from the
Eedi corpus, then applies it to every distinct wrong answer in the dataset. The
taxonomy is derived from an *independent* corpus so that applying it to
FoundationalASSIST is a genuine test rather than a circular fit.

**Stage 2 — sequence classifier.** An LSTM trained from scratch predicts
misconception categories per attempt from the student's preceding history. Eight
sigmoid outputs with masked binary cross-entropy, since one wrong answer often
admits more than one explanation.

### The eight categories

Terminology and notation confusion · Faulty fact or formula recall · Corrupted
procedure execution · Invalid rule generalisation · Conceptual meaning error ·
Representation misreading · Appearance-based spatial reasoning · Task
interpretation and monitoring

Clustering was constrained to describe the **nature of the error** rather than
the mathematical topic — without that constraint the model returns groupings like
"fractions" and "geometry," duplicating information the curriculum skill tag
already carries.

---

## Repository structure

```
aai590-capstone/
├── notebooks/
│   ├── 01_data_cleaning_eda.ipynb                # Cleaning, EDA, Stage 1 readiness
│   ├── 02_stage1a_taxonomy.ipynb                 # Taxonomy generation + validation
│   ├── 03_stage1b_pilot.ipynb                    # Single-label pilot; reliability
│   ├── 04_stage1b_multilabel_validation.ipynb    # Multi-label validation
│   ├── 05_stage1b_full_run.ipynb                 # Full labeling run (20,585 pairs)
│   └── 06_stage2_lstm.ipynb                      # LSTM, baselines, three ablations
├── misconception_taxonomy_v1.json                # Stage 1a artifact
├── requirements.txt
└── data/                                          # Not tracked (CC-BY-NC-4.0)
```

Run the notebooks in order. Each is self-contained and works in Google Colab or
locally; 02 through 05 require an Anthropic API key, 06 does not.

---

## Reproducing

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**Credentials.** `ANTHROPIC_API_KEY` for notebooks 02–05, a HuggingFace token for
the gated dataset, and Kaggle credentials for the Eedi corpus in notebook 02. In
Colab, add these to secrets rather than hardcoding them.

**Datasets.** Neither is redistributed here.

- FoundationalASSIST — gated, CC-BY-NC-4.0
  https://huggingface.co/datasets/ASSISTments/FoundationalASSIST
- Eedi misconception corpus — accept the competition rules to download
  https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics

**Cost.** The full labeling run in notebook 05 is the only significant expense:
**$55** and roughly 7 hours at the validated settings. Notebook 05 calibrates on
a small sample and projects the cost before committing, enforces a hard spend
cap, and checkpoints to Drive so an interrupted run resumes rather than restarts.

---

## Method notes

**Labeling is multi-label.** An initial single-label design reached only κ = 0.560
between independent passes. The disagreements clustered on answers where several
categories were simultaneously defensible — a forced choice, not genuine
uncertainty. Permitting multiple categories raised agreement to **κ = 0.675**.

**Reasoning effort was chosen empirically.** Reliability was measured across four
settings rather than defaulting to the most expensive. The middle setting matched
the highest to within 0.004 at a fifth of the computation. The two lowest were
rejected not for poor agreement but because they collapsed to one category per
answer, silently reverting to the formulation already shown inadequate —
reliability alone would have accepted them.

**Coverage is 59%.** Select-all problems are excluded because a wrong subset can
encode several errors at once and cannot carry one coherent label. Free-response
labeling covers the ten most common wrong answers per problem. Unlabeled attempts
are **masked in the loss, not dropped** — they remain in the sequence as
behavioural context.

**Splits are by student**, never by attempt, since the model's task is explicitly
to generalise to students it has not seen.

---

## Evaluation caveats

- Two fifths of wrong attempts carry no label
- Labels are imperfect: roughly a third of individual category decisions are
  unstable between passes, which caps achievable performance
- Problem text was excluded from model inputs to keep the sequential question
  uncontaminated; including it would likely raise the single-interaction baseline
- Reported figures use initial hyperparameters, not a tuned configuration
- No comparison against published knowledge-tracing systems

---

## References

Brown, J. S., & Burton, R. B. (1978). Diagnostic models for procedural bugs in
basic mathematical skills. *Cognitive Science*, *2*(2), 155–192.

Corbett, A. T., & Anderson, J. R. (1994). Knowledge tracing: Modeling the
acquisition of procedural knowledge. *User Modeling and User-Adapted
Interaction*, *4*(4), 253–278.

Eedi. (2024). *Eedi — Mining misconceptions in mathematics* [Data set]. Kaggle.

Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural
Computation*, *9*(8), 1735–1780.

Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization.
*ICLR*.

Paszke, A., et al. (2019). PyTorch: An imperative style, high-performance deep
learning library. *NeurIPS*, *32*, 8024–8035.

Piech, C., Bassen, J., Huang, J., Ganguli, S., Sahami, M., Guibas, L. J., &
Sohl-Dickstein, J. (2015). Deep knowledge tracing. *NeurIPS*, *28*, 505–513.

Wang, Z., et al. (2020). Diagnostic questions: The NeurIPS 2020 education
challenge. *arXiv*. https://arxiv.org/abs/2007.12061

Worden, E., Heffernan, C., Heffernan, N., & Sonkar, S. (2026).
FoundationalASSIST: An educational dataset for foundational knowledge tracing and
pedagogical grounding of LLMs. *arXiv*. https://arxiv.org/abs/2602.00070
