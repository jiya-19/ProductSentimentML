# Comparative Evaluation of Classical and Transformer-Based Approaches for Product Review Sentiment Classification

### ProductSentimentML

---

## Overview

ProductSentimentML is a comparative NLP research project **and** a deployable web application. It
builds and rigorously evaluates several classical machine-learning approaches to 3-class
(Negative/Neutral/Positive) product-review sentiment classification, adds a pretrained transformer baseline for comparison, and deploys the most practical model behind a FastAPI backend with a
simple web frontend.

This is presented as a **comparative experimental study** — not a novel algorithm, not a
state-of-the-art claim. Its contribution is a controlled comparison across approaches, with honest,
reproducible reporting.

## Research Question

> **How do classical TF-IDF-based machine-learning models compare with a pretrained
> transformer approach for product-review sentiment classification?**

Specifically, this project investigates:

1. Which classical ML model performs best?
2. Does hyperparameter tuning improve performance?
3. Do bigrams improve sentiment classification compared with unigrams?
4. How does the transformer approach compare with classical ML?
5. Which sentiment class is most difficult to classify?
6. What types of reviews are commonly misclassified?
7. What are the trade-offs between classical ML and transformer approaches?
8. Which approach is more practical for lightweight deployment?

## Objectives

- Build a reproducible, leakage-free classical ML pipeline for review sentiment classification.
- Run controlled experiments (n-grams, algorithm choice, hyperparameter tuning) and report real,
  executed results.
- Add a transformer baseline for comparison, clearly disclosing whether it was executed.
- Fix probability/confidence handling so the application never mislabels raw decision scores as
  probabilities.
- Deploy a lightweight, practical model behind a real web application on Render.

## Dataset

[Women's Clothing E-Commerce Reviews](https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews)
— 23,486 real customer reviews with free-text content and a 1-5 star rating, loaded automatically
from a public GitHub-hosted CSV (no manual download or API key required).

- **Records after cleaning:** 22,634 (duplicates and empty reviews removed).
- **Columns used:** `Review Text` (model input), `Rating` (1-5 stars, used to derive the label).
- **Missing values:** rows missing `Review Text` or `Rating` are dropped during cleaning.
- **Duplicates:** exact-duplicate review texts are dropped.
- **Rating distribution:** heavily skewed toward 4-5 stars (see `results/figures/04_rating_distribution.png`).
- **Sentiment distribution** (after mapping): Positive ≈ 78%, Neutral ≈ 11%, Negative ≈ 11% — see
  `results/figures/01_sentiment_distribution.png`.

**Important:** sentiment labels are **derived from the star rating**, not manually annotated by a
human reading each review's text:

```text
1-2 stars -> Negative
3   stars -> Neutral
4-5 stars -> Positive
```

A 3-star review is *assumed* neutral by this convention, not verified as neutral by a human
annotator — see [Limitations](#limitations).

Full provenance: [`data/README.md`](data/README.md).

## Methodology

```
Raw Dataset
     |
Basic Cleaning
     |
Train/Test Split (80/20, stratified, random_state=42)
     |
TF-IDF fitted ONLY on training data
     |
Cross-Validation / Hyperparameter Tuning
     |
Final Classical Model (calibrated for deployment)
     |
Untouched Test-Set Evaluation
     |
Transformer Baseline (where executable) + Master Comparison
```

TF-IDF is always fit inside a scikit-learn `Pipeline`, exclusively on the training fold — never on
the full dataset before splitting, and never using test data for feature fitting, hyperparameter
selection, or model selection.

## Classical Machine Learning

| Model | Role |
|---|---|
| Dummy Classifier | Baseline (predicts the majority class) |
| Logistic Regression | Linear model, `class_weight="balanced"` |
| Multinomial Naive Bayes | Classic probabilistic bag-of-words model |
| Linear SVM (`LinearSVC`) | Strong linear model for high-dimensional sparse text |

Random Forest was intentionally **not** included — tree ensembles generally underperform linear
models on very high-dimensional sparse TF-IDF data and add unnecessary compute overhead; the goal
here is a controlled comparison, not an exhaustive model zoo.

**TF-IDF configuration:** `ngram_range=(1,2)`, `min_df=5`, `max_df=0.9`, `max_features=20000`,
`sublinear_tf=True` (see `src/feature_engineering.py` for the rationale behind each choice).

## Transformer Baseline

**Preferred model:** `distilbert-base-uncased`, fine-tuned with a 3-class classification head on a
fixed, documented, reproducible subset (2,000 train / 500 test reviews, `random_state=42`, 2
epochs) drawn from the same cleaned train/test split as the classical models.

**Execution status:** the notebook contains a guarded DistilBERT baseline. When Hugging Face access is available (for example, in Google Colab), it fine-tunes `distilbert-base-uncased` once on 2,000 training reviews and evaluates once on the fixed 500-review test subset. **DistilBERT does not use 5-fold cross-validation in this project.** The exact same 500 held-out reviews are evaluated by the Linear SVM in the matched comparison section. If Hugging Face access is unavailable, the notebook completes without fabricating transformer metrics and leaves those cells blank in the comparison CSV.

## Results

Classical results below come from the held-out (20%) test set, with 5-fold stratified cross-validation used for model comparison and lightweight `GridSearchCV` hyperparameter tuning. DistilBERT is evaluated separately, once, on the fixed matched 500-review test subset; it does **not** use 5-fold cross-validation. All executed numbers below come directly from `notebooks/Product_Sentiment_Analysis.ipynb`; the generated CSV files under `results/metrics/` are the authoritative machine-readable outputs.

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---:|---:|---:|---:|
| **Linear SVM (tuned)** | **0.819** | **0.614** | **0.595** | **0.603** |
| Logistic Regression (tuned) | 0.783 | 0.585 | 0.628 | 0.601 |
| Multinomial Naive Bayes | 0.776 | 0.597 | 0.352 | 0.327 |
| Dummy Classifier (baseline) | 0.770 | 0.257 | 0.333 | 0.290 |
| DistilBERT (fine-tuned, matched 500-review test set) | *See `results/metrics/master_comparison.csv` after a Colab run* | | | |

**Best classical model: Linear SVM** (Research Question 1), selected on macro-averaged F1-score —
the right criterion for this imbalanced 3-class problem, not accuracy alone. The Dummy Classifier
already reaches 77% accuracy by always predicting "Positive," while completely failing on Negative
and Neutral reviews (0.29 macro F1) — a concrete illustration of why accuracy alone is misleading
here.

### Deployed (calibrated) model performance

The model actually served by the FastAPI application is a `CalibratedClassifierCV`-wrapped Linear
SVM (see [Confidence & Probability Handling](#confidence--probability-handling)), retrained and
re-evaluated separately from the research-comparison pipeline above:

| | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---:|---:|---:|---:|
| Linear SVM (uncalibrated, research) | 0.819 | 0.614 | 0.595 | 0.603 |
| **Linear SVM (calibrated, deployed)** | **0.826** | **0.633** | **0.562** | **0.585** |

Calibration slightly changes the precision/recall balance (macro F1 0.603 -> 0.585) in exchange for
genuine, usable probability output — a deliberate, disclosed trade-off, not an oversight.

### Experiments

| Experiment | Comparison | Result |
|---|---|---|
| A — N-grams (Q3) | Unigrams only vs. Unigrams+Bigrams (Logistic Regression) | Bigrams improved macro F1 by **+0.013** (0.588 -> 0.601) |
| B — Algorithm comparison (Q1) | LR vs. Naive Bayes vs. Linear SVM | Linear SVM highest (0.603 macro F1) |
| C — Default vs. tuned (Q2) | Linear SVM default vs. `GridSearchCV`-tuned `C` | Tuning improved macro F1 by **+0.023** (0.580 -> 0.603) |

Full experiment table: `results/metrics/experiment_summary.csv`.

### Which class is hardest to classify? (Q5)

**Neutral.** For the best classical model, Neutral reviews have the lowest recall of the three
classes (~35.6%) — see `results/figures/10_confusion_matrix_linear_svm.png`. This lines up with
Neutral being both the most linguistically ambiguous ("it's okay," "decent but...") and the class
with the fewest training examples.

### Apples-to-apples 500-review comparison

The notebook creates `results/metrics/matched_500_comparison.csv`. It evaluates the best classical Linear SVM and DistilBERT on the **exact same 500 held-out reviews**, selected deterministically from the classical test split with `random_state=42`.

This is the appropriate table for the direct model-performance comparison. The training budgets are intentionally reported separately: Linear SVM uses the full classical training split, while DistilBERT uses a 2,000-review training subset for a practical transformer baseline. No 5-fold CV is claimed for DistilBERT.

The exact 500 source rows are also saved as `results/metrics/matched_500_test_reviews.csv` for reproducibility.

| Model | Training setup | Evaluation reviews | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---|---:|---:|---:|---:|---:|
| Linear SVM | Full classical training split | 500 | See generated CSV | See generated CSV | See generated CSV | See generated CSV |
| DistilBERT | 2,000-review subset, 2 epochs | 500 | Generated by notebook | Generated by notebook | Generated by notebook | Generated by notebook |

### Computational comparison

Measured directly (`results/metrics/computational_comparison.csv`), not assumed:

| Model | Train Time (s) | Inference Time, full test set (s) | Reviews/sec | Serialized Size (KB) |
|---|---:|---:|---:|---:|
| Linear SVM | 1.2 | 0.12 | ~36,300 | ~1,130 |
| Logistic Regression | 1.6 | 0.13 | ~34,100 | ~1,130 |
| Multinomial Naive Bayes | 0.9 | 0.12 | ~37,100 | ~1,552 |

All three classical models train and predict on the full ~18K/4.5K train/test split in a couple of
seconds on a standard CPU, with serialized sizes around 1.1-1.6MB. DistilBERT weights alone are
~250MB+ and fine-tuning benefits substantially from a GPU — see the Classical vs. Transformer
Comparison section below.

## Classical vs. Transformer Comparison (Q7, Q8)

| | Classical ML (deployed) | Transformer (DistilBERT) |
|---|---|---|
| Training | Seconds on CPU | Minutes-to-hours; GPU strongly recommended even for a small subset |
| Inference | Sub-millisecond per review, CPU-only | Tens of milliseconds per review on CPU, faster on GPU |
| Model size | ~1.1MB | ~250MB+ |
| Deployment | Fits comfortably in a Render free-tier web service | Needs more memory/CPU; slower cold starts |
| Typical accuracy ceiling | Solid, bounded by bag-of-n-grams representation | Usually higher on nuanced/contextual language, given adequate fine-tuning |

**Practical conclusion (Q8):** the classical model is more practical for lightweight deployment,
which is exactly why the deployed FastAPI application uses it, not DistilBERT — see the Deployment
Decision section below.

## Error Analysis

On the held-out test set, misclassified reviews are inspected directly (not assumed) — see notebook
Section 15. Patterns actually observed in this data:

- **Neutral-boundary confusion:** a large share of errors involve the Neutral class as either the
  true or predicted label — consistent with Neutral being the hardest class to separate from its
  neighbors.
- **Short reviews:** very short reviews (<=5 words after cleaning) give TF-IDF little signal to work
  with, and are checked explicitly against the overall test-set rate rather than assumed to be
  over-represented among errors.
- **Negation and mixed sentiment:** reviews with a positive opening and a negative qualifier (or
  vice versa — e.g. "great fabric but sizing is way off") are a recurring, visible error pattern in
  the sampled misclassifications.

Only error categories actually supported by sampled examples are reported — see notebook Section 15
for the sampled misclassified reviews table this analysis is based on.

## Explainability

**Classical models:** top TF-IDF features per class are extracted directly from the trained linear
model's coefficients (`src/evaluation.py::get_top_features_linear`) — see
`results/figures/11_feature_importance.png`. These represent **model-learned associations from this
dataset**, not universal definitions of sentiment. A per-prediction explainer
(`explain_prediction` in the notebook) shows which words in a specific review pushed the prediction
toward or away from its predicted class.

**Transformer:** no complicated explainability framework was added for the transformer baseline
purely for appearance. If reliable transformer explanations (e.g. attention visualization,
integrated gradients) are implemented, that belongs in future work (see below) — the classical
feature analysis and error analysis above are considered sufficient for this project's scope.

## Confidence & Probability Handling

`LinearSVC`'s `decision_function` output is an unbounded signed distance from the separating
hyperplane — **not** a probability. This project does not describe it as one anywhere. The deployed
model is a `CalibratedClassifierCV`-wrapped Linear SVM (sigmoid/Platt calibration via internal
cross-validation on the training data), which provides genuine `predict_proba` output. The FastAPI
`/predict` endpoint always returns real calibrated probabilities:

```json
{
  "sentiment": "Positive",
  "confidence": 0.9358,
  "probabilities": {
    "Negative": 0.0201,
    "Neutral": 0.0441,
    "Positive": 0.9358
  }
}
```

If a future model swap ever removes `predict_proba` support, the API is written to fail loudly
(`500` with a clear message) rather than silently mislabeling a decision score as a probability.

## Deployment Decision

**The deployed FastAPI application uses the calibrated classical Linear SVM — not the
transformer.** This is deliberate:

- The classical pipeline is ~1.1MB, trains and predicts in milliseconds on CPU, and fits
  comfortably in a Render free-tier web service.
- DistilBERT is ~250MB+ and needs meaningfully more memory/compute for comparable latency —
  unnecessary overhead for this project's deployment scale.
- The research notebook is where classical-vs-transformer comparison belongs; the deployed
  application's job is to be fast, cheap to host, and reliable.

This split — compare multiple approaches in research, deploy the efficient one — is documented
explicitly rather than left implicit.

## Web Application

```
Browser (HTML / CSS / JS)
          |
          v
   FastAPI (app/main.py)
     GET  /            -> static frontend
     GET  /health       -> health check
     POST /predict      -> { sentiment, confidence, probabilities }
          |
          v
Saved, calibrated scikit-learn Pipeline (models/sentiment_model.pkl)
```

One FastAPI application serves both the static frontend and the `/predict` API — a single Render
web service, one public URL, no separate frontend hosting, no cross-origin complexity.

- `app/main.py` — FastAPI backend, Pydantic request validation (rejects empty/oversized reviews),
  project-relative model path (`Path(__file__).resolve().parent.parent / "models" / ...`), no
  hardcoded ports.
- `app/static/index.html` / `style.css` / `script.js` — vanilla HTML/CSS/JS frontend: a review text
  area, an "Analyze sentiment" button, a calibrated-probability bar chart per class, loading and
  error states ("Analyzing review…", "Unable to connect to the prediction service. Please try
  again."), and a short "About this model" section.

## Live Demo

The application is configured for Render deployment through `render.yaml`.

## Technologies Used

Python 3, pandas, numpy, scikit-learn, matplotlib, seaborn, wordcloud, scipy, joblib, FastAPI,
uvicorn, pydantic. (Research-only, Colab-side: `transformers`, `datasets`, `accelerate`, and `torch` — see `requirements-notebook.txt`.)

## Running Locally

### Web application

From the repository root:

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000`. Do not use VS Code Live Server on port 5500 for this project: the frontend is served by FastAPI and the `/predict` endpoint is provided by the same application.

### Research notebook

The notebook needs the research packages in `requirements-notebook.txt`. The DistilBERT section additionally requires `transformers`, `datasets`, `accelerate`, and `torch`, plus access to Hugging Face. The notebook itself contains the exact configuration and evaluation cells.

## Project Structure

```
ProductSentimentML/
|
├── notebooks/
|   └── Product_Sentiment_Analysis.ipynb   # Full research pipeline and matched transformer comparison
|
├── results/
|   ├── figures/                # Generated plots
|   └── metrics/                # Generated CSV metrics and matched-test audit file
|
├── src/
|   ├── preprocessing.py       # Data cleaning + text-cleaning pipeline
|   ├── feature_engineering.py # TF-IDF vectorizer builder
|   ├── models.py               # Model + calibrated-pipeline definitions
|   └── evaluation.py           # Metrics, plots, feature-importance helpers
|
├── app/
|   ├── main.py                 # FastAPI backend
|   └── static/
|       ├── index.html
|       ├── style.css
|       └── script.js
|
├── data/
|   └── README.md               # Dataset provenance (raw CSV not committed)
|
├── models/
|   └── sentiment_model.pkl     # Deployed, calibrated pipeline (TF-IDF + Linear SVM)
|
├── requirements.txt
├── requirements-notebook.txt
├── render.yaml
├── README.md
├── .gitignore
└── LICENSE
```

Not committed: the raw dataset CSV (auto-downloaded), transformer weights/checkpoints, virtual
environments, cache files, and Colab checkpoints — see `.gitignore`.

## Installation

After cloning the repository, open a terminal in its root and run:

```bash
pip install -r requirements.txt
```

For the research notebook, install the additional research dependencies:

```bash
pip install -r requirements-notebook.txt
```

The DistilBERT section additionally requires `transformers`, `datasets`, `accelerate`, and `torch`, plus access to Hugging Face. The notebook contains the exact configuration and evaluation cells.

## Usage

### Run the research notebook (Google Colab)

1. Open `notebooks/Product_Sentiment_Analysis.ipynb` in Google Colab.
2. Run all cells top to bottom (`Runtime > Run all`). The dataset downloads automatically.
3. GPU is **not required** for the classical ML pipeline. GPU is optional but recommended for the
   transformer section; CPU also works for the small documented subset, just slower.
4. The transformer experiment uses 2,000 training reviews and the exact same 500 held-out reviews as
   the matched Linear SVM evaluation. It does **not** use 5-fold cross-validation.

### Run the research notebook locally

```bash
pip install -r requirements-notebook.txt
jupyter notebook notebooks/Product_Sentiment_Analysis.ipynb
```

### Run the web application locally

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Then open **http://localhost:8000** in a browser. Do not use VS Code Live Server on port 5500: the
frontend is served by FastAPI and `/predict` is provided by the same application.

The API can also be tested directly:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"review": "This product is amazing, best purchase ever!"}'
```

## Key Findings

- Classical ML with TF-IDF features achieves strong performance on this task: **81.9% accuracy**
  and **0.603 macro F1** (research pipeline) with a tuned Linear SVM, far above the 0.290 macro-F1
  majority-class baseline.
- **Linear SVM** consistently outperformed Logistic Regression and Multinomial Naive Bayes on this
  dataset (Q1).
- **Hyperparameter tuning** improved Linear SVM's macro F1 by +0.023 (Q2).
- **Bigrams** measurably improved performance over unigrams alone (+0.013 macro F1), indicating
  short phrases (including negations) carry sentiment signal beyond individual words (Q3).
- The **transformer comparison (Q4)** is implemented as a guarded Colab-ready experiment. It was not executed in this delivery environment because external dataset/model downloads are unavailable here; no transformer metrics are fabricated. The Linear SVM matched evaluation cell is implemented and validated with a smoke test.
- The **Neutral class is hardest to classify** (Q5) — lowest recall of the three classes, consistent
  with 3-star reviews being both linguistically ambiguous and underrepresented in training data.
- **Error analysis (Q6)** shows Neutral-boundary confusion, short-review ambiguity, and mixed
  positive/negative-clause reviews as recurring, evidence-based patterns.
- On measured **computational trade-offs (Q7)**, the classical pipeline trains/predicts in seconds
  on CPU at ~1.1MB, versus DistilBERT's 250MB+ footprint and GPU-preferred fine-tuning.
- The classical model is the **more practical choice for lightweight deployment (Q8)** — which is
  why it, not the transformer, powers the deployed FastAPI application.

Full narrative discussion generated from the executed results is in Section 26 of the notebook.

## Limitations

- **Rating-derived labels, not human annotations.** A 3-star review is assumed neutral by
  convention, not verified as neutral by a human reader — the "ground truth" itself is a proxy.
- **Domain dependency**: trained entirely on women's clothing reviews; vocabulary and sentiment
  expression likely differ in other product categories.
- **Class imbalance**: Neutral/Negative classes have far fewer training examples than Positive, even
  with `class_weight="balanced"` and macro-averaged metrics.
- **Sarcasm/irony** are not handled by TF-IDF-based linear models.
- **Negation is only partially captured** — bigrams help, but there's no real understanding of
  negation scope in longer or more complex sentences.
- **Context-dependent language**: TF-IDF cannot capture long-range context or coreference the way
  transformer models can.
- **Transformer evaluation subset**: DistilBERT is evaluated once on a fixed 500-review subset for compute-time reasons. Linear SVM is evaluated on those exact same 500 reviews, so the evaluation set is matched; the training budgets remain different and are reported explicitly.
- **Transformer computational requirements**: CPU-only fine-tuning is slow even for the small
  documented subset; a GPU runtime is recommended.

## Future Work

1. Fully fine-tune BERT/DistilBERT on the complete dataset (not a subset) with GPU resources, and
   re-run the comparison at matched test-set size.
2. Aspect-based sentiment analysis (fit, fabric quality, shipping, etc. separately).
3. Explicit sarcasm/irony detection as a pre-filtering step.
4. Multilingual sentiment classification.
5. Domain-adaptation experiments across product categories.
6. Explainable transformer models (attention visualization, integrated gradients).
7. A larger, human-annotated dataset to validate the rating-to-sentiment proxy.
8. Real-time review analytics on top of the deployed API.

## Reproducibility

- Fixed random seed (`RANDOM_STATE = 42`) throughout data splitting, model training, and the
  (documented) transformer subset sampling.
- TF-IDF is always fit inside a `Pipeline`, exclusively on training folds — no leakage into
  cross-validation, tuning, or the held-out test set.
- The included classical result CSVs and deployed `models/sentiment_model.pkl` come from the project's prior validated classical run. This delivery could not re-run the full notebook because the raw dataset is intentionally not bundled and external downloads are unavailable in the current environment. The modified notebook was syntax-checked, its new comparison cells were smoke-tested, and the deployed FastAPI application was tested locally.
- The transformer section is implemented for Colab and explicitly leaves transformer metrics blank when Hugging Face access is unavailable, rather than filling them with invented numbers. DistilBERT uses one fixed 500-review evaluation, not 5-fold CV.
- The FastAPI backend, static frontend, and all documented API behaviors (health check, prediction,
  input validation, error handling) were tested directly against a running local server — including
  from a **clean virtual environment installing only the pinned `requirements.txt`**, to confirm the
  app works on a machine that has never had this project's dependencies installed before (see
  below).
- **No NLTK runtime download.** Stopword removal uses scikit-learn's built-in `ENGLISH_STOP_WORDS`
  (a static object bundled with scikit-learn) instead of NLTK's downloadable "stopwords" corpus. The
  project has no `nltk.download(...)` call anywhere — notebook or application — so a completely
  fresh Render deploy (or any fresh environment) never needs to reach an external data host at
  import time, on first request, or on a cold start. This was previously a real risk: the prior
  implementation fell back to `nltk.download("stopwords")` the first time the corpus wasn't already
  cached, which is exactly the kind of runtime network dependency a deployed API should not have.
- **Pinned dependencies.** `requirements.txt` pins exact versions for every package (`numpy==2.4.4`,
  `pandas==3.0.2`, `scikit-learn==1.8.0`, `fastapi==0.141.1`, etc.) rather than leaving them
  unconstrained, so `pip install -r requirements.txt` reproduces the exact environment this project
  was tested against, both locally and on Render (`render.yaml` pins the matching Python version,
  3.12.3).

### Clean-environment verification

Before this delivery, the full stack was verified from a **fresh virtual environment containing
none of this project's dependencies beforehand** — not just the already-populated global
environment used during development:

```bash
python3 -m venv /tmp/psml_clean_env
source /tmp/psml_clean_env/bin/activate
pip install -r requirements.txt        # only the pinned requirements — nothing else
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Confirmed in that clean environment:
- `pip install -r requirements.txt` succeeds with no missing dependencies (in particular, no NLTK
  package is required or installed).
- The app imports and starts with **no network calls** during startup — no corpus download, no
  external fetch.
- `GET /` returns the frontend; `GET /health` returns `{"status": "ok", "model_loaded": true}`.
- `POST /predict` returns correct, genuinely-calibrated sentiment predictions for real review text.
- Input validation correctly rejects empty reviews and oversized input (422), and malformed JSON
  (422).
- `models/sentiment_model.pkl` loads correctly and its predictions match the notebook's saved
  evaluation results.
