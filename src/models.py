"""
models.py

Model definitions and training helpers for the ProductSentimentML project.

All models are wrapped in a scikit-learn `Pipeline` (TF-IDF -> classifier)
so that the vectorizer is always fit exclusively on the training fold,
whether that's the train/test split or a cross-validation fold. This
prevents data leakage from the test set (or held-out CV fold) into the
feature vocabulary.

Note on confidence/probability: `LinearSVC` does not produce genuine
probabilities — its `decision_function` output is an unbounded signed
distance from the separating hyperplane, not a calibrated probability.
For research comparison (Experiments A/B/C) we use plain `LinearSVC` for
speed. For the model that is actually deployed (see
`build_calibrated_svm_pipeline`), we wrap it in `CalibratedClassifierCV`
so the application can honestly report class probabilities instead of
mislabeling raw decision scores as "confidence".
"""

from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from feature_engineering import build_tfidf_vectorizer

RANDOM_STATE = 42


def get_model_pipelines() -> dict:
    """Return a dict of {model_name: Pipeline} for every model compared in
    this project's research experiments.

    A fresh TfidfVectorizer instance is created per pipeline so that each
    model's vectorizer is fit independently and no fitted state is shared
    across models. These pipelines use plain (uncalibrated) classifiers,
    which is appropriate for research-comparison purposes where only
    accuracy/precision/recall/F1 are reported, not probabilities.
    """
    pipelines = {
        "Dummy Classifier": Pipeline(
            [
                ("tfidf", build_tfidf_vectorizer()),
                ("clf", DummyClassifier(strategy="most_frequent")),
            ]
        ),
        "Logistic Regression": Pipeline(
            [
                ("tfidf", build_tfidf_vectorizer()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Multinomial Naive Bayes": Pipeline(
            [
                ("tfidf", build_tfidf_vectorizer()),
                ("clf", MultinomialNB()),
            ]
        ),
        "Linear SVM": Pipeline(
            [
                ("tfidf", build_tfidf_vectorizer()),
                (
                    "clf",
                    LinearSVC(
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }
    return pipelines


def build_calibrated_svm_pipeline(C: float = 1.0, cv: int = 5) -> Pipeline:
    """Build a deployable Linear-SVM pipeline with genuine, calibrated
    class probabilities.

    `LinearSVC` has no `predict_proba`. Wrapping it in
    `CalibratedClassifierCV` (Platt/sigmoid calibration via internal
    cross-validation on the training data) produces a classifier that
    exposes real, probability-like `predict_proba` output, so the
    deployed application can honestly display per-class confidence
    instead of relabeling raw decision-function scores as "confidence".
    """
    base_svm = LinearSVC(C=C, class_weight="balanced", random_state=RANDOM_STATE)
    calibrated = CalibratedClassifierCV(base_svm, method="sigmoid", cv=cv)
    return Pipeline(
        [
            ("tfidf", build_tfidf_vectorizer()),
            ("clf", calibrated),
        ]
    )

