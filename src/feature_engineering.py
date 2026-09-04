"""
feature_engineering.py

TF-IDF feature extraction helpers.

TF-IDF (Term Frequency - Inverse Document Frequency) represents each review
as a sparse vector where every dimension is a word (or n-gram). The value
combines:
  - Term Frequency: how often the term appears in that review.
  - Inverse Document Frequency: a down-weighting of terms that appear in
    many reviews (e.g. "the", "product") because they carry little
    discriminative signal, and an up-weighting of terms that are rarer and
    therefore more informative about a specific review's sentiment.

This makes TF-IDF a natural fit for classical ML sentiment classification:
it turns free text into a fixed-length numeric representation while
automatically discounting uninformative, high-frequency words, without the
need for dense embeddings or deep learning.
"""

from sklearn.feature_extraction.text import TfidfVectorizer


def build_tfidf_vectorizer(
    ngram_range=(1, 2),
    min_df=5,
    max_df=0.9,
    max_features=20000,
    sublinear_tf=True,
) -> TfidfVectorizer:
    """Create a `TfidfVectorizer` with sensible defaults for this project.

    Parameter choices:
        ngram_range=(1, 2): unigrams capture individual sentiment words
            ("great", "terrible"); bigrams capture short phrases and
            negation patterns ("not good", "no problem").
        min_df=5: drop terms that appear in fewer than 5 reviews - removes
            typos and rare tokens that would otherwise blow up the
            vocabulary without adding generalizable signal.
        max_df=0.9: drop terms that appear in more than 90% of reviews -
            removes near-universal words that carry little sentiment
            information.
        max_features=20000: caps the vocabulary size so the resulting
            matrix stays small enough to comfortably fit in a Colab
            runtime.
        sublinear_tf=True: applies 1 + log(tf) instead of raw term
            frequency, which reduces the influence of a term that is
            repeated many times in a single review.
    """
    return TfidfVectorizer(
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        max_features=max_features,
        sublinear_tf=sublinear_tf,
    )
