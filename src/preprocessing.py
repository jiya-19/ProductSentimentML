"""
preprocessing.py

Data cleaning and text-preprocessing utilities for the ProductSentimentML
project.

Two separate concerns are kept apart on purpose:
  1. Row-level dataframe cleaning (missing values, duplicates, invalid
     ratings) -> `load_and_clean_data`.
  2. Text-level cleaning (HTML, URLs, punctuation, stopwords) -> `clean_text`.

Negation words (e.g. "not", "no", "never") are deliberately kept in the
stopword-removal step because they carry sentiment information ("not good"
should not collapse to "good").
"""

import re

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Stopwords come from scikit-learn's built-in list rather than NLTK's
# downloadable corpus. scikit-learn is already a hard dependency of this
# project and ships ENGLISH_STOP_WORDS as a static Python object with the
# package itself - no network access or `nltk.download(...)` call is
# required at import time or first request. This matters specifically for
# a fresh deployment (e.g. Render): the previous NLTK-based implementation
# attempted to download the "stopwords" corpus the first time it was
# imported if it wasn't already cached, which is a runtime network
# dependency that can fail or add latency on a cold start. Using
# scikit-learn's built-in list removes that risk entirely while keeping
# the same stopword-removal behavior.
STOPWORDS = set(ENGLISH_STOP_WORDS)

# Words that flip or scope sentiment. These are removed from the stopword
# list so that negated phrases ("not good", "never buying again") are not
# stripped down to their non-negated form.
NEGATION_WORDS = {
    "no",
    "not",
    "nor",
    "never",
    "none",
    "nobody",
    "nothing",
    "neither",
    "nowhere",
    "cannot",
    "can't",
    "won't",
    "don't",
    "didn't",
    "doesn't",
    "isn't",
    "wasn't",
    "aren't",
    "weren't",
    "couldn't",
    "shouldn't",
    "wouldn't",
    "hasn't",
    "haven't",
    "hadn't",
    "without",
}

SENTIMENT_STOPWORDS = STOPWORDS - NEGATION_WORDS

HTML_TAG_RE = re.compile(r"<[^>]+>")
URL_RE = re.compile(r"http\S+|www\.\S+")
WHITESPACE_RE = re.compile(r"\s+")
# Keep letters, digits, apostrophes (for negation contractions like "don't")
NON_ALPHA_RE = re.compile(r"[^a-z0-9'\s]")


def rating_to_sentiment(rating: int) -> str:
    """Map a 1-5 star rating to a 3-class sentiment label.

    Mapping used throughout the project:
        1-2 stars -> Negative
        3   stars -> Neutral
        4-5 stars -> Positive
    """
    if rating <= 2:
        return "Negative"
    if rating == 3:
        return "Neutral"
    return "Positive"


def load_and_clean_data(
    path: str,
    text_col: str = "Review Text",
    rating_col: str = "Rating",
) -> pd.DataFrame:
    """Load the raw CSV and perform dataframe-level cleaning.

    Steps:
        - Drop rows with missing review text or missing rating.
        - Drop exact duplicate reviews.
        - Keep only valid ratings (1-5).
        - Drop reviews that are empty after stripping whitespace.
        - Create the `Sentiment` label column from the rating.

    Returns a cleaned dataframe with (at least) `text_col`, `rating_col`
    and a new `Sentiment` column.
    """
    df = pd.read_csv(path, index_col=0)

    df = df.dropna(subset=[text_col, rating_col]).copy()

    df[text_col] = df[text_col].astype(str).str.strip()
    df = df[df[text_col].str.len() > 0]

    df = df[df[rating_col].between(1, 5)]

    before = len(df)
    df = df.drop_duplicates(subset=[text_col])
    duplicates_removed = before - len(df)

    df["Sentiment"] = df[rating_col].apply(rating_to_sentiment)

    df = df.reset_index(drop=True)
    df.attrs["duplicates_removed"] = duplicates_removed
    return df


def clean_text(text: str, remove_stopwords: bool = True) -> str:
    """Clean and normalize a single review string.

    Pipeline:
        1. Lowercase
        2. Strip HTML tags
        3. Strip URLs
        4. Remove punctuation except apostrophes (keeps negations intact,
           e.g. "don't" stays as "don't" rather than becoming "dont")
        5. Collapse repeated whitespace
        6. Optionally remove stopwords (negation words are preserved)
    """
    text = str(text).lower()
    text = HTML_TAG_RE.sub(" ", text)
    text = URL_RE.sub(" ", text)
    text = NON_ALPHA_RE.sub(" ", text)
    text = WHITESPACE_RE.sub(" ", text).strip()

    if remove_stopwords:
        tokens = [tok for tok in text.split() if tok not in SENTIMENT_STOPWORDS]
        text = " ".join(tokens)

    return text


def preprocess_dataframe(
    df: pd.DataFrame,
    text_col: str = "Review Text",
    output_col: str = "clean_text",
    remove_stopwords: bool = True,
) -> pd.DataFrame:
    """Apply `clean_text` to an entire dataframe column and drop rows that
    become empty after cleaning."""
    df = df.copy()
    df[output_col] = df[text_col].apply(lambda t: clean_text(t, remove_stopwords))
    df = df[df[output_col].str.len() > 0].reset_index(drop=True)
    return df
