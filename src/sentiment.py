"""
sentiment.py — FinBERT-based Financial Sentiment Scoring for RELIANCE News.

Primary model
-------------
  **FinBERT** (``ProsusAI/finbert``) — a BERT model fine-tuned on ~10 000
  financial sentences from Reuters, Bloomberg, and the Financial PhraseBank.
  It produces three softmax probabilities for each input text:

      P(positive), P(negative), P(neutral)

  The signed *sentiment score* used as an ML feature in Phase 3 is:

      score = P(positive) − P(negative)   ∈ [−1, +1]

  A score near +1 → strong positive news; near −1 → strong negative news.

Fallback chain
--------------
  If the HuggingFace ``transformers`` / ``torch`` stack is unavailable the
  module degrades gracefully:

  1. **TextBlob** — uses NLTK / pattern-based polarity (−1 to +1).
  2. **VADER** (``nltk.sentiment.vader``) — rule-based financial polarity.
  3. **Neutral** — returns a score of 0.0 for all texts if all else fails.

Daily aggregation
-----------------
  Individual article scores are aggregated per calendar day to produce:

      daily_sentiment_score  : mean(P(pos) − P(neg)) across all day's articles
      daily_sentiment_label  : "Positive" / "Negative" / "Neutral" based on
                               the mean score vs. SENTIMENT_NEUTRAL_BAND
      daily_article_count    : number of articles that day

Caching
-------
  Inference is slow on CPU (~0.2–0.5 s/article). Results are written to
  ``data/news/sentiment_scores.csv`` so subsequent calls are instant.
"""

from __future__ import annotations

import os
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import (
    FINBERT_BATCH_SIZE,
    FINBERT_MODEL_NAME,
    NEWS_DATA_DIR,
    SENTIMENT_CACHE_PATH,
    SENTIMENT_NEUTRAL_BAND,
)


# ──────────────────────────────────────────────────────────────────────────────
# Backend detection
# ──────────────────────────────────────────────────────────────────────────────


def _finbert_available() -> bool:
    try:
        import transformers  # noqa: F401
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def _textblob_available() -> bool:
    try:
        from textblob import TextBlob  # noqa: F401
        return True
    except ImportError:
        return False


def _vader_available() -> bool:
    try:
        from nltk.sentiment.vader import SentimentIntensityAnalyzer  # noqa: F401
        return True
    except (ImportError, LookupError):
        return False


# ──────────────────────────────────────────────────────────────────────────────
# FinBERT backend
# ──────────────────────────────────────────────────────────────────────────────


def _load_finbert(model_name: str = FINBERT_MODEL_NAME):
    """
    Load the FinBERT tokenizer and model.

    Returns
    -------
    (tokenizer, model, device_str)
    """
    from transformers import AutoTokenizer, AutoModelForSequenceClassification  # type: ignore
    import torch  # type: ignore

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    return tokenizer, model, device


def _score_batch_finbert(
    texts: List[str],
    tokenizer,
    model,
    device: str,
    batch_size: int = FINBERT_BATCH_SIZE,
) -> List[Dict[str, float]]:
    """
    Run FinBERT inference on a list of texts in batches.

    Returns
    -------
    List of dicts ``{"positive": p, "negative": n, "neutral": u}``
    for each input text.
    """
    import torch  # type: ignore
    import torch.nn.functional as F  # type: ignore

    # FinBERT label order from the model config
    id2label: Dict[int, str] = model.config.id2label

    results: List[Dict[str, float]] = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs).logits

        probs = F.softmax(logits, dim=-1).cpu().numpy()

        for row in probs:
            scores = {id2label[i].lower(): float(row[i]) for i in range(len(row))}
            # Ensure all three keys exist
            for key in ("positive", "negative", "neutral"):
                scores.setdefault(key, 0.0)
            results.append(scores)

    return results


# ──────────────────────────────────────────────────────────────────────────────
# TextBlob fallback backend
# ──────────────────────────────────────────────────────────────────────────────


def _score_textblob(texts: List[str]) -> List[Dict[str, float]]:
    """
    Use TextBlob polarity as a lightweight sentiment fallback.

    TextBlob returns ``polarity ∈ [−1, 1]``.  We map it to the same schema:

      * polarity > BAND  → positive: |polarity|, negative: 0
      * polarity < -BAND → negative: |polarity|, positive: 0
      * otherwise        → neutral: 1 − |polarity|
    """
    from textblob import TextBlob  # type: ignore

    results = []
    for text in texts:
        polarity: float = TextBlob(text).sentiment.polarity
        pos = max(polarity, 0.0)
        neg = max(-polarity, 0.0)
        neu = 1.0 - pos - neg
        results.append({"positive": pos, "negative": neg, "neutral": max(neu, 0.0)})
    return results


# ──────────────────────────────────────────────────────────────────────────────
# VADER fallback backend
# ──────────────────────────────────────────────────────────────────────────────


def _score_vader(texts: List[str]) -> List[Dict[str, float]]:
    """
    Use VADER compound score as a fallback.

    VADER ``compound ∈ [−1, 1]``.  Mapping is identical to TextBlob fallback.
    """
    try:
        from nltk.sentiment.vader import SentimentIntensityAnalyzer  # type: ignore
        import nltk
        try:
            nltk.data.find("sentiment/vader_lexicon.zip")
        except LookupError:
            nltk.download("vader_lexicon", quiet=True)
    except ImportError:
        return [{"positive": 0.0, "negative": 0.0, "neutral": 1.0}] * len(texts)

    sia = SentimentIntensityAnalyzer()
    results = []
    for text in texts:
        compound: float = sia.polarity_scores(text)["compound"]
        pos = max(compound, 0.0)
        neg = max(-compound, 0.0)
        neu = 1.0 - pos - neg
        results.append({"positive": pos, "negative": neg, "neutral": max(neu, 0.0)})
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Dispatcher
# ──────────────────────────────────────────────────────────────────────────────


def score_articles(
    texts: List[str],
    model_name: str = FINBERT_MODEL_NAME,
    batch_size: int = FINBERT_BATCH_SIZE,
    verbose: bool = True,
) -> List[Dict[str, float]]:
    """
    Score a list of text strings with the best available sentiment backend.

    Tries FinBERT first, then TextBlob, then VADER, then returns neutral zeros.

    Parameters
    ----------
    texts      : List of headline (or headline + description) strings.
    model_name : HuggingFace model identifier (default: ``ProsusAI/finbert``).
    batch_size : Batch size for FinBERT GPU/CPU inference.
    verbose    : If True, show a tqdm progress bar and backend info.

    Returns
    -------
    List of dicts ``{"positive": float, "negative": float, "neutral": float}``
    — one per input text.
    """
    if not texts:
        return []

    if _finbert_available():
        if verbose:
            print(f"[sentiment] Loading FinBERT model: {model_name} ...")
        try:
            tokenizer, model, device = _load_finbert(model_name)
            if verbose:
                print(f"[sentiment] Running FinBERT inference on {len(texts)} texts "
                      f"(device={device}, batch_size={batch_size})...")
            # Wrap with tqdm for visibility
            all_results: List[Dict[str, float]] = []
            for start in tqdm(
                range(0, len(texts), batch_size),
                desc="FinBERT",
                disable=not verbose,
            ):
                batch = texts[start : start + batch_size]
                all_results.extend(
                    _score_batch_finbert(batch, tokenizer, model, device, batch_size)
                )
            return all_results
        except Exception as exc:
            warnings.warn(
                f"FinBERT inference failed ({exc}). Falling back to TextBlob.",
                RuntimeWarning,
                stacklevel=2,
            )

    if _textblob_available():
        if verbose:
            print("[sentiment] Using TextBlob fallback...")
        return _score_textblob(texts)

    if _vader_available():
        if verbose:
            print("[sentiment] Using VADER fallback...")
        return _score_vader(texts)

    warnings.warn(
        "No sentiment backend available (FinBERT, TextBlob, VADER). "
        "Returning neutral scores for all texts. "
        "Install at least one: pip install transformers torch textblob",
        RuntimeWarning,
        stacklevel=2,
    )
    return [{"positive": 0.0, "negative": 0.0, "neutral": 1.0}] * len(texts)


# ──────────────────────────────────────────────────────────────────────────────
# Article-level scoring
# ──────────────────────────────────────────────────────────────────────────────


def score_dataframe(
    df: pd.DataFrame,
    text_col: str = "headline",
    desc_col: Optional[str] = "description",
    model_name: str = FINBERT_MODEL_NAME,
    batch_size: int = FINBERT_BATCH_SIZE,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Add per-article sentiment columns to a news DataFrame.

    The input text for each article is ``headline + " " + description``
    (if *desc_col* is present and non-empty), truncated to 512 characters.

    Parameters
    ----------
    df         : DataFrame with at least a *text_col* column.
    text_col   : Column containing the main article text (headline).
    desc_col   : Optional column with secondary text (description / summary).
    model_name : FinBERT model identifier.
    batch_size : Batch size for inference.
    verbose    : Print progress.

    Returns
    -------
    df with three new columns added in-place:
      ``sentiment_positive``, ``sentiment_negative``, ``sentiment_neutral``,
      ``sentiment_score``  (= positive − negative),
      ``sentiment_label``  ("Positive" / "Negative" / "Neutral")
    """
    if df.empty:
        for col in (
            "sentiment_positive", "sentiment_negative", "sentiment_neutral",
            "sentiment_score", "sentiment_label",
        ):
            df[col] = pd.Series(dtype=float if col != "sentiment_label" else str)
        return df

    # Build combined texts
    texts: List[str] = []
    for _, row in df.iterrows():
        headline = str(row.get(text_col, "") or "")
        desc = ""
        if desc_col and desc_col in df.columns:
            desc = str(row.get(desc_col, "") or "")
        combined = (headline + " " + desc).strip()[:512]
        texts.append(combined)

    scores = score_articles(
        texts,
        model_name=model_name,
        batch_size=batch_size,
        verbose=verbose,
    )

    result = df.copy()
    result["sentiment_positive"] = [s["positive"] for s in scores]
    result["sentiment_negative"] = [s["negative"] for s in scores]
    result["sentiment_neutral"] = [s["neutral"] for s in scores]
    result["sentiment_score"] = (
        result["sentiment_positive"] - result["sentiment_negative"]
    )
    result["sentiment_label"] = result["sentiment_score"].apply(_label_from_score)
    return result


def _label_from_score(score: float, band: float = SENTIMENT_NEUTRAL_BAND) -> str:
    """Map a sentiment score to a human-readable label."""
    if score > band:
        return "Positive"
    if score < -band:
        return "Negative"
    return "Neutral"


# ──────────────────────────────────────────────────────────────────────────────
# Daily aggregation
# ──────────────────────────────────────────────────────────────────────────────


def aggregate_daily_sentiment(
    scored_df: pd.DataFrame,
    date_col: str = "date",
) -> pd.DataFrame:
    """
    Aggregate per-article sentiment scores to a daily summary.

    Parameters
    ----------
    scored_df : DataFrame returned by :func:`score_dataframe` (must have
                ``sentiment_score``, ``sentiment_positive``,
                ``sentiment_negative`` columns).
    date_col  : Column that holds the publication date (datetime or date).

    Returns
    -------
    pd.DataFrame with one row per calendar day:

      date                  : datetime (date only, time zeroed out)
      daily_sentiment_score : mean sentiment score for the day
      daily_sentiment_label : "Positive" / "Negative" / "Neutral"
      daily_article_count   : number of articles on that day
      daily_pos_mean        : mean P(positive)
      daily_neg_mean        : mean P(negative)
      daily_neu_mean        : mean P(neutral)
    """
    if scored_df.empty or "sentiment_score" not in scored_df.columns:
        return pd.DataFrame(
            columns=[
                "date",
                "daily_sentiment_score",
                "daily_sentiment_label",
                "daily_article_count",
                "daily_pos_mean",
                "daily_neg_mean",
                "daily_neu_mean",
            ]
        )

    df = scored_df.copy()
    df[date_col] = pd.to_datetime(df[date_col]).dt.normalize()

    daily = (
        df.groupby(date_col)
        .agg(
            daily_sentiment_score=("sentiment_score", "mean"),
            daily_pos_mean=("sentiment_positive", "mean"),
            daily_neg_mean=("sentiment_negative", "mean"),
            daily_neu_mean=("sentiment_neutral", "mean"),
            daily_article_count=(date_col, "count"),
        )
        .reset_index()
        .rename(columns={date_col: "date"})
    )

    daily["daily_sentiment_label"] = daily["daily_sentiment_score"].apply(
        _label_from_score
    )

    return daily.sort_values("date").reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────────
# Caching helpers
# ──────────────────────────────────────────────────────────────────────────────


def load_sentiment_cache(cache_path: str = SENTIMENT_CACHE_PATH) -> pd.DataFrame:
    """Load cached daily sentiment scores.  Returns empty DataFrame if absent."""
    if not os.path.exists(cache_path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(cache_path, parse_dates=["date"])
        df["date"] = pd.to_datetime(df["date"])
        return df
    except Exception:
        return pd.DataFrame()


def save_sentiment_cache(
    daily_df: pd.DataFrame,
    cache_path: str = SENTIMENT_CACHE_PATH,
) -> None:
    """Persist daily sentiment scores to CSV (append and de-duplicate)."""
    os.makedirs(NEWS_DATA_DIR, exist_ok=True)
    if os.path.exists(cache_path):
        existing = load_sentiment_cache(cache_path)
        if not existing.empty:
            combined = pd.concat([existing, daily_df], ignore_index=True)
            combined["date"] = pd.to_datetime(combined["date"])
            combined = (
                combined.sort_values("date")
                .drop_duplicates(subset=["date"], keep="last")
                .reset_index(drop=True)
            )
            combined.to_csv(cache_path, index=False)
            return
    daily_df.to_csv(cache_path, index=False)


# ──────────────────────────────────────────────────────────────────────────────
# Convenience wrapper: score + aggregate in one call
# ──────────────────────────────────────────────────────────────────────────────


def run_sentiment_scoring(
    news_df: pd.DataFrame,
    model_name: str = FINBERT_MODEL_NAME,
    batch_size: int = FINBERT_BATCH_SIZE,
    use_cache: bool = True,
    cache_path: str = SENTIMENT_CACHE_PATH,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Score all articles and aggregate to daily sentiment.

    This is the top-level function called by :mod:`src.sentiment_pipeline`.

    Parameters
    ----------
    news_df    : Raw news DataFrame (from :func:`~src.news_fetcher.fetch_news`).
    model_name : FinBERT HuggingFace model name.
    batch_size : Inference batch size.
    use_cache  : If True, try to load existing scores from *cache_path* and
                 only score new (unseen) dates.
    cache_path : Path to daily sentiment cache CSV.
    verbose    : Print progress messages.

    Returns
    -------
    (scored_df, daily_df)
      * *scored_df* — article-level DataFrame with sentiment columns added.
      * *daily_df*  — one row per day with aggregated sentiment stats.
    """
    if news_df.empty:
        warnings.warn("Empty news DataFrame passed to run_sentiment_scoring.", stacklevel=2)
        return news_df, pd.DataFrame()

    # ── Try to use existing cache ─────────────────────────────────────────
    cached_daily = load_sentiment_cache(cache_path) if use_cache else pd.DataFrame()

    if not cached_daily.empty:
        cached_dates = set(pd.to_datetime(cached_daily["date"]).dt.date)
        news_df["date"] = pd.to_datetime(news_df["date"])
        new_articles = news_df[
            ~news_df["date"].dt.date.isin(cached_dates)
        ]
        if new_articles.empty:
            if verbose:
                print(
                    f"[sentiment] All dates already cached "
                    f"({len(cached_daily)} days). Returning cache."
                )
            return news_df, cached_daily
    else:
        new_articles = news_df

    if verbose:
        print(f"[sentiment] Scoring {len(new_articles)} new articles...")

    # ── Score articles ────────────────────────────────────────────────────
    scored = score_dataframe(
        new_articles,
        model_name=model_name,
        batch_size=batch_size,
        verbose=verbose,
    )

    # ── Daily aggregation ─────────────────────────────────────────────────
    new_daily = aggregate_daily_sentiment(scored)

    # Merge with cached data
    if not cached_daily.empty:
        all_daily = pd.concat([cached_daily, new_daily], ignore_index=True)
        all_daily["date"] = pd.to_datetime(all_daily["date"])
        all_daily = (
            all_daily.sort_values("date")
            .drop_duplicates(subset=["date"], keep="last")
            .reset_index(drop=True)
        )
    else:
        all_daily = new_daily

    # ── Save cache ────────────────────────────────────────────────────────
    if use_cache:
        save_sentiment_cache(all_daily, cache_path)
        if verbose:
            print(f"[sentiment] Sentiment cache saved: {cache_path}")

    return scored, all_daily
