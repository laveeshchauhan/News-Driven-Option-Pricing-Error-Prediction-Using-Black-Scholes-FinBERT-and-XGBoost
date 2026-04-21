"""
sentiment_pipeline.py — Phase 2 Orchestrator: FinBERT Sentiment Analysis.

Steps performed by :func:`run_sentiment_pipeline`:
  1. Load / fetch news articles for RELIANCE (RSS feeds or NewsAPI)
  2. Run FinBERT inference on each headline (TextBlob/VADER as fallback)
  3. Aggregate article scores to daily sentiment (mean P(pos)−P(neg))
  4. Merge daily sentiment onto the Phase 1 option chain output (join on date)
  5. Export enriched DataFrame to ``outputs/results_with_sentiment.csv``
  6. Print a formatted Phase 2 summary to terminal
  7. Optionally generate sentiment timeline chart (``--plot``)

All monetary values are in ₹ (INR).
"""

from __future__ import annotations

import os
import warnings
from datetime import date, timedelta
from typing import List, Optional

import numpy as np
import pandas as pd

from config import (
    DEFAULT_OUTPUT_CSV,
    FINBERT_BATCH_SIZE,
    FINBERT_MODEL_NAME,
    NEWS_DATA_DIR,
    NEWS_KEYWORDS,
    NEWS_LOOKBACK_DAYS,
    NEWS_RSS_FEEDS,
    OUTPUT_DIR,
    RAW_NEWS_CACHE_PATH,
    SENTIMENT_CACHE_PATH,
    SENTIMENT_OUTPUT_CSV,
)
from src.news_fetcher import fetch_news, generate_sample_news
from src.sentiment import run_sentiment_scoring


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────


def _ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# Main pipeline function
# ──────────────────────────────────────────────────────────────────────────────


def run_sentiment_pipeline(
    phase1_csv: str = DEFAULT_OUTPUT_CSV,
    demo: bool = False,
    keywords: Optional[List[str]] = None,
    lookback_days: int = NEWS_LOOKBACK_DAYS,
    rss_feeds: Optional[List[str]] = None,
    newsapi_key: Optional[str] = None,
    model_name: str = FINBERT_MODEL_NAME,
    batch_size: int = FINBERT_BATCH_SIZE,
    output_csv: str = SENTIMENT_OUTPUT_CSV,
    use_cache: bool = True,
    plot: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run the full Phase 2 sentiment analysis pipeline.

    Parameters
    ----------
    phase1_csv   : Path to the Phase 1 results CSV
                   (``outputs/results.csv`` by default).  If the file does
                   not exist the pipeline continues without merging and
                   exports a sentiment-only CSV.
    demo         : If True, skip network calls and use synthetic sample news.
    keywords     : Keywords to search for in news articles.
    lookback_days: How many days back to fetch news.
    rss_feeds    : RSS feed URLs to use (defaults to config).
    newsapi_key  : Optional NewsAPI.org developer key; reads the environment
                   variable ``NEWSAPI_KEY`` if not provided.
    model_name   : HuggingFace model for sentiment inference.
    batch_size   : FinBERT inference batch size.
    output_csv   : Path for the enriched output CSV.
    use_cache    : Cache raw news and sentiment scores to CSV.
    plot         : Generate and save sentiment charts.
    verbose      : Print progress and summary to stdout.

    Returns
    -------
    pd.DataFrame — the enriched option chain with sentiment columns, or a
    daily sentiment DataFrame if the Phase 1 results CSV is unavailable.
    """
    keywords = keywords or NEWS_KEYWORDS
    rss_feeds = rss_feeds or NEWS_RSS_FEEDS

    # ── Step 1: Fetch / load news articles ────────────────────────────────
    if verbose:
        print("\n[Phase 2 — Step 1/6] Loading news articles...")

    if demo:
        if verbose:
            print("  Using synthetic sample news data (demo mode).")
        news_df = generate_sample_news()
    else:
        news_df = fetch_news(
            keywords=keywords,
            lookback_days=lookback_days,
            rss_feeds=rss_feeds,
            newsapi_key=newsapi_key,
            use_cache=use_cache,
            cache_path=RAW_NEWS_CACHE_PATH,
            verbose=verbose,
        )

    if news_df.empty:
        warnings.warn(
            "No news articles available. Cannot run sentiment pipeline.",
            RuntimeWarning,
            stacklevel=2,
        )
        return pd.DataFrame()

    if verbose:
        print(f"  Articles loaded: {len(news_df)}")

    # ── Step 2: FinBERT inference ──────────────────────────────────────────
    if verbose:
        print("[Phase 2 — Step 2/6] Running sentiment inference...")

    scored_df, daily_df = run_sentiment_scoring(
        news_df=news_df,
        model_name=model_name,
        batch_size=batch_size,
        use_cache=use_cache,
        cache_path=SENTIMENT_CACHE_PATH,
        verbose=verbose,
    )

    if daily_df.empty:
        warnings.warn(
            "Sentiment aggregation produced no results.",
            RuntimeWarning,
            stacklevel=2,
        )
        return pd.DataFrame()

    if verbose:
        print(f"  Daily sentiment computed for {len(daily_df)} trading days.")

    # ── Step 3: Load Phase 1 results ───────────────────────────────────────
    if verbose:
        print("[Phase 2 — Step 3/6] Loading Phase 1 results for merge...")

    phase1_df: Optional[pd.DataFrame] = None
    if os.path.exists(phase1_csv):
        try:
            phase1_df = pd.read_csv(phase1_csv, parse_dates=["date"])
            phase1_df["date"] = pd.to_datetime(phase1_df["date"]).dt.normalize()
            if verbose:
                print(f"  Phase 1 data loaded: {len(phase1_df)} rows from {phase1_csv}")
        except Exception as exc:
            warnings.warn(
                f"Could not load Phase 1 results from {phase1_csv}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
    else:
        warnings.warn(
            f"Phase 1 results CSV not found at {phase1_csv}. "
            "Exporting daily sentiment only.",
            RuntimeWarning,
            stacklevel=2,
        )

    # ── Step 4: Merge sentiment onto option chain ──────────────────────────
    if verbose:
        print("[Phase 2 — Step 4/6] Merging sentiment with option chain...")

    daily_df["date"] = pd.to_datetime(daily_df["date"]).dt.normalize()

    if phase1_df is not None:
        enriched_df = phase1_df.merge(daily_df, on="date", how="left")
        # Fill missing sentiment for dates with no news
        for col in (
            "daily_sentiment_score",
            "daily_pos_mean",
            "daily_neg_mean",
            "daily_neu_mean",
        ):
            if col in enriched_df.columns:
                enriched_df[col] = enriched_df[col].fillna(0.0)
        if "daily_article_count" in enriched_df.columns:
            enriched_df["daily_article_count"] = (
                enriched_df["daily_article_count"].fillna(0).astype(int)
            )
        if "daily_sentiment_label" in enriched_df.columns:
            enriched_df["daily_sentiment_label"] = enriched_df[
                "daily_sentiment_label"
            ].fillna("Neutral")
    else:
        # No Phase 1 data — export sentiment only
        enriched_df = daily_df

    if verbose:
        n_merged = enriched_df["daily_sentiment_score"].notna().sum() if "daily_sentiment_score" in enriched_df.columns else 0
        print(f"  Rows with sentiment data: {n_merged}")

    # ── Step 5: Print summary ──────────────────────────────────────────────
    if verbose:
        print("[Phase 2 — Step 5/6] Generating Phase 2 summary...")
        _print_summary(news_df, scored_df, daily_df, enriched_df)

    # ── Step 6: Export CSV ─────────────────────────────────────────────────
    if verbose:
        print(f"[Phase 2 — Step 6/6] Exporting results to: {output_csv}")

    _ensure_dir(output_csv)
    enriched_df.to_csv(output_csv, index=False)
    if verbose:
        print(f"  Saved {len(enriched_df)} rows → {output_csv}")

    # ── Optional: charts ──────────────────────────────────────────────────
    if plot:
        _generate_plots(daily_df, scored_df, output_dir=OUTPUT_DIR, verbose=verbose)

    return enriched_df


# ──────────────────────────────────────────────────────────────────────────────
# Summary printing
# ──────────────────────────────────────────────────────────────────────────────


def _print_summary(
    news_df: pd.DataFrame,
    scored_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    enriched_df: pd.DataFrame,
) -> None:
    sep = "=" * 65

    total_articles = len(news_df)
    n_days = len(daily_df)

    # Label distribution
    label_counts = (
        daily_df["daily_sentiment_label"].value_counts()
        if not daily_df.empty
        else pd.Series(dtype=int)
    )
    n_pos = label_counts.get("Positive", 0)
    n_neg = label_counts.get("Negative", 0)
    n_neu = label_counts.get("Neutral", 0)

    mean_score = daily_df["daily_sentiment_score"].mean() if not daily_df.empty else 0.0
    std_score = daily_df["daily_sentiment_score"].std() if not daily_df.empty else 0.0

    print(f"\n{sep}")
    print("  PHASE 2 — FINBERT SENTIMENT ANALYSIS RESULTS")
    print(f"  RELIANCE Industries (RELIANCE.NS) | NSE India")
    print(sep)
    print(f"  Total news articles processed    : {total_articles:>8,}")
    print(f"  Unique trading days              : {n_days:>8,}")
    print(sep)
    print(f"  Mean daily sentiment score       : {mean_score:>10.4f}")
    print(f"  Std  daily sentiment score       : {std_score:>10.4f}")
    print(sep)
    print(f"  Positive sentiment days          : {n_pos:>8,}")
    print(f"  Negative sentiment days          : {n_neg:>8,}")
    print(f"  Neutral  sentiment days          : {n_neu:>8,}")
    print(sep)

    # Most positive / negative days
    if not daily_df.empty:
        top_pos = daily_df.nlargest(3, "daily_sentiment_score")[
            ["date", "daily_sentiment_score", "daily_article_count"]
        ]
        top_neg = daily_df.nsmallest(3, "daily_sentiment_score")[
            ["date", "daily_sentiment_score", "daily_article_count"]
        ]
        print("\n  Top 3 most POSITIVE days:")
        print(
            top_pos.to_string(
                index=False,
                float_format="{:.4f}".format,
            )
        )
        print("\n  Top 3 most NEGATIVE days:")
        print(
            top_neg.to_string(
                index=False,
                float_format="{:.4f}".format,
            )
        )
    print(f"\n{sep}\n")


# ──────────────────────────────────────────────────────────────────────────────
# Visualisation
# ──────────────────────────────────────────────────────────────────────────────


def _generate_plots(
    daily_df: pd.DataFrame,
    scored_df: pd.DataFrame,
    output_dir: str,
    verbose: bool = True,
) -> None:
    """Generate and save Phase 2 sentiment visualisation charts."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        import seaborn as sns
    except ImportError:
        warnings.warn(
            "matplotlib / seaborn not installed. Skipping Phase 2 plots.",
            RuntimeWarning,
        )
        return

    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # ── Plot 1: Daily sentiment score timeline ────────────────────────────
    if not daily_df.empty and "daily_sentiment_score" in daily_df.columns:
        fig, ax = plt.subplots(figsize=(14, 5))
        daily_sorted = daily_df.sort_values("date")

        # Colour bars by label
        colors = daily_sorted["daily_sentiment_label"].map(
            {"Positive": "steelblue", "Negative": "tomato", "Neutral": "grey"}
        ).fillna("grey")

        ax.bar(
            daily_sorted["date"],
            daily_sorted["daily_sentiment_score"],
            color=colors,
            alpha=0.75,
            label="_nolegend_",
        )
        ax.axhline(0, color="black", linewidth=1.2, linestyle="--")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b"))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
        plt.xticks(rotation=45)
        ax.set_title("Daily Sentiment Score — RELIANCE News (FinBERT)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Sentiment Score (Positive − Negative)")

        from matplotlib.patches import Patch
        legend_handles = [
            Patch(color="steelblue", label="Positive"),
            Patch(color="tomato", label="Negative"),
            Patch(color="grey", label="Neutral"),
        ]
        ax.legend(handles=legend_handles)
        plt.tight_layout()
        path1 = os.path.join(output_dir, "sentiment_timeline.png")
        plt.savefig(path1, dpi=150, bbox_inches="tight")
        plt.close()
        if verbose:
            print(f"  Saved chart: {path1}")

    # ── Plot 2: Sentiment score distribution ─────────────────────────────
    if not daily_df.empty and "daily_sentiment_score" in daily_df.columns:
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.histplot(
            daily_df["daily_sentiment_score"],
            bins=20,
            kde=True,
            ax=ax,
            color="steelblue",
            alpha=0.65,
        )
        ax.axvline(0, color="red", linewidth=1.5, linestyle="--", label="Score = 0")
        mean_score = daily_df["daily_sentiment_score"].mean()
        ax.axvline(
            mean_score,
            color="darkblue",
            linewidth=1.5,
            linestyle="-",
            label=f"Mean = {mean_score:.3f}",
        )
        ax.set_title("Distribution of Daily Sentiment Score")
        ax.set_xlabel("Daily Sentiment Score")
        ax.set_ylabel("Count (trading days)")
        ax.legend()
        plt.tight_layout()
        path2 = os.path.join(output_dir, "sentiment_distribution.png")
        plt.savefig(path2, dpi=150, bbox_inches="tight")
        plt.close()
        if verbose:
            print(f"  Saved chart: {path2}")

    # ── Plot 3: Article count per day ────────────────────────────────────
    if not daily_df.empty and "daily_article_count" in daily_df.columns:
        fig, ax = plt.subplots(figsize=(14, 4))
        daily_sorted = daily_df.sort_values("date")
        ax.bar(
            daily_sorted["date"],
            daily_sorted["daily_article_count"],
            color="teal",
            alpha=0.65,
        )
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b"))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
        plt.xticks(rotation=45)
        ax.set_title("Number of RELIANCE News Articles per Day")
        ax.set_xlabel("Date")
        ax.set_ylabel("Article Count")
        plt.tight_layout()
        path3 = os.path.join(output_dir, "sentiment_article_count.png")
        plt.savefig(path3, dpi=150, bbox_inches="tight")
        plt.close()
        if verbose:
            print(f"  Saved chart: {path3}")
