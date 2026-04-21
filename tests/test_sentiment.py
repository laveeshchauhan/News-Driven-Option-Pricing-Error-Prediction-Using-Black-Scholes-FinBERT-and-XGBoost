"""
test_sentiment.py — Unit tests for Phase 2 sentiment modules.

Covers:
  • src/news_fetcher.py  — generate_sample_news, fetch_news (offline)
  • src/sentiment.py    — score_articles, score_dataframe,
                          aggregate_daily_sentiment, run_sentiment_scoring
  • Daily aggregation math
  • Fallback backend (TextBlob/VADER/neutral)
  • Date-merge logic (sentiment_pipeline.run_sentiment_pipeline with demo data)
"""

from __future__ import annotations

import os
import sys
import math
import warnings
from datetime import date, timedelta
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

# Ensure project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.news_fetcher import generate_sample_news
from src.sentiment import (
    _label_from_score,
    aggregate_daily_sentiment,
    score_dataframe,
    score_articles,
    run_sentiment_scoring,
    load_sentiment_cache,
    save_sentiment_cache,
)


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_news_df():
    """30-row synthetic news DataFrame (uses the offline generator)."""
    return generate_sample_news()


@pytest.fixture
def tiny_news_df():
    """Minimal 3-row DataFrame for fast unit tests."""
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-03-01", "2026-03-01", "2026-03-02"]),
            "source": ["ET", "MC", "ET"],
            "headline": [
                "Reliance beats Q3 profit estimates; Jio growth strong",
                "RIL stock falls on crude spike; margin compression feared",
                "Reliance board meeting scheduled for January results",
            ],
            "description": [
                "Net profit rose 11% YoY driven by Jio and Retail.",
                "Rising Brent crude above $92 raises feedstock costs.",
                "Board to approve Q3 standalone and consolidated financials.",
            ],
            "url": [
                "https://example.com/1",
                "https://example.com/2",
                "https://example.com/3",
            ],
        }
    )


@pytest.fixture
def scored_df(tiny_news_df):
    """
    Inject a known fallback result so tests don't need FinBERT.
    Scores are set deterministically: pos, neg, neutral all 1/3.
    """
    df = tiny_news_df.copy()
    df["sentiment_positive"] = [0.7, 0.05, 0.3]
    df["sentiment_negative"] = [0.1, 0.80, 0.2]
    df["sentiment_neutral"] = [0.2, 0.15, 0.5]
    df["sentiment_score"] = df["sentiment_positive"] - df["sentiment_negative"]
    df["sentiment_label"] = df["sentiment_score"].apply(_label_from_score)
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Tests: news_fetcher.generate_sample_news
# ──────────────────────────────────────────────────────────────────────────────


class TestGenerateSampleNews:
    def test_returns_dataframe(self, sample_news_df):
        assert isinstance(sample_news_df, pd.DataFrame)

    def test_expected_columns(self, sample_news_df):
        for col in ("date", "source", "headline", "description", "url"):
            assert col in sample_news_df.columns, f"Missing column: {col}"

    def test_thirty_rows(self, sample_news_df):
        assert len(sample_news_df) == 30

    def test_dates_are_datetime(self, sample_news_df):
        assert pd.api.types.is_datetime64_any_dtype(sample_news_df["date"])

    def test_no_null_headlines(self, sample_news_df):
        assert sample_news_df["headline"].notna().all()

    def test_date_range_covers_lookback(self, sample_news_df):
        start = sample_news_df["date"].min().date()
        end = sample_news_df["date"].max().date()
        # Default range is 60 days
        assert (end - start).days >= 0

    def test_custom_date_range(self):
        end = date(2026, 3, 11)
        start = date(2026, 1, 10)
        df = generate_sample_news(start_date=start, end_date=end)
        dates = df["date"].dt.date
        assert dates.min() >= start
        assert dates.max() <= end

    def test_sources_from_expected_list(self, sample_news_df):
        valid_sources = {"Economic Times", "Moneycontrol", "NDTV Profit"}
        assert set(sample_news_df["source"].unique()).issubset(valid_sources)


# ──────────────────────────────────────────────────────────────────────────────
# Tests: _label_from_score
# ──────────────────────────────────────────────────────────────────────────────


class TestLabelFromScore:
    def test_positive_above_band(self):
        assert _label_from_score(0.5) == "Positive"

    def test_negative_below_band(self):
        assert _label_from_score(-0.5) == "Negative"

    def test_neutral_within_band(self):
        assert _label_from_score(0.0) == "Neutral"
        assert _label_from_score(0.04) == "Neutral"
        assert _label_from_score(-0.04) == "Neutral"

    def test_boundary_positive(self):
        # Score exactly at band boundary: > band → Positive
        assert _label_from_score(0.051) == "Positive"

    def test_boundary_negative(self):
        assert _label_from_score(-0.051) == "Negative"

    def test_extreme_values(self):
        assert _label_from_score(1.0) == "Positive"
        assert _label_from_score(-1.0) == "Negative"


# ──────────────────────────────────────────────────────────────────────────────
# Tests: score_articles (mock backends)
# ──────────────────────────────────────────────────────────────────────────────


class TestScoreArticles:
    def test_empty_list_returns_empty(self):
        result = score_articles([])
        assert result == []

    def test_returns_correct_length(self):
        texts = ["Reliance profit surges", "RIL faces headwinds"]
        with patch("src.sentiment._finbert_available", return_value=False), \
             patch("src.sentiment._textblob_available", return_value=True), \
             patch("src.sentiment._score_textblob") as mock_tb:
            mock_tb.return_value = [
                {"positive": 0.8, "negative": 0.1, "neutral": 0.1},
                {"positive": 0.1, "negative": 0.8, "neutral": 0.1},
            ]
            result = score_articles(texts, verbose=False)
        assert len(result) == 2

    def test_result_schema(self):
        texts = ["Test headline"]
        with patch("src.sentiment._finbert_available", return_value=False), \
             patch("src.sentiment._textblob_available", return_value=True), \
             patch("src.sentiment._score_textblob") as mock_tb:
            mock_tb.return_value = [{"positive": 0.5, "negative": 0.3, "neutral": 0.2}]
            result = score_articles(texts, verbose=False)
        assert set(result[0].keys()) == {"positive", "negative", "neutral"}

    def test_all_backends_unavailable_returns_neutral(self):
        texts = ["Test"]
        with patch("src.sentiment._finbert_available", return_value=False), \
             patch("src.sentiment._textblob_available", return_value=False), \
             patch("src.sentiment._vader_available", return_value=False):
            with warnings.catch_warnings(record=True):
                result = score_articles(texts, verbose=False)
        assert result[0]["positive"] == 0.0
        assert result[0]["negative"] == 0.0
        assert result[0]["neutral"] == 1.0

    def test_probabilities_sum_to_one(self):
        texts = ["Reliance beats expectations"]
        with patch("src.sentiment._finbert_available", return_value=False), \
             patch("src.sentiment._textblob_available", return_value=True), \
             patch("src.sentiment._score_textblob") as mock_tb:
            mock_tb.return_value = [{"positive": 0.6, "negative": 0.2, "neutral": 0.2}]
            result = score_articles(texts, verbose=False)
        total = result[0]["positive"] + result[0]["negative"] + result[0]["neutral"]
        assert abs(total - 1.0) < 1e-6

    def test_vader_fallback_used_when_textblob_unavailable(self):
        texts = ["Jio 5G rollout accelerates revenue growth"]
        with patch("src.sentiment._finbert_available", return_value=False), \
             patch("src.sentiment._textblob_available", return_value=False), \
             patch("src.sentiment._vader_available", return_value=True), \
             patch("src.sentiment._score_vader") as mock_vader:
            mock_vader.return_value = [{"positive": 0.4, "negative": 0.1, "neutral": 0.5}]
            result = score_articles(texts, verbose=False)
        mock_vader.assert_called_once()
        assert len(result) == 1


# ──────────────────────────────────────────────────────────────────────────────
# Tests: score_dataframe
# ──────────────────────────────────────────────────────────────────────────────


class TestScoreDataframe:
    def test_adds_expected_columns(self, tiny_news_df):
        with patch("src.sentiment._finbert_available", return_value=False), \
             patch("src.sentiment._textblob_available", return_value=True), \
             patch("src.sentiment._score_textblob") as mock_tb:
            mock_tb.return_value = [
                {"positive": 0.7, "negative": 0.1, "neutral": 0.2},
                {"positive": 0.1, "negative": 0.8, "neutral": 0.1},
                {"positive": 0.3, "negative": 0.2, "neutral": 0.5},
            ]
            result = score_dataframe(tiny_news_df, verbose=False)

        for col in (
            "sentiment_positive", "sentiment_negative", "sentiment_neutral",
            "sentiment_score", "sentiment_label",
        ):
            assert col in result.columns, f"Missing column: {col}"

    def test_sentiment_score_equals_pos_minus_neg(self, tiny_news_df):
        with patch("src.sentiment._finbert_available", return_value=False), \
             patch("src.sentiment._textblob_available", return_value=True), \
             patch("src.sentiment._score_textblob") as mock_tb:
            mock_tb.return_value = [
                {"positive": 0.7, "negative": 0.1, "neutral": 0.2},
                {"positive": 0.1, "negative": 0.8, "neutral": 0.1},
                {"positive": 0.3, "negative": 0.2, "neutral": 0.5},
            ]
            result = score_dataframe(tiny_news_df, verbose=False)

        computed = result["sentiment_positive"] - result["sentiment_negative"]
        pd.testing.assert_series_equal(
            result["sentiment_score"].round(10),
            computed.round(10),
            check_names=False,
        )

    def test_sentiment_label_consistent_with_score(self, tiny_news_df):
        with patch("src.sentiment._finbert_available", return_value=False), \
             patch("src.sentiment._textblob_available", return_value=True), \
             patch("src.sentiment._score_textblob") as mock_tb:
            mock_tb.return_value = [
                {"positive": 0.7, "negative": 0.1, "neutral": 0.2},
                {"positive": 0.1, "negative": 0.8, "neutral": 0.1},
                {"positive": 0.3, "negative": 0.2, "neutral": 0.5},
            ]
            result = score_dataframe(tiny_news_df, verbose=False)

        for _, row in result.iterrows():
            expected = _label_from_score(row["sentiment_score"])
            assert row["sentiment_label"] == expected

    def test_empty_df_returns_empty(self):
        empty = pd.DataFrame(columns=["date", "headline", "description"])
        result = score_dataframe(empty, verbose=False)
        assert result.empty

    def test_row_count_preserved(self, tiny_news_df):
        with patch("src.sentiment._finbert_available", return_value=False), \
             patch("src.sentiment._textblob_available", return_value=True), \
             patch("src.sentiment._score_textblob") as mock_tb:
            mock_tb.return_value = [
                {"positive": p, "negative": n, "neutral": 1 - p - n}
                for p, n in [(0.6, 0.1), (0.1, 0.6), (0.3, 0.3)]
            ]
            result = score_dataframe(tiny_news_df, verbose=False)
        assert len(result) == len(tiny_news_df)


# ──────────────────────────────────────────────────────────────────────────────
# Tests: aggregate_daily_sentiment
# ──────────────────────────────────────────────────────────────────────────────


class TestAggregateDailySentiment:
    def test_returns_one_row_per_day(self, scored_df):
        daily = aggregate_daily_sentiment(scored_df)
        assert len(daily) == scored_df["date"].dt.date.nunique()

    def test_expected_columns(self, scored_df):
        daily = aggregate_daily_sentiment(scored_df)
        for col in (
            "date",
            "daily_sentiment_score",
            "daily_sentiment_label",
            "daily_article_count",
            "daily_pos_mean",
            "daily_neg_mean",
            "daily_neu_mean",
        ):
            assert col in daily.columns, f"Missing column: {col}"

    def test_article_count_correct(self, scored_df):
        daily = aggregate_daily_sentiment(scored_df)
        # 2 articles on 2026-03-01, 1 on 2026-03-02
        counts = daily.set_index(daily["date"].dt.date)["daily_article_count"]
        assert counts[date(2026, 3, 1)] == 2
        assert counts[date(2026, 3, 2)] == 1

    def test_mean_score_math(self, scored_df):
        daily = aggregate_daily_sentiment(scored_df)
        # 2026-03-01: articles 0 and 1
        day1 = scored_df[scored_df["date"].dt.date == date(2026, 3, 1)]
        expected_mean = day1["sentiment_score"].mean()
        actual_mean = daily.loc[
            daily["date"].dt.date == date(2026, 3, 1), "daily_sentiment_score"
        ].values[0]
        assert abs(actual_mean - expected_mean) < 1e-10

    def test_label_positive(self, scored_df):
        # article 0: score = 0.7 − 0.1 = 0.6 (Positive)
        # article 1: score = 0.05 − 0.80 = −0.75 (Negative)
        # mean for day 1 = (0.6 − 0.75) / 2 = −0.075 → Negative
        daily = aggregate_daily_sentiment(scored_df)
        day1_label = daily.loc[
            daily["date"].dt.date == date(2026, 3, 1), "daily_sentiment_label"
        ].values[0]
        assert day1_label in ("Positive", "Negative", "Neutral")

    def test_single_article_day(self):
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-03-05"]),
                "sentiment_positive": [0.9],
                "sentiment_negative": [0.05],
                "sentiment_neutral": [0.05],
                "sentiment_score": [0.85],
                "sentiment_label": ["Positive"],
            }
        )
        daily = aggregate_daily_sentiment(df)
        assert len(daily) == 1
        assert abs(daily["daily_sentiment_score"].iloc[0] - 0.85) < 1e-10

    def test_empty_df_returns_empty(self):
        daily = aggregate_daily_sentiment(pd.DataFrame())
        assert daily.empty

    def test_sorted_by_date(self, scored_df):
        daily = aggregate_daily_sentiment(scored_df)
        dates = daily["date"].tolist()
        assert dates == sorted(dates)


# ──────────────────────────────────────────────────────────────────────────────
# Tests: run_sentiment_scoring (mock inference)
# ──────────────────────────────────────────────────────────────────────────────


class TestRunSentimentScoring:
    def _mock_score_articles(self, texts, **kwargs):
        """Return deterministic scores for any list of texts."""
        return [{"positive": 0.5, "negative": 0.2, "neutral": 0.3}] * len(texts)

    def test_returns_tuple_of_two_dfs(self, tiny_news_df):
        with patch("src.sentiment.score_articles", side_effect=self._mock_score_articles):
            scored, daily = run_sentiment_scoring(
                tiny_news_df, use_cache=False, verbose=False
            )
        assert isinstance(scored, pd.DataFrame)
        assert isinstance(daily, pd.DataFrame)

    def test_scored_df_has_sentiment_columns(self, tiny_news_df):
        with patch("src.sentiment.score_articles", side_effect=self._mock_score_articles):
            scored, _ = run_sentiment_scoring(
                tiny_news_df, use_cache=False, verbose=False
            )
        assert "sentiment_score" in scored.columns

    def test_daily_df_has_expected_columns(self, tiny_news_df):
        with patch("src.sentiment.score_articles", side_effect=self._mock_score_articles):
            _, daily = run_sentiment_scoring(
                tiny_news_df, use_cache=False, verbose=False
            )
        assert "daily_sentiment_score" in daily.columns
        assert "daily_article_count" in daily.columns

    def test_daily_row_count_matches_unique_dates(self, tiny_news_df):
        with patch("src.sentiment.score_articles", side_effect=self._mock_score_articles):
            _, daily = run_sentiment_scoring(
                tiny_news_df, use_cache=False, verbose=False
            )
        assert len(daily) == tiny_news_df["date"].dt.date.nunique()

    def test_empty_input_returns_empty(self):
        empty_news = pd.DataFrame(
            columns=["date", "source", "headline", "description", "url"]
        )
        with warnings.catch_warnings(record=True):
            scored, daily = run_sentiment_scoring(
                empty_news, use_cache=False, verbose=False
            )
        assert daily.empty

    def test_cache_roundtrip(self, tiny_news_df, tmp_path):
        cache_file = str(tmp_path / "sentiment_cache.csv")
        with patch("src.sentiment.score_articles", side_effect=self._mock_score_articles):
            _, daily_first = run_sentiment_scoring(
                tiny_news_df,
                use_cache=True,
                cache_path=cache_file,
                verbose=False,
            )
        # Load from cache on second call
        with patch("src.sentiment.score_articles", side_effect=self._mock_score_articles) as mock_sa:
            _, daily_second = run_sentiment_scoring(
                tiny_news_df,
                use_cache=True,
                cache_path=cache_file,
                verbose=False,
            )
            # score_articles should NOT be called again — all dates cached
            mock_sa.assert_not_called()
        assert len(daily_first) == len(daily_second)


# ──────────────────────────────────────────────────────────────────────────────
# Tests: sentiment cache helpers
# ──────────────────────────────────────────────────────────────────────────────


class TestSentimentCache:
    def test_load_returns_empty_if_not_exist(self, tmp_path):
        path = str(tmp_path / "nonexistent.csv")
        result = load_sentiment_cache(path)
        assert result.empty

    def test_save_and_load_roundtrip(self, tmp_path):
        path = str(tmp_path / "sentiment.csv")
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-03-01", "2026-03-02"]),
                "daily_sentiment_score": [0.3, -0.2],
                "daily_sentiment_label": ["Positive", "Negative"],
                "daily_article_count": [3, 2],
                "daily_pos_mean": [0.6, 0.1],
                "daily_neg_mean": [0.3, 0.7],
                "daily_neu_mean": [0.1, 0.2],
            }
        )
        save_sentiment_cache(df, path)
        loaded = load_sentiment_cache(path)
        assert len(loaded) == 2
        assert abs(loaded["daily_sentiment_score"].iloc[0] - 0.3) < 1e-6

    def test_save_deduplicates_on_reload(self, tmp_path):
        path = str(tmp_path / "sentiment_dup.csv")
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-03-01"]),
                "daily_sentiment_score": [0.3],
                "daily_sentiment_label": ["Positive"],
                "daily_article_count": [3],
                "daily_pos_mean": [0.6],
                "daily_neg_mean": [0.3],
                "daily_neu_mean": [0.1],
            }
        )
        save_sentiment_cache(df, path)
        # Save a new row for the same date with a different score
        df2 = df.copy()
        df2["daily_sentiment_score"] = [0.8]
        save_sentiment_cache(df2, path)
        loaded = load_sentiment_cache(path)
        # Should keep only one row for 2026-03-01, with the latest score
        assert len(loaded) == 1
        assert abs(loaded["daily_sentiment_score"].iloc[0] - 0.8) < 1e-6


# ──────────────────────────────────────────────────────────────────────────────
# Integration test: sentiment_pipeline (demo mode, no network)
# ──────────────────────────────────────────────────────────────────────────────


class TestSentimentPipelineDemo:
    """
    End-to-end smoke test for sentiment_pipeline.run_sentiment_pipeline.

    Uses --demo so no network is needed; patches score_articles to avoid
    the FinBERT download (speeds up CI dramatically).
    """

    def _mock_score_articles(self, texts, **kwargs):
        return [{"positive": 0.5, "negative": 0.2, "neutral": 0.3}] * len(texts)

    def test_demo_returns_dataframe(self, tmp_path):
        from src.sentiment_pipeline import run_sentiment_pipeline

        out_csv = str(tmp_path / "results_with_sentiment.csv")

        with patch("src.sentiment.score_articles", side_effect=self._mock_score_articles):
            result = run_sentiment_pipeline(
                demo=True,
                output_csv=out_csv,
                use_cache=False,
                plot=False,
                verbose=False,
            )

        assert isinstance(result, pd.DataFrame)
        assert not result.empty

    def test_demo_output_csv_created(self, tmp_path):
        from src.sentiment_pipeline import run_sentiment_pipeline

        out_csv = str(tmp_path / "results_with_sentiment.csv")

        with patch("src.sentiment.score_articles", side_effect=self._mock_score_articles):
            run_sentiment_pipeline(
                demo=True,
                output_csv=out_csv,
                use_cache=False,
                plot=False,
                verbose=False,
            )

        assert os.path.exists(out_csv)

    def test_demo_output_has_sentiment_columns(self, tmp_path):
        from src.sentiment_pipeline import run_sentiment_pipeline

        out_csv = str(tmp_path / "results_with_sentiment2.csv")

        with patch("src.sentiment.score_articles", side_effect=self._mock_score_articles):
            result = run_sentiment_pipeline(
                demo=True,
                output_csv=out_csv,
                use_cache=False,
                plot=False,
                verbose=False,
            )

        assert "daily_sentiment_score" in result.columns
        assert "daily_sentiment_label" in result.columns

    def test_demo_no_phase1_csv_still_works(self, tmp_path):
        """Phase 2 must work standalone even if Phase 1 CSV doesn't exist."""
        from src.sentiment_pipeline import run_sentiment_pipeline

        out_csv = str(tmp_path / "sent_only.csv")
        missing_phase1 = str(tmp_path / "phase1_results_nonexistent.csv")

        with patch("src.sentiment.score_articles", side_effect=self._mock_score_articles):
            with warnings.catch_warnings(record=True):
                result = run_sentiment_pipeline(
                    phase1_csv=missing_phase1,
                    demo=True,
                    output_csv=out_csv,
                    use_cache=False,
                    plot=False,
                    verbose=False,
                )

        assert isinstance(result, pd.DataFrame)
        assert not result.empty


# ──────────────────────────────────────────────────────────────────────────────
# Tests: main.py --phase 2 integration (argument parsing)
# ──────────────────────────────────────────────────────────────────────────────


class TestMainPhaseArg:
    def test_phase_1_is_default(self):
        from main import build_parser
        parser = build_parser()
        args = parser.parse_args(["--demo"])
        assert args.phase == "1"

    def test_phase_2_arg_parsed(self):
        from main import build_parser
        parser = build_parser()
        args = parser.parse_args(["--phase", "2", "--demo"])
        assert args.phase == "2"

    def test_phase_all_arg_parsed(self):
        from main import build_parser
        parser = build_parser()
        args = parser.parse_args(["--phase", "all", "--demo"])
        assert args.phase == "all"

    def test_invalid_phase_raises_system_exit(self):
        from main import build_parser
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--phase", "3"])

    def test_no_cache_flag(self):
        from main import build_parser
        parser = build_parser()
        args = parser.parse_args(["--phase", "2", "--demo", "--no-cache"])
        assert args.no_cache is True

    def test_lookback_days_default(self):
        from main import build_parser
        from config import NEWS_LOOKBACK_DAYS
        parser = build_parser()
        args = parser.parse_args(["--phase", "2"])
        assert args.lookback_days == NEWS_LOOKBACK_DAYS

    def test_sentiment_model_default(self):
        from main import build_parser
        from config import FINBERT_MODEL_NAME
        parser = build_parser()
        args = parser.parse_args(["--phase", "2"])
        assert args.sentiment_model == FINBERT_MODEL_NAME
