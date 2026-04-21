"""
news_fetcher.py — Fetch financial news articles about RELIANCE Industries.

Supports two data sources (in priority order):
  1. **NewsAPI** (newsapi.org) — requires a free/paid API key set via the
     environment variable ``NEWSAPI_KEY``. Provides structured JSON with full
     metadata.
  2. **RSS feeds** (Economic Times, Moneycontrol, NDTV Profit) — completely
     free, no credentials required. Used as the default when no API key is
     present.

Output schema
-------------
Each article is stored as a row with columns:

    date        : datetime.date  — publication date (UTC)
    source      : str            — feed / outlet name
    headline    : str            — article title
    description : str            — short summary / lead paragraph
    url         : str            — canonical article URL

Results are cached to ``data/news/raw_news.csv`` to avoid redundant network
calls on subsequent runs.

Offline / testing
-----------------
Call :func:`generate_sample_news` for a deterministic synthetic DataFrame that
requires no network access.
"""

from __future__ import annotations

import os
import warnings
from datetime import date, datetime, timedelta
from typing import List, Optional

import pandas as pd

from config import (
    NEWS_DATA_DIR,
    NEWS_KEYWORDS,
    NEWS_LOOKBACK_DAYS,
    NEWS_MAX_ARTICLES,
    NEWS_RSS_FEEDS,
    RAW_NEWS_CACHE_PATH,
)


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────


_ARTICLE_COLUMNS = ["date", "source", "headline", "description", "url"]


def _ensure_news_dir() -> None:
    os.makedirs(NEWS_DATA_DIR, exist_ok=True)


def _to_date(value) -> Optional[date]:
    """Convert various date representations to :class:`datetime.date`."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return pd.to_datetime(value, utc=True).date()
    except Exception:
        return None


def _is_relevant(text: str, keywords: List[str]) -> bool:
    """Return True if *text* contains at least one keyword (case-insensitive)."""
    lower = text.lower()
    return any(kw.lower() in lower for kw in keywords)


def _empty_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=_ARTICLE_COLUMNS)


# ──────────────────────────────────────────────────────────────────────────────
# RSS feed fetcher
# ──────────────────────────────────────────────────────────────────────────────


def _fetch_rss(
    feed_urls: List[str],
    keywords: List[str],
    since: date,
) -> pd.DataFrame:
    """
    Download and filter articles from a list of RSS feed URLs.

    Parameters
    ----------
    feed_urls : List of RSS feed URL strings.
    keywords  : Keep only articles whose title or summary contains a keyword.
    since     : Discard articles published before this date.

    Returns
    -------
    DataFrame with columns ``date, source, headline, description, url``.
    """
    try:
        import feedparser  # type: ignore
    except ImportError:
        warnings.warn(
            "feedparser is not installed — cannot fetch RSS feeds. "
            "Install it with: pip install feedparser",
            RuntimeWarning,
            stacklevel=3,
        )
        return _empty_frame()

    records: list = []

    for url in feed_urls:
        try:
            feed = feedparser.parse(url)
            source_name = feed.feed.get("title", url)

            for entry in feed.entries:
                title = entry.get("title", "")
                summary = entry.get("summary", entry.get("description", ""))

                if not _is_relevant(f"{title} {summary}", keywords):
                    continue

                # Parse publication date
                pub = entry.get("published_parsed") or entry.get("updated_parsed")
                if pub:
                    article_date = date(*pub[:3])
                else:
                    article_date = date.today()

                if article_date < since:
                    continue

                records.append(
                    {
                        "date": article_date,
                        "source": source_name,
                        "headline": title.strip(),
                        "description": summary.strip(),
                        "url": entry.get("link", ""),
                    }
                )
        except Exception as exc:
            warnings.warn(
                f"Failed to parse RSS feed {url}: {exc}",
                RuntimeWarning,
                stacklevel=3,
            )

    if not records:
        return _empty_frame()

    df = pd.DataFrame(records, columns=_ARTICLE_COLUMNS)
    df["date"] = pd.to_datetime(df["date"])
    return df.drop_duplicates(subset=["headline", "date"]).reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────────
# NewsAPI fetcher
# ──────────────────────────────────────────────────────────────────────────────


def _fetch_newsapi(
    api_key: str,
    keywords: List[str],
    since: date,
    until: date,
    max_articles: int,
) -> pd.DataFrame:
    """
    Fetch articles from NewsAPI.org for the given keyword list and date range.

    Requires the ``newsapi-python`` package and a valid ``NEWSAPI_KEY``
    environment variable.

    Parameters
    ----------
    api_key      : NewsAPI developer key.
    keywords     : Search terms (OR-joined).
    since        : Start date (inclusive).
    until        : End date (inclusive).
    max_articles : Cap on total articles (0 = unlimited).

    Returns
    -------
    DataFrame with columns ``date, source, headline, description, url``.
    """
    try:
        from newsapi import NewsApiClient  # type: ignore
    except ImportError:
        warnings.warn(
            "newsapi-python is not installed. "
            "Install with: pip install newsapi-python",
            RuntimeWarning,
            stacklevel=3,
        )
        return _empty_frame()

    client = NewsApiClient(api_key=api_key)
    query = " OR ".join(f'"{kw}"' for kw in keywords)

    records: list = []
    page = 1

    while True:
        try:
            response = client.get_everything(
                q=query,
                from_param=since.isoformat(),
                to=until.isoformat(),
                language="en",
                sort_by="publishedAt",
                page=page,
                page_size=100,
            )
        except Exception as exc:
            warnings.warn(
                f"NewsAPI request failed (page {page}): {exc}",
                RuntimeWarning,
                stacklevel=3,
            )
            break

        articles = response.get("articles", [])
        if not articles:
            break

        for art in articles:
            pub_str = art.get("publishedAt", "")
            article_date = _to_date(pub_str) or date.today()
            records.append(
                {
                    "date": article_date,
                    "source": (art.get("source") or {}).get("name", "NewsAPI"),
                    "headline": (art.get("title") or "").strip(),
                    "description": (art.get("description") or "").strip(),
                    "url": art.get("url", ""),
                }
            )
            if max_articles and len(records) >= max_articles:
                break

        if max_articles and len(records) >= max_articles:
            break

        total_results = response.get("totalResults", 0)
        if page * 100 >= total_results:
            break
        page += 1

    if not records:
        return _empty_frame()

    df = pd.DataFrame(records, columns=_ARTICLE_COLUMNS)
    df["date"] = pd.to_datetime(df["date"])
    return df.drop_duplicates(subset=["headline", "date"]).reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────


def fetch_news(
    keywords: Optional[List[str]] = None,
    lookback_days: int = NEWS_LOOKBACK_DAYS,
    since: Optional[date] = None,
    until: Optional[date] = None,
    use_cache: bool = True,
    cache_path: str = RAW_NEWS_CACHE_PATH,
    rss_feeds: Optional[List[str]] = None,
    newsapi_key: Optional[str] = None,
    max_articles: int = NEWS_MAX_ARTICLES,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Fetch Reliance-related news articles from NewsAPI or RSS feeds.

    The function first checks for a cached copy at *cache_path*. If a fresh
    cache exists, it is loaded directly. Otherwise it downloads from the network
    and writes the result back to the cache.

    Parameters
    ----------
    keywords     : Keywords to filter articles. Defaults to :data:`NEWS_KEYWORDS`.
    lookback_days: How many calendar days back to retrieve (used only when
                   *since* is not provided).
    since        : Start date for article retrieval (inclusive).
    until        : End date for article retrieval (inclusive, defaults to today).
    use_cache    : Load from / save to *cache_path* (default True).
    cache_path   : Path for the raw-news cache CSV.
    rss_feeds    : List of RSS feed URLs. Defaults to :data:`NEWS_RSS_FEEDS`.
    newsapi_key  : NewsAPI.org developer key. If not supplied, the function
                   reads the environment variable ``NEWSAPI_KEY``. Falls back
                   to RSS when neither is present.
    max_articles : Maximum number of articles to return (0 = no limit).
    verbose      : Print progress messages.

    Returns
    -------
    pd.DataFrame with columns ``date, source, headline, description, url``.
    """
    keywords = keywords or NEWS_KEYWORDS
    rss_feeds = rss_feeds or NEWS_RSS_FEEDS
    until = until or date.today()
    since = since or (until - timedelta(days=lookback_days))

    # ── Load from cache if available ──────────────────────────────────────
    if use_cache and os.path.exists(cache_path):
        try:
            cached = pd.read_csv(cache_path, parse_dates=["date"])
            if not cached.empty:
                if verbose:
                    print(
                        f"[news_fetcher] Loaded {len(cached)} cached articles "
                        f"from {cache_path}"
                    )
                # Filter to requested date range
                cached["date"] = pd.to_datetime(cached["date"])
                mask = (cached["date"].dt.date >= since) & (
                    cached["date"].dt.date <= until
                )
                filtered = cached[mask].reset_index(drop=True)
                if not filtered.empty:
                    return filtered
        except Exception as exc:
            warnings.warn(
                f"Could not read news cache: {exc}. Fetching fresh data.",
                RuntimeWarning,
                stacklevel=2,
            )

    # ── Determine data source ─────────────────────────────────────────────
    api_key = newsapi_key or os.environ.get("NEWSAPI_KEY", "")

    if api_key:
        if verbose:
            print("[news_fetcher] Using NewsAPI as data source...")
        df = _fetch_newsapi(
            api_key=api_key,
            keywords=keywords,
            since=since,
            until=until,
            max_articles=max_articles,
        )
    else:
        if verbose:
            print("[news_fetcher] Using RSS feeds as data source...")
        df = _fetch_rss(
            feed_urls=rss_feeds,
            keywords=keywords,
            since=since,
        )

    if df.empty:
        warnings.warn(
            "No news articles retrieved. Returning empty DataFrame.",
            RuntimeWarning,
            stacklevel=2,
        )
        return _empty_frame()

    # Apply max_articles cap
    if max_articles and len(df) > max_articles:
        df = df.head(max_articles)

    if verbose:
        print(f"[news_fetcher] Retrieved {len(df)} articles.")

    # ── Save to cache ─────────────────────────────────────────────────────
    if use_cache:
        _ensure_news_dir()
        try:
            if os.path.exists(cache_path):
                existing = pd.read_csv(cache_path, parse_dates=["date"])
                combined = pd.concat([existing, df], ignore_index=True)
                combined = combined.drop_duplicates(
                    subset=["headline", "date"]
                ).reset_index(drop=True)
                combined.to_csv(cache_path, index=False)
            else:
                df.to_csv(cache_path, index=False)
            if verbose:
                print(f"[news_fetcher] Cache updated: {cache_path}")
        except Exception as exc:
            warnings.warn(
                f"Could not write news cache: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )

    return df.reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────────
# Sample / offline data generator
# ──────────────────────────────────────────────────────────────────────────────

# 30 synthetic Reliance news headlines that span a range of sentiments
_SAMPLE_HEADLINES = [
    # Positive
    (
        "Reliance Industries Q3 profit beats estimates; Jio subscriber growth strong",
        "RIL Q3 standalone net profit rose 11% year-on-year, driven by robust growth "
        "in Jio and Retail segments, surpassing analyst consensus by a wide margin.",
        "positive",
    ),
    (
        "Mukesh Ambani unveils ₹75,000 crore green energy push at AGM",
        "Reliance chairman announced aggressive capex plans in solar and hydrogen, "
        "sending shares to a 52-week high.",
        "positive",
    ),
    (
        "Jio Platforms inks strategic partnership with global AI chip maker",
        "The deal is expected to accelerate Jio's 5G rollout and position RIL as a "
        "key player in India's emerging AI infrastructure market.",
        "positive",
    ),
    (
        "RELIANCE.NS technical breakout: analysts set ₹1,550 target",
        "Multiple brokerages raised price targets after the stock broke above a key "
        "resistance zone on heavy volumes.",
        "positive",
    ),
    (
        "Reliance Retail records highest-ever quarterly revenues",
        "Revenue crossed ₹80,000 crore for the first time, underpinned by festive "
        "season demand and rapid store expansion.",
        "positive",
    ),
    (
        "RIL dividend payout raises investor confidence ahead of FY27",
        "The board declared a special dividend of ₹10 per share, signalling confidence "
        "in the company's cash flow generation.",
        "positive",
    ),
    (
        "Jio's 5G subscriber base doubles, boosting ARPU outlook for Reliance",
        "Average Revenue Per User improved sharply as premium 5G plans gained traction "
        "in metro cities.",
        "positive",
    ),
    (
        "Reliance acquires majority stake in leading EV charging network",
        "The acquisition aligns with RIL's broader clean-energy strategy and gives it "
        "access to 10,000+ charging points across India.",
        "positive",
    ),
    (
        "Foreign institutional investors increase RELIANCE.NS holdings to multi-year high",
        "FII net buying in Reliance hit ₹4,200 crore in the month, reflecting renewed "
        "confidence in Indian large-cap energy and telecom conglomerates.",
        "positive",
    ),
    (
        "Reliance O2C segment benefits as crude-to-petrochem spreads widen",
        "Refining margins improved significantly quarter-on-quarter, adding ₹3,500 crore "
        "to operating profit.",
        "positive",
    ),
    # Negative
    (
        "RIL stock falls 3% as crude oil prices spike; margin compression feared",
        "Rising Brent crude above $92/bbl raises feedstock costs for Reliance's O2C "
        "division, triggering profit warnings from analysts.",
        "negative",
    ),
    (
        "Reliance Industries under regulatory scrutiny over telecom pricing",
        "TRAI has initiated a review of Jio's bundled tariff plans, raising concerns "
        "about potential penalties and forced restructuring.",
        "negative",
    ),
    (
        "RELIANCE.NS drops 2.4% as broader market selloff hits heavyweight stocks",
        "Nifty 50 fell 1.8% amid global risk-off sentiment; Reliance, being the index's "
        "largest component, led the decline in absolute points.",
        "negative",
    ),
    (
        "Analysts warn of earnings risk as Jio ARPU growth stalls",
        "Sequential ARPU improvement moderated sharply, disappointing bulls who had "
        "expected a sustained uptrend.",
        "negative",
    ),
    (
        "Reliance stake sale rumours weigh on stock price",
        "Unconfirmed reports of a possible promoter pledge invoked by lenders sent "
        "shares lower during intraday trade.",
        "negative",
    ),
    (
        "RIL Q2 results miss expectations; Retail segment drags on margins",
        "Revenue from operations missed the street estimate by 4%, with Retail EBITDA "
        "margins contracting 120 basis points.",
        "negative",
    ),
    (
        "High FII outflows from India hit Reliance; stock at 6-month low",
        "Dollar strength and rising US Treasury yields prompted foreign fund managers "
        "to reduce exposure to Indian equities, with Reliance bearing the brunt.",
        "negative",
    ),
    (
        "Competition intensifies: Airtel 5G gains market share from Jio",
        "Bharti Airtel's aggressive pricing and network quality improvement are eroding "
        "Jio's subscriber growth momentum.",
        "negative",
    ),
    (
        "Global PE funds put Reliance Retail IPO on hold — report",
        "Plans to list Reliance Retail at a premium valuation are reportedly delayed "
        "due to unfavourable market conditions.",
        "negative",
    ),
    (
        "Crude-refining spreads narrow to 18-month low, hurting RIL O2C margins",
        "The GRM (Gross Refining Margin) fell to $7.2/bbl from $11.4/bbl a year ago, "
        "a key negative for Reliance's petrochemical segment.",
        "negative",
    ),
    # Neutral
    (
        "Reliance Industries to hold Annual General Meeting on September 5",
        "Shareholders can register to attend the 47th AGM of Reliance Industries "
        "Limited via the company's official investor portal.",
        "neutral",
    ),
    (
        "RELIANCE.NS trades flat ahead of RBI policy announcement",
        "The stock was range-bound in a tight band as traders awaited the Monetary "
        "Policy Committee verdict expected later in the day.",
        "neutral",
    ),
    (
        "Reliance Industries updates investor relations page with Q3 presentations",
        "The company uploaded quarterly earnings presentation slides, analyst Q&A "
        "transcripts and segment-wise financial tables.",
        "neutral",
    ),
    (
        "NSE revises lot size for RELIANCE futures contracts from October expiry",
        "The change follows NSE's periodic review of F&O contract parameters and takes "
        "effect from the October 2026 expiry series.",
        "neutral",
    ),
    (
        "Reliance Industries board meeting scheduled for January 16 to consider results",
        "The board will consider and approve audited standalone and consolidated "
        "financial results for the quarter ending December 2025.",
        "neutral",
    ),
    (
        "RELIANCE.NS option chain shows elevated put-call ratio ahead of expiry",
        "The PCR for near-month contracts is at 1.35, suggesting mild defensive "
        "positioning but no strong directional bias.",
        "neutral",
    ),
    (
        "Reliance clarifies no mergers planned; rumours unfounded — BSE filing",
        "The company issued a clarification on the stock exchange denying media "
        "speculation about a proposed merger with another entity.",
        "neutral",
    ),
    (
        "Reliance Industries announces CSR spend of ₹1,200 crore in FY25",
        "Mandatory CSR disclosures show spending in education, healthcare and "
        "rural infrastructure across 18 Indian states.",
        "neutral",
    ),
    (
        "Benchmark Nifty 50 rebalancing keeps RELIANCE.NS weighting unchanged",
        "NSE semi-annual index review results showed no change to Reliance's index "
        "weight, which remains the largest at approximately 9.8%.",
        "neutral",
    ),
    (
        "SEBI issues routine notice to Reliance on disclosure timing compliance",
        "The regulator's notice relates to a technical filing delay and is not "
        "indicative of any substantive governance concern, the company said.",
        "neutral",
    ),
]


def generate_sample_news(
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> pd.DataFrame:
    """
    Return a deterministic synthetic news DataFrame for offline testing.

    The sample contains 30 pre-written headlines (10 positive, 10 negative,
    10 neutral) spread evenly across the requested date range.

    Parameters
    ----------
    start_date : First date to assign articles (default: 60 days ago).
    end_date   : Last date to assign articles (default: today).

    Returns
    -------
    pd.DataFrame with columns ``date, source, headline, description, url``.
    """
    end_date = end_date or date.today()
    start_date = start_date or (end_date - timedelta(days=60))

    total_days = (end_date - start_date).days or 1
    n = len(_SAMPLE_HEADLINES)

    records = []
    for i, (headline, description, _sentiment) in enumerate(_SAMPLE_HEADLINES):
        # Spread articles evenly across the date range
        day_offset = int(i * total_days / n)
        article_date = start_date + timedelta(days=day_offset)
        records.append(
            {
                "date": pd.Timestamp(article_date),
                "source": ["Economic Times", "Moneycontrol", "NDTV Profit"][i % 3],
                "headline": headline,
                "description": description,
                "url": f"https://example.com/reliance-news-{i + 1}",
            }
        )

    df = pd.DataFrame(records, columns=_ARTICLE_COLUMNS)
    df["date"] = pd.to_datetime(df["date"])
    return df
