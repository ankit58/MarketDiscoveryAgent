"""
Market Discovery Agent - Python Script Template
Purpose: Search social platforms (Twitter/X, Reddit, Product Hunt, Indie Hackers, tech blogs)
for live apps/SaaS reporting revenue or MRR, extract structured data, rank and export results.

This is a template with modular functions. Some platform scrapers require API keys or
3rd-party services (Instagram/TikTok) — placeholders are provided.

How to use:
- Create a .env file with required API keys (TWITTER_BEARER_TOKEN, REDDIT_CLIENT_ID, etc.)
- Install requirements from requirements.txt (see header of file for suggested packages)
- Run: python market_discovery_agent.py --output results.csv

Note: This script is a starting point. Adapt selectors/regexes and add robust error handling
for production scraping. Respect each site's robots.txt and rate limits.
"""

import os
import re
import json
import time
import argparse
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Optional, Dict, Tuple

import requests
import pandas as pd
from dateutil import parser as dateparser
from dotenv import load_dotenv

# Optional libraries (uncomment in requirements.txt)
# snscrape for twitter scraping without API: pip install snscrape
# praw for reddit: pip install praw
# beautifulsoup4 for HTML parsing: pip install beautifulsoup4
# spacy for NER (optional): pip install spacy && python -m spacy download en_core_web_sm

try:
    import snscrape.modules.twitter as sntwitter
except Exception:
    sntwitter = None

try:
    import praw
except Exception:
    praw = None

from bs4 import BeautifulSoup

# --------------------------- Configuration & Utils ---------------------------

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("market_discovery_agent")

# Environment variables (put these in .env)
TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN")
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT", "market-discovery-agent/0.1")
PRODUCTHUNT_API_KEY = os.getenv("PRODUCTHUNT_API_KEY")
INDIEHACKERS_COOKIE = os.getenv("INDIEHACKERS_COOKIE")  # if needed

# Regex patterns for revenue detection
CURRENCY_RE = r"(?:\$|USD|Rs\.?|INR|€|EUR|£|GBP)\s?[0-9,]+(?:\.?\d+)?"
MRR_PATTERNS = [
    r"(\$[0-9,]+(?:\.[0-9]+)?\s*(?:k|K|m|M)?\s*(?:MRR|per month|/mo|monthly))",
    r"([0-9,]+(?:\.[0-9]+)?\s*(?:k|K|m|M)?\s*(?:MRR|monthly))",
    r"(made\s+\$[0-9,]+(\.[0-9]+)?\s*(?:in|this month|this week)?)",
]

DATE_PATTERNS = [r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s*\d{4}",
                 r"\b\d{4}-\d{2}-\d{2}\b"]

# Dataclass for structured results
@dataclass
class AppFinding:
    name: str
    category: Optional[str]
    mrr: Optional[str]
    mrr_value_usd: Optional[float]
    reported_date: Optional[str]
    source: str
    link: str
    growth_highlights: Optional[str]
    monetization: Optional[str]
    notes: Optional[str]

    def to_row(self):
        return asdict(self)

DEFAULT_KEYWORDS = ["SaaS", "startup", "MRR", "bootstrapped"]

def parse_args():
    parser = argparse.ArgumentParser(description="Market Discovery Agent")
    parser.add_argument(
        "--keywords",
        type=str,
        help="Comma-separated keywords for market discovery",
        default=",".join(DEFAULT_KEYWORDS)
    )
    args = parser.parse_args()
    keywords = [kw.strip() for kw in args.keywords.split(",") if kw.strip()]
    return keywords
# --------------------------- Helper parsing functions ---------------------------

def parse_currency_to_float(s: str) -> Optional[float]:
    """Try to normalize common currency strings to USD numeric value (best-effort).
    This function does not handle exchange rates. If currency is non-USD, return None or
    implement a conversion lookup.
    Examples it handles: '$45k', '$12,345', '45k MRR', '€5k'
    """
    if not s:
        return None
    s = s.replace(",", "").strip()
    # find number and multiplier
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*([kKmM]?)", s)
    if not m:
        return None
    num = float(m.group(1))
    mul = m.group(2).lower()
    if mul == 'k':
        num *= 1_000
    elif mul == 'm':
        num *= 1_000_000
    # If currency symbol is not $, we skip conversion here
    if s.startswith('$') or 'USD' in s.upper():
        return num
    # heuristics: if 'Rs' or 'INR' appears, do not convert automatically
    if 'rs' in s.lower() or 'inr' in s.lower():
        return None
    # default: if currency symbol absent, return the number
    return num


def extract_mrr_from_text(text: str) -> Tuple[Optional[str], Optional[float], Optional[str]]:
    """Search text for MRR-like statements. Returns (raw_match, normalized_value_usd, date_hint)
    """
    if not text:
        return None, None, None
    # simple search for MRR-like patterns
    for pat in MRR_PATTERNS:
        match = re.search(pat, text, flags=re.IGNORECASE)
        if match:
            raw = match.group(0)
            val = parse_currency_to_float(raw)
            # find nearby date
            date_match = None
            for dpat in DATE_PATTERNS:
                dm = re.search(dpat, text)
                if dm:
                    date_match = dm.group(0)
                    break
            return raw, val, date_match
    # fallback: search for free-form currency
    match = re.search(CURRENCY_RE, text)
    if match:
        raw = match.group(0)
        val = parse_currency_to_float(raw)
        return raw, val, None
    return None, None, None

# --------------------------- Twitter (snscrape optional) ---------------------------

def search_twitter_snscrape(query: str, limit: int = 200) -> List[Dict]:
    """Use snscrape to fetch tweets matching the query. Returns list of dicts with content and url.
    snscrape is helpful if you don't have elevated Twitter API access.
    """
    results = []
    if sntwitter is None:
        logger.warning("snscrape not available. Install snscrape to enable Twitter scraping.")
        return results
    logger.info(f"Searching Twitter for: {query}")
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(query).get_items()):
        if i >= limit:
            break
        results.append({
            'id': tweet.id,
            'date': tweet.date.isoformat() if tweet.date else None,
            'content': tweet.content,
            'url': f"https://twitter.com/i/web/status/{tweet.id}",
            'user': getattr(tweet, 'user', None).username if getattr(tweet, 'user', None) else None
        })
    return results

# If you prefer Twitter API v2, implement a function using requests with TWITTER_BEARER_TOKEN

# --------------------------- Reddit (PRAW) ---------------------------

def search_reddit(subreddits: List[str], query: str, limit: int = 100) -> List[Dict]:
    results = []
    if praw is None:
        logger.warning("praw not available - install praw and configure Reddit API credentials.")
        return results
    reddit = praw.Reddit(client_id=REDDIT_CLIENT_ID,
                         client_secret=REDDIT_CLIENT_SECRET,
                         user_agent=REDDIT_USER_AGENT)

    for sub in subreddits:
        logger.info(f"Searching r/{sub} for: {query}")
        try:
            for submission in reddit.subreddit(sub).search(query, limit=limit, sort='new'):
                results.append({
                    'id': submission.id,
                    'title': submission.title,
                    'selftext': submission.selftext,
                    'url': submission.url,
                    'date': datetime.fromtimestamp(submission.created_utc).isoformat()
                })
        except Exception as e:
            logger.exception(f"Reddit search failed for {sub}: {e}")
    return results

# --------------------------- Product Hunt (API) ---------------------------

def fetch_producthunt_recent(page: int = 1) -> List[Dict]:
    """Fetch recent Product Hunt posts (requires API key). Simple placeholder using public web scraping.
    Product Hunt API requires OAuth; for prototyping, we can scrape the site (be mindful of ToS).
    """
    results = []
    headers = {"User-Agent": "market-discovery-agent/0.1"}
    url = f"https://www.producthunt.com/" if PRODUCTHUNT_API_KEY is None else None
    try:
        r = requests.get(url, headers=headers, timeout=15)
        soup = BeautifulSoup(r.text, 'html.parser')
        # NOTE: Product Hunt's markup is dynamic; this is a simplified example
        posts = soup.select('div.styles_card__')[:30]
        for p in posts:
            title = p.get_text(strip=True)[:200]
            link = 'https://www.producthunt.com'  # placeholder
            results.append({'title': title, 'url': link, 'raw': str(p)})
    except Exception as e:
        logger.warning(f"Product Hunt scrape failed: {e}")
    return results

# --------------------------- Indie Hackers (scrape) ---------------------------

def fetch_indiehackers_recent() -> List[Dict]:
    results = []
    headers = {"User-Agent": "market-discovery-agent/0.1"}
    url = "https://www.indiehackers.com/"
    try:
        r = requests.get(url, headers=headers, timeout=15)
        soup = BeautifulSoup(r.text, 'html.parser')
        posts = soup.select('.topic')[:40]
        for p in posts:
            title = p.get_text(strip=True)
            link = p.get('href')
            results.append({'title': title, 'url': link, 'raw': str(p)})
    except Exception as e:
        logger.warning(f"Indie Hackers scrape failed: {e}")
    return results

# --------------------------- Tech blogs (RSS) ---------------------------

def fetch_rss_feed(feed_url: str, limit: int = 20) -> List[Dict]:
    try:
        r = requests.get(feed_url, timeout=10)
        soup = BeautifulSoup(r.text, 'xml')
        items = soup.find_all('item')[:limit]
        results = []
        for it in items:
            results.append({
                'title': it.title.get_text() if it.title else None,
                'link': it.link.get_text() if it.link else None,
                'desc': it.description.get_text() if it.description else None,
                'pubDate': it.pubDate.get_text() if it.pubDate else None
            })
        return results
    except Exception as e:
        logger.warning(f"RSS fetch failed for {feed_url}: {e}")
        return []

# --------------------------- Instagram / TikTok (placeholder) ---------------------------

def fetch_instagram_tiktok_trends(keywords: List[str]) -> List[Dict]:
    """Instagram and TikTok are harder to scrape. Suggest using 3rd-party tools like Phantombuster,
    Apify, or manual collection. This stub returns empty list.
    """
    logger.info("Instagram/TikTok scraping is not implemented. Use Phantombuster/Apify or manual sources.")
    return []

# --------------------------- Extraction & Merging ---------------------------

def extract_findings_from_source(source_name: str, items: List[Dict]) -> List[AppFinding]:
    findings: List[AppFinding] = []
    for item in items:
        text = ''
        link = item.get('url') or item.get('link') or ''
        if 'content' in item:
            text = item.get('content')
            title = item.get('user') or ''
        else:
            title = item.get('title') or ''
            text = (item.get('selftext') or '') + '\n' + (item.get('desc') or '') + '\n' + title

        raw_mrr, val_usd, date_hint = extract_mrr_from_text(text + ' ' + title)

        # find app name heuristics: title first, else try to detect 'AppName' from text using quotes
        name = None
        if title:
            name = title.strip()[:120]
        else:
            # try to extract capitalized tokens or words in backticks/quotes
            m = re.search(r"['\"`]{1}([A-Za-z0-9 _\-]{2,60})['\"`]{1}", text)
            if m:
                name = m.group(1)
        if not name:
            name = 'Unknown'

        # basic category detection from keywords
        category = None
        if re.search(r"ai |llm|large language model|chatgpt|gpt|machine learning", text, re.I):
            category = 'AI / LLM'
        elif re.search(r"e[- ]?commerce|shopify|store", text, re.I):
            category = 'E-commerce SaaS'
        elif re.search(r"productivity|task|todo|notion|clickup", text, re.I):
            category = 'Productivity'
        else:
            category = 'Other'

        monetization = None
        if re.search(r"subscription|monthly|per month|MRR|/mo|annual|freemium|paid plan", text, re.I):
            monetization = 'Subscription'
        elif re.search(r"one[- ]time|lifetime|pay once", text, re.I):
            monetization = 'One-time'

        growth = None
        # capture short growth cues
        growth_cues = []
        if re.search(r"tiktok|viral|reddit|producthunt|launch|launching", text, re.I):
            growth_cues.append('viral/launch traction')
        if re.search(r"niche|vertical|shopify|marketplace|specialized", text, re.I):
            growth_cues.append('niche targeting')
        if re.search(r"ads|facebook ads|tiktok ads|paid ads", text, re.I):
            growth_cues.append('paid ads')
        if re.search(r"organic|word of mouth|referral", text, re.I):
            growth_cues.append('organic/referrals')
        if growth_cues:
            growth = ', '.join(growth_cues)

        notes = item.get('raw') if item.get('raw') else None

        app = AppFinding(
            name=name,
            category=category,
            mrr=raw_mrr,
            mrr_value_usd=val_usd,
            reported_date=date_hint,
            source=source_name,
            link=link,
            growth_highlights=growth,
            monetization=monetization,
            notes=notes
        )
        # filter: ignore if no mrr detected
        if app.mrr is None and app.mrr_value_usd is None:
            continue
        findings.append(app)
    return findings

# --------------------------- Merge duplicates & ranking ---------------------------

def merge_findings(findings: List[AppFinding]) -> List[AppFinding]:
    # simple merge by name: keep the one with latest reported_date or highest mrr_value_usd
    merged: Dict[str, AppFinding] = {}
    for f in findings:
        key = f.name.lower()
        if key not in merged:
            merged[key] = f
        else:
            existing = merged[key]
            # prefer numeric mrr value if available
            if f.mrr_value_usd and (not existing.mrr_value_usd or f.mrr_value_usd > existing.mrr_value_usd):
                merged[key] = f
            # else prefer later date if parsable
            try:
                if f.reported_date and existing.reported_date:
                    d1 = dateparser.parse(f.reported_date)
                    d2 = dateparser.parse(existing.reported_date)
                    if d1 > d2:
                        merged[key] = f
            except Exception:
                pass
    return list(merged.values())


def rank_findings(findings: List[AppFinding]) -> List[AppFinding]:
    # Score by (1) numeric MRR, (2) presence of growth_highlights, (3) uniqueness heuristic
    scored = []
    for f in findings:
        score = 0
        if f.mrr_value_usd:
            # log scale
            import math
            score += math.log10(f.mrr_value_usd + 1) * 10
        if f.growth_highlights:
            score += 5
        # uniqueness: penalize generic categories
        if f.category and f.category.lower() in ('other', 'productivity'):
            score += 1
        else:
            score += 2
        scored.append((score, f))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [s[1] for s in scored]

# --------------------------- Exporting ---------------------------

def export_to_csv(findings: List[AppFinding], path: str):
    df = pd.DataFrame([f.to_row() for f in findings])
    df.to_csv(path, index=False)
    logger.info(f"Exported {len(findings)} findings to {path}")


def export_to_markdown(findings: List[AppFinding], path: str):
    lines = ["| App Name | Category | MRR/Revenue | Source | Link | Growth Highlights |\n",
             "|---|---|---|---|---|---|\n"]
    for f in findings:
        lines.append(f"| {f.name} | {f.category} | {f.mrr or ''} | {f.source} | {f.link} | {f.growth_highlights or ''} |\n")
    with open(path, 'w', encoding='utf-8') as fh:
        fh.writelines(lines)
    logger.info(f"Exported markdown to {path}")

# --------------------------- Main orchestration ---------------------------

def run_discovery(limit_twitter=200, limit_reddit=100, output_csv='results.csv'):
    findings: List[AppFinding] = []

    # Twitter search examples
    twitter_queries = [
        'MRR OR "monthly recurring revenue" OR "$ per month" OR "bootstrapped SaaS"',
        '"made $" "this month" site:twitter.com'
    ]
    if sntwitter:
        for q in twitter_queries:
            tw = search_twitter_snscrape(q, limit=limit_twitter)
            findings += extract_findings_from_source('Twitter', tw)
            time.sleep(1)
    else:
        logger.info('Skipping Twitter scraper (snscrape not installed)')

    # Reddit
    subreddits = ['Entrepreneur', 'SideProject', 'SaaS', 'startups', 'IndieHackers']
    if praw:
        reddit_items = search_reddit(subreddits, 'MRR OR "revenue report" OR "made $"', limit=limit_reddit)
        findings += extract_findings_from_source('Reddit', reddit_items)
    else:
        logger.info('Skipping Reddit (praw not installed)')

    # Product Hunt
    ph = fetch_producthunt_recent()
    findings += extract_findings_from_source('ProductHunt', ph)

    # Indie Hackers
    ih = fetch_indiehackers_recent()
    findings += extract_findings_from_source('IndieHackers', ih)

    # Tech blogs (example feeds)
    feeds = [
        'https://techcrunch.com/feed/',
        'https://thenextweb.com/feed',
        'https://www.producthunt.com/feed'
    ]
    for f in feeds:
        items = fetch_rss_feed(f)
        findings += extract_findings_from_source('TechBlog', items)

    # Instagram / TikTok (placeholder)
    findings += extract_findings_from_source('Instagram/TikTok', fetch_instagram_tiktok_trends(['app making $']))

    # Merge and rank
    merged = merge_findings(findings)
    ranked = rank_findings(merged)

    # Export results
    export_to_csv(ranked, output_csv)
    export_to_markdown(ranked, output_csv.replace('.csv', '.md'))

    # Print top N
    for i, f in enumerate(ranked[:25], 1):
        print(f"{i}. {f.name} — {f.mrr or f.mrr_value_usd} — {f.source} — {f.link}\n   Growth: {f.growth_highlights}\n")

    return ranked

# --------------------------- CLI ---------------------------

if __name__ == '__main__':
    keywords = parse_args()
    print(f"Running market discovery with keywords: {keywords}")
    parser = argparse.ArgumentParser(description='Market Discovery Agent')
    parser.add_argument('--output', '-o', help='CSV output path', default='results.csv')
    parser.add_argument('--twitter', type=int, default=200, help='Twitter scrape limit')
    parser.add_argument('--reddit', type=int, default=100, help='Reddit search limit')
    args = parser.parse_args()

    run_discovery(limit_twitter=args.twitter, limit_reddit=args.reddit, output_csv=args.output)

# --------------------------- Requirements (for your requirements.txt) ---------------------------
# requests
# python-dotenv
# pandas
# beautifulsoup4
# dateutil
# snscrape  # optional
# praw  # optional
# spacy  # optional for advanced NER
# lxml  # optional

# --------------------------- Next steps & TODOs ---------------------------
# - Implement Product Hunt API OAuth flow for reliable data
# - Improve NER for app names using spaCy and custom entity rules
# - Implement currency exchange conversion for non-USD claims
# - Add retry/backoff and persistent caching (sqlite) for deduplication and rate limiting
# - Add image capture / screenshot capability (e.g., Playwright) if you want quotes/screenshots
# - Add tests and CI pipeline
