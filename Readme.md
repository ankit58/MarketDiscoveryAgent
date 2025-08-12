# Market Discovery Agent

This project is a Python-based automation tool to discover trending apps, SaaS products, and ideas that are generating significant Monthly Recurring Revenue (MRR) by scraping and analyzing content from social media and startup platforms.

## Project Structure

```
market_discovery_agent/
├── market_discovery_agent.py   # Main script for data collection and analysis
├── requirements.txt            # Python dependencies
├── README.md                   # Documentation and usage guide
```

## How It Works

1. **Data Collection**
   - Scrapes Twitter/X posts using `snscrape` for keywords like "MRR" or "monthly revenue".
   - Collects Reddit posts via `PRAW` from entrepreneurial subreddits.
   - (Optional) Integrates Product Hunt API and Indie Hackers scraping.

2. **Data Processing**
   - Extracts potential MRR mentions from text using regex and heuristics.
   - Normalizes values into numeric monthly USD revenue.
   - Cleans and deduplicates entries.

3. **Analysis & Ranking**
   - Ranks products by detected MRR.
   - Identifies trends by keyword frequency and platform origin.

4. **Output**
   - Exports findings to CSV for further analysis.
   - Generates Markdown reports with top discoveries.

## Business Logic

- **Goal:** Identify profitable ideas early by spotting public MRR disclosures.
- **MRR Extraction:** Detect numbers and currency, normalize to USD.
- **Ranking:** Sort by MRR, but also capture engagement metrics (likes, upvotes).
- **Use Cases:**
  - Entrepreneurs hunting for product inspiration.
  - Investors scouting early-stage profitable businesses.
  - Market researchers studying trends.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python3 market_discovery_agent.py --keywords "SaaS,MRR,startup"
```

---

**requirements.txt**
```
snscrape
praw
requests
pandas
beautifulsoup4
lxml
python-dotenv
```
