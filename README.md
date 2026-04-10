# BTS Skytrain — Aspect-Based Sentiment Analysis

> **Topic:** Enhancing BTS Skytrain Services through Aspect-Based Sentiment Analysis of Passenger Reviews

---

## Project Overview

This project analyzes passenger reviews of the BTS Skytrain (Bangkok, Thailand) using **Aspect-Based Sentiment Analysis (ABSA)**. Rather than classifying a review as simply "positive" or "negative," ABSA identifies which **specific service aspects** passengers are talking about — and what they feel about each one.

**Example:**
> *"The BTS is fast and convenient, but the ticket machines are confusing."*

| Aspect | Sentiment |
|--------|-----------|
| Infrastructure (speed) | Positive |
| Fare & Payment | Negative |

---

## Repository Structure

```
├── README.md                                      ← You are here
├── BTS_Aspect_Based_Sentiment_Labeling_Documentation.md
│                                                    ← Full labeling methodology & results
├── label_bts_reviews.py                            ← Automated aspect + sentiment labeling script
├── merge_raw.py                                     ← Raw file merge pipeline (Reddit + TripAdvisor)
│
├── raw_data/
│   ├── bts_reddit_reviews_expanded_nlp.csv          ← Reddit source (2,000 rows)
│   └── 11k_bts_skytrain_reviews.csv                ← TripAdvisor source (11,073 rows)
│
├── bts_merged_reviews_v2.csv                       ← Merged raw data (13,073 rows × 29 cols)
├── bts_labeled_reviews_v2.csv                      ← Fully labeled dataset (13,073 rows × 44 cols)
└── BTS_Aspect_Based_Sentiment_Labeling_Documentation.md
```

---

## Dataset

The project uses two raw data sources merged into a single dataset:

| Source File | Category | Records | Description |
|---|---|---|---|
| `raw_data/bts_reddit_reviews_expanded_nlp.csv` | `reddit` | 2,000 | Reddit posts mentioning BTS Skytrain |
| `raw_data/11k_bts_skytrain_reviews.csv` | `tripadvisor` | 11,073 | TripAdvisor BTS Skytrain reviews |
| **Merged total** | — | **13,073** | — |

### Data Source Notes

| Field | Reddit Source | TripAdvisor Source |
|-------|-------------|-------------------|
| `review_rating` | NLP-derived (VADER sentiment → 1–5 stars) | Platform star rating |
| `like_count` | Reddit upvotes (community agreement signal) | Helpful votes count |

> **Why this matters:** A Reddit post like *"The BTS Skytrain is a terrible nightmare!"* with 5,000 upvotes is a **highly agreed-upon complaint** — not a 5-star positive review. The crawler (`bts_scraper.py`) runs VADER sentiment analysis on the post text to generate a proper 1–5 star rating, while upvotes are preserved separately in `like_count` for community agreement analysis.

| Property | Value |
|---|---|
| **Output** | `bts_merged_reviews_v2.csv` (raw) / `bts_labeled_reviews_v2.csv` (labeled) |
| **Languages** | English, Thai, others |
| **Rating scale** | 1–5 stars |

### Rating Distribution

#### TripAdvisor (NLP: VADER Sentiment)
| Rating | Count | Share |
|--------|-------|-------|
| ⭐ 5 | 6,915 | 62.5% |
| ⭐⭐⭐⭐ 4 | 3,269 | 29.5% |
| ⭐⭐⭐ 3 | 693 | 6.3% |
| ⭐⭐ 2 | 111 | 1.0% |
| ⭐ 1 | 85 | 0.8% |

#### Reddit (Platform Stars)
| Rating | Count | Share |
|--------|-------|-------|
| ⭐ 5 | 1,434 | 71.7% |
| ⭐⭐⭐⭐ 4 | 201 | 10.1% |
| ⭐⭐⭐ 3 | 156 | 7.8% |
| ⭐⭐ 2 | 100 | 5.0% |
| ⭐ 1 | 109 | 5.5% |

---

## Labeling Schema

The pipeline produces **4 types of labels** per review:

```
┌──────────────────────────────────────────────────────────┐
│  relevant                 Is the review BTS-specific?  │
│  overall_sentiment        Positive / Neutral / Negative │
│  primary_aspect           Most-mentioned aspect          │
│  aspects_detected         All aspects found (multi)     │
│  aspect_count             # of aspects (0–10)           │
│  sentiment_<aspect>        Per-aspect sentiment          │
└──────────────────────────────────────────────────────────┘
```

### 10 Service Aspects

| # | Aspect | What it covers |
|---|--------|---------------|
| 1 | Staff & Customer Service | Helpfulness, politeness, attitude |
| 2 | Punctuality & Reliability | Delays, frequency, service disruptions |
| 3 | Crowding & Comfort | Density, seat availability, rush hour |
| 4 | Cleanliness & Hygiene | Train/station cleanliness, odor |
| 5 | Fare & Payment System | Ticket machines, pricing, queues, cards |
| 6 | Safety & Security | Security presence, CCTV, theft |
| 7 | Route Coverage & Connectivity | Network coverage, transfers, operating hours |
| 8 | Signage & Navigation | Wayfinding, maps, announcements |
| 9 | Infrastructure & Facilities | AC, elevators, escalators, platforms |
| 10 | Overall Experience | General satisfaction, recommendations |

---

## Labeling Results

### Sentiment Distribution
```
Positive     11,246  (86.0%)  ████████████████████████████████
Negative         555  ( 4.2%)  ██
Neutral          389  ( 3.0%)  █
```

### Primary Aspect Breakdown
```
Fare & Payment System         2,628  (20.1%)  ████████████████████
Crowding & Comfort            2,592  (19.8%)  ████████████████████
Cleanliness & Hygiene         1,734  (13.3%)  █████████████
Staff & Customer Service      1,570  (12.0%)  ████████████
Punctuality & Reliability     1,513  (11.6%)  ████████████
Overall Experience            1,046  ( 8.0%)  ████████
Route Coverage & Connectivity   977  ( 7.5%)  ███████
Infrastructure & Facilities     826  ( 6.3%)  ██████
Signage & Navigation            129  ( 1.0%)  █
Safety & Security                57  ( 0.4%)
```

### Per-Aspect Sentiment Highlights

| Aspect | Positive | Neutral | Negative |
|--------|----------|---------|----------|
| Overall Experience | 57.2% | 5.4% | 2.1% |
| Fare & Payment | 28.9% | 19.6% | 4.0% |
| Infrastructure | 25.4% | 23.0% | 3.3% |
| Cleanliness & Hygiene | 23.9% | 1.5% | 0.4% |
| Route Coverage | 16.7% | 20.5% | 2.2% |
| Staff & Customer Service | 9.6% | 2.0% | 0.5% |
| Crowding & Comfort | 8.0% | 11.8% | 7.9% |
| Punctuality & Reliability | 5.5% | 6.5% | 1.4% |
| Safety & Security | 1.1% | 1.7% | 0.5% |

> **Average aspects detected per review: 3.04** — most reviews discuss multiple service dimensions simultaneously.

---

## Quick Start

### 1. Install dependencies

```bash
pip install pandas
```

### 2. Merge raw files (optional — v2 files are pre-built)

```bash
python merge_raw.py
```

Outputs `bts_merged_reviews_v2.csv` and `bts_labeled_reviews_v2.csv`.

### 3. Run the labeling script on a fresh merge

```bash
python label_bts_reviews.py
```

> Reads `bts_merged_reviews.csv` by default. Edit the `INPUT` variable at the top of `label_bts_reviews.py` to point to a different file.

### 4. Load and explore the labeled data

```python
import pandas as pd

df = pd.read_csv("bts_labeled_reviews.csv")

# BTS-relevant reviews only
df = df[df["relevant"] == True]

# Reviews with negative fare/payment sentiment
df[df["sentiment_fare_payment"] == "Negative"][[
    "review_rating", "review_text",
    "sentiment_fare_payment", "sentiment_crowding"
]].head(10)

# Sentiment split per aspect
aspect_cols = [c for c in df.columns if c.startswith("sentiment_")]
for col in aspect_cols:
    print(df[col].value_counts(normalize=True).round(3) * 100)
```

---

## Output Columns Reference

| Column | Type | Description |
|--------|------|-------------|
| `relevant` | bool | `True` = BTS service review; `False` = off-topic |
| `overall_sentiment` | str | Positive / Neutral / Negative |
| `primary_aspect` | str | Most-mentioned aspect name |
| `aspects_detected` | str | Comma-separated list of all detected aspects |
| `aspect_count` | int | Number of distinct aspects found (0–10) |
| `sentiment_staff` | str | Sentiment for Staff aspect |
| `sentiment_punctuality` | str | Sentiment for Punctuality aspect |
| `sentiment_crowding` | str | Sentiment for Crowding aspect |
| `sentiment_cleanliness` | str | Sentiment for Cleanliness aspect |
| `sentiment_fare_payment` | str | Sentiment for Fare & Payment aspect |
| `sentiment_safety` | str | Sentiment for Safety aspect |
| `sentiment_route_connectivity` | str | Sentiment for Route Coverage aspect |
| `sentiment_signage` | str | Sentiment for Signage aspect |
| `sentiment_infrastructure` | str | Sentiment for Infrastructure aspect |
| `sentiment_overall` | str | Sentiment for Overall Experience |

> All original columns from `bts_merged_reviews.csv` are preserved unchanged.

---

## Methodology

The labeling pipeline uses a **rule-based keyword extraction** approach:

1. **Aspect Detection** — Match multi-word keywords (EN + TH) per aspect category using substring search on normalized text
2. **Overall Sentiment** — Keyword frequency comparison (positive vs. negative word counts) with star-rating fallback
3. **Per-Aspect Sentiment** — Extract sentiment from the **specific sentence(s)** containing each aspect's keywords (not the whole review), enabling multi-aspect + mixed-sentiment detection
4. **Negation Handling** — Detect negation prefixes ("not clean", "don't have to wait") to flip sentiment polarity correctly
5. **Relevance Filter** — Flag reviews that do not substantively discuss BTS services (e.g., "near BTS" used only as a location reference)

See `BTS_Aspect_Based_Sentiment_Labeling_Documentation.md` for the full methodology.

---

## Key Findings

1. **Fare & Payment** is the most-discussed aspect (20.1%) — ticket machine usability and payment limitations (e.g., no credit card top-up) generate strong passenger reactions
2. **Crowding** is highly polarized — passengers recognize peak-hour crowding as inevitable but still express frustration
3. **Cleanliness** receives predominantly positive sentiment — BTS is seen as clean relative to other Bangkok transport options
4. **Safety** is rarely explicitly mentioned — passengers likely feel generally safe without commenting on it
5. **86% of reviews are Positive** — consistent with the 5-star rating skew; low-rated reviews are disproportionately informative for service improvement

---

## Limitations

- **Rule-based** keyword matching → no deep semantic understanding; context may be missed
- **English-heavy** keyword set → Thai-language reviews may have lower aspect detection coverage
- **No aspect weighting** → a 1-sentence mention counts equally as a full paragraph complaint
- **Rating fallback bias** → 5-star reviews with no positive keywords are labeled Positive
- **Off-topic content** — 11.7% of reviews are filtered via `relevant = False`

---

## Future Work

- [ ] Fine-tune a **BERT/RoBERTa ABSA model** for higher accuracy
- [ ] Add **aspect term extraction** (e.g., identify "ticket machine at Siam" as a term)
- [ ] Implement **aspect importance weighting** based on sentence prominence
- [ ] Extend to **5-class sentiment intensity** (Very Positive / Positive / Neutral / Negative / Very Negative)
- [ ] Build **aspect correlation matrix** (do crowding complaints co-occur with punctuality issues?)
- [ ] **Temporal analysis** — track how aspect sentiments evolve over time
- [ ] **Manual validation** — label 500–1000 samples for precision/recall benchmarking

---

## Files

| File | Description |
|------|-------------|
| `raw_data/bts_reddit_reviews_expanded_nlp.csv` | Raw Reddit source (2,000 rows) |
| `raw_data/11k_bts_skytrain_reviews.csv` | Raw TripAdvisor source (11,073 rows) |
| `bts_merged_reviews_v2.csv` | Merged raw dataset (13,073 rows, 29 cols) |
| `bts_labeled_reviews_v2.csv` | Fully labeled dataset (13,073 rows, 44 cols) |
| `label_bts_reviews.py` | Automated labeling script |
| `merge_raw.py` | Raw file merge pipeline |
| `BTS_Aspect_Based_Sentiment_Labeling_Documentation.md` | Full methodology & results |
| `README.md` | This file |
