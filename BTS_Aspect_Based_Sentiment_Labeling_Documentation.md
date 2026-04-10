# BTS Skytrain Aspect-Based Sentiment Analysis
## Dataset Labeling Documentation

**Topic:** Enhancing BTS Skytrain Services through Aspect-Based Sentiment Analysis of Passenger Reviews
**Author:** ML Project Pipeline
**Date:** April 2026
**Dataset:** `bts_merged_reviews.csv` → `bts_labeled_reviews.csv`
**Records Labeled:** 13,072 reviews

---

## Table of Contents

1. [Overview](#1-overview)
2. [Dataset Description](#2-dataset-description)
3. [Labeling Schema](#3-labeling-schema)
4. [Aspect Definitions](#4-aspect-definitions)
5. [Sentiment Detection Methodology](#5-sentiment-detection-methodology)
6. [Labeling Results](#6-labeling-results)
7. [Output File Structure](#7-output-file-structure)
8. [Usage Guide](#8-usage-guide)
9. [Limitations & Future Improvements](#9-limitations--future-improvements)

---

## 1. Overview

This document describes the automated aspect-based sentiment labeling pipeline applied to passenger reviews of the BTS Skytrain system in Bangkok, Thailand. The goal is to enrich raw review data with structured labels that enable:

- **Aspect-level analysis** — understanding which specific service aspects passengers are talking about
- **Sentiment classification** — determining whether the sentiment toward each aspect is positive, neutral, or negative
- **Relevance filtering** — distinguishing BTS-relevant reviews from off-topic content
- **Actionable insights** — enabling service improvement decisions based on passenger feedback

---

## 2. Dataset Description

### 2.1 Source Data

| Property | Value |
|---|---|
| **Input File** | `bts_merged_reviews_v2.csv` |
| **Total Raw Rows** | 13,073 |
| **Total Columns** | 29 |
| **After Filtering** | 13,072 rows (1 row removed: no usable text) |
| **Sources** | TripAdvisor (11,073), Reddit (2,000) |
| **BTS Lines** | Sukhumvit Line, Silom Line, Both Lines |

### 2.2 Rating vs. Upvotes Decoupling

> ⚠️ **Critical design note:** Reddit has no native star-rating system. A post's numerical "rating" was never meant to be upvotes — it is a **community agreement signal**. A complaint with 5,000 upvotes would incorrectly look like a 5-star positive review if upvotes were used as the rating.
>
> The crawler (`bts_scraper.py`) addresses this by running **VADER sentiment analysis** on the post text to generate a proper **1–5 star rating** before saving to CSV. The original upvote count is preserved in `like_count`.

| Field | Reddit Source | TripAdvisor Source |
|-------|-------------|-------------------|
| `review_rating` | NLP-derived (VADER → 1–5 stars) | Platform star rating |
| `like_count` | Reddit upvotes (0–1,353 range) | Helpful votes count |

### 2.2 Original Columns

```
entity_name, entity_id, bts_line, category, review_id, review_title,
review_text, review_rating, published_date, created_at_date, trip_type,
stay_date, review_language, is_translated, original_language, like_count,
images_count, reviewer_id, reviewer_name, reviewer_username,
reviewer_contribution_count, reviewer_hometown, reviewer_profile_link,
has_owner_response, owner_response_text, owner_response_date,
review_link, entity_link, source
```

### 2.3 Rating Distribution (Original 1–5 Stars)

#### TripAdvisor (platform stars)
| Rating | Count | Percentage |
|--------|-------|------------|
| ⭐ 5 | 6,915 | 62.5% |
| ⭐⭐⭐⭐ 4 | 3,269 | 29.5% |
| ⭐⭐⭐ 3 | 693 | 6.3% |
| ⭐⭐ 2 | 111 | 1.0% |
| ⭐ 1 | 85 | 0.8% |

#### Reddit (VADER NLP sentiment → stars)
| Rating | Count | Percentage |
|--------|-------|------------|
| ⭐ 5 | 1,434 | 71.7% |
| ⭐⭐⭐⭐ 4 | 201 | 10.1% |
| ⭐⭐⭐ 3 | 156 | 7.8% |
| ⭐⭐ 2 | 100 | 5.0% |
| ⭐ 1 | 109 | 5.5% |

> The dataset is skewed toward positive ratings, reflecting the self-selection bias common in public review platforms. Reddit's VADER-distributed ratings show a wider spread (including more 1–2 star complaints) than TripAdvisor's platform stars.

---

## 3. Labeling Schema

### 3.1 Label Types

The pipeline generates **4 categories of labels** for each review:

```
┌─────────────────────────────────────────────────────────┐
│  RELEVANCE FLAG                                          │
│  Is this review about BTS Skytrain services?            │
│  Values: True / False                                    │
├─────────────────────────────────────────────────────────┤
│  OVERALL SENTIMENT                                       │
│  Holistic sentiment of the entire review                 │
│  Values: Positive / Neutral / Negative                 │
├─────────────────────────────────────────────────────────┤
│  ASPECT LABELS (Multi-Label)                            │
│  Which service aspects are discussed?                   │
│  10 aspect categories (see Section 4)                  │
├─────────────────────────────────────────────────────────┤
│  PER-ASPECT SENTIMENT                                    │
│  Sentiment toward each detected aspect                  │
│  Values: Positive / Neutral / Negative                 │
└─────────────────────────────────────────────────────────┘
```

### 3.2 Multi-Label Design Rationale

A single review can discuss multiple service aspects simultaneously. For example:

> *"The BTS is fast and convenient, but the ticket machines are confusing and the queues are too long."*

This review touches **3 aspects** with **mixed sentiment**:
- **Infrastructure** → Positive (fast, convenient)
- **Fare & Payment** → Negative (confusing queues)
- **Crowding** → Negative (long queues)

The multi-label schema captures this granularity, which is lost in a single-label sentiment classification approach.

---

## 4. Aspect Definitions

### 4.1 Aspect Overview

| # | Aspect Key | Display Name | Mention Freq. (Primary) |
|---|-----------|-----------------------------------|--------------------------|
| 1 | `staff` | Staff & Customer Service | 1,570 (12.0%) |
| 2 | `punctuality` | Punctuality & Reliability | 1,513 (11.6%) |
| 3 | `crowding` | Crowding & Comfort | 2,592 (19.8%) |
| 4 | `cleanliness` | Cleanliness & Hygiene | 1,734 (13.3%) |
| 5 | `fare_payment` | Fare & Payment System | 2,628 (20.1%) |
| 6 | `safety` | Safety & Security | 57 (0.4%) |
| 7 | `route_connectivity` | Route Coverage & Connectivity | 977 (7.5%) |
| 8 | `signage` | Signage & Navigation | 129 (1.0%) |
| 9 | `infrastructure` | Infrastructure & Facilities | 826 (6.3%) |
| 10 | `overall` | Overall Experience | 1,046 (8.0%) |

### 4.2 Detailed Aspect Definitions

---

#### Aspect 1: Staff & Customer Service (`staff`)

**Definition:** All aspects of human interaction with BTS staff, including ticket sellers, platform attendants, security personnel, and customer service representatives.

**Keywords (EN):** staff, employee, workers, crew, attendant, conductor, ticket seller, ticket officer, gate staff, helpful, polite, rude, friendly, courteous, unfriendly, attitude, customer service

**Keywords (TH/Other):** ผู้โดยสาร, พนักงาน, เจ้าหน้าที่, staffs

**Example Patterns:**
- *"The staff was very helpful and polite."* → Positive
- *"Staff members were rude and dismissive."* → Negative
- *"No staff available to help at the station."* → Negative

---

#### Aspect 2: Punctuality & Reliability (`punctuality`)

**Definition:** The reliability of BTS train services in terms of on-time performance, service frequency, headway between trains, and service disruptions.

**Keywords (EN):** on time, punctual, delay, delayed, late, wait time, waiting time, frequency, service disruption, breakdown, cancelled, train jam, traffic jam

**Keywords (TH):** แพ้, delay, มาสาย, รถมาช้า, รอรถ, รอนาน

**Example Patterns:**
- *"Trains run every 5 minutes — always on time."* → Positive
- *"Waited 20 minutes for a train during peak hour."* → Negative
- *"Service disruption caused massive delays."* → Negative

---

#### Aspect 3: Crowding & Comfort (`crowding`)

**Definition:** The level of crowding inside trains and at stations, seat availability, standing room conditions, and overall passenger comfort during the journey.

**Keywords (EN):** crowd, crowded, busy, packed, overcrowded, standing room, no seat, rush hour, peak hour, morning rush, evening rush, congested

**Keywords (TH):** แออัด, คนเยอะ, คนแน่น, ยืน, ไม่มีที่นั่ง

**Example Patterns:**
- *"Rush hour is incredibly crowded — no room to breathe."* → Negative
- *"Non-peak times are comfortable with plenty of seats."* → Positive
- *"Busy but manageable on the Sukhumvit line."* → Neutral

---

#### Aspect 4: Cleanliness & Hygiene (`cleanliness`)

**Definition:** The cleanliness of trains, stations, platforms, restrooms, and surrounding areas, including odor, litter, and general hygiene maintenance.

**Keywords (EN):** clean, cleanliness, dirty, hygiene, smell, smelly, trash, garbage, dust, maintenance, filthy, neat, tidy

**Keywords (TH):** สะอาด, สกปรก, กลิ่น, ฝุ่น

**Example Patterns:**
- *"The trains are spotless and smell fresh."* → Positive
- *"Stations are dirty and poorly maintained."* → Negative
- *"Restrooms at the station need better cleaning."* → Negative

---

#### Aspect 5: Fare & Payment System (`fare_payment`)

**Definition:** All aspects related to BTS pricing, ticketing, payment methods, top-up procedures, ticket machines, and queuing behavior.

**Keywords (EN):** fare, ticket, price, expensive, credit card, top up, reload, BTS card, Rabbit card, token, queue, ticket machine, ATM, coins, cash, payment

**Keywords (TH):** ราคา, ตั๋ว, บัตร, เติมเงิน, คิว, ATM, เสียค่าโดยสาร, ไม่รับบัตร

**Example Patterns:**
- *"Ticket machines are confusing and only accept cash."* → Negative
- *"Rabbit card makes payment seamless and fast."* → Positive
- *"Queues at the ticket counter are always too long."* → Negative

> ⚠️ **Most frequently mentioned primary aspect (20.1%)** — reflects strong passenger interest in ticketing and payment experience.

---

#### Aspect 6: Safety & Security (`safety`)

**Definition:** Perceived safety at stations and on trains, security presence, CCTV coverage, theft/pickpocketing concerns, and emergency preparedness.

**Keywords (EN):** safety, secure, security, unsafe, dangerous, theft, stolen, robbery, police, guard, CCTV, pickpocket, harassment, crime

**Keywords (TH):** ความปลอดภัย, ปลอดภัย, ขโมย, กรรโชก

**Example Patterns:**
- *"Feel safe with visible security guards at every station."* → Positive
- *"Pickpockets are common on the Silom line during rush hour."* → Negative
- *"No CCTV coverage in some areas of the station."* → Negative

---

#### Aspect 7: Route Coverage & Connectivity (`route_connectivity`)

**Definition:** The extent of BTS network coverage, connectivity to other transit modes (MRT, Airport Rail Link, buses), accessibility of stations, and first/last train operating hours.

**Keywords (EN):** station, line, connect, connection, transfer, interchange, first train, last train, operating hours, coverage, route, destination, no station, accessible

**Keywords (TH):** สถานี, เชื่อม, เปลี่ยน, ต่อรถ, ไม่มีสถานี, ไกล

**Example Patterns:**
- *"The BTS connects seamlessly with the MRT at Asoke."* → Positive
- *"No BTS station near the Grand Palace — need a taxi."* → Negative
- *"First train at 6 AM is too late for morning commuters."* → Negative

---

#### Aspect 8: Signage & Navigation (`signage`)

**Definition:** Quality of station signage, wayfinding, maps, direction boards, multilingual signs, PA announcements, and information displays.

**Keywords (EN):** sign, signage, signs, direction, directions, wayfinding, map, maps, guide, information board, announcement, display board

**Keywords (TH):** แผ่นป้าย, ป้ายบอก, เครื่องหมาย, ชั้น

**Example Patterns:**
- *"All signs are in Thai, English, and Chinese — very helpful."* → Positive
- *"Confusing signage makes it hard to find the exit."* → Negative
- *"PA announcements are barely audible."* → Negative

---

#### Aspect 9: Infrastructure & Facilities (`infrastructure`)

**Definition:** Physical infrastructure quality including air conditioning, elevators, escalators, platform conditions, gap between platform and train, seating, restrooms, lighting, and accessibility for disabled passengers.

**Keywords (EN):** air conditioning, a/c, aircon, elevator, lift, escalator, stairs, platform, platform gap, ventilation, toilet, restroom, seating, seat, lighting, wheelchair, ramp

**Keywords (TH):** เครื่องปรับอากาศ, บันไดเลื่อน, ลิฟต์, ไม้กั้น, ชานชาลา, ห้องน้ำ

**Example Patterns:**
- *"Air conditioning is excellent — a relief from Bangkok heat."* → Positive
- *"Broken escalator forces passengers to use steep stairs."* → Negative
- *"Platform gaps are dangerous for elderly passengers."* → Negative

---

#### Aspect 10: Overall Experience (`overall`)

**Definition:** Holistic assessment of the BTS journey, general satisfaction, likelihood to recommend, and overall value perception.

**Keywords (EN):** overall, in general, recommend, best, worst, excellent, terrible, amazing, awful, experience, good, great, bad, satisfied, disappointed

**Keywords (TH):** ไม่แนะนำ, แนะนำ, ดี, ประทับใจ, ผิดหวัง, พอใจ, รวม

**Example Patterns:**
- *"Overall, the BTS is the best way to get around Bangkok."* → Positive
- *"Would not recommend during peak hours."* → Negative
- *"Decent experience overall, but improvements needed."* → Neutral

---

## 5. Sentiment Detection Methodology

### 5.1 Overall Sentiment Classification

The overall sentiment is determined by a **multi-signal approach** combining keyword frequency analysis with the original star rating:

```
┌────────────────────────────────────────────────────────┐
│  STEP 1: Count positive and negative keywords in text  │
│          positive_words_set  vs  negative_words_set    │
├────────────────────────────────────────────────────────┤
│  STEP 2: Check for qualifier phrases (but, however)   │
│          Handle mixed/masked sentiments               │
├────────────────────────────────────────────────────────┤
│  STEP 3: Compare counts                               │
│          pos > neg+1 → Positive                        │
│          neg > pos+1 → Negative                       │
│          pos == neg  → Neutral (if both > 0)          │
│          else        → Rating Fallback                │
├────────────────────────────────────────────────────────┤
│  STEP 4: Rating Fallback (if keywords inconclusive)  │
│          Rating 5,4 → Positive                        │
│          Rating 3   → Neutral                         │
│          Rating 2,1 → Negative                        │
└────────────────────────────────────────────────────────┘
```

### 5.2 Per-Aspect Sentiment

For each detected aspect, sentiment is extracted from the **specific sentence(s)** containing aspect-related keywords, rather than the full review. This prevents one aspect's sentiment from contaminating another:

```
Review: "The train was clean and fast, but the ticket machine was broken."

→ Aspect: infrastructure  → Sentiment: Positive (from "clean and fast")
→ Aspect: fare_payment   → Sentiment: Negative (from "ticket machine was broken")
```

### 5.3 Negation Handling

Negation words are detected to correctly classify negative sentiment expressions:
- "not clean" → treated as negative
- "don't have to wait" → treated as positive
- "hardly crowded" → treated as positive

### 5.4 Qualifier-Based Sentiment Handling

Reviews containing both positive and negative keywords with qualifier words (but, however, although) are handled by the qualifier logic. When positive and negative keyword counts are approximately equal, the sentiment defaults to **Neutral**.

---

## 6. Labeling Results

### 6.1 Overview

| Metric | Value |
|--------|-------|
| Total Reviews Labeled | 13,072 |
| BTS-Relevant Reviews | 11,536 (88.3%) |
| Off-Topic Reviews | 1,536 (11.7%) |
| Average Aspects per Review | 3.04 |
| Reviews with ≥3 Aspects | ~50% |

### 6.2 Overall Sentiment Distribution

| Sentiment | Count | Percentage | Bar |
|-----------|-------|------------|-----|
| Positive | 11,246 | 86.0% | ████████████████████████████████ |
| Negative | 555 | 4.2% | ██ |
| Neutral | 389 | 3.0% | █ |

> The high positive ratio aligns with the rating distribution (64% are 5-star ratings) and reflects that BTS Skytrain is generally well-regarded by passengers.

### 6.3 Primary Aspect Distribution

| Rank | Aspect | Count | % |
|------|--------|-------|---|
| 🥇 1 | Fare & Payment System | 2,628 | 20.1% |
| 🥈 2 | Crowding & Comfort | 2,592 | 19.8% |
| 🥉 3 | Cleanliness & Hygiene | 1,734 | 13.3% |
| 4 | Staff & Customer Service | 1,570 | 12.0% |
| 5 | Punctuality & Reliability | 1,513 | 11.6% |
| 6 | Overall Experience | 1,046 | 8.0% |
| 7 | Route Coverage & Connectivity | 977 | 7.5% |
| 8 | Infrastructure & Facilities | 826 | 6.3% |
| 9 | Signage & Navigation | 129 | 1.0% |
| 10 | Safety & Security | 57 | 0.4% |

### 6.4 Per-Aspect Sentiment Breakdown

#### Staff & Customer Service
| Sentiment | Count | % |
|------------|-------|---|
| Positive | 1,250 | 9.6% |
| Neutral | 259 | 2.0% |
| Negative | 61 | 0.5% |

#### Punctuality & Reliability
| Sentiment | Count | % |
|------------|-------|---|
| Positive | 724 | 5.5% |
| Neutral | 850 | 6.5% |
| Negative | 187 | 1.4% |

#### Crowding & Comfort
| Sentiment | Count | % |
|------------|-------|---|
| Positive | 1,040 | 8.0% |
| Neutral | 1,548 | 11.8% |
| Negative | 1,030 | 7.9% |

#### Cleanliness & Hygiene
| Sentiment | Count | % |
|------------|-------|---|
| Positive | 3,125 | 23.9% |
| Neutral | 190 | 1.5% |
| Negative | 49 | 0.4% |

#### Fare & Payment System
| Sentiment | Count | % |
|------------|-------|---|
| Positive | 3,773 | 28.9% |
| Neutral | 2,561 | 19.6% |
| Negative | 526 | 4.0% |

#### Safety & Security
| Sentiment | Count | % |
|------------|-------|---|
| Positive | 142 | 1.1% |
| Neutral | 225 | 1.7% |
| Negative | 68 | 0.5% |

#### Route Coverage & Connectivity
| Sentiment | Count | % |
|------------|-------|---|
| Positive | 2,183 | 16.7% |
| Neutral | 2,680 | 20.5% |
| Negative | 284 | 2.2% |

#### Signage & Navigation
| Sentiment | Count | % |
|------------|-------|---|
| Positive | 804 | 6.2% |
| Neutral | 929 | 7.1% |
| Negative | 87 | 0.7% |

#### Infrastructure & Facilities
| Sentiment | Count | % |
|------------|-------|---|
| Positive | 3,314 | 25.4% |
| Neutral | 3,005 | 23.0% |
| Negative | 430 | 3.3% |

#### Overall Experience
| Sentiment | Count | % |
|------------|-------|---|
| Positive | 7,471 | 57.2% |
| Neutral | 712 | 5.4% |
| Negative | 275 | 2.1% |

### 6.5 Key Insights from Labeling

1. **Fare & Payment** is the most discussed primary aspect — passengers frequently comment on ticket machine usability, queue lengths, and payment method limitations (especially the lack of credit card support).

2. **Crowding & Comfort** is the second most-discussed aspect with near-equal positive and negative split — passengers recognize it as unavoidable during peak hours but still express frustration.

3. **Cleanliness & Hygiene** receives predominantly positive sentiment (23.9%) — the BTS is perceived as clean relative to other transport options in Bangkok.

4. **Safety & Security** is rarely explicitly mentioned — passengers likely feel generally safe without commenting on it.

5. **Infrastructure** has a high mention rate (48.4%) with strong positive sentiment for air conditioning, balanced by neutral/negative comments about escalator breakdowns and platform gaps.

6. **Off-topic reviews (11.7%)** — many Reddit posts mention BTS only incidentally (e.g., "near the BTS station" as a location reference) without discussing the service itself. The `relevant` flag enables filtering these out.

---

## 7. Output File Structure

### 7.1 Output File

```
bts_labeled_reviews.csv
```

### 7.2 New Columns Added

| Column Name | Type | Description |
|-------------|------|-------------|
| `relevant` | Boolean | `True` if review discusses BTS transit; `False` if off-topic |
| `overall_sentiment` | String | Positive / Neutral / Negative |
| `primary_aspect` | String | The most-mentioned aspect in the review |
| `aspects_detected` | String | Comma-separated list of all detected aspect names |
| `aspect_count` | Integer | Number of distinct aspects detected (0–10) |
| `sentiment_staff` | String | Sentiment toward Staff aspect |
| `sentiment_punctuality` | String | Sentiment toward Punctuality aspect |
| `sentiment_crowding` | String | Sentiment toward Crowding aspect |
| `sentiment_cleanliness` | String | Sentiment toward Cleanliness aspect |
| `sentiment_fare_payment` | String | Sentiment toward Fare & Payment aspect |
| `sentiment_safety` | String | Sentiment toward Safety aspect |
| `sentiment_route_connectivity` | String | Sentiment toward Route Connectivity aspect |
| `sentiment_signage` | String | Sentiment toward Signage aspect |
| `sentiment_infrastructure` | String | Sentiment toward Infrastructure aspect |
| `sentiment_overall` | String | Sentiment toward Overall Experience |

### 7.3 Sample Labeled Records

```
review_rating | overall_sentiment | primary_aspect              | aspects_detected                                          | aspect_count
--------------+-------------------+-----------------------------+-----------------------------------------------------------+-------------
5             | Positive          | Crowding & Comfort          | Crowding & Comfort, Overall Experience                    | 2
5             | Positive          | Crowding & Comfort          | Crowding & Comfort, Route Coverage, Signage, Infrastructure | 5
4             | Negative          | Fare & Payment System       | Fare & Payment, Route Coverage, Signage, Infrastructure   | 4
5             | Positive          | Staff & Customer Service    | Staff, Cleanliness, Fare, Route, Signage, Infrastructure, Overall | 7
3             | Neutral           | Fare & Payment System       | Fare & Payment, Route Coverage                            | 2
```

---

## 8. Usage Guide

### 8.1 Loading the Labeled Dataset

```python
import pandas as pd

df = pd.read_csv("bts_labeled_reviews.csv")
print(f"Loaded {len(df):,} rows × {len(df.columns)} columns")
```

### 8.2 Filter BTS-Relevant Reviews Only

```python
df_relevant = df[df["relevant"] == True]
```

### 8.3 Get Negative Reviews Only

```python
df_negative = df[df["overall_sentiment"] == "Negative"]
```

### 8.4 Filter Reviews Mentioning a Specific Aspect

```python
# Reviews where passengers complained about Fare & Payment
df_fare_negative = df[
    (df["sentiment_fare_payment"] == "Negative") &
    (df["relevant"] == True)
]
```

### 8.5 Analyze Aspect Sentiment Distribution

```python
aspect_cols = [c for c in df.columns if c.startswith("sentiment_")]
for col in aspect_cols:
    print(df[col].value_counts())
```

### 8.6 Aspect-Count Statistics

```python
print(df["aspect_count"].describe())
# Output: mean=3.04, median=3.0, max=10
```

### 8.7 Cross-Aspect Analysis

```python
# Reviews with multi-aspect complaints (≥ 3 aspects)
df_complex = df[df["aspect_count"] >= 3]
print(f"Complex multi-aspect reviews: {len(df_complex):,}")
```

---

## 9. Limitations & Future Improvements

### 9.1 Current Limitations

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| **Rule-based keyword matching** | May miss context-dependent meanings; no deep semantic understanding | Keyword lists expanded with Thai/multilingual terms |
| **Rating as sentiment fallback** | 5-star ratings labeled as Positive even without positive keywords | Sentiment detection via keyword frequency + rating fallback |
| **Off-topic reviews (11.7%)** | Some reviews mention BTS only as a location marker | `relevant` flag enables filtering |
| **English bias in keywords** | Non-English reviews may have fewer aspects detected | Thai keywords added; language field available |
| **No aspect weighting** | All detected aspects treated equally | `primary_aspect` and `aspect_count` provide structure |
| **Single-review labeling** | Aspect-level sentiment from same review may be correlated | Sentence-level extraction used for per-aspect sentiment |

### 9.2 Recommended Future Improvements

1. **Fine-tuned Transformer Model** — Train a BERT/RoBERTa-based aspect extractor (e.g., BERT4ABS) for higher accuracy than keyword matching

2. **Aspect Term Extraction** — Identify exact phrases passengers mention (e.g., "ticket machine at Siam") rather than just aspect categories

3. **Aspect Importance Weighting** — Weight each aspect by how central it is to the review (e.g., a 1-paragraph complaint about fare weighs more than a 1-sentence mention)

4. **Sentiment Intensity Scale** — Extend from 3-class (Positive/Neutral/Negative) to 5-class (Very Positive → Very Negative) using sentiment intensity scoring

5. **Aspect Correlation Analysis** — Discover hidden correlations (e.g., crowding complaints often co-occur with punctuality complaints)

6. **Temporal Analysis** — Track how aspect sentiments change over time using `published_date`

7. **Manual Validation Set** — Human-label 500–1000 samples as a gold-standard test set to compute precision/recall per aspect

---

## Appendix A: Keyword Reference

### Positive Sentiment Keywords
```
excellent, perfect, amazing, wonderful, fantastic, great, good, nice,
love, best, awesome, impressive, convenient, efficient, fast, quick,
smooth, clean, comfortable, reliable, friendly, polite, helpful,
recommend, easy, simple, worth, pleasant, outstanding, superb, brilliant
```

### Negative Sentiment Keywords
```
bad, worst, terrible, awful, horrible, hate, poor, disappointed,
annoying, rude, expensive, overcrowded, slow, dirty, unreliable,
broken, unsafe, dangerous, difficult, complicated, confusing, filthy,
smelly, problem, issue, avoid, never again, waste, useless, pathetic
```

### Qualifier Words (Sentiment Handling)
```
but, however, although, though, except, aside from, apart from, besides,
still, yet, unfortunately, sadly, the only problem, only issue,
the problem is, thing is
```

---

*Generated by `label_bts_reviews.py` — BTS Skytrain Aspect-Based Sentiment Labeling Pipeline*
