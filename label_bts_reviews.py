#!/usr/bin/env python3
"""
BTS Skytrain Aspect-Based Sentiment Labeling Script
===================================================
Topic: Enhancing BTS Skytrain Services through Aspect-Based Sentiment Analysis of Passenger Reviews

Labels each review with:
  - Overall Sentiment      (Positive / Neutral / Negative)
  - Aspect + Aspect Sentiment pairs (multi-label)
  - Aspect Categories       (Service Quality / Cleanliness / Staff / etc.)
  - Relevance Flag          (BTS-related vs. off-topic)

Aspect Definitions:
  1. Staff & Customer Service   | Staff helpfulness, politeness, responsiveness
  2. Punctuality & Reliability  | On-time performance, service disruptions, wait times
  3. Crowding & Comfort         | Density, seat availability, ride comfort
  4. Cleanliness & Hygiene      | Cleanliness of trains, stations, restrooms
  5. Fare & Payment System      | Ticket pricing, card systems, queues, payment convenience
  6. Safety & Security          | Security presence, station safety, theft concerns
  7. Route Coverage & Connectivity | Station accessibility, first/last train times
  8. Signage & Navigation        | Wayfinding, clear directions, multilingual signs
  9. Infrastructure & Facilities  | Air-conditioning, escalators, elevators, platforms
 10. Overall Experience           | General satisfaction / recommendation
"""

import pandas as pd
import re
import warnings
from collections import Counter

warnings.filterwarnings("ignore")

# ── 1. Aspect keyword dictionaries ────────────────────────────────────────────
ASPECT_KEYWORDS = {
    "staff": [
        "staff", "employee", "workers", "crew", "attendant", "conductor",
        "ticket seller", "ticket officer", "gate staff", "platform staff",
        "helpful", "polite", "rude", "friendly", "courteous", "unfriendly",
        "attitude", "customer service", "service staff", "workers",
        "ผู้โดยสาร", "พนักงาน", "เจ้าหน้าที่", "staffs",
    ],
    "punctuality": [
        "punctual", "on time", "on-time", "delay", "delayed", "late",
        "wait time", "waiting time", "waiting", "headway", "frequency",
        "service disruption", "disruption", "breakdown", "cancelled",
        "canceled", "train jam", "traffic jam", "bts jam",
        "แพ้", "delay", "มาสาย", "รถมาช้า", "รถไฟฟ้า delay",
        "เที่ยวรถ", "รอรถ", "รอนาน",
    ],
    "crowding": [
        "crowd", "crowded", "crowding", "busy", "packed", "overcrowded",
        "full", "standing room", "standing", "no seat", "no seats",
        "too many people", "too crowded", "congested", "peak hour",
        "rush hour", "peak time", "peak hours", "morning rush",
        "evening rush", "挤", "滿", "人多",
        "แออัด", "คนเยอะ", "คนแน่น", "ยืน", "ไม่มีที่นั่ง",
    ],
    "cleanliness": [
        "clean", "cleanliness", "dirty", "hygiene", "hygienic", "smell",
        "smelly", "stink", "stinky", "garbage", "litter", "trash",
        "dust", "maintain", "maintenance", "filthy", "neat", "tidy",
        "สะอาด", "สกปรก", "กลิ่น", "ฝุ่น",
    ],
    "fare_payment": [
        "fare", "ticket", "price", "cost", "expensive", "cheap",
        "credit card", "debit card", "payment", "pay", "top up",
        "top-up", "充值", "reload", "rabbit card", "bts card",
        "magnetic card", "token", "queue", "queues", "queuing",
        "ticket machine", "vending machine", "atm", "change",
        "machine", "machines", "coins", "cash",
        "ราคา", "ตั๋ว", "บัตร", "เติมเงิน", "จ่าย", "คิว", "ATM",
        "เสียค่าโดยสาร", "ราคาแพง", "ไม่รับบัตร",
    ],
    "safety": [
        "safety", "secure", "security", "unsafe", "dangerous",
        "danger", "accident", "theft", "theft", "stolen", "robbery",
        "police", "guard", "cctv", "camera", "pickpocket", "pick pocket",
        "harassment", "harassed", "crime", "criminal",
        "ความปลอดภัย", "ปลอดภัย", "ขโมย", "กรรโชก",
    ],
    "route_connectivity": [
        "station", "stations", "line", "connect", "connection",
        "transfer", "transfer to", "interchange", "integrat",
        "first train", "last train", "operating hours", "service hours",
        "coverage", "reach", "accessible", "accessibility",
        "route", "routes", "destination", "no station",
        "ป้าย", "สถานี", "เชื่อม", "เปลี่ยน", "ต่อรถ", "รถไฟฟ้า",
        "ไม่มีสถานี", "ไกล", "ไปถึง", "ครอบคลุม",
    ],
    "signage": [
        "sign", "signage", "signs", "direction", "directions",
        "wayfinding", "map", "maps", "guide", "information board",
        "display", "notice", "notice board", "announcement",
        "announcements", "announce", "display board",
        "แผ่นป้าย", "ป้ายบอก", "ป้าย", "เครื่องหมาย",
        "ชั้น", "direction", "นับ",
    ],
    "infrastructure": [
        "air conditioning", "a/c", "ac", "aircon", "air-condition",
        "elevator", "lift", "escalator", "stairs", "stairs",
        "platform", "platform gap", "gap", "ventilation", "fan",
        "toilet", "restroom", "bathroom", "seating", "seat",
        "seat", "bench", "shelter", "roof", "lighting", "lights",
        "wheelchair", "disabled access", "ramp", "ไม้กั้น",
        "เครื่องปรับอากาศ", "บันไดเลื่อน", "ลิฟต์", "แอร์",
        "พื้นที่", "ชานชาลา", "ห้องน้ำ",
    ],
    "overall": [
        "overall", "in general", "generally", "recommend", "recommended",
        "not recommend", "worst", "best", "excellent", "terrible",
        "amazing", "awful", "fantastic", "horrible", "impressed",
        "disappointed", "satisfied", "dissatisfied", "experience",
        "good", "great", "bad", "nice", "love", "hate", "enjoy",
        "ไม่แนะนำ", "แนะนำ", "ดี", "เลว", "ห่วย", "ประทับใจ",
        "ผิดหวัง", "พอใจ", "ไม่พอใจ", "ทั้งหมด", "รวม",
    ],
}

# Aspect display names
ASPECT_DISPLAY = {
    "staff":             "Staff & Customer Service",
    "punctuality":       "Punctuality & Reliability",
    "crowding":          "Crowding & Comfort",
    "cleanliness":       "Cleanliness & Hygiene",
    "fare_payment":      "Fare & Payment System",
    "safety":            "Safety & Security",
    "route_connectivity":"Route Coverage & Connectivity",
    "signage":           "Signage & Navigation",
    "infrastructure":    "Infrastructure & Facilities",
    "overall":           "Overall Experience",
}

# ── 2. Sentiment keyword dictionaries ─────────────────────────────────────────
POSITIVE_WORDS = {
    "excellent", "perfect", "amazing", "wonderful", "fantastic", "great",
    "good", "nice", "love", "loved", "best", "awesome", "impressive",
    "convenient", "efficient", "fast", "quick", "smooth", "clean",
    "comfortable", "reliable", "friendly", "polite", "helpful",
    "recommend", "recommended", "painless", "easy", "simple",
    "worth", "value", "pleasant", "outstanding", "superb", "brilliant",
    "แม่น", "ดีมาก", "สะดวก", "เร็ว", "สะอาด", "ง่าย", "ดี",
    "love it", "best way", "love the", "no complaints",
    "highly recommend", "definitely recommend", "would recommend",
}
NEGATIVE_WORDS = {
    "bad", "worst", "terrible", "awful", "horrible", "hate", "poor",
    "disappointed", "disappointing", "annoying", "annoyed", "frustrated",
    "frustrating", "rude", "expensive", "overpriced", "crowded", "slow",
    "dirty", "unreliable", "broken", "unsafe", "dangerous", "difficult",
    "complicated", "confusing", "overcrowded", "filthy", "smelly",
    "fault", "faulty", "problem", "problems", "issue", "issues",
    "avoid", "never again", "waste", "useless", "pathetic",
    "แย่", "ห่วย", "แพง", "สกปรก", "ยาก", "ไม่ดี", "ผิดหวัง",
    "ไม่แนะนำ", "ไม่ Worth", "ไม่ Worth it",
    "do not recommend", "not recommended", "would not recommend",
    "would not return", "never going back", "stay away",
}
QUALIFIERS = {
    "but", "however", "although", "though", "except", "aside from",
    "apart from", "besides", "still", "yet", "unfortunately", "sadly",
    "the only problem", "only issue", "only thing", "main issue",
    "the only", "the problem is", "issue is", "thing is",
}

# ── 3. Utility functions ───────────────────────────────────────────────────────

def normalize(text: str) -> str:
    """Lowercase + collapse whitespace."""
    return re.sub(r"\s+", " ", str(text).lower().strip())

def find_aspects(text: str) -> list[str]:
    """Return list of aspect keys detected in text."""
    text_lower = normalize(text)
    found = []
    for aspect, keywords in ASPECT_KEYWORDS.items():
        for kw in keywords:
            if kw.lower() in text_lower:
                found.append(aspect)
                break
    return list(dict.fromkeys(found))   # preserve order, deduplicate

def detect_sentiment(text: str, rating: int) -> str:
    """
    Determine overall sentiment using keyword matching + rating fallback.
    Handles mixed sentiments via qualifier detection.
    """
    text_lower = normalize(text)
    pos_count = sum(1 for w in POSITIVE_WORDS if w in text_lower)
    neg_count = sum(1 for w in NEGATIVE_WORDS if w in text_lower)

    # Rating-based adjustment (5-star scale)
    rating_sentiment = {1: "Negative", 2: "Negative", 3: "Neutral",
                        4: "Positive", 5: "Positive"}.get(rating, "Neutral")

    # Check for qualifiers indicating mixed sentiment
    has_qualifier = any(q in text_lower for q in QUALIFIERS)
    mixed_pos = has_qualifier and pos_count > 0 and neg_count == 0
    mixed_neg = has_qualifier and neg_count > 0 and pos_count == 0

    if mixed_pos:
        return "Positive"   # positive with a "but..." caveat → still positive
    if mixed_neg:
        return "Negative"   # negative with a "but..." caveat → still negative

    if pos_count > neg_count + 1:
        return "Positive"
    elif neg_count > pos_count + 1:
        return "Negative"
    elif pos_count == neg_count and pos_count > 0:
        return "Mixed"       # equal positive and negative keywords → Mixed
    else:
        return rating_sentiment   # fallback to rating

def extract_aspect_sentiment(text: str, aspect: str) -> str:
    """Determine sentiment toward a specific aspect within a review."""
    text_lower = normalize(text)
    keywords = ASPECT_KEYWORDS.get(aspect, [])

    # Extract the sentence containing aspect keywords for targeted analysis
    sentences = re.split(r"[.!?\n]", text_lower)
    aspect_sentences = [
        s for s in sentences
        if any(kw in s for kw in keywords)
    ]
    if not aspect_sentences:
        return "Not Mentioned"

    joined = " ".join(aspect_sentences)
    pos = sum(1 for w in POSITIVE_WORDS if w in joined)
    neg = sum(1 for w in NEGATIVE_WORDS if w in joined)

    # Negation handling
    negations = ["not ", "no ", "never ", "don't ", "doesn't ", "isn't ",
                 "wasn't ", "weren't ", "can't ", "couldn't ", "won't ",
                 "hardly ", "rarely ", "barely "]
    if neg > 0:
        for neg_word in NEGATIVE_WORDS:
            for neg_prefix in negations:
                if neg_prefix + neg_word in joined:
                    neg -= 1
                    pos += 1
                    break

    if pos > neg:
        return "Positive"
    elif neg > pos:
        return "Negative"
    else:
        return "Neutral"

def is_bts_relevant(text: str) -> bool:
    """Check if the review is about BTS Skytrain services."""
    text_lower = normalize(text)
    bts_signals = [
        "bts", "skytrain", "sukhumvit line", "silom line",
        "bts station", "bts line", " BTS ", " BTS",
        "รถไฟฟ้า", " BTS",
    ]
    general_signals = [
        "train", "station", "transit", "transport", "mrt", "arl",
        "bus", "taxi", "tuktuk", "ferry", "bangkok",
    ]
    has_bts = any(sig in text_lower for sig in bts_signals)
    has_general = any(sig in text_lower for sig in general_signals)
    return has_bts or has_general

# ── 4. Main labeling function ──────────────────────────────────────────────────

def label_review(row: dict) -> dict:
    """Label a single review and return a dict of label columns."""
    text  = str(row.get("review_text", "")) + " " + str(row.get("review_title", ""))
    rating = int(row["review_rating"]) if pd.notna(row["review_rating"]) else 3

    aspects_found  = find_aspects(text)
    overall_sent   = detect_sentiment(text, rating)
    relevant       = is_bts_relevant(text)

    # Per-aspect sentiment
    aspect_sentiments = {}
    for aspect in ASPECT_KEYWORDS:
        aspect_sentiments[f"sentiment_{aspect}"] = extract_aspect_sentiment(text, aspect)

    # Primary aspect (most mentioned)
    primary_aspect = aspects_found[0] if aspects_found else "overall"

    return {
        "relevant":               relevant,
        "overall_sentiment":       overall_sent,
        "primary_aspect":          ASPECT_DISPLAY.get(primary_aspect, "Overall Experience"),
        "aspects_detected":        ", ".join(ASPECT_DISPLAY.get(a, a) for a in aspects_found) if aspects_found else "General",
        "aspect_count":            len(aspects_found),
        **aspect_sentiments,
    }

# ── 5. Main execution ──────────────────────────────────────────────────────────

def main():
    INPUT  = "/Users/paul/Desktop/ChuyenDe4/main/bts_merged_reviews.csv"
    OUTPUT = "/Users/paul/Desktop/ChuyenDe4/main/bts_labeled_reviews.csv"

    print("Loading dataset …")
    df = pd.read_csv(INPUT, low_memory=False)
    print(f"  Loaded {len(df):,} rows × {len(df.columns)} columns")

    # Drop rows with no usable text content
    df = df.dropna(subset=["review_text"])
    df = df[df["review_text"].str.strip().str.len() > 10].copy()
    print(f"  After filtering: {len(df):,} rows with usable text")

    print("Labeling reviews (this may take ~1-2 min) …")
    labels = df.apply(label_review, axis=1, result_type="expand")
    df_labeled = pd.concat([df.reset_index(drop=True),
                            labels.reset_index(drop=True)], axis=1)

    # ── Summary statistics ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("LABELING SUMMARY REPORT")
    print("=" * 60)

    print(f"\n{'Total labeled reviews:':<35} {len(df_labeled):,}")
    print(f"\n{'[1] Relevance:':}")
    print(df_labeled["relevant"].value_counts().to_string())

    print(f"\n{'[2] Overall Sentiment:':}")
    print(df_labeled["overall_sentiment"].value_counts().to_string())

    print(f"\n{'[3] Rating Distribution:':}")
    print(df_labeled["review_rating"].value_counts().sort_index().to_string())

    print(f"\n{'[4] Primary Aspect Distribution:':}")
    print(df_labeled["primary_aspect"].value_counts().to_string())

    print(f"\n{'[5] Aspects Detected per review:':}")
    print(df_labeled["aspect_count"].describe().round(2).to_string())

    print(f"\n{'[6] Cross-Aspect Sentiment Breakdown:':}")
    sentiment_cols = [c for c in df_labeled.columns if c.startswith("sentiment_")]
    for col in sentiment_cols:
        aspect_name = col.replace("sentiment_", "")
        display_name = ASPECT_DISPLAY.get(aspect_name, aspect_name)
        counts = df_labeled[col].value_counts()
        total = counts.sum()
        print(f"\n  {display_name}:")
        for sent in ["Positive", "Neutral", "Negative", "Not Mentioned"]:
            cnt = counts.get(sent, 0)
            pct = cnt / total * 100 if total else 0
            bar = "█" * int(pct / 2)
            print(f"    {sent:<15} {cnt:>5,} ({pct:5.1f}%) {bar}")

    # Save
    df_labeled.to_csv(OUTPUT, index=False)
    print(f"\n{'✓':} Saved labeled dataset → {OUTPUT}")

    # Show sample labeled rows
    print("\n--- Sample Labeled Reviews ---")
    sample = df_labeled[["review_rating", "overall_sentiment",
                          "primary_aspect", "aspects_detected"]].head(10)
    print(sample.to_string())


if __name__ == "__main__":
    main()
