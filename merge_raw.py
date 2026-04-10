#!/usr/bin/env python3
"""
merge_raw.py
============
Merges the two raw BTS Skytrain review files into a single clean dataset.

Sources:
  • bts_reddit_reviews_expanded_nlp.csv   → category: "reddit",     source: "reddit"
  • 11k_bts_skytrain_reviews.csv          → category: "tripadvisor", source: "tripadvisor"

Steps:
  1. Load both raw files (shared 28-column schema)
  2. Add / overwrite `source` column (not present in raw files)
  3. Deduplicate across both files
  4. Concatenate
  5. Sort by source + published_date (desc)
  6. Drop duplicate review_ids within each source
  7. Save to bts_merged_reviews_v2.csv
"""

import pandas as pd
import sys

RAW_REDDIT   = "/Users/paul/Desktop/ChuyenDe4/main/raw_data/bts_reddit_reviews_expanded_nlp.csv"
RAW_TRIP     = "/Users/paul/Desktop/ChuyenDe4/main/raw_data/11k_bts_skytrain_reviews.csv"
OUT_MERGED   = "/Users/paul/Desktop/ChuyenDe4/main/bts_merged_reviews.csv"
OUT_LABELED  = "/Users/paul/Desktop/ChuyenDe4/main/bts_labeled_reviews.csv"

# ── 1. Load raw files ─────────────────────────────────────────────────────────
print("Loading raw files …")
df_reddit = pd.read_csv(RAW_REDDIT, low_memory=False)
df_trip   = pd.read_csv(RAW_TRIP,   low_memory=False)

print(f"  Reddit      : {len(df_reddit):,} rows × {len(df_reddit.columns)} cols")
print(f"  TripAdvisor : {len(df_trip):,}  rows × {len(df_trip.columns)}  cols")

# ── 2. Assign source + normalise category ─────────────────────────────────────
df_reddit["source"] = "reddit"
df_trip["source"]   = "tripadvisor"

# Normalise category names for TripAdvisor to match desired labels
df_trip["category"] = df_trip["category"].replace({
    "transit_skytrain": "tripadvisor",
    "transit_line":     "tripadvisor",
    "transit_market":    "tripadvisor",
    "transit_info":      "tripadvisor",
})

# Reddit already has "Reddit Post" — normalise to lowercase "reddit"
df_reddit["category"] = df_reddit["category"].str.lower().str.replace(" ", "_")

# ── 3. Column alignment ────────────────────────────────────────────────────────
shared_cols = list(df_reddit.columns.intersection(df_trip.columns))
df_reddit   = df_reddit[shared_cols]
df_trip     = df_trip[shared_cols]
print(f"\nShared columns ({len(shared_cols)}): {shared_cols}")

# ── 4. Deduplicate within each source ─────────────────────────────────────────
before_reddit = len(df_reddit)
before_trip   = len(df_trip)

df_reddit = df_reddit.drop_duplicates(subset=["review_id"], keep="first")
df_trip   = df_trip.drop_duplicates(subset=["review_id"], keep="first")

print(f"\nDeduplication:")
print(f"  Reddit      : {before_reddit:,} → {len(df_reddit):,}  (removed {before_reddit - len(df_reddit):,})")
print(f"  TripAdvisor  : {before_trip:,} → {len(df_trip):,}  (removed {before_trip - len(df_trip):,})")

# ── 5. Concatenate ─────────────────────────────────────────────────────────────
df_merged = pd.concat([df_reddit, df_trip], ignore_index=True)
print(f"\nTotal merged (before cross-source dedup): {len(df_merged):,}")

# ── 6. Cross-source deduplication ─────────────────────────────────────────────
df_merged = df_merged.drop_duplicates(subset=["review_id"], keep="first")
print(f"Total merged (after  cross-source dedup): {len(df_merged):,}")

# ── 7. Parse dates and sort ────────────────────────────────────────────────────
df_merged["published_date_parsed"] = pd.to_datetime(
    df_merged["published_date"], errors="coerce"
)
df_merged = df_merged.sort_values(
    ["source", "published_date_parsed"],
    ascending=[True, False],
    na_position="last"
).reset_index(drop=True)

df_merged = df_merged.drop(columns=["published_date_parsed"])

# ── 8. Save merged CSV ─────────────────────────────────────────────────────────
df_merged.to_csv(OUT_MERGED, index=False)
print(f"\n✓ Merged CSV saved → {OUT_MERGED}")
print(f"  Final shape: {df_merged.shape}")

# ── 9. Carry over labels from existing labeled file ───────────────────────────
print("\nChecking existing labels for carry-over …")
try:
    df_old_labels = pd.read_csv(LABELED_OLD, low_memory=False,
                                 usecols=["review_id"] + [
                                     c for c in [
                                         "relevant", "overall_sentiment",
                                         "primary_aspect", "aspects_detected",
                                         "aspect_count",
                                         "sentiment_staff", "sentiment_punctuality",
                                         "sentiment_crowding", "sentiment_cleanliness",
                                         "sentiment_fare_payment", "sentiment_safety",
                                         "sentiment_route_connectivity",
                                         "sentiment_signage",
                                         "sentiment_infrastructure",
                                         "sentiment_overall",
                                     ]
                                     if c in pd.read_csv(LABELED_OLD, low_memory=False, nrows=1).columns
                                 ])
    print(f"  Old labels shape: {df_old_labels.shape}")
except Exception as e:
    print(f"  Could not load old labels: {e}")
    df_old_labels = pd.DataFrame()

if not df_old_labels.empty:
    matched = df_merged["review_id"].isin(df_old_labels["review_id"]).sum()
    print(f"  Review IDs in new merge also in old labels: {matched:,}")
    df_merged = df_merged.merge(df_old_labels, on="review_id", how="left")
    label_cols = [c for c in df_merged.columns if c.startswith("relevant")
                  or c.startswith("overall_")
                  or c.startswith("primary_")
                  or c.startswith("aspects_")
                  or c.startswith("aspect_")
                  or c.startswith("sentiment_")]

    already_labeled = df_merged["overall_sentiment"].notna().sum()
    new_to_label    = df_merged["overall_sentiment"].isna().sum()
    print(f"  Already labeled (carry-over): {already_labeled:,}")
    print(f"  Need labeling (new rows):     {new_to_label:,}")

    # ── 10. Label only the new (unlabeled) rows ────────────────────────────────
    if new_to_label > 0:
        print(f"\nLabeling {new_to_label:,} new rows …")
        # Import and run labeling on the new rows only
        import importlib, label_bts_reviews as lbr
        importlib.reload(lbr)

        df_unlabeled = df_merged[df_merged["overall_sentiment"].isna()].copy()
        new_labels = df_unlabeled.apply(lbr.label_review, axis=1, result_type="expand")
        new_labels.index = df_unlabeled.index
        for col in new_labels.columns:
            df_merged.loc[df_merged.index.isin(df_unlabeled.index), col] = new_labels[col].values

    # ── 11. Save labeled CSV ───────────────────────────────────────────────────
    df_merged.to_csv(OUT_LABELED, index=False)
    print(f"\n✓ Labeled CSV saved → {OUT_LABELED}")
else:
    print("\n⚠ No old labels found — skipping carry-over. "
          "Run label_bts_reviews.py on the new merge to label all rows.")

# ── 12. Final summary ─────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("FINAL MERGE SUMMARY")
print("=" * 55)
print(f"\nSource breakdown:")
print(df_merged["source"].value_counts().to_string())
print(f"\nCategory breakdown:")
print(df_merged["category"].value_counts().to_string())
print(f"\nRating distribution:")
print(df_merged["review_rating"].value_counts().sort_index().to_string())
print(f"\nBTS Line breakdown:")
print(df_merged["bts_line"].value_counts().to_string())
print(f"\nAll columns ({len(df_merged.columns)}):")
print(df_merged.columns.tolist())
