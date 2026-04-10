import requests
import pandas as pd
import time
from datetime import datetime
from langdetect import detect, LangDetectException
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ==========================================
# 1. CONFIGURATION & MASSIVE QUERY LIST
# ==========================================
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

TARGET_RECORDS = 10000
# Added broader travel subreddits to capture tourist reviews
subreddits = "Bangkok+Thailand+ThailandTourism+travel+solotravel+digitalnomad+bkk"

# EXHAUSTIVE list of stations and common terms to force maximum data retrieval
search_queries =[
    # General terms
    {"query": "BTS Skytrain", "bts_line": "General"},
    {"query": "Bangkok BTS", "bts_line": "General"},
    {"query": "Rabbit Card", "bts_line": "General"},
    {"query": "Skytrain Bangkok", "bts_line": "General"},
    {"query": "Sukhumvit line", "bts_line": "Sukhumvit"},
    {"query": "Silom line", "bts_line": "Silom"},
    
    # Sukhumvit Line Stations (North to South)
    {"query": "Khu Khot BTS", "bts_line": "Sukhumvit"},
    {"query": "Mo Chit BTS", "bts_line": "Sukhumvit"},
    {"query": "Mochit BTS", "bts_line": "Sukhumvit"}, # Misspelling
    {"query": "Saphan Khwai BTS", "bts_line": "Sukhumvit"},
    {"query": "Ari BTS", "bts_line": "Sukhumvit"},
    {"query": "Sanam Pao BTS", "bts_line": "Sukhumvit"},
    {"query": "Victory Monument BTS", "bts_line": "Sukhumvit"},
    {"query": "Phaya Thai BTS", "bts_line": "Sukhumvit"},
    {"query": "Ratchathewi BTS", "bts_line": "Sukhumvit"},
    {"query": "Siam BTS", "bts_line": "Sukhumvit/Silom"}, # Interchange
    {"query": "Chit Lom BTS", "bts_line": "Sukhumvit"},
    {"query": "Chidlom BTS", "bts_line": "Sukhumvit"}, # Misspelling
    {"query": "Phloen Chit BTS", "bts_line": "Sukhumvit"},
    {"query": "Nana BTS", "bts_line": "Sukhumvit"},
    {"query": "Asok BTS", "bts_line": "Sukhumvit"},
    {"query": "Asoke BTS", "bts_line": "Sukhumvit"}, # Misspelling
    {"query": "Phrom Phong BTS", "bts_line": "Sukhumvit"},
    {"query": "Thong Lo BTS", "bts_line": "Sukhumvit"},
    {"query": "Thonglor BTS", "bts_line": "Sukhumvit"}, # Misspelling
    {"query": "Ekkamai BTS", "bts_line": "Sukhumvit"},
    {"query": "Ekamai BTS", "bts_line": "Sukhumvit"}, # Misspelling
    {"query": "Phra Khanong BTS", "bts_line": "Sukhumvit"},
    {"query": "On Nut BTS", "bts_line": "Sukhumvit"},
    {"query": "Bang Chak BTS", "bts_line": "Sukhumvit"},
    {"query": "Punnawithi BTS", "bts_line": "Sukhumvit"},
    {"query": "Udom Suk BTS", "bts_line": "Sukhumvit"},
    {"query": "Bang Na BTS", "bts_line": "Sukhumvit"},
    {"query": "Bearing BTS", "bts_line": "Sukhumvit"},
    {"query": "Samrong BTS", "bts_line": "Sukhumvit"},
    
    # Silom Line Stations
    {"query": "National Stadium BTS", "bts_line": "Silom"},
    {"query": "Ratchadamri BTS", "bts_line": "Silom"},
    {"query": "Sala Daeng BTS", "bts_line": "Silom"},
    {"query": "Chong Nonsi BTS", "bts_line": "Silom"},
    {"query": "Saint Louis BTS", "bts_line": "Silom"},
    {"query": "Surasak BTS", "bts_line": "Silom"},
    {"query": "Saphan Taksin BTS", "bts_line": "Silom"},
    {"query": "Krung Thon Buri BTS", "bts_line": "Silom"},
    {"query": "Wongwian Yai BTS", "bts_line": "Silom"}
]

# Adding Time Filters multiplies our results (Bypasses the 1000 limit)
time_filters = ["all", "year", "month"]
sort_methods =["relevance", "new", "top", "comments"]

data_rows =[]
seen_ids = set() # To track duplicates exactly

analyzer = SentimentIntensityAnalyzer()

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def detect_language(text):
    if not text or text in ["[deleted]", "[removed]"]: return "en"
    try:
        return detect(text)
    except LangDetectException:
        return "en"

def get_image_count(post_data):
    if post_data.get("is_gallery"):
        return len(post_data.get("gallery_data", {}).get("items",[]))
    elif post_data.get("post_hint") == "image":
        return 1
    return 0

def calculate_star_rating(text):
    if not text: return 3
    score = analyzer.polarity_scores(text)['compound']
    if score <= -0.5: return 1
    elif score <= -0.05: return 2
    elif score < 0.05: return 3
    elif score < 0.5: return 4
    else: return 5

# ==========================================
# 3. 10K DEEP CRAWLING LOGIC
# ==========================================
print(f"Starting deep crawl to reach {TARGET_RECORDS} Reddit records. This will take some time...")

for search in search_queries:
    if len(seen_ids) >= TARGET_RECORDS:
        break
        
    for t_filter in time_filters:
        if len(seen_ids) >= TARGET_RECORDS:
            break
            
        for sort_method in sort_methods:
            if len(seen_ids) >= TARGET_RECORDS:
                break
                
            print(f"\nSearching: '{search['query']}' | Time: {t_filter} | Sort: {sort_method}...")
            after_token = None
            
            while True:
                url = f"https://www.reddit.com/r/{subreddits}/search.json"
                params = {
                    "q": search['query'],
                    "restrict_sr": "on",
                    "limit": 100,
                    "sort": sort_method, 
                    "t": t_filter           
                }
                
                if after_token:
                    params["after"] = after_token
                    
                try:
                    response = requests.get(url, headers=HEADERS, params=params, timeout=15)
                except requests.exceptions.RequestException as e:
                    print(f"Network error: {e}")
                    time.sleep(5)
                    break
                    
                if response.status_code == 429:
                    print("⚠️ Rate limited by Reddit. Sleeping for 30 seconds to recover...")
                    time.sleep(30)
                    continue # Retry same request
                    
                if response.status_code != 200:
                    print(f"Failed to fetch (Status {response.status_code}). Skipping...")
                    time.sleep(5)
                    break
                    
                data = response.json().get("data", {})
                posts = data.get("children",[])
                
                if not posts:
                    break # No more posts for this specific combination
                    
                new_records_count = 0
                
                for post in posts:
                    p = post["data"]
                    post_id = p.get("id")
                    
                    if post_id in seen_ids or p.get("author") == "[deleted]":
                        continue
                        
                    text_content = p.get("selftext", "")
                    title_content = p.get("title", "")
                    
                    full_text = f"{title_content}. {text_content}".strip()
                    lang = detect_language(full_text)
                    calculated_rating = calculate_star_rating(full_text)
                    
                    row = {
                        "entity_name": "BTS Skytrain",
                        "entity_id": f"r/{p.get('subreddit')}",
                        "bts_line": search['bts_line'],
                        "category": "Reddit Post",
                        "review_id": post_id,
                        "review_title": title_content,
                        "review_text": text_content if text_content else "(Title only post)",
                        "review_rating": calculated_rating, 
                        "published_date": datetime.utcfromtimestamp(p.get("created_utc", 0)).strftime('%Y-%m-%d %H:%M:%S'),
                        "created_at_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        "trip_type": None, 
                        "stay_date": None, 
                        "review_language": lang,
                        "is_translated": False,
                        "original_language": lang,
                        "like_count": p.get("score", 0), 
                        "images_count": get_image_count(p),
                        "reviewer_id": p.get("author_fullname"),
                        "reviewer_name": p.get("author"),
                        "reviewer_username": p.get("author"),
                        "reviewer_contribution_count": p.get("num_comments", 0), # Store Number of Comments here!
                        "reviewer_hometown": p.get("author_flair_text"),
                        "reviewer_profile_link": f"https://www.reddit.com/user/{p.get('author')}",
                        "has_owner_response": False,
                        "owner_response_text": None,
                        "owner_response_date": None,
                        "review_link": f"https://www.reddit.com{p.get('permalink')}",
                        "entity_link": f"https://www.reddit.com/r/{p.get('subreddit')}"
                    }
                    
                    data_rows.append(row)
                    seen_ids.add(post_id)
                    new_records_count += 1
                    
                    if len(seen_ids) >= TARGET_RECORDS:
                        break
                        
                print(f"  -> Scraped {new_records_count} new posts. Total unique records: {len(seen_ids)}")
                
                if len(seen_ids) >= TARGET_RECORDS:
                    break
                    
                after_token = data.get("after")
                if not after_token:
                    break 
                    
                # Strict 3-second delay to prevent IP bans during a 10K scrape
                time.sleep(3) 

# ==========================================
# 4. EXPORT TO CSV
# ==========================================
df = pd.DataFrame(data_rows)

columns_order =[
    "entity_name", "entity_id", "bts_line", "category", "review_id", 
    "review_title", "review_text", "review_rating", "published_date", 
    "created_at_date", "trip_type", "stay_date", "review_language", 
    "is_translated", "original_language", "like_count", "images_count", 
    "reviewer_id", "reviewer_name", "reviewer_username", 
    "reviewer_contribution_count", "reviewer_hometown", "reviewer_profile_link", 
    "has_owner_response", "owner_response_text", "owner_response_date", 
    "review_link", "entity_link"
]
df = df[columns_order]

# Save the massive dataset
df.to_csv("bts_10k_reddit_reviews.csv", index=False, encoding="utf-8-sig")
print(f"\n*** COMPLETE ***\nSuccessfully scraped and exported {len(df)} records to 'bts_10k_reddit_reviews.csv'.")