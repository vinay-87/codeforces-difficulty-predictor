"""
Data collection module for Codeforces Problem Difficulty Predictor.
Fetches problem data from the Codeforces API and saves to CSV.
"""

import requests
import pandas as pd
import time
import re
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CF_PROBLEMS_ENDPOINT, RAW_DATA_PATH, DATA_DIR


def fetch_problems() -> pd.DataFrame:
    """
    Fetch all problems from the Codeforces API.
    
    Returns:
        pd.DataFrame with columns: contest_id, index, name, rating, tags,
        time_limit, memory_limit, solved_count
    """
    print("[1/3] Fetching problems from Codeforces API...")
    
    response = requests.get(CF_PROBLEMS_ENDPOINT, timeout=30)
    response.raise_for_status()
    data = response.json()
    
    if data["status"] != "OK":
        raise RuntimeError(f"API returned status: {data['status']}")
    
    problems = data["result"]["problems"]
    statistics = data["result"]["problemStatistics"]
    
    # Build solved_count lookup
    solved_map = {}
    for stat in statistics:
        key = (stat["contestId"], stat["index"])
        solved_map[key] = stat.get("solvedCount", 0)
    
    records = []
    for prob in problems:
        contest_id = prob.get("contestId")
        index = prob.get("index", "")
        name = prob.get("name", "")
        rating = prob.get("rating")
        tags = prob.get("tags", [])
        
        # Skip problems without ratings (unrated)
        if rating is None:
            continue
        
        solved_count = solved_map.get((contest_id, index), 0)
        
        records.append({
            "contest_id": contest_id,
            "index": index,
            "name": name,
            "rating": rating,
            "tags": "|".join(tags),
            "tag_count": len(tags),
            "solved_count": solved_count,
        })
    
    df = pd.DataFrame(records)
    print(f"   Fetched {len(df)} rated problems.")
    return df


def fetch_problem_statements(df: pd.DataFrame, max_problems: int = 5000) -> pd.DataFrame:
    """
    Fetch problem statements via web scraping for text features.
    Falls back to using problem name + tags if scraping fails.
    
    Args:
        df: DataFrame with contest_id and index columns
        max_problems: Maximum number of statements to fetch
    
    Returns:
        DataFrame with added 'statement' column
    """
    print(f"[2/3] Generating text features for {min(len(df), max_problems)} problems...")
    
    statements = []
    for idx, row in df.head(max_problems).iterrows():
        # Create a synthetic statement from available metadata
        # This avoids aggressive web scraping while providing text features
        tag_text = row["tags"].replace("|", " ")
        name_text = row["name"]
        
        # Combine name and tags as text features
        # In production, you would scrape actual problem statements
        statement = f"{name_text} {tag_text}"
        statements.append(statement)
        
        if (idx + 1) % 1000 == 0:
            print(f"   Processed {idx + 1}/{min(len(df), max_problems)} problems...")
    
    # For problems beyond max_problems, use just name + tags
    for idx in range(min(len(df), max_problems), len(df)):
        row = df.iloc[idx]
        tag_text = row["tags"].replace("|", " ")
        statements.append(f"{row['name']} {tag_text}")
    
    df["statement"] = statements
    return df


def enrich_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived metadata features.
    """
    print("[3/3] Enriching metadata features...")
    
    # Text statistics
    df["name_length"] = df["name"].str.len()
    df["name_word_count"] = df["name"].str.split().str.len()
    df["statement_length"] = df["statement"].str.len()
    df["statement_word_count"] = df["statement"].str.split().str.len()
    
    # Unique words in statement
    df["unique_words"] = df["statement"].apply(
        lambda x: len(set(x.lower().split())) if isinstance(x, str) else 0
    )
    
    # Average word length
    df["avg_word_length"] = df["statement"].apply(
        lambda x: (
            sum(len(w) for w in x.split()) / max(len(x.split()), 1)
            if isinstance(x, str) else 0
        )
    )
    
    # Has numbers in name (often indicates mathematical problems)
    df["name_has_numbers"] = df["name"].str.contains(r"\d", regex=True).astype(int)
    
    # Contest ID as a proxy for time period (higher = more recent)
    df["contest_id_normalized"] = (
        (df["contest_id"] - df["contest_id"].min())
        / (df["contest_id"].max() - df["contest_id"].min() + 1)
    )
    
    # Log-transformed solved count (popularity proxy)
    df["log_solved_count"] = df["solved_count"].apply(
        lambda x: max(0, pd.np.log1p(x)) if hasattr(pd, "np") else 0
    )
    # Fallback for newer pandas without pd.np
    import numpy as np
    df["log_solved_count"] = np.log1p(df["solved_count"].values)
    
    return df


def collect_and_save():
    """Main data collection pipeline."""
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Step 1: Fetch problems from API
    df = fetch_problems()
    
    # Step 2: Add text features
    df = fetch_problem_statements(df)
    
    # Step 3: Enrich with derived features
    df = enrich_metadata(df)
    
    # Save to CSV
    df.to_csv(RAW_DATA_PATH, index=False)
    print(f"\n✅ Saved {len(df)} problems to {RAW_DATA_PATH}")
    print(f"   Rating distribution:")
    print(df["rating"].value_counts().sort_index().to_string())
    
    return df


if __name__ == "__main__":
    collect_and_save()
