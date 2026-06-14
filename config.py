"""
Configuration constants for Codeforces Difficulty Predictor.
"""

import os

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

RAW_DATA_PATH = os.path.join(DATA_DIR, "problems.csv")
FEATURES_PATH = os.path.join(DATA_DIR, "features.pkl")
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.joblib")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib")
LDA_PATH = os.path.join(MODEL_DIR, "lda_model.joblib")
SVD_PATH = os.path.join(MODEL_DIR, "svd_model.joblib")
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.joblib")
PIPELINE_PATH = os.path.join(MODEL_DIR, "feature_pipeline.joblib")

# --- Codeforces API ---
CF_API_BASE = "https://codeforces.com/api"
CF_PROBLEMS_ENDPOINT = f"{CF_API_BASE}/problemset.problems"

# --- Rating Bins (difficulty classes) ---
# We bin ratings into 8 classes for classification
RATING_BINS = [0, 1000, 1200, 1400, 1600, 1800, 2100, 2500, 4000]
RATING_LABELS = [
    "800-1000",
    "1100-1200",
    "1300-1400",
    "1500-1600",
    "1700-1800",
    "1900-2100",
    "2200-2500",
    "2600+",
]

# --- Feature Engineering ---
TFIDF_MAX_FEATURES = 5000
TFIDF_NGRAM_RANGE = (1, 2)
SVD_COMPONENTS = 20
LDA_N_TOPICS = 15

# --- Model Training ---
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5
OPTUNA_N_TRIALS = 100

# --- All known Codeforces tags ---
ALL_TAGS = [
    "implementation", "dp", "math", "greedy", "brute force",
    "data structures", "constructive algorithms", "graphs",
    "sortings", "binary search", "dfs and similar", "trees",
    "strings", "number theory", "combinatorics", "geometry",
    "bitmasks", "two pointers", "dsu", "shortest paths",
    "probabilities", "divide and conquer", "hashing",
    "games", "flows", "interactive", "matrices",
    "string suffix structures", "fft", "graph matchings",
    "ternary search", "expression parsing", "meet-in-the-middle",
    "2-sat", "chinese remainder theorem", "schedules",
]

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
