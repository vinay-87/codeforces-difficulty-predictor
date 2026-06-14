"""
Streamlit web application for Codeforces Problem Difficulty Predictor.
Interactive demo for predicting problem difficulty ratings.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import MODEL_PATH, PIPELINE_PATH, RATING_LABELS, RAW_DATA_PATH, ALL_TAGS


@st.cache_resource
def load_model():
    """Load the trained model and feature pipeline."""
    model = joblib.load(MODEL_PATH)
    pipeline = joblib.load(PIPELINE_PATH)
    return model, pipeline


@st.cache_data
def load_problem_data():
    """Load the problem dataset."""
    if os.path.exists(RAW_DATA_PATH):
        return pd.read_csv(RAW_DATA_PATH)
    return None


def predict_difficulty(model, pipeline, problem_data: dict):
    """Predict difficulty for a single problem."""
    df = pd.DataFrame([problem_data])
    
    # Ensure all required columns exist
    for col in ["statement", "tags", "tag_count", "name_length", "name_word_count",
                 "statement_length", "statement_word_count", "unique_words",
                 "avg_word_length", "name_has_numbers", "contest_id_normalized",
                 "log_solved_count"]:
        if col not in df.columns:
            df[col] = 0
    
    X = pipeline.transform(df)
    prediction = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]
    
    predicted_label = pipeline.label_encoder.inverse_transform([prediction])[0]
    
    return predicted_label, probabilities


def fetch_problem_info(contest_id: int, index: str):
    """Fetch problem info from Codeforces API."""
    url = f"https://codeforces.com/api/contest.standings?contestId={contest_id}&from=1&count=1"
    try:
        resp = requests.get(url, timeout=10)
        data = resp.json()
        if data["status"] == "OK":
            for prob in data["result"]["problems"]:
                if prob["index"] == index:
                    return prob
    except Exception:
        pass
    return None


def main():
    st.set_page_config(
        page_title="CF Difficulty Predictor",
        page_icon="🎯",
        layout="wide",
    )
    
    st.title("🎯 Codeforces Problem Difficulty Predictor")
    st.markdown(
        "Predict the difficulty rating of a Codeforces problem using "
        "machine learning. Enter problem details below or select from "
        "the dataset."
    )
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH) or not os.path.exists(PIPELINE_PATH):
        st.error(
            "⚠️ Model not found! Please run training first:\n"
            "```bash\npython -m src.data_collector\npython -m src.train\n```"
        )
        return
    
    model, pipeline = load_model()
    df = load_problem_data()
    
    st.sidebar.header("⚙️ Input Method")
    input_method = st.sidebar.radio(
        "Choose input method:",
        ["Manual Input", "Browse Dataset", "Codeforces URL"]
    )
    
    if input_method == "Manual Input":
        st.subheader("📝 Enter Problem Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            problem_name = st.text_input(
                "Problem Name",
                placeholder="e.g., Two Sum"
            )
            selected_tags = st.multiselect(
                "Problem Tags",
                options=ALL_TAGS,
                default=["implementation"]
            )
        
        with col2:
            solved_count = st.number_input(
                "Approximate Solved Count", min_value=0, value=5000, step=100
            )
            contest_id = st.number_input(
                "Contest ID (approximate)", min_value=1, value=1900, step=1
            )
        
        if st.button("🔮 Predict Difficulty", type="primary"):
            if not problem_name:
                st.warning("Please enter a problem name.")
            else:
                tag_str = "|".join(selected_tags)
                statement = f"{problem_name} {' '.join(selected_tags)}"
                
                problem_data = {
                    "name": problem_name,
                    "tags": tag_str,
                    "tag_count": len(selected_tags),
                    "statement": statement,
                    "solved_count": solved_count,
                    "contest_id": contest_id,
                    "name_length": len(problem_name),
                    "name_word_count": len(problem_name.split()),
                    "statement_length": len(statement),
                    "statement_word_count": len(statement.split()),
                    "unique_words": len(set(statement.lower().split())),
                    "avg_word_length": sum(len(w) for w in statement.split()) / max(len(statement.split()), 1),
                    "name_has_numbers": int(any(c.isdigit() for c in problem_name)),
                    "contest_id_normalized": min(contest_id / 2000, 1.0),
                    "log_solved_count": np.log1p(solved_count),
                }
                
                predicted_label, probabilities = predict_difficulty(
                    model, pipeline, problem_data
                )
                
                st.success(f"### Predicted Difficulty: **{predicted_label}**")
                
                # Show probability distribution
                st.subheader("📊 Confidence Distribution")
                labels = pipeline.label_encoder.classes_
                prob_df = pd.DataFrame({
                    "Difficulty Class": labels,
                    "Probability": probabilities
                }).sort_values("Probability", ascending=True)
                
                st.bar_chart(prob_df.set_index("Difficulty Class"))
    
    elif input_method == "Browse Dataset":
        if df is not None:
            st.subheader("📚 Problem Dataset")
            
            # Filters
            col1, col2, col3 = st.columns(3)
            with col1:
                min_rating = st.selectbox("Min Rating", [800, 1000, 1200, 1400, 1600, 1800, 2000], index=0)
            with col2:
                max_rating = st.selectbox("Max Rating", [1200, 1400, 1600, 1800, 2000, 2500, 3500], index=6)
            with col3:
                tag_filter = st.selectbox("Filter by Tag", ["All"] + ALL_TAGS[:20])
            
            filtered = df[(df["rating"] >= min_rating) & (df["rating"] <= max_rating)]
            if tag_filter != "All":
                filtered = filtered[filtered["tags"].str.contains(tag_filter, na=False)]
            
            st.dataframe(
                filtered[["name", "rating", "tags", "solved_count"]].head(100),
                use_container_width=True
            )
            
            st.metric("Problems shown", len(filtered.head(100)))
            st.metric("Total matching", len(filtered))
        else:
            st.warning("Dataset not found. Run data collection first.")
    
    elif input_method == "Codeforces URL":
        st.subheader("🔗 Predict from Codeforces URL")
        url = st.text_input(
            "Codeforces Problem URL",
            placeholder="e.g., https://codeforces.com/problemset/problem/1/A"
        )
        
        if st.button("🔮 Predict", type="primary"):
            if url:
                try:
                    parts = url.rstrip("/").split("/")
                    contest_id = int(parts[-2])
                    index = parts[-1]
                    
                    info = fetch_problem_info(contest_id, index)
                    if info:
                        tags = info.get("tags", [])
                        name = info.get("name", "Unknown")
                        tag_str = "|".join(tags)
                        statement = f"{name} {' '.join(tags)}"
                        
                        problem_data = {
                            "name": name,
                            "tags": tag_str,
                            "tag_count": len(tags),
                            "statement": statement,
                            "solved_count": 0,
                            "contest_id": contest_id,
                            "name_length": len(name),
                            "name_word_count": len(name.split()),
                            "statement_length": len(statement),
                            "statement_word_count": len(statement.split()),
                            "unique_words": len(set(statement.lower().split())),
                            "avg_word_length": sum(len(w) for w in statement.split()) / max(len(statement.split()), 1),
                            "name_has_numbers": int(any(c.isdigit() for c in name)),
                            "contest_id_normalized": min(contest_id / 2000, 1.0),
                            "log_solved_count": 0,
                        }
                        
                        predicted_label, probabilities = predict_difficulty(
                            model, pipeline, problem_data
                        )
                        
                        st.info(f"**Problem:** {name}\n**Tags:** {', '.join(tags)}")
                        
                        actual_rating = info.get("rating")
                        if actual_rating:
                            st.info(f"**Actual Rating:** {actual_rating}")
                        
                        st.success(f"### Predicted Difficulty: **{predicted_label}**")
                        
                        labels = pipeline.label_encoder.classes_
                        prob_df = pd.DataFrame({
                            "Difficulty Class": labels,
                            "Probability": probabilities
                        })
                        st.bar_chart(prob_df.set_index("Difficulty Class"))
                    else:
                        st.error("Could not fetch problem info. Check the URL.")
                except Exception as e:
                    st.error(f"Error parsing URL: {e}")
    
    # Sidebar info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📈 Model Info")
    st.sidebar.markdown("- **Algorithm:** XGBoost")
    st.sidebar.markdown("- **Features:** 80+")
    st.sidebar.markdown("- **Classes:** 8 difficulty levels")
    st.sidebar.markdown("- **Optimization:** Optuna (Bayesian)")
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "Built by [Vinay Kumar](https://github.com/vinay-87)"
    )


if __name__ == "__main__":
    main()
