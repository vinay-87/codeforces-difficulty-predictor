"""
Feature engineering pipeline for Codeforces Problem Difficulty Predictor.
Transforms raw problem data into ML-ready feature matrices.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    TFIDF_MAX_FEATURES, TFIDF_NGRAM_RANGE, SVD_COMPONENTS,
    LDA_N_TOPICS, ALL_TAGS, RATING_BINS, RATING_LABELS,
    VECTORIZER_PATH, LDA_PATH, SVD_PATH, LABEL_ENCODER_PATH,
    PIPELINE_PATH, MODEL_DIR
)


class FeatureEngineer:
    """
    Feature engineering pipeline that extracts text, tag, and metadata
    features from Codeforces problem data.
    """
    
    def __init__(self):
        self.tfidf = TfidfVectorizer(
            max_features=TFIDF_MAX_FEATURES,
            ngram_range=TFIDF_NGRAM_RANGE,
            stop_words="english",
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
        )
        self.svd = TruncatedSVD(n_components=SVD_COMPONENTS, random_state=42)
        self.lda = LatentDirichletAllocation(
            n_components=LDA_N_TOPICS,
            random_state=42,
            max_iter=20,
            learning_method="online",
        )
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.is_fitted = False
    
    def _extract_tag_features(self, df: pd.DataFrame) -> np.ndarray:
        """One-hot encode problem tags."""
        tag_matrix = np.zeros((len(df), len(ALL_TAGS)), dtype=np.float32)
        
        for i, tags_str in enumerate(df["tags"]):
            if isinstance(tags_str, str):
                tags = tags_str.split("|")
                for tag in tags:
                    tag = tag.strip()
                    if tag in ALL_TAGS:
                        tag_idx = ALL_TAGS.index(tag)
                        tag_matrix[i, tag_idx] = 1.0
        
        return tag_matrix
    
    def _extract_text_features(self, df: pd.DataFrame, fit: bool = True) -> np.ndarray:
        """Extract TF-IDF + SVD features from problem statements."""
        texts = df["statement"].fillna("").values
        
        if fit:
            tfidf_matrix = self.tfidf.fit_transform(texts)
            svd_features = self.svd.fit_transform(tfidf_matrix)
        else:
            tfidf_matrix = self.tfidf.transform(texts)
            svd_features = self.svd.transform(tfidf_matrix)
        
        return svd_features
    
    def _extract_topic_features(self, df: pd.DataFrame, fit: bool = True) -> np.ndarray:
        """Extract LDA topic distributions."""
        texts = df["statement"].fillna("").values
        
        # Use a separate TF vectorizer for LDA (no IDF, no sublinear)
        from sklearn.feature_extraction.text import CountVectorizer
        
        if fit:
            if not hasattr(self, "_count_vec"):
                self._count_vec = CountVectorizer(
                    max_features=3000, stop_words="english", min_df=2
                )
            count_matrix = self._count_vec.fit_transform(texts)
            topic_features = self.lda.fit_transform(count_matrix)
        else:
            count_matrix = self._count_vec.transform(texts)
            topic_features = self.lda.transform(count_matrix)
        
        return topic_features
    
    def _extract_metadata_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract numerical metadata features."""
        feature_cols = [
            "tag_count",
            "name_length",
            "name_word_count",
            "statement_length",
            "statement_word_count",
            "unique_words",
            "avg_word_length",
            "name_has_numbers",
            "contest_id_normalized",
            "log_solved_count",
        ]
        
        # Only use columns that exist
        available_cols = [c for c in feature_cols if c in df.columns]
        meta_features = df[available_cols].fillna(0).values.astype(np.float32)
        
        return meta_features
    
    def _create_labels(self, df: pd.DataFrame, fit: bool = True) -> np.ndarray:
        """Bin ratings into difficulty classes."""
        rating_classes = pd.cut(
            df["rating"],
            bins=RATING_BINS,
            labels=RATING_LABELS,
            right=True,
        ).astype(str)
        
        if fit:
            labels = self.label_encoder.fit_transform(rating_classes)
        else:
            labels = self.label_encoder.transform(rating_classes)
        
        return labels
    
    def fit_transform(self, df: pd.DataFrame):
        """
        Fit the feature pipeline and transform data.
        
        Returns:
            X: Feature matrix (n_samples, n_features)
            y: Label array (n_samples,)
            feature_names: List of feature names
        """
        print("Engineering features...")
        
        # Extract all feature groups
        print("  - Tag features (one-hot)...")
        tag_features = self._extract_tag_features(df)
        
        print("  - Text features (TF-IDF + SVD)...")
        text_features = self._extract_text_features(df, fit=True)
        
        print("  - Topic features (LDA)...")
        topic_features = self._extract_topic_features(df, fit=True)
        
        print("  - Metadata features...")
        meta_features = self._extract_metadata_features(df)
        
        # Combine all features
        X = np.hstack([tag_features, text_features, topic_features, meta_features])
        
        # Scale features
        X = self.scaler.fit_transform(X)
        
        # Create labels
        y = self._create_labels(df, fit=True)
        
        # Build feature names
        feature_names = (
            [f"tag_{t}" for t in ALL_TAGS]
            + [f"tfidf_svd_{i}" for i in range(SVD_COMPONENTS)]
            + [f"lda_topic_{i}" for i in range(LDA_N_TOPICS)]
            + [
                "tag_count", "name_length", "name_word_count",
                "statement_length", "statement_word_count",
                "unique_words", "avg_word_length", "name_has_numbers",
                "contest_id_normalized", "log_solved_count",
            ]
        )
        
        self.is_fitted = True
        self.feature_names = feature_names
        
        print(f"  ✅ Feature matrix shape: {X.shape}")
        print(f"     ({X.shape[0]} samples × {X.shape[1]} features)")
        
        return X, y, feature_names
    
    def transform(self, df: pd.DataFrame):
        """Transform new data using fitted pipeline."""
        if not self.is_fitted:
            raise RuntimeError("Pipeline not fitted. Call fit_transform first.")
        
        tag_features = self._extract_tag_features(df)
        text_features = self._extract_text_features(df, fit=False)
        topic_features = self._extract_topic_features(df, fit=False)
        meta_features = self._extract_metadata_features(df)
        
        X = np.hstack([tag_features, text_features, topic_features, meta_features])
        X = self.scaler.transform(X)
        
        return X
    
    def save(self):
        """Save fitted pipeline components."""
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(self, PIPELINE_PATH)
        print(f"  💾 Pipeline saved to {PIPELINE_PATH}")
    
    @staticmethod
    def load():
        """Load a saved pipeline."""
        return joblib.load(PIPELINE_PATH)
