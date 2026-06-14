"""
Evaluation module for Codeforces Problem Difficulty Predictor.
Generates classification reports, confusion matrices, and SHAP analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, f1_score
)
import joblib
import shap
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODEL_PATH, PIPELINE_PATH, RATING_LABELS, MODEL_DIR


def load_artifacts():
    """Load trained model and test data."""
    model = joblib.load(MODEL_PATH)
    pipeline = joblib.load(PIPELINE_PATH)
    test_data = joblib.load(os.path.join(MODEL_DIR, "test_data.joblib"))
    return model, pipeline, test_data


def plot_confusion_matrix(y_true, y_pred, labels, save_path=None):
    """Plot and save confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # Raw counts
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels, ax=axes[0]
    )
    axes[0].set_title("Confusion Matrix (Counts)", fontsize=14)
    axes[0].set_xlabel("Predicted", fontsize=12)
    axes[0].set_ylabel("Actual", fontsize=12)
    
    # Normalized
    sns.heatmap(
        cm_normalized, annot=True, fmt=".2f", cmap="Oranges",
        xticklabels=labels, yticklabels=labels, ax=axes[1]
    )
    axes[1].set_title("Confusion Matrix (Normalized)", fontsize=14)
    axes[1].set_xlabel("Predicted", fontsize=12)
    axes[1].set_ylabel("Actual", fontsize=12)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  📊 Confusion matrix saved to {save_path}")
    plt.close()


def plot_shap_analysis(model, X_test, feature_names, save_dir=None):
    """Generate SHAP summary and feature importance plots."""
    print("\n  Computing SHAP values (this may take a minute)...")
    
    # Use a sample for speed
    sample_size = min(500, len(X_test))
    X_sample = X_test[:sample_size]
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    
    # Summary plot (bar)
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # For multiclass, average absolute SHAP values across classes
    if isinstance(shap_values, list):
        mean_shap = np.mean([np.abs(sv) for sv in shap_values], axis=0)
    else:
        mean_shap = np.abs(shap_values)
    
    feature_importance = np.mean(mean_shap, axis=0)
    
    # Top 20 features
    top_indices = np.argsort(feature_importance)[-20:]
    top_features = [feature_names[i] for i in top_indices]
    top_importances = feature_importance[top_indices]
    
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_features)))
    ax.barh(range(len(top_features)), top_importances, color=colors)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features)
    ax.set_xlabel("Mean |SHAP Value|", fontsize=12)
    ax.set_title("Top 20 Feature Importances (SHAP)", fontsize=14)
    plt.tight_layout()
    
    if save_dir:
        path = os.path.join(save_dir, "shap_feature_importance.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  📊 SHAP importance plot saved to {path}")
    plt.close()
    
    return feature_importance


def plot_class_distribution(y_train, y_test, labels, save_path=None):
    """Plot training and test set class distributions."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for ax, y, title in [
        (axes[0], y_train, "Training Set"),
        (axes[1], y_test, "Test Set"),
    ]:
        unique, counts = np.unique(y, return_counts=True)
        label_names = [labels[i] for i in unique]
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique)))
        ax.bar(label_names, counts, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_title(f"Class Distribution — {title}", fontsize=12)
        ax.set_xlabel("Difficulty Class")
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=45)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  📊 Class distribution saved to {save_path}")
    plt.close()


def evaluate():
    """Run full evaluation pipeline."""
    print("=" * 60)
    print("CODEFORCES PROBLEM DIFFICULTY PREDICTOR — EVALUATION")
    print("=" * 60)
    
    # Load artifacts
    model, pipeline, test_data = load_artifacts()
    X_test = test_data["X_test"]
    y_test = test_data["y_test"]
    X_train = test_data["X_train"]
    y_train = test_data["y_train"]
    feature_names = test_data["feature_names"]
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Classification report
    print("\n📋 CLASSIFICATION REPORT")
    print("-" * 50)
    
    # Map encoded labels back to rating strings
    label_names = pipeline.label_encoder.classes_
    report = classification_report(y_test, y_pred, target_names=label_names)
    print(report)
    
    # Overall metrics
    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")
    
    print(f"Overall Accuracy:  {accuracy:.4f}")
    print(f"Macro F1 Score:    {macro_f1:.4f}")
    print(f"Weighted F1 Score: {weighted_f1:.4f}")
    
    # Create output directory for plots
    plot_dir = os.path.join(os.path.dirname(MODEL_PATH), "plots")
    os.makedirs(plot_dir, exist_ok=True)
    
    # Confusion matrix
    print("\n📊 Generating plots...")
    plot_confusion_matrix(
        y_test, y_pred, label_names,
        save_path=os.path.join(plot_dir, "confusion_matrix.png")
    )
    
    # Class distribution
    plot_class_distribution(
        y_train, y_test, label_names,
        save_path=os.path.join(plot_dir, "class_distribution.png")
    )
    
    # SHAP analysis
    try:
        feature_importance = plot_shap_analysis(
            model, X_test, feature_names, save_dir=plot_dir
        )
        
        # Print top 10 most important features
        print("\n🔍 TOP 10 MOST IMPORTANT FEATURES (SHAP)")
        print("-" * 40)
        top_indices = np.argsort(feature_importance)[-10:][::-1]
        for rank, idx in enumerate(top_indices, 1):
            print(f"  {rank:2d}. {feature_names[idx]:<30s} {feature_importance[idx]:.4f}")
    except Exception as e:
        print(f"  ⚠️  SHAP analysis failed: {e}")
    
    print(f"\n✅ All plots saved to {plot_dir}/")


if __name__ == "__main__":
    evaluate()
