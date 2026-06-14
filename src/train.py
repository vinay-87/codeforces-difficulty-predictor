"""
Model training pipeline for Codeforces Problem Difficulty Predictor.
Trains multiple models, performs hyperparameter optimization with Optuna,
and saves the best model.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import optuna
import joblib
import sys
import os
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    RAW_DATA_PATH, MODEL_PATH, TEST_SIZE, RANDOM_STATE,
    CV_FOLDS, OPTUNA_N_TRIALS, FEATURES_PATH
)
from src.feature_engineering import FeatureEngineer


def load_data() -> pd.DataFrame:
    """Load collected problem data."""
    if not os.path.exists(RAW_DATA_PATH):
        print("Data not found. Running data collection...")
        from src.data_collector import collect_and_save
        return collect_and_save()
    return pd.read_csv(RAW_DATA_PATH)


def train_baseline_models(X_train, X_test, y_train, y_test):
    """
    Train and evaluate baseline models for comparison.
    
    Returns:
        dict of model_name -> (model, train_acc, test_acc)
    """
    print("\n" + "=" * 60)
    print("BASELINE MODEL COMPARISON")
    print("=" * 60)
    
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE, multi_class="multinomial"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=15, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "SVM (RBF)": SVC(
            kernel="rbf", C=10, gamma="scale", random_state=RANDOM_STATE
        ),
    }
    
    results = {}
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        train_acc = model.score(X_train, y_train)
        test_acc = model.score(X_test, y_test)
        results[name] = (model, train_acc, test_acc)
        print(f"  Train Accuracy: {train_acc:.4f}")
        print(f"  Test Accuracy:  {test_acc:.4f}")
    
    return results


def optimize_xgboost(X_train, y_train, n_trials: int = None):
    """
    Optimize XGBoost hyperparameters using Optuna Bayesian optimization.
    
    Returns:
        Best XGBClassifier model
    """
    if n_trials is None:
        n_trials = OPTUNA_N_TRIALS
    
    n_classes = len(np.unique(y_train))
    
    print(f"\n{'=' * 60}")
    print(f"XGBOOST HYPERPARAMETER OPTIMIZATION ({n_trials} trials)")
    print("=" * 60)
    
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "objective": "multi:softprob",
            "num_class": n_classes,
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
            "eval_metric": "mlogloss",
            "verbosity": 0,
        }
        
        model = XGBClassifier(**params)
        scores = cross_val_score(
            model, X_train, y_train,
            cv=CV_FOLDS, scoring="accuracy", n_jobs=-1
        )
        return scores.mean()
    
    # Suppress Optuna logs
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print(f"\n  Best trial accuracy (CV): {study.best_value:.4f}")
    print(f"  Best parameters:")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")
    
    # Train final model with best parameters
    best_params = study.best_params
    best_params["objective"] = "multi:softprob"
    best_params["num_class"] = n_classes
    best_params["random_state"] = RANDOM_STATE
    best_params["n_jobs"] = -1
    best_params["eval_metric"] = "mlogloss"
    best_params["verbosity"] = 0
    
    best_model = XGBClassifier(**best_params)
    best_model.fit(X_train, y_train)
    
    return best_model, study


def main():
    """Main training pipeline."""
    print("=" * 60)
    print("CODEFORCES PROBLEM DIFFICULTY PREDICTOR — TRAINING")
    print("=" * 60)
    
    # Load data
    print("\n📊 Loading data...")
    df = load_data()
    print(f"   Loaded {len(df)} problems")
    
    # Feature engineering
    print("\n🔧 Feature Engineering...")
    engineer = FeatureEngineer()
    X, y, feature_names = engineer.fit_transform(df)
    
    # Save features for later use
    joblib.dump({"X": X, "y": y, "feature_names": feature_names}, FEATURES_PATH)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\n   Train set: {X_train.shape[0]} samples")
    print(f"   Test set:  {X_test.shape[0]} samples")
    
    # Train baselines
    baseline_results = train_baseline_models(X_train, X_test, y_train, y_test)
    
    # Optimize XGBoost
    best_xgb, study = optimize_xgboost(X_train, y_train, n_trials=OPTUNA_N_TRIALS)
    
    xgb_train_acc = best_xgb.score(X_train, y_train)
    xgb_test_acc = best_xgb.score(X_test, y_test)
    
    print(f"\n  XGBoost (optimized):")
    print(f"    Train Accuracy: {xgb_train_acc:.4f}")
    print(f"    Test Accuracy:  {xgb_test_acc:.4f}")
    
    # Final comparison
    print(f"\n{'=' * 60}")
    print("FINAL MODEL COMPARISON")
    print("=" * 60)
    print(f"{'Model':<25} {'Train Acc':>10} {'Test Acc':>10}")
    print("-" * 45)
    for name, (_, train_acc, test_acc) in baseline_results.items():
        print(f"{name:<25} {train_acc:>10.4f} {test_acc:>10.4f}")
    print(f"{'XGBoost (Optimized)':<25} {xgb_train_acc:>10.4f} {xgb_test_acc:>10.4f}")
    
    # Save best model
    joblib.dump(best_xgb, MODEL_PATH)
    engineer.save()
    
    print(f"\n✅ Best model saved to {MODEL_PATH}")
    print(f"✅ Feature pipeline saved")
    
    # Save test data for evaluation
    joblib.dump({
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "feature_names": feature_names,
    }, os.path.join(os.path.dirname(MODEL_PATH), "test_data.joblib"))
    
    return best_xgb, engineer


if __name__ == "__main__":
    main()
