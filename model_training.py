"""
model_training.py
=================
Defines, trains, persists, and loads Machine Learning pipelines (Logistic Regression, Random Forest, XGBoost)
using scikit-learn Pipelines with leak-free preprocessing (SimpleImputer and StandardScaler).

Supports both:
  1. Single-Measurement Point-in-Time Models: For real-time triage and single bedside checks.
  2. Longitudinal Time-Aware Trajectory Models: For multi-hour observation windows.
"""

import os
from typing import Dict, Any, Tuple, Optional, List
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from preprocessing import (
    SINGLE_MEASUREMENT_FEATURES,
    prepare_single_measurement_dataset,
    extract_time_aware_features,
    split_patient_data
)
from evaluation import evaluate_models, extract_feature_importances

MODELS_DIR = "models"


def build_models(pos_weight: float = 1.0, seed: int = 42) -> Dict[str, Any]:
    """
    Constructs scikit-learn pipelines with SimpleImputer and StandardScaler
    for Logistic Regression, Random Forest, and XGBoost.
    """
    return {
        "Logistic Regression": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed)
        ),
        "Random Forest": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(
                n_estimators=120,
                max_depth=6,
                class_weight="balanced",
                random_state=seed
            )
        ),
        "XGBoost": make_pipeline(
            SimpleImputer(strategy="median"),
            XGBClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.08,
                scale_pos_weight=max(1.0, float(pos_weight)),
                eval_metric="logloss",
                random_state=seed
            )
        )
    }


def train_all_models(X_train: pd.DataFrame, y_train: pd.Series, seed: int = 42) -> Dict[str, Any]:
    """
    Fits all candidate models strictly on the training fold.
    Calculates class balance weights to address clinical sepsis prevalence.
    """
    neg_count = int((y_train == 0).sum())
    pos_count = int((y_train == 1).sum())
    pos_weight = neg_count / max(1, pos_count)

    models = build_models(pos_weight=pos_weight, seed=seed)
    trained_models = {}

    for name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        trained_models[name] = pipeline

    return trained_models


def save_models(models: Dict[str, Any], meta: Dict[str, Any], model_type: str = "single") -> None:
    """
    Persists trained model pipelines and associated metadata to disk.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, f"{model_type}_models.joblib")
    meta_path = os.path.join(MODELS_DIR, f"{model_type}_meta.joblib")

    joblib.dump(models, model_path)
    joblib.dump(meta, meta_path)


def load_models(model_type: str = "single") -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Loads persisted model pipelines and metadata from disk if available.
    """
    model_path = os.path.join(MODELS_DIR, f"{model_type}_models.joblib")
    meta_path = os.path.join(MODELS_DIR, f"{model_type}_meta.joblib")

    if os.path.exists(model_path) and os.path.exists(meta_path):
        try:
            models = joblib.load(model_path)
            meta = joblib.load(meta_path)
            return models, meta
        except Exception:
            return None, None
    return None, None


def predict_single_measurement(trained_models: Dict[str, Any], feature_df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Runs model inference on an incoming feature vector using neutral, non-diagnostic labels.
    """
    predictions = {}
    for name, pipeline in trained_models.items():
        probs = pipeline.predict_proba(feature_df)[:, 1]
        prob = float(probs[0])
        pred_class = int(prob >= 0.5)

        predictions[name] = {
            "prediction": pred_class,
            "probability": prob,
            "prob_percent": f"{prob:.1%}",
            "label": "Model output: Positive class" if pred_class == 1 else "Model output: Negative class",
            "is_positive": pred_class == 1
        }
    return predictions


def predict_patient_record(trained_models: Dict[str, Any], input_features_df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Computes class prediction and calibrated probability for an incoming patient feature vector.
    Maintained for complete backward compatibility.
    """
    return predict_single_measurement(trained_models, input_features_df)


def predict_batch(trained_models: Dict[str, Any], feature_df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs batch inference on a DataFrame of multiple patient observations.
    Computes predictions, probabilities, and consensus classification across candidate models.
    """
    results_df = feature_df.copy()

    for name, pipeline in trained_models.items():
        probs = pipeline.predict_proba(feature_df)[:, 1]
        preds = (probs >= 0.5).astype(int)

        short_name = name.replace(" ", "_")
        results_df[f"{short_name}_probability"] = np.round(probs, 4)
        results_df[f"{short_name}_output"] = [
            "Positive class" if p == 1 else "Negative class" for p in preds
        ]

    # Consensus (majority vote across candidate models)
    pred_cols = [f"{name.replace(' ', '_')}_probability" for name in trained_models.keys()]
    mean_prob = results_df[pred_cols].mean(axis=1)
    results_df["mean_model_probability"] = np.round(mean_prob, 4)
    results_df["consensus_output"] = [
        "Positive class" if p >= 0.5 else "Negative class" for p in mean_prob
    ]

    return results_df


def init_or_load_all_models(raw_df: pd.DataFrame, force_retrain: bool = False) -> Dict[str, Any]:
    """
    Ensures that both single-measurement and longitudinal model pipelines are available.
    Loads from disk if available, or trains and persists them cleanly using patient-level splits.
    """
    single_models, single_meta = load_models("single")
    long_models, long_meta = load_models("longitudinal")

    # If already persisted and not forcing retrain, return existing
    if not force_retrain and single_models is not None and long_models is not None:
        return {
            "single_models": single_models,
            "single_meta": single_meta,
            "long_models": long_models,
            "long_meta": long_meta
        }

    # 1. Train Single-Measurement Models (Point-in-Time)
    X_single, y_single, pids_single = prepare_single_measurement_dataset(raw_df)
    X_s_train, X_s_test, y_s_train, y_s_test = split_patient_data(X_single, y_single, pids_single, seed=42)
    single_models = train_all_models(X_s_train, y_s_train, seed=42)
    single_eval = evaluate_models(single_models, X_s_test, y_s_test)
    single_imp = extract_feature_importances(single_models, SINGLE_MEASUREMENT_FEATURES)

    single_meta = {
        "features": SINGLE_MEASUREMENT_FEATURES,
        "eval_results": single_eval,
        "importances": single_imp,
        "n_train_records": len(X_s_train),
        "n_test_records": len(X_s_test)
    }
    save_models(single_models, single_meta, "single")

    # 2. Train Longitudinal Models (Time-Aggregated)
    X_long, y_long, pids_long = extract_time_aware_features(raw_df)
    X_l_train, X_l_test, y_l_train, y_l_test = split_patient_data(X_long, y_long, pids_long, seed=42)
    long_models = train_all_models(X_l_train, y_l_train, seed=42)
    long_eval = evaluate_models(long_models, X_l_test, y_l_test)
    long_imp = extract_feature_importances(long_models, list(X_long.columns))

    long_meta = {
        "features": list(X_long.columns),
        "eval_results": long_eval,
        "importances": long_imp,
        "n_train_patients": len(X_l_train),
        "n_test_patients": len(X_l_test)
    }
    save_models(long_models, long_meta, "longitudinal")

    return {
        "single_models": single_models,
        "single_meta": single_meta,
        "long_models": long_models,
        "long_meta": long_meta
    }
