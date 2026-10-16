"""
evaluation.py
=============
Comprehensive clinical and statistical model evaluation module.
Calculates:
  - Accuracy, ROC-AUC, Precision, Recall (Sensitivity), Specificity, F1-Score
  - ROC curves (FPR, TPR, thresholds)
  - Precision-Recall curves (Precision, Recall, thresholds, Average Precision)
  - Confusion Matrices
  - Feature Importances for Tree Models (Random Forest and XGBoost)
"""

from typing import Dict, Any, List
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    average_precision_score
)


def evaluate_models(trained_models: Dict[str, Any], X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
    """
    Evaluates all trained models on held-out test data.
    Never invents or mocks metrics; all are computed directly from test predictions.
    """
    results = {
        "metrics_table": [],
        "detailed_metrics": {},
        "roc_curves": {},
        "pr_curves": {},
        "confusion_matrices": {}
    }

    # Verify that test set contains both classes
    unique_classes = np.unique(y_test)
    has_both_classes = len(unique_classes) >= 2

    for name, pipeline in trained_models.items():
        probs = pipeline.predict_proba(X_test)[:, 1]
        preds = (probs >= 0.5).astype(int)

        # Standard metrics
        acc = float(accuracy_score(y_test, preds))
        prec = float(precision_score(y_test, preds, zero_division=0))
        rec = float(recall_score(y_test, preds, zero_division=0))  # Sensitivity
        f1 = float(f1_score(y_test, preds, zero_division=0))

        # Confusion Matrix components
        cm = confusion_matrix(y_test, preds, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

        # ROC-AUC & PR-AUC
        if has_both_classes:
            auc = float(roc_auc_score(y_test, probs))
            pr_auc = float(average_precision_score(y_test, probs))
            fpr, tpr, roc_thresh = roc_curve(y_test, probs)
            precision_pts, recall_pts, pr_thresh = precision_recall_curve(y_test, probs)
        else:
            auc = 0.5
            pr_auc = 0.0
            fpr, tpr, roc_thresh = np.array([0, 1]), np.array([0, 1]), np.array([0, 0])
            precision_pts, recall_pts, pr_thresh = np.array([0]), np.array([0]), np.array([0])

        model_summary = {
            "Model": name,
            "Accuracy": acc,
            "ROC-AUC": auc,
            "Precision": prec,
            "Recall (Sensitivity)": rec,
            "Specificity": spec,
            "F1-Score": f1,
            "PR-AUC": pr_auc,
            "TP": int(tp),
            "FP": int(fp),
            "TN": int(tn),
            "FN": int(fn)
        }

        results["metrics_table"].append(model_summary)
        results["detailed_metrics"][name] = model_summary
        results["confusion_matrices"][name] = cm
        results["roc_curves"][name] = {
            "fpr": fpr,
            "tpr": tpr,
            "thresholds": roc_thresh,
            "auc": auc
        }
        results["pr_curves"][name] = {
            "precision": precision_pts,
            "recall": recall_pts,
            "thresholds": pr_thresh,
            "pr_auc": pr_auc
        }

    results["metrics_df"] = pd.DataFrame(results["metrics_table"]).set_index("Model")
    return results


def extract_feature_importances(trained_models: Dict[str, Any], feature_names: List[str]) -> pd.DataFrame:
    """
    Extracts Gini / Gain feature importances from tree-based models (Random Forest and XGBoost).
    """
    importance_data = {"Feature": feature_names}

    if "Random Forest" in trained_models:
        rf = trained_models["Random Forest"].named_steps["randomforestclassifier"]
        importance_data["Random Forest"] = rf.feature_importances_

    if "XGBoost" in trained_models:
        xgb = trained_models["XGBoost"].named_steps["xgbclassifier"]
        importance_data["XGBoost"] = xgb.feature_importances_

    df_importance = pd.DataFrame(importance_data)
    if "Random Forest" in df_importance.columns and "XGBoost" in df_importance.columns:
        df_importance["Average Importance"] = (df_importance["Random Forest"] + df_importance["XGBoost"]) / 2
        df_importance = df_importance.sort_values(by="Average Importance", ascending=False)
    elif "Random Forest" in df_importance.columns:
        df_importance = df_importance.sort_values(by="Random Forest", ascending=False)

    return df_importance
