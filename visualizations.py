"""
visualizations.py
=================
Generates interactive, accessible Plotly visualizations for the healthcare dashboard:
  - Multi-model ROC Curves with AUC and chance baseline
  - Precision-Recall Curves
  - Model Performance Comparison Bar Charts
  - Confusion Matrix Heatmaps
  - Top Feature Importance Bar Charts
  - Patient Longitudinal Physiological Trajectories
"""

from typing import Dict, Any
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


# Cohesive Clinical Color Palette (Navy, Teal, Cyan, Coral, Gray)
COLOR_PALETTE = {
    "Logistic Regression": "#0284c7",  # Sky Blue
    "Random Forest": "#0d9488",        # Teal
    "XGBoost": "#1e3a8a",              # Navy Blue
    "Baseline": "#94a3b8",             # Slate Gray
    "Accent": "#06b6d4",               # Cyan
    "Danger": "#ef4444"                # Coral Red
}


def plot_roc_curves(roc_data: Dict[str, Dict[str, Any]]) -> go.Figure:
    """
    Renders multi-model ROC Curves with readable tooltips and chance baseline.
    """
    fig = go.Figure()

    # Random chance baseline
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        line=dict(color=COLOR_PALETTE["Baseline"], width=2, dash="dash"),
        name="Chance Baseline (AUC = 0.50)",
        hoverinfo="name"
    ))

    for name, data in roc_data.items():
        fpr = data["fpr"]
        tpr = data["tpr"]
        auc = data["auc"]
        color = COLOR_PALETTE.get(name, "#4f46e5")

        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode="lines",
            line=dict(color=color, width=3.0),
            name=f"{name} (AUC = {auc:.3f})",
            hovertemplate=f"<b>{name}</b><br>False Positive Rate: %{{x:.3f}}<br>True Positive Rate (Sensitivity): %{{y:.3f}}<extra></extra>"
        ))

    fig.update_layout(
        title=dict(
            text="Receiver Operating Characteristic (ROC) Curves",
            font=dict(size=18, family="Arial, sans-serif", color="#0f172a")
        ),
        xaxis=dict(
            title="False Positive Rate (1 - Specificity)",
            range=[-0.02, 1.02],
            gridcolor="#f1f5f9",
            zerolinecolor="#cbd5e1"
        ),
        yaxis=dict(
            title="True Positive Rate (Sensitivity / Recall)",
            range=[-0.02, 1.05],
            gridcolor="#f1f5f9",
            zerolinecolor="#cbd5e1"
        ),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        legend=dict(
            orientation="v",
            yanchor="bottom",
            y=0.05,
            xanchor="right",
            x=0.98,
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor="#cbd5e1",
            borderwidth=1,
            font=dict(size=11, color="#0f172a")
        ),
        margin=dict(l=60, r=40, t=60, b=60),
        height=500
    )
    return fig


def plot_pr_curves(pr_data: Dict[str, Dict[str, Any]]) -> go.Figure:
    """
    Renders Precision-Recall curves. Essential for imbalanced clinical cohorts.
    """
    fig = go.Figure()

    for name, data in pr_data.items():
        precision = data["precision"]
        recall = data["recall"]
        pr_auc = data["pr_auc"]
        color = COLOR_PALETTE.get(name, "#4f46e5")

        fig.add_trace(go.Scatter(
            x=recall, y=precision,
            mode="lines",
            line=dict(color=color, width=2.8),
            name=f"{name} (PR-AUC = {pr_auc:.3f})",
            hovertemplate=f"<b>{name}</b><br>Recall: %{{x:.3f}}<br>Precision: %{{y:.3f}}<extra></extra>"
        ))

    fig.update_layout(
        title=dict(
            text="Precision-Recall Curves (Clinical Imbalance Evaluation)",
            font=dict(size=18, color="#0f172a")
        ),
        xaxis=dict(title="Recall (Sensitivity)", range=[-0.02, 1.02], gridcolor="#f1f5f9"),
        yaxis=dict(title="Precision (Positive Predictive Value)", range=[-0.02, 1.05], gridcolor="#f1f5f9"),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        legend=dict(
            orientation="v",
            yanchor="bottom",
            y=0.05,
            xanchor="left",
            x=0.03,
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor="#cbd5e1",
            borderwidth=1,
            font=dict(size=11, color="#0f172a")
        ),
        margin=dict(l=60, r=40, t=60, b=60),
        height=500
    )
    return fig


def plot_model_comparison_bar(metrics_df: pd.DataFrame) -> go.Figure:
    """
    Renders grouped bar chart comparing multiple evaluation metrics across candidate algorithms.
    Uses exact consistent model colors across charts.
    """
    display_metrics = ["Accuracy", "ROC-AUC", "Precision", "Recall (Sensitivity)", "Specificity", "F1-Score"]
    available_metrics = [m for m in display_metrics if m in metrics_df.columns]

    fig = go.Figure()

    for model_name in metrics_df.index:
        model_color = COLOR_PALETTE.get(model_name, "#4f46e5")
        fig.add_trace(go.Bar(
            name=model_name,
            x=available_metrics,
            y=[metrics_df.loc[model_name, m] for m in available_metrics],
            marker_color=model_color,
            text=[f"{metrics_df.loc[model_name, m]:.3f}" for m in available_metrics],
            textposition="auto"
        ))

    fig.update_layout(
        title=dict(
            text="Model Performance Metric Comparison",
            font=dict(size=18, color="#0f172a")
        ),
        barmode="group",
        yaxis=dict(title="Score (0.0 to 1.0)", range=[0, 1.15], gridcolor="#f1f5f9"),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=40, t=70, b=60),
        height=480
    )
    return fig


def plot_confusion_matrices(cm_dict: Dict[str, np.ndarray]) -> go.Figure:
    """
    Renders subplots of confusion matrices with labeled True/False Positive/Negative counts.
    """
    model_names = list(cm_dict.keys())
    fig = make_subplots(
        rows=1, cols=len(model_names),
        subplot_titles=[f"<b>{name}</b>" for name in model_names],
        horizontal_spacing=0.08
    )

    labels = ["Negative", "Positive"]

    for i, name in enumerate(model_names, start=1):
        cm = cm_dict[name]
        # Text annotations
        text_vals = [
            [f"TN: {cm[0, 0]}", f"FP: {cm[0, 1]}"],
            [f"FN: {cm[1, 0]}", f"TP: {cm[1, 1]}"]
        ]

        fig.add_trace(go.Heatmap(
            z=cm,
            x=labels,
            y=labels,
            text=text_vals,
            texttemplate="%{text}",
            textfont=dict(size=14, color="#ffffff"),
            colorscale=[[0, "#e2e8f0"], [0.5, "#0284c7"], [1, "#1e3a8a"]],
            showscale=False
        ), row=1, col=i)

        fig.update_xaxes(title_text="Predicted Label", row=1, col=i)
        if i == 1:
            fig.update_yaxes(title_text="True Label", row=1, col=i)

    fig.update_layout(
        title=dict(text="Confusion Matrix Comparison (Held-Out Test Cohort)", font=dict(size=18, color="#0f172a")),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        height=380,
        margin=dict(l=60, r=40, t=80, b=50)
    )
    return fig


def plot_feature_importance(importance_df: pd.DataFrame, top_n: int = 12) -> go.Figure:
    """
    Renders horizontal bar chart of top predictive biomarkers from tree ensembles.
    """
    df_top = importance_df.head(top_n).sort_values(by=importance_df.columns[1], ascending=True)
    metric_col = importance_df.columns[1]

    fig = go.Figure(go.Bar(
        x=df_top[metric_col],
        y=df_top["Feature"],
        orientation="h",
        marker=dict(
            color=df_top[metric_col],
            colorscale=[[0, "#99f6e4"], [1, "#0d9488"]],
            line=dict(color="#14b8a6", width=1)
        ),
        text=[f"{v:.4f}" for v in df_top[metric_col]],
        textposition="outside"
    ))

    fig.update_layout(
        title=dict(text=f"Top {top_n} Early Warning Predictors ({metric_col})", font=dict(size=18, color="#0f172a")),
        xaxis=dict(title="Relative Feature Importance Score", gridcolor="#f1f5f9"),
        yaxis=dict(title="", tickfont=dict(size=11)),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        height=450,
        margin=dict(l=140, r=60, t=60, b=50)
    )
    return fig


def plot_patient_trajectory(patient_df: pd.DataFrame, patient_id: str) -> go.Figure:
    """
    Plots multi-hour vital signs (Heart Rate, MAP) and labs (Lactate) for a single patient trajectory.
    """
    p_data = patient_df[patient_df["patient_id"] == patient_id].sort_values("timestamp")
    if len(p_data) == 0:
        return go.Figure()

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Heart Rate
    fig.add_trace(go.Scatter(
        x=p_data["timestamp"], y=p_data["heart_rate"],
        name="Heart Rate (bpm)", mode="lines+markers",
        line=dict(color="#ef4444", width=2.5)
    ), secondary_y=False)

    # MAP
    fig.add_trace(go.Scatter(
        x=p_data["timestamp"], y=p_data["map"],
        name="MAP (mmHg)", mode="lines+markers",
        line=dict(color="#0284c7", width=2.5)
    ), secondary_y=False)

    # Blood Lactate
    if "lactate" in p_data.columns:
        fig.add_trace(go.Scatter(
            x=p_data["timestamp"], y=p_data["lactate"],
            name="Lactate (mmol/L)", mode="lines+markers",
            line=dict(color="#f59e0b", width=2.5, dash="dot")
        ), secondary_y=True)

    fig.update_layout(
        title=dict(text=f"Longitudinal Hemodynamics Timeline: Patient {patient_id}", font=dict(size=16, color="#0f172a")),
        xaxis=dict(title="ICU Hour (timestamp)", gridcolor="#f1f5f9"),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=400,
        margin=dict(l=60, r=60, t=70, b=50)
    )
    fig.update_yaxes(title_text="Vitals (bpm / mmHg)", secondary_y=False)
    fig.update_yaxes(title_text="Lactate (mmol/L)", secondary_y=True)

    return fig


def plot_patient_prediction_comparison(predictions: Dict[str, Dict[str, Any]]) -> go.Figure:
    """
    Renders a clear bar chart comparing estimated probabilities and predicted classes across candidate models.
    Includes the 50% classification decision boundary line.
    """
    model_names = list(predictions.keys())
    probs = [predictions[m]["probability"] for m in model_names]
    colors = [COLOR_PALETTE.get(m, "#0284c7") for m in model_names]
    bar_texts = [f"{p:.1%}<br>({predictions[m]['label'].replace('Model output: ', '')})" for m, p in zip(model_names, probs)]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=model_names,
        y=probs,
        marker=dict(color=colors, line=dict(color="#0f172a", width=1.5)),
        text=bar_texts,
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Estimated Probability: %{y:.1%}<extra></extra>"
    ))

    # Add 50% classification threshold line
    fig.add_hline(
        y=0.5,
        line_dash="dash",
        line_color="#ef4444",
        line_width=2,
        annotation_text="50% Classification Decision Boundary",
        annotation_position="top right",
        annotation_font=dict(color="#b91c1c", size=11)
    )

    fig.update_layout(
        title=dict(
            text="Model Probability & Classification Comparison",
            font=dict(size=17, color="#0f172a")
        ),
        xaxis=dict(title="", tickfont=dict(size=13)),
        yaxis=dict(
            title="Estimated Positive Class Probability",
            range=[0, 1.18],
            tickformat=".0%",
            gridcolor="#f1f5f9"
        ),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        height=380,
        margin=dict(l=60, r=40, t=60, b=50)
    )

    return fig


def plot_batch_predictions_summary(batch_results_df: pd.DataFrame) -> go.Figure:
    """
    Renders summary breakdown of positive vs negative classifications across models in batch mode.
    """
    models = ["Logistic_Regression", "Random_Forest", "XGBoost"]
    display_names = ["Logistic Regression", "Random Forest", "XGBoost"]

    pos_counts = []
    neg_counts = []

    for m in models:
        col = f"{m}_output"
        if col in batch_results_df.columns:
            pos = int((batch_results_df[col] == "Positive class").sum())
            neg = int((batch_results_df[col] == "Negative class").sum())
        else:
            pos, neg = 0, 0
        pos_counts.append(pos)
        neg_counts.append(neg)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Positive Class",
        x=display_names,
        y=pos_counts,
        marker_color="#ef4444",
        text=pos_counts,
        textposition="auto"
    ))
    fig.add_trace(go.Bar(
        name="Negative Class",
        x=display_names,
        y=neg_counts,
        marker_color="#0284c7",
        text=neg_counts,
        textposition="auto"
    ))

    fig.update_layout(
        barmode="group",
        title=dict(text="Batch Classification Outcome Distribution", font=dict(size=16, color="#0f172a")),
        yaxis=dict(title="Number of Patient Records", gridcolor="#f1f5f9"),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=360,
        margin=dict(l=50, r=40, t=60, b=50)
    )

    return fig
