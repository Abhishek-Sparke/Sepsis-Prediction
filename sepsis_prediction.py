import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, roc_curve,
    f1_score, confusion_matrix, ConfusionMatrixDisplay
)
from xgboost import XGBClassifier

np.random.seed(42)
n = 800

label = np.random.choice([0, 1], n, p=[0.8, 0.2])

df = pd.DataFrame({
    "patient_id": np.random.randint(1, 81, n),
    "timestamp": np.random.randint(1, 24, n),
    "heart_rate": np.random.normal(78, 10, n) + label * 18,
    "resp_rate": np.random.normal(16, 3, n) + label * 6,
    "temperature": np.random.normal(36.8, 0.4, n) + label * 1.2,
    "sbp": np.random.normal(122, 12, n) - label * 16,
    "dbp": np.random.normal(78, 8, n) - label * 10,
    "map": np.random.normal(92, 9, n) - label * 12,
    "oxygen_saturation": np.clip(
        np.random.normal(98, 1.5, n) - label * 4, 70, 100
    ),
    "wbc": np.clip(np.random.normal(7.5, 2, n) + label * 5.5, 1, 35),
    "lactate": np.clip(np.random.normal(1.2, 0.4, n) + label * 2.2, 0.5, 15),
    "creatinine": np.clip(np.random.normal(0.9, 0.2, n) + label * 0.8, 0.3, 10),
    "platelets": np.clip(np.random.normal(240, 40, n) - label * 65, 20, 500),
    "sepsis_label": label
}).sort_values(["patient_id", "timestamp"])

df.to_csv("sepsis_dataset.csv", index=False)

features = [
    "heart_rate", "resp_rate", "temperature", "sbp", "dbp",
    "map", "oxygen_saturation", "wbc", "lactate",
    "creatinine", "platelets"
]

X = df.groupby("patient_id")[features].agg(
    ["mean", "min", "max", "last"]
)
X.columns = [f"{col}_{stat}" for col, stat in X.columns]

X["shock_index"] = X["heart_rate_last"] / X["sbp_last"].replace(0, np.nan)
y = df.groupby("patient_id")["sepsis_label"].max()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

models = {
    "Logistic Regression": make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced")
    ),
    "Random Forest": make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(
            n_estimators=100, class_weight="balanced", random_state=42
        )
    ),
    "XGBoost": make_pipeline(
        SimpleImputer(strategy="median"),
        XGBClassifier(
            n_estimators=100, max_depth=3,
            eval_metric="logloss", random_state=42
        )
    )
}

results, roc_data, predictions = {}, {}, {}

for name, model in models.items():
    model.fit(X_train, y_train)
    prob = model.predict_proba(X_test)[:, 1]
    pred = (prob >= 0.5).astype(int)

    auc = roc_auc_score(y_test, prob)
    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, zero_division=0)

    results[name] = {"Accuracy": acc, "ROC-AUC": auc, "F1-Score": f1}
    roc_data[name] = (*roc_curve(y_test, prob), auc)
    predictions[name] = pred

plt.figure(figsize=(9, 7))

for name, (fpr, tpr, thresholds, auc) in roc_data.items():
    plt.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC={auc:.3f})")
    plt.fill_between(fpr, tpr, alpha=0.08)

    valid = np.flatnonzero(np.isfinite(thresholds))
    if len(valid):
        i = valid[np.argmin(abs(thresholds[valid] - 0.5))]
        plt.scatter(fpr[i], tpr[i], s=55)
        plt.annotate(f"t={thresholds[i]:.2f}", (fpr[i], tpr[i]))

plt.plot([0, 1], [0, 1], "k--", label="Random baseline (AUC=0.50)")
plt.xlabel("False Positive Rate (1 - Specificity)")
plt.ylabel("True Positive Rate (Sensitivity)")
plt.title("Sepsis Prediction - Detailed ROC Curves")
plt.xlim(0, 1)
plt.ylim(0, 1.05)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("sepsis_detailed_roc.png", dpi=300)
plt.show()

scores = pd.DataFrame(results).T
scores.plot(kind="bar", figsize=(10, 5), ylim=(0, 1.05))
plt.title("Model Performance Comparison")
plt.ylabel("Score")
plt.xticks(rotation=15)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("sepsis_model_comparison.png", dpi=300)
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

for ax, (name, pred) in zip(axes, predictions.items()):
    ConfusionMatrixDisplay(
        confusion_matrix(y_test, pred, labels=[0, 1]),
        display_labels=["Negative", "Positive"]
    ).plot(ax=ax, colorbar=False)
    ax.set_title(name)

plt.tight_layout()
plt.savefig("sepsis_confusion_matrices.png", dpi=300)
plt.show()

pd.DataFrame(results).T.to_csv("sepsis_model_results.csv")
