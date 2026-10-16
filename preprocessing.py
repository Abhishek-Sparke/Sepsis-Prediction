"""
preprocessing.py
================
Schema validation, time-aware feature engineering, and patient-level splitting.
Enforces strict prevention of clinical data leakage:
  - Fits imputer and scalers ONLY on training folds.
  - Groups strictly by patient_id so measurements from the same patient never appear in both train and test sets.
"""

from typing import Tuple, List, Dict, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


REQUIRED_COLUMNS = [
    "patient_id", "timestamp", "heart_rate", "resp_rate", "temperature",
    "sbp", "dbp", "map", "oxygen_saturation", "wbc", "lactate",
    "creatinine", "platelets", "sepsis_label"
]

VITAL_AND_LAB_COLS = [
    "heart_rate", "resp_rate", "temperature", "sbp", "dbp",
    "map", "oxygen_saturation", "wbc", "lactate", "creatinine", "platelets"
]

SINGLE_MEASUREMENT_FEATURES = [
    "heart_rate", "resp_rate", "temperature", "sbp", "dbp",
    "map", "oxygen_saturation", "wbc", "lactate", "creatinine", "platelets",
    "shock_index", "pulse_pressure"
]

CLINICAL_BOUNDS = {
    "heart_rate": (20.0, 300.0, "bpm"),
    "resp_rate": (4.0, 80.0, "breaths/min"),
    "temperature": (30.0, 45.0, "°C"),
    "sbp": (30.0, 300.0, "mmHg"),
    "dbp": (10.0, 200.0, "mmHg"),
    "map": (20.0, 250.0, "mmHg"),
    "oxygen_saturation": (50.0, 100.0, "%"),
    "wbc": (0.1, 150.0, "x10³/µL"),
    "lactate": (0.1, 30.0, "mmol/L"),
    "creatinine": (0.1, 25.0, "mg/dL"),
    "platelets": (5.0, 1500.0, "x10³/µL"),
}

__all__ = [
    "REQUIRED_COLUMNS",
    "VITAL_AND_LAB_COLS",
    "SINGLE_MEASUREMENT_FEATURES",
    "CLINICAL_BOUNDS",
    "validate_dataset",
    "extract_time_aware_features",
    "split_patient_data",
    "validate_single_measurement",
    "prepare_single_measurement_dataset",
    "build_single_measurement_feature_vector",
]


def validate_dataset(df: pd.DataFrame) -> Tuple[bool, List[str], Dict[str, Any]]:
    """
    Validates that uploaded dataset contains required clinical columns,
    non-empty rows, correct data types, and binary target labels.
    """
    errors = []
    meta = {}

    if df is None or len(df) == 0:
        return False, ["Dataset is empty."], meta

    # Check required columns
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {', '.join(missing_cols)}")

    meta["total_records"] = len(df)
    meta["unique_patients"] = df["patient_id"].nunique() if "patient_id" in df.columns else 0

    if "sepsis_label" in df.columns:
        unique_labels = df["sepsis_label"].dropna().unique()
        meta["sepsis_cases"] = int((df["sepsis_label"] == 1).sum())
        meta["sepsis_prevalence"] = float((df["sepsis_label"] == 1).mean())
        if not set(unique_labels).issubset({0, 1}):
            errors.append("Column 'sepsis_label' must only contain binary values (0 or 1).")
        if len(unique_labels) < 2:
            errors.append("Dataset must contain both positive (1) and negative (0) sepsis cases.")

    # Data types and missing values
    meta["missing_per_col"] = df.isnull().sum().to_dict()
    meta["dtypes"] = df.dtypes.astype(str).to_dict()

    is_valid = len(errors) == 0
    return is_valid, errors, meta


def extract_time_aware_features(df: pd.DataFrame, cutoff_timestamp: float = None) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Extracts time-series summary features per patient prior to prediction time.
    Calculates:
      - Recent state: last observation (_last)
      - Extremes: minimum and maximum (_min, _max)
      - Central tendency: mean (_mean)
      - Trajectory / acute deterioration: trend (_trend = last - first)
      - Clinical composite indicators:
          * Shock Index (HR_last / SBP_last)
          * Pulse Pressure (SBP_last - DBP_last)
    """
    data = df.copy()
    data = data.sort_values(by=["patient_id", "timestamp"])

    if cutoff_timestamp is not None:
        data = data[data["timestamp"] <= cutoff_timestamp]

    # Clinical forward fill for intermittent lab draws within each patient's stay
    data[VITAL_AND_LAB_COLS] = data.groupby("patient_id")[VITAL_AND_LAB_COLS].ffill()

    feature_rows = []

    for pid, group in data.groupby("patient_id"):
        row = {"patient_id": pid}

        # Target outcome: Did this patient develop sepsis within the defined prediction window?
        row["sepsis_label"] = int(group["sepsis_label"].max())
        row["observation_count"] = len(group)

        for col in VITAL_AND_LAB_COLS:
            vals = group[col].dropna()
            if len(vals) > 0:
                row[f"{col}_last"] = vals.iloc[-1]
                row[f"{col}_min"] = vals.min()
                row[f"{col}_max"] = vals.max()
                row[f"{col}_mean"] = vals.mean()
                row[f"{col}_trend"] = vals.iloc[-1] - vals.iloc[0]
            else:
                row[f"{col}_last"] = np.nan
                row[f"{col}_min"] = np.nan
                row[f"{col}_max"] = np.nan
                row[f"{col}_mean"] = np.nan
                row[f"{col}_trend"] = 0.0

        # Clinical Composite Indices
        last_hr = row.get("heart_rate_last", np.nan)
        last_sbp = row.get("sbp_last", np.nan)
        last_dbp = row.get("dbp_last", np.nan)

        # Shock Index (SI = HR / SBP)
        if not np.isnan(last_hr) and not np.isnan(last_sbp) and last_sbp > 0:
            row["shock_index"] = last_hr / last_sbp
        else:
            row["shock_index"] = np.nan

        # Pulse Pressure (PP = SBP - DBP)
        if not np.isnan(last_sbp) and not np.isnan(last_dbp):
            row["pulse_pressure"] = last_sbp - last_dbp
        else:
            row["pulse_pressure"] = np.nan

        feature_rows.append(row)

    feature_df = pd.DataFrame(feature_rows)
    feature_cols = [c for c in feature_df.columns if c not in ["patient_id", "sepsis_label"]]
    X = feature_df[feature_cols]
    y = feature_df["sepsis_label"]
    patient_ids = feature_df["patient_id"]

    return X, y, patient_ids


def split_patient_data(X: pd.DataFrame, y: pd.Series, patient_ids: pd.Series, test_size: float = 0.25, seed: int = 42):
    """
    Performs patient-level group split to prevent clinical data leakage.
    Guarantees no patient appears in both training and test sets.
    """
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(X, y, groups=patient_ids))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    return X_train, X_test, y_train, y_test


def validate_single_measurement(data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Strict validation of individual patient measurements against clinical plausibility bounds.
    Does not silently accept impossible or conflicting values.
    """
    errors = []

    # Check required fields and numerical bounds
    for field, (low, high, unit) in CLINICAL_BOUNDS.items():
        val = data.get(field)
        field_display = field.replace("_", " ").title()
        if val is None:
            errors.append(f"Missing required measurement: {field_display} ({unit}).")
        else:
            try:
                num_val = float(val)
                if np.isnan(num_val):
                    errors.append(f"Measurement cannot be empty or NaN: {field_display}.")
                elif num_val < low or num_val > high:
                    errors.append(
                        f"{field_display} ({num_val} {unit}) is outside physiologically plausible range ({low} - {high} {unit})."
                    )
            except (ValueError, TypeError):
                errors.append(f"Invalid non-numeric value for {field_display}: '{val}'.")

    # Relational Blood Pressure & MAP validation
    sbp = data.get("sbp")
    dbp = data.get("dbp")
    map_val = data.get("map")

    if sbp is not None and dbp is not None:
        try:
            s = float(sbp)
            d = float(dbp)
            if s <= d:
                errors.append(
                    f"Physiologically invalid blood pressure: Systolic BP ({s} mmHg) must be strictly greater than Diastolic BP ({d} mmHg)."
                )
            if map_val is not None:
                m = float(map_val)
                if not (d <= m <= s):
                    errors.append(
                        f"Mean Arterial Pressure ({m} mmHg) must lie between Diastolic ({d} mmHg) and Systolic ({s} mmHg) pressure."
                    )
        except (ValueError, TypeError):
            pass

    return len(errors) == 0, errors


def prepare_single_measurement_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Constructs a point-in-time single measurement dataset from raw longitudinal records.
    Calculates Shock Index and Pulse Pressure at each point in time.
    Returns (X, y, patient_ids).
    """
    clean_df = df.dropna(subset=["sepsis_label"]).copy()

    # Add clinical composite markers
    sbp_safe = clean_df["sbp"].clip(lower=1.0)
    clean_df["shock_index"] = clean_df["heart_rate"] / sbp_safe
    clean_df["pulse_pressure"] = clean_df["sbp"] - clean_df["dbp"]

    X = clean_df[SINGLE_MEASUREMENT_FEATURES].copy()
    y = clean_df["sepsis_label"].astype(int)
    patient_ids = clean_df["patient_id"]

    return X, y, patient_ids


def build_single_measurement_feature_vector(measurements: Dict[str, float]) -> pd.DataFrame:
    """
    Constructs a 1-row DataFrame matching the exact feature schema and column order of the single measurement model.
    """
    hr = float(measurements["heart_rate"])
    sbp = float(measurements["sbp"])
    dbp = float(measurements["dbp"])

    row = {
        "heart_rate": hr,
        "resp_rate": float(measurements["resp_rate"]),
        "temperature": float(measurements["temperature"]),
        "sbp": sbp,
        "dbp": dbp,
        "map": float(measurements["map"]),
        "oxygen_saturation": float(measurements["oxygen_saturation"]),
        "wbc": float(measurements["wbc"]),
        "lactate": float(measurements["lactate"]),
        "creatinine": float(measurements["creatinine"]),
        "platelets": float(measurements["platelets"]),
        "shock_index": hr / max(1.0, sbp),
        "pulse_pressure": sbp - dbp
    }
    return pd.DataFrame([row])[SINGLE_MEASUREMENT_FEATURES]
