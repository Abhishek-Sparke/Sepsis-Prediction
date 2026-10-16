# Sepsis Early Warning Prediction — Clinical ML Dashboard

A professional, responsive healthcare web application built with **Streamlit**, **Scikit-learn**, **XGBoost**, and **Plotly** to demonstrate time-aware early warning prediction of sepsis in critical care and emergency medicine.

---

## 📌 Project Overview

- **Domain:** Critical Care / Emergency Medicine
- **Problem Statement:** Sepsis progresses rapidly into septic shock. Early warning signals hidden within longitudinal vital signs and laboratory measurements provide critical study of physiological deterioration.
- **Objective:** Developing a time-aware machine-learning model to study early warning patterns associated with sepsis using patient vital signs and laboratory measurements.
- **Validation Note:** Research demonstration using synthetic cohort; model outputs are not clinically certified for patient diagnosis.

---

## 🏗️ Project Architecture

```
sepsis-project/
│
├── app.py                  # Main Streamlit web application & healthcare dashboard
├── data_generator.py       # Longitudinal ICU multi-hour trajectory generator (SIRS/SOFA dynamics)
├── preprocessing.py        # Schema validation, time-aware feature engineering & patient-level split
├── model_training.py       # Scikit-learn pipelines (Imputation + Scaling + LR / RF / XGBoost)
├── evaluation.py           # Real metrics (ROC-AUC, PR-AUC, Sensitivity, Specificity, F1, Importances)
├── visualizations.py       # Interactive Plotly figures (ROC, PR Curves, Confusion Matrices, Trajectories)
│
models/
│   ├── single_models.joblib    # Trained pipelines for point-in-time single measurements
│   ├── single_meta.joblib      # Metadata & test validation results
│   ├── longitudinal_models.joblib # Trained pipelines for longitudinal trajectories
│   └── longitudinal_meta.joblib   # Metadata & test validation results
│
├── data/
│   └── sample_sepsis_data.csv # Pre-generated clinical dataset (800 records, 80 patients)
│
├── requirements.txt        # Pinned project dependencies
└── README.md               # Setup and deployment documentation
```

---

## 📋 Input Dataset Specification

The pipeline expects 14 clinical fields:
1. `patient_id` — Patient identifier
2. `timestamp` — Measurement hour (e.g., ICU hour 1 to 24)
3. `heart_rate` — Heart rate (bpm)
4. `resp_rate` — Respiratory rate (breaths/min)
5. `temperature` — Body temperature (°C)
6. `sbp` — Systolic blood pressure (mmHg)
7. `dbp` — Diastolic blood pressure (mmHg)
8. `map` — Mean arterial pressure (mmHg)
9. `oxygen_saturation` — Peripheral oxygen saturation (SpO2, %)
10. `wbc` — White blood cell count (x10^3/µL)
11. `lactate` — Blood lactate measurement (mmol/L)
12. `creatinine` — Serum creatinine (mg/dL)
13. `platelets` — Platelet count (x10^3/µL)
14. `sepsis_label` — Binary outcome label (1 = sepsis onset in window, 0 = non-septic)

---

## 🚀 Step-by-Step Setup & Running Locally (Windows)

### 1. Prerequisites
Ensure you have **Python 3.10+** installed on Windows.

### 2. Open PowerShell in the Project Directory
```powershell
cd c:\Users\abhis\OneDrive\Videos\vac
```

### 3. Activate Virtual Environment (or create one)
```powershell
# If .venv exists:
.\.venv\Scripts\Activate.ps1

# Or create a new environment:
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 4. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 5. Launch the Streamlit Dashboard
```powershell
streamlit run app.py
```
The application will automatically launch in your default web browser at `http://localhost:8501`.

---

## 📊 Dashboard Modules

1. **🏠 Project Overview:** Executive clinical summary, machine learning workflow, algorithm cards, and live dataset statistics.
2. **📊 Dataset Management:** CSV file upload, schema validation, missing-value summary, patient trajectory explorer, and one-click dataset export.
3. **⚙️ Model Training & Evaluation:**
   - Multi-metric cards: Accuracy, ROC-AUC, Sensitivity (Recall), Specificity, F1-Score.
   - Interactive Plotly ROC Curves with chance baseline.
   - Precision-Recall curves.
   - Grouped performance bar chart.
   - Confusion matrix heatmaps for all 3 models.
   - Feature importance rankings (Lactate trend, Shock Index, WBC, etc.).
4. **🧪 Patient Prediction Lab:**
   - **Patient Measurement Analysis:** Dedicated point-in-time form with sections for Vital Signs, Blood Pressure, and Laboratory Measurements.
   - **Strict Validation:** Checks physiological bounds (e.g. SBP > DBP, valid heart rates, SpO2 <= 100%) and rejects invalid values.
   - **Trained Model Inference:** Predicts using persisted Logistic Regression, Random Forest, and XGBoost models without retraining on click.
   - **Neutral Outputs:** Displays `"Model output: Positive class"` vs `"Model output: Negative class"`, probabilities, and interactive comparison chart with the 50% decision boundary.
   - **Input Verification Table:** Audit trail of all submitted values and calculated indices (Shock Index, Pulse Pressure, MAP).
   - **Batch CSV Prediction Mode:** Upload multi-patient CSV files, preview, process, view distribution charts, and download predictions CSV.
5. **📖 Clinical & ML Guide:** College-friendly educational guide explaining sepsis pathophysiology, biomarker meanings, algorithm trade-offs, and clinical research limitations.

---

## ⚠️ Clinical Disclaimer
This application is a **research and academic demonstration** developed to showcase time-aware machine learning methodologies. It is **not a clinical diagnostic device** and must not be used for direct clinical decisions, diagnosis, or treatment.
