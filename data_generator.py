"""
data_generator.py
=================
Generates realistic multi-hour ICU time-series data for Sepsis Early Warning research.
Simulates physiological deterioration (SIRS and SOFA dynamics):
  - Tachycardia (elevated Heart Rate)
  - Tachypnea (elevated Respiratory Rate)
  - Hypothermia or Fever
  - Hypotension (dropping SBP, DBP, and MAP)
  - Hypoxia (dropping Oxygen Saturation)
  - Organ dysfunction lab markers: rising Lactate, rising Creatinine, dropping Platelets, elevated WBC
  - Intermittent laboratory testing intervals (clinical missingness)
"""

import numpy as np
import pandas as pd


def generate_synthetic_sepsis_data(n_patients: int = 120, seed: int = 42) -> pd.DataFrame:
    """
    Generates longitudinal measurement-level ICU data for N patients.
    Each patient has multi-hour measurements prior to a defined outcome window.
    """
    np.random.seed(seed)
    records = []

    for pid in range(1, n_patients + 1):
        # ~20% sepsis incidence
        is_septic = np.random.rand() < 0.22
        stay_hours = np.random.randint(10, 24)

        # Baseline physiology for patient
        base_hr = np.random.normal(76, 7)
        base_rr = np.random.normal(16, 2)
        base_temp = np.random.normal(36.8, 0.3)
        base_sbp = np.random.normal(122, 9)
        base_dbp = np.random.normal(76, 6)
        base_spo2 = np.random.normal(98, 1.2)

        # Baseline labs
        base_wbc = np.random.normal(7.5, 1.5)
        base_lactate = np.random.normal(1.2, 0.3)
        base_creatinine = np.random.normal(0.9, 0.2)
        base_platelets = np.random.normal(245, 35)

        # Non-septic confounders (e.g. mild fever from surgery or transient tachycardia)
        mild_fever_only = (not is_septic) and (np.random.rand() < 0.15)
        if mild_fever_only:
            base_temp = np.random.normal(38.0, 0.2)
            base_hr += 10

        # Sepsis clinical onset hour
        onset_hr = stay_hours - np.random.randint(2, 5) if is_septic else 999

        for t in range(1, stay_hours + 1):
            # Hours until overt sepsis
            hrs_to_onset = onset_hr - t
            # Deterioration starts in the 4-hour window before clinical onset
            in_early_warning = is_septic and (0 <= hrs_to_onset <= 4)
            drift = max(0.0, float(4 - hrs_to_onset)) if (is_septic and hrs_to_onset <= 4) else 0.0

            # Perturbations
            hr = base_hr + (drift * 4.5) + np.random.normal(0, 3.0)
            rr = base_rr + (drift * 1.6) + np.random.normal(0, 1.2)
            temp = base_temp + (drift * 0.25) + np.random.normal(0, 0.15)
            sbp = base_sbp - (drift * 4.2) + np.random.normal(0, 4.0)
            dbp = base_dbp - (drift * 2.8) + np.random.normal(0, 3.0)
            map_val = (2 * dbp + sbp) / 3.0
            spo2 = np.clip(base_spo2 - (drift * 0.8) + np.random.normal(0, 0.8), 70.0, 100.0)

            # Laboratory tests (measured intermittently: every 4-6 hours or upon admission)
            is_lab_hour = (t == 1) or (t % 4 == 0) or (t == stay_hours)
            is_lactate_hour = (t == 1) or (t % 3 == 0) or (t == stay_hours)

            wbc = (base_wbc + (drift * 1.8) + np.random.normal(0, 0.8)) if is_lab_hour else np.nan
            lactate = (base_lactate + (drift * 0.65) + np.random.normal(0, 0.2)) if is_lactate_hour else np.nan
            creatinine = (base_creatinine + (drift * 0.2) + np.random.normal(0, 0.05)) if is_lab_hour else np.nan
            platelets = (base_platelets - (drift * 18.0) + np.random.normal(0, 10.0)) if is_lab_hour else np.nan

            # Sepsis label: 1 if patient develops sepsis in the prediction horizon
            sepsis_label = 1 if in_early_warning else 0

            records.append({
                "patient_id": f"P_{pid:03d}",
                "timestamp": t,
                "heart_rate": round(float(np.clip(hr, 35.0, 220.0)), 1),
                "resp_rate": round(float(np.clip(rr, 6.0, 60.0)), 1),
                "temperature": round(float(np.clip(temp, 33.0, 42.0)), 2),
                "sbp": round(float(np.clip(sbp, 45.0, 240.0)), 1),
                "dbp": round(float(np.clip(dbp, 25.0, 140.0)), 1),
                "map": round(float(np.clip(map_val, 30.0, 160.0)), 1),
                "oxygen_saturation": round(float(spo2), 1),
                "wbc": round(float(np.clip(wbc, 0.5, 50.0)), 2) if not np.isnan(wbc) else np.nan,
                "lactate": round(float(np.clip(lactate, 0.3, 20.0)), 2) if not np.isnan(lactate) else np.nan,
                "creatinine": round(float(np.clip(creatinine, 0.2, 12.0)), 2) if not np.isnan(creatinine) else np.nan,
                "platelets": round(float(np.clip(platelets, 15.0, 600.0)), 1) if not np.isnan(platelets) else np.nan,
                "sepsis_label": int(sepsis_label)
            })

    return pd.DataFrame(records)


if __name__ == "__main__":
    sample_df = generate_synthetic_sepsis_data(n_patients=100)
    sample_df.to_csv("sepsis_dataset.csv", index=False)
    print("Generated synthetic dataset:", sample_df.shape)
