"""
Clinical Data Analysis Pipeline
=================================
Covers:
  1. Missingness Analysis  — correlating missing/sentinel values with patient severity
  2. Feature Engineering   — Comorbidity Score from clinical risk columns
  3. Outlier Detection     — Z-score + IQR on vital signs, with impute/drop decisions

Dataset: cvs_project_csv.xlsx  (18 000 rows × 25 columns)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# 0. LOAD DATA
# ─────────────────────────────────────────────────────────────

df = pd.read_csv("dataset/cvs_project.csv")
print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}\n")

# ═══════════════════════════════════════════════════════════════
# PART 1: MISSINGNESS ANALYSIS
# ═══════════════════════════════════════════════════════════════
"""
This dataset uses SENTINEL values instead of NaN to encode
"test not ordered / not applicable":
  • Creatinine_Level  : negative values (-0.42 min) → impossible clinically
  • HbA1c_Level       : very low values (< 2) may indicate "not tested"
  • Hemoglobin_Level  : values near 0 → impossible
  • Time_Since_Last_Discharge : 0 may mean "first admission" (not missing)
  • ICU_Stay_Flag / High_Risk_Medication_Flag : binary; 0 = absence, not missing

Strategy: define sentinel thresholds → flag rows → correlate with Severity_Score
"""

print("=" * 60)
print("PART 1: MISSINGNESS ANALYSIS")
print("=" * 60)

# ── 1a. Define clinical sentinel / impossible-value thresholds ──
sentinel_rules = {
    "Creatinine_Level": lambda s: s < 0,  # physically impossible
    "HbA1c_Level": lambda s: s < 2.0,  # < 2 is non-physiological
    "Hemoglobin_Level": lambda s: s < 3.0,  # < 3 g/dL is incompatible with life
    "Average_Systolic_BP": lambda s: (s < 60) | (s > 300),  # impossible BP range
}

# Create missingness indicator columns
for col, rule in sentinel_rules.items():
    flag_col = f"{col}_MISSING"
    df[flag_col] = rule(df[col]).astype(int)
    n_missing = df[flag_col].sum()
    pct = 100 * n_missing / len(df)
    print(f"  {col:30s}  sentinel/missing: {n_missing:5d}  ({pct:.2f}%)")

# ── 1b. Aggregate: how many tests are "missing" per patient? ──
missing_flag_cols = [f"{c}_MISSING" for c in sentinel_rules]
df["Total_Missing_Tests"] = df[missing_flag_cols].sum(axis=1)

print("\nMissing-test count distribution:")
print(df["Total_Missing_Tests"].value_counts().sort_index().to_string())

# ── 1c. Correlate missingness with Severity_Score ──
print("\n--- Severity Score by number of missing tests ---")
severity_by_missing = (
    df.groupby("Total_Missing_Tests")["Severity_Score"]
    .agg(["mean", "median", "count"])
    .rename(columns={"mean": "Mean_Severity", "median": "Median_Severity", "count": "N"})
)
print(severity_by_missing.to_string())

# ── 1d. Per-flag correlation with severity ──
print("\n--- Point-biserial correlation: each missing flag vs Severity_Score ---")
for flag_col in missing_flag_cols:
    corr, pval = stats.pointbiserialr(df[flag_col], df["Severity_Score"])
    sig = "**" if pval < 0.01 else ("*" if pval < 0.05 else "")
    print(f"  {flag_col:40s}  r={corr:+.4f}  p={pval:.4f}  {sig}")

# ── 1e. Visualise ──
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Box-plot: severity vs total missing tests
df.boxplot(column="Severity_Score", by="Total_Missing_Tests", ax=axes[0])
axes[0].set_title("Severity Score by # Missing Tests")
axes[0].set_xlabel("Number of Missing / Sentinel Tests")
axes[0].set_ylabel("Severity Score")
plt.sca(axes[0])
plt.xticks(rotation=0)

# Heat-map of mean severity per individual missing flag
heatmap_data = pd.DataFrame({
    f.replace("_MISSING", "").replace("_Level", ""): [
        df[df[f] == 0]["Severity_Score"].mean(),
        df[df[f] == 1]["Severity_Score"].mean()
    ]
    for f in missing_flag_cols
}, index=["Test Present", "Test Missing/Sentinel"])

sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="RdYlGn_r",
            linewidths=0.5, ax=axes[1])
axes[1].set_title("Mean Severity Score\n(Test Present vs Missing/Sentinel)")

plt.suptitle("Missingness vs Patient Severity", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig("missingness_analysis.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n[Plot saved: missingness_analysis.png]")

# ═══════════════════════════════════════════════════════════════
# PART 2: FEATURE ENGINEERING — COMORBIDITY SCORE
# ═══════════════════════════════════════════════════════════════
"""
Available comorbidity-related columns in this dataset:
  • Comorbidity_Index          (0–4+, pre-computed index)
  • Chronic_Disease_Count      (count of chronic diseases)
  • ICU_Stay_Flag              (1 = severe enough to need ICU)
  • High_Risk_Medication_Flag  (1 = on high-risk meds, proxy for complex disease)
  • HbA1c_Level                (diabetes surrogate; > 6.5 = diabetic range)
  • Creatinine_Level           (renal function; > 1.5 = elevated / CKD proxy)
  • Average_Systolic_BP        (hypertension; > 140 = stage 2)

Scoring philosophy (weighted):
  • Each sub-component is normalised to [0, 1] then multiplied by a clinical weight.
  • Final score is scaled 0 – 10 for interpretability.
"""

print("\n" + "=" * 60)
print("PART 2: FEATURE ENGINEERING — COMORBIDITY SCORE")
print("=" * 60)

df_feat = df.copy()

# ── Component 1: Normalised Comorbidity Index (max observed = 4) ──
df_feat["comp_comorbidity_index"] = (
        df_feat["Comorbidity_Index"].clip(0, 4) / 4
)

# ── Component 2: Normalised Chronic Disease Count (max observed ~6) ──
chronic_max = df_feat["Chronic_Disease_Count"].quantile(0.99)  # robust max
df_feat["comp_chronic_disease"] = (
        df_feat["Chronic_Disease_Count"].clip(0, chronic_max) / chronic_max
)

# ── Component 3: ICU Stay (binary, direct) ──
df_feat["comp_icu"] = df_feat["ICU_Stay_Flag"].astype(float)

# ── Component 4: High-Risk Medication (binary, direct) ──
df_feat["comp_high_risk_med"] = df_feat["High_Risk_Medication_Flag"].astype(float)


# ── Component 5: Diabetes proxy — HbA1c > 6.5 ──
#    Grade: normal (<5.7)=0, pre-diabetic(5.7-6.5)=0.5, diabetic(>6.5)=1
def hba1c_score(x):
    if x < 5.7:
        return 0.0
    elif x <= 6.5:
        return 0.5
    else:
        return 1.0


df_feat["comp_diabetes"] = df_feat["HbA1c_Level"].apply(hba1c_score)


# ── Component 6: Renal function — Creatinine proxy for CKD ──
#    Normal (<1.2)=0, mildly elevated(1.2–1.5)=0.5, high(>1.5)=1
def creatinine_score(x):
    if x < 1.2:
        return 0.0
    elif x <= 1.5:
        return 0.5
    else:
        return 1.0


df_feat["comp_renal"] = df_feat["Creatinine_Level"].apply(creatinine_score)


# ── Component 7: Hypertension proxy — Systolic BP ──
#    Normal(<120)=0, elevated(120-139)=0.5, stage2 HTN(>=140)=1
def sbp_score(x):
    if x < 120:
        return 0.0
    elif x < 140:
        return 0.5
    else:
        return 1.0


df_feat["comp_hypertension"] = df_feat["Average_Systolic_BP"].apply(sbp_score)

# ── Weighted aggregation ──
weights = {
    "comp_comorbidity_index": 2.5,  # pre-computed clinical index → high weight
    "comp_chronic_disease": 2.0,  # direct count of chronic diseases
    "comp_icu": 1.5,  # ICU need = high severity marker
    "comp_high_risk_med": 1.0,  # medication complexity
    "comp_diabetes": 1.0,  # HbA1c-based diabetes
    "comp_renal": 1.0,  # Creatinine-based CKD
    "comp_hypertension": 1.0,  # SBP-based hypertension
}

weight_total = sum(weights.values())  # = 10.0 → score naturally 0–10

df_feat["Comorbidity_Score"] = sum(
    df_feat[col] * w for col, w in weights.items()
)

print("\nComorbidity Score statistics:")
print(df_feat["Comorbidity_Score"].describe().round(3).to_string())

# ── Validation: correlation with Severity_Score & Readmission ──
r_severity, p_severity = stats.pearsonr(
    df_feat["Comorbidity_Score"], df_feat["Severity_Score"])
r_readmit, p_readmit = stats.pointbiserialr(
    df_feat["Comorbidity_Score"], df_feat["Readmitted_Within_30_Days"])

print(f"\nValidation correlations:")
print(f"  Comorbidity_Score vs Severity_Score         r={r_severity:+.4f}  p={p_severity:.4e}")
print(f"  Comorbidity_Score vs Readmitted_Within_30D  r={r_readmit:+.4f}  p={p_readmit:.4e}")

# ── Bin into risk categories ──
df_feat["Comorbidity_Risk_Category"] = pd.cut(
    df_feat["Comorbidity_Score"],
    bins=[-0.01, 2.5, 5.0, 7.5, 10.0],
    labels=["Low", "Moderate", "High", "Very High"]
)
print("\nRisk category distribution:")
print(df_feat["Comorbidity_Risk_Category"].value_counts().sort_index().to_string())

# ── Visualise ──
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Distribution of Comorbidity Score
axes[0].hist(df_feat["Comorbidity_Score"], bins=40, color="steelblue", edgecolor="white")
axes[0].set_title("Comorbidity Score Distribution")
axes[0].set_xlabel("Comorbidity Score (0–10)")
axes[0].set_ylabel("Count")

# Score vs Severity
scatter_sample = df_feat.sample(1000, random_state=42)
axes[1].scatter(scatter_sample["Comorbidity_Score"],
                scatter_sample["Severity_Score"],
                alpha=0.3, s=15, color="teal")
axes[1].set_title(f"Comorbidity Score vs Severity\n(r={r_severity:.3f})")
axes[1].set_xlabel("Comorbidity Score")
axes[1].set_ylabel("Severity Score")

# Risk category vs Readmission rate
readmit_by_cat = (
        df_feat.groupby("Comorbidity_Risk_Category", observed=True)["Readmitted_Within_30_Days"]
        .mean() * 100
)
readmit_by_cat.plot(kind="bar", ax=axes[2], color=["green", "gold", "orange", "red"],
                    edgecolor="black")
axes[2].set_title("30-Day Readmission Rate\nby Comorbidity Risk Category")
axes[2].set_xlabel("Risk Category")
axes[2].set_ylabel("Readmission Rate (%)")
axes[2].tick_params(axis="x", rotation=0)

plt.suptitle("Comorbidity Score Feature Engineering", fontsize=13)
plt.tight_layout()
plt.savefig("comorbidity_score.png", dpi=150, bbox_inches="tight")
plt.close()
print("[Plot saved: comorbidity_score.png]")

# ═══════════════════════════════════════════════════════════════
# PART 3: OUTLIER DETECTION — VITAL SIGNS
# ═══════════════════════════════════════════════════════════════
"""
Vital-sign columns in this dataset and their physiologically valid ranges:
  Column                  |  Plausible range        |  Decision if outlier
  ─────────────────────────────────────────────────────────────────────────
  Average_Systolic_BP     |  60 – 250 mmHg          |  Cap (Winsorise)
  HbA1c_Level             |  2.0 – 14.0 %           |  Cap
  Creatinine_Level        |  0.3 – 3.0  mg/dL       |  Cap (negatives → impute)
  Hemoglobin_Level        |  3.0 – 20.0 g/dL        |  Cap
  Medication_Adherence_Score | 0.0 – 1.0            |  Cap / drop if > 1

Two methods used:
  • Z-score  : |z| > 3   → statistical outlier
  • IQR      : x < Q1 – 1.5*IQR  OR  x > Q3 + 1.5*IQR → distributional outlier
"""

print("\n" + "=" * 60)
print("PART 3: OUTLIER DETECTION — VITAL SIGNS")
print("=" * 60)

vital_cols = [
    "Average_Systolic_BP",
    "HbA1c_Level",
    "Creatinine_Level",
    "Hemoglobin_Level",
    "Medication_Adherence_Score",
]

# Clinical hard limits (used after statistical detection for final capping)
clinical_limits = {
    "Average_Systolic_BP": (60, 250),
    "HbA1c_Level": (2.0, 14.0),
    "Creatinine_Level": (0.3, 3.0),
    "Hemoglobin_Level": (3.0, 20.0),
    "Medication_Adherence_Score": (0.0, 1.0),
}

df_clean = df_feat.copy()


def detect_outliers_zscore(series, threshold=3.0):
    """Return boolean mask: True = outlier by Z-score."""
    z = np.abs(stats.zscore(series.dropna()))
    mask = pd.Series(False, index=series.index)
    mask.iloc[series.dropna().index] = z > threshold
    return mask


def detect_outliers_iqr(series, factor=1.5):
    """Return boolean mask: True = outlier by IQR rule."""
    Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
    IQR = Q3 - Q1
    return (series < Q1 - factor * IQR) | (series > Q3 + factor * IQR)


print(f"\n{'Column':<30} {'Z>3':>7} {'IQR':>7} {'Union':>7} {'Min_orig':>10} {'Max_orig':>10}")
print("-" * 75)

outlier_summary = {}

for col in vital_cols:
    z_mask = detect_outliers_zscore(df_clean[col])
    iq_mask = detect_outliers_iqr(df_clean[col])
    union = z_mask | iq_mask

    outlier_summary[col] = {
        "z_mask": z_mask, "iq_mask": iq_mask, "union": union
    }

    print(f"{col:<30} {z_mask.sum():>7} {iq_mask.sum():>7} {union.sum():>7} "
          f"{df_clean[col].min():>10.3f} {df_clean[col].max():>10.3f}")

# ── Decision logic ──
print("\n--- Outlier handling decisions ---")

for col in vital_cols:
    lo, hi = clinical_limits[col]
    union = outlier_summary[col]["union"]

    # Identify impossible (clinically invalid) vs extreme-but-possible
    impossible_mask = (df_clean[col] < lo) | (df_clean[col] > hi)
    n_impossible = impossible_mask.sum()
    n_stat_only = (union & ~impossible_mask).sum()

    decision = ""

    # Negative Creatinine → biologically impossible → impute with median
    if col == "Creatinine_Level" and n_impossible > 0:
        median_val = df_clean.loc[~impossible_mask, col].median()
        df_clean.loc[impossible_mask, col] = median_val
        decision = f"Imputed {n_impossible} impossible values with median ({median_val:.3f})"

    else:
        # All other impossible values → Winsorise (cap to clinical bounds)
        if n_impossible > 0:
            df_clean[col] = df_clean[col].clip(lo, hi)
            decision += f"Capped {n_impossible} impossible values to [{lo}, {hi}]. "

        # Extreme-but-possible statistical outliers → Winsorise to 1st/99th pct
        if n_stat_only > 0:
            p01 = df_clean[col].quantile(0.01)
            p99 = df_clean[col].quantile(0.99)
            n_winsorise = ((df_clean[col] < p01) | (df_clean[col] > p99)).sum()
            df_clean[col] = df_clean[col].clip(p01, p99)
            decision += f"Winsorised {n_winsorise} statistical outliers to [P1={p01:.2f}, P99={p99:.2f}]."

        if not decision:
            decision = "No action needed."

    print(f"\n  {col}")
    print(f"    Clinically impossible: {n_impossible}  |  Statistical-only: {n_stat_only}")
    print(f"    Action: {decision}")

# ── Visualise: before vs after ──
fig, axes = plt.subplots(2, len(vital_cols), figsize=(20, 8))

for i, col in enumerate(vital_cols):
    # Before (original)
    axes[0, i].hist(df_feat[col].clip(-1, df_feat[col].quantile(0.999) * 1.1),
                    bins=40, color="salmon", edgecolor="white")
    axes[0, i].set_title(f"{col}\n(Original)", fontsize=9)
    axes[0, i].set_xlabel("")

    # After (cleaned)
    axes[1, i].hist(df_clean[col], bins=40, color="steelblue", edgecolor="white")
    axes[1, i].set_title(f"{col}\n(After Cleaning)", fontsize=9)

axes[0, 0].set_ylabel("Count (Original)")
axes[1, 0].set_ylabel("Count (Cleaned)")

plt.suptitle("Outlier Detection & Treatment — Vital Signs", fontsize=13)
plt.tight_layout()
plt.savefig("outlier_detection.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n[Plot saved: outlier_detection.png]")

# ── Final summary of cleaned dataset ──
print("\n" + "=" * 60)
print("FINAL DATASET SUMMARY")
print("=" * 60)
print(f"Original rows:  {len(df)}")
print(f"Cleaned rows:   {len(df_clean)}")
print(f"New columns added: Comorbidity_Score, Comorbidity_Risk_Category, "
      f"*_MISSING flags ({len(missing_flag_cols)}), Total_Missing_Tests")
print(f"\nComorbidity_Score range: "
      f"{df_clean['Comorbidity_Score'].min():.2f} – {df_clean['Comorbidity_Score'].max():.2f}")
print(f"Cleaned vital stats:")
print(df_clean[vital_cols].describe().round(3).to_string())

# ── Export enriched dataset ──
export_cols = list(df.columns) + [
    "Total_Missing_Tests", "Comorbidity_Score", "Comorbidity_Risk_Category"
] + missing_flag_cols

df_clean[export_cols].to_csv("cvs_project_cleaned.csv", index=False)
print("\n[Cleaned dataset saved: cvs_project_cleaned.csv]")
print("\nDone. All three analyses complete.")

