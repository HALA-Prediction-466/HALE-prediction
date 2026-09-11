import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import MinMaxScaler, StandardScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLEANED_DATA_PATH = os.path.join(BASE_DIR, "data", "cleaned_merged_dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "results", "models", "best_model.pkl")
MODEL_RESULTS_PATH = os.path.join(BASE_DIR, "results", "tables", "model_results.csv")
FEATURE_IMPORTANCE_PATH = os.path.join(
    BASE_DIR, "results", "tables", "feature_importance.csv"
)

TARGET_COL = "HALE_60"
CATEGORICAL_COLS = ["ParentLocation", "Location", "Dim1"]

ZSCORE_COLUMNS = [
    "HALE_Birth",
    "infant mortality rate (between birth and 11 months per 1000 live births)",
    "Age-standardized suicide rates (per 100 000 population)",
    "Alcohol, total per capita (15+ years) consumption (in litres of pure alcohol) (SDG Indicator 3.5.2)",
    "Estimates of rate of homicides (per 100 000 population)",
    "Mean Total Cholesterol (crude estimate)",
    "PM2.5_Exposure",
    "Health_Expenditure_pct_GDP",
]

PERCENTAGE_COLUMNS = [
    "Prevalence of insufficient physical activity among adults aged 18+ years (age-standardized estimate) (%)",
    "Prevalence of obesity among adults, BMI ³ 30 (age-standardized estimate) (%)",
    "Prevalence of raised blood pressure among adults aged 30-79 years",
    "Probability of dying between the exact ages 30 and 70 years from cardiovascular diseases, cancer, diabetes, or chronic respiratory diseases (SDG 3.4.1)",
    "Undernourishment_Prevalence",
]

RAW_NUMERIC_COLUMNS = ZSCORE_COLUMNS + PERCENTAGE_COLUMNS


FIELD_GROUPS = {
    "Mortality & Disease Risk": [
        "HALE_Birth",
        "infant mortality rate (between birth and 11 months per 1000 live births)",
        "Age-standardized suicide rates (per 100 000 population)",
        "Estimates of rate of homicides (per 100 000 population)",
        "Probability of dying between the exact ages 30 and 70 years from cardiovascular diseases, cancer, diabetes, or chronic respiratory diseases (SDG 3.4.1)",
    ],
    "Clinical & Behavioral Risk Factors": [
        "Mean Total Cholesterol (crude estimate)",
        "Prevalence of raised blood pressure among adults aged 30-79 years",
        "Prevalence of obesity among adults, BMI ³ 30 (age-standardized estimate) (%)",
        "Prevalence of insufficient physical activity among adults aged 18+ years (age-standardized estimate) (%)",
        "Alcohol, total per capita (15+ years) consumption (in litres of pure alcohol) (SDG Indicator 3.5.2)",
    ],
    "Environment & Health System": [
        "PM2.5_Exposure",
        "Undernourishment_Prevalence",
        "Health_Expenditure_pct_GDP",
    ],
}

SHORT_LABELS = {
    "HALE_Birth": "Healthy Life Expectancy at Birth (years)",
    "infant mortality rate (between birth and 11 months per 1000 live births)": "Infant Mortality Rate (per 1,000 live births)",
    "Age-standardized suicide rates (per 100 000 population)": "Suicide Rate (per 100,000 population)",
    "Estimates of rate of homicides (per 100 000 population)": "Homicide Rate (per 100,000 population)",
    "Probability of dying between the exact ages 30 and 70 years from cardiovascular diseases, cancer, diabetes, or chronic respiratory diseases (SDG 3.4.1)": "Probability of Premature NCD Death, ages 30-70 (%)",
    "Mean Total Cholesterol (crude estimate)": "Mean Total Cholesterol (mmol/L)",
    "Prevalence of raised blood pressure among adults aged 30-79 years": "Raised Blood Pressure Prevalence, ages 30-79 (%)",
    "Prevalence of obesity among adults, BMI ³ 30 (age-standardized estimate) (%)": "Obesity Prevalence, adults (%)",
    "Prevalence of insufficient physical activity among adults aged 18+ years (age-standardized estimate) (%)": "Insufficient Physical Activity, adults 18+ (%)",
    "Alcohol, total per capita (15+ years) consumption (in litres of pure alcohol) (SDG Indicator 3.5.2)": "Alcohol Consumption per Capita (litres, 15+ years)",
    "PM2.5_Exposure": "PM2.5 Exposure (µg/m³)",
    "Undernourishment_Prevalence": "Undernourishment Prevalence (%)",
    "Health_Expenditure_pct_GDP": "Health Expenditure (% of GDP)",
}


@st.cache_resource(show_spinner="Loading model and preprocessing pipeline...")
def load_pipeline():
    """Load the trained model and rebuild the exact fitted scalers / column
    schema used at training time, from the same cleaned dataset that produced
    them. This guarantees new inputs are transformed identically to training
    data without needing to separately persist scaler objects."""

    data = pd.read_csv(CLEANED_DATA_PATH)

    max_period = data["Period"].max()
    engineered = data.copy()
    engineered["Period_sin"] = np.sin(2 * np.pi * engineered["Period"] / max_period)
    engineered["Period_cos"] = np.cos(2 * np.pi * engineered["Period"] / max_period)

    encoded = pd.get_dummies(engineered, columns=CATEGORICAL_COLS, drop_first=True)
    train_rows = encoded[encoded["Period"] <= 2014].reset_index(drop=True)

    x_train = train_rows.drop(columns=[TARGET_COL])
    feature_columns = x_train.columns.tolist()

    scaler_z = StandardScaler()
    scaler_z.fit(x_train[ZSCORE_COLUMNS])

    scaler_mm = MinMaxScaler()
    scaler_mm.fit(x_train[PERCENTAGE_COLUMNS])

    model = joblib.load(MODEL_PATH)

    location_meta = (
        data.drop_duplicates("Location")[["Location", "ParentLocation"]]
        .set_index("Location")["ParentLocation"]
        .to_dict()
    )

    feature_stats = data[RAW_NUMERIC_COLUMNS].describe().T[["min", "50%", "max"]]
    feature_stats.columns = ["min", "median", "max"]

    return {
        "model": model,
        "scaler_z": scaler_z,
        "scaler_mm": scaler_mm,
        "feature_columns": feature_columns,
        "max_period": int(max_period),
        "min_period": int(data["Period"].min()),
        "location_to_parent": location_meta,
        "feature_stats": feature_stats,
        "history": data,
    }


def build_feature_row(pipeline: dict, location: str, sex: str, period: int, raw_values: dict) -> pd.DataFrame:
    row = {col: 0.0 for col in pipeline["feature_columns"]}

    row["Period"] = period
    max_period = pipeline["max_period"]
    row["Period_sin"] = np.sin(2 * np.pi * period / max_period)
    row["Period_cos"] = np.cos(2 * np.pi * period / max_period)

    for col, value in raw_values.items():
        row[col] = value

    parent_location = pipeline["location_to_parent"].get(location)
    parent_dummy = f"ParentLocation_{parent_location}"
    if parent_dummy in row:
        row[parent_dummy] = 1

    location_dummy = f"Location_{location}"
    if location_dummy in row:
        row[location_dummy] = 1

    if sex != "Both sexes":
        sex_dummy = f"Dim1_{sex}"
        if sex_dummy in row:
            row[sex_dummy] = 1

    frame = pd.DataFrame([row], columns=pipeline["feature_columns"])
    frame[ZSCORE_COLUMNS] = pipeline["scaler_z"].transform(frame[ZSCORE_COLUMNS])
    frame[PERCENTAGE_COLUMNS] = pipeline["scaler_mm"].transform(frame[PERCENTAGE_COLUMNS])
    return frame


@st.cache_data
def load_model_results():
    try:
        df = pd.read_csv(MODEL_RESULTS_PATH, index_col=0)
        return df
    except FileNotFoundError:
        return None


@st.cache_data
def load_feature_importance():
    try:
        return pd.read_csv(FEATURE_IMPORTANCE_PATH)
    except FileNotFoundError:
        return None


def latest_record_for(history: pd.DataFrame, location: str, sex: str):
    subset = history[(history["Location"] == location) & (history["Dim1"] == sex)]
    if subset.empty:
        return None
    return subset.sort_values("Period").iloc[-1]


st.set_page_config(page_title="HALE_60 Predictor", page_icon="\U0001FA7A", layout="wide")

st.title("\U0001FA7A Healthy Life Expectancy at 60 Predictor")
st.caption(
    "Estimate the expected remaining healthy years for a 60-year-old, from WHO Global "
    "Health Observatory indicators, using a Gradient Boosting model trained on "
    "2000-2019 country-level data."
)

try:
    pipeline = load_pipeline()
except FileNotFoundError as exc:
    st.error(
        "Not load the trained model or its supporting data files. "
        f"Missing file: {exc.filename}\n\n"
        "Make sure `results/models/best_model.pkl` and `data/cleaned_merged_dataset.csv` "
        "exist relative to this app, then restart it."
    )
    st.stop()

locations = sorted(pipeline["location_to_parent"].keys())

with st.sidebar:
    st.header("About this model")
    results_df = load_model_results()
    if results_df is not None and "Gradient Boosting" in results_df.index:
        gb_row = results_df.loc["Gradient Boosting"]
        c1, c2 = st.columns(2)
        c1.metric("Test R²", f"{gb_row['Test R²']:.3f}")
        c2.metric("Test MAE", f"{gb_row['Test MAE']:.2f} yrs")
    st.markdown(
        "**Model:** Gradient Boosting Regressor\n\n"
        "**Trained on:** 185 countries, 2000-2014\n\n"
        "**Validated on:** 2015-2019 (held-out years)\n\n"
        "**Target:** healthy life expectancy at age 60, in years"
    )

    importance_df = load_feature_importance()
    if importance_df is not None:
        st.subheader("Top drivers of the prediction")
        top = importance_df.head(8).set_index("Feature")
        st.bar_chart(top["Importance"])

    st.info(
        "This tool is trained on historical population-level WHO data and is "
        "intended for exploratory and educational use, not individual clinical "
        "or policy decisions.",
        icon="ℹ️",
    )

st.subheader("1. Population")
c1, c2, c3 = st.columns([2, 1, 1])
with c1:
    location = st.selectbox("Country / Area", locations, index=locations.index("Saudi Arabia") if "Saudi Arabia" in locations else 0)
with c2:
    sex = st.radio("Sex", ["Both sexes", "Female", "Male"], horizontal=True)
with c3:
    period = st.slider("Year", pipeline["min_period"], pipeline["max_period"], pipeline["max_period"])

parent_location = pipeline["location_to_parent"].get(location, "Unknown")
st.caption(f"Region: **{parent_location}**")

st.divider()

load_col, _ = st.columns([1, 3])
if load_col.button("\U0001F4E5 Load actual data for this selection", use_container_width=True):
    record = latest_record_for(pipeline["history"], location, sex)
    if record is not None:
        for col in RAW_NUMERIC_COLUMNS:
            st.session_state[f"input_{col}"] = float(record[col])
        st.toast(f"Loaded {int(record['Period'])} data for {location} ({sex}).", icon="✅")
    else:
        st.toast("No historical record found for that combination.", icon="⚠️")

st.subheader("2. Health & Development Indicators")
raw_values = {}
tabs = st.tabs(list(FIELD_GROUPS.keys()))
for tab, (_, fields) in zip(tabs, FIELD_GROUPS.items()):
    with tab:
        cols = st.columns(2)
        for i, field in enumerate(fields):
            stats = pipeline["feature_stats"].loc[field]
            key = f"input_{field}"
            default = st.session_state.get(key, float(stats["median"]))
            with cols[i % 2]:
                raw_values[field] = st.number_input(
                    SHORT_LABELS.get(field, field),
                    min_value=0.0,
                    max_value=float(stats["max"]) * 1.5,
                    value=float(default),
                    step=0.1,
                    key=key,
                    help=f"Observed range in training data: {stats['min']:.2f} – {stats['max']:.2f}",
                )

st.divider()

if st.button("\U0001F52E Predict HALE_60", type="primary", use_container_width=True):
    feature_row = build_feature_row(pipeline, location, sex, period, raw_values)
    prediction = float(pipeline["model"].predict(feature_row)[0])

    st.subheader("3. Prediction")
    m1, m2, m3 = st.columns(3)
    m1.metric("Predicted HALE at 60", f"{prediction:.1f} years")
    m2.metric("Implied total lifespan", f"{60 + prediction:.1f} years")
    dataset_mean = pipeline["history"][TARGET_COL].mean()
    m3.metric("vs. global average", f"{prediction - dataset_mean:+.1f} years", delta=f"{prediction - dataset_mean:+.1f}")

    history = pipeline["history"]
    country_history = history[(history["Location"] == location) & (history["Dim1"] == sex)].sort_values("Period")
    if not country_history.empty:
        chart_df = country_history[["Period", TARGET_COL]].rename(columns={TARGET_COL: "Actual HALE_60"}).set_index("Period")
        chart_df.loc[period, "Predicted HALE_60"] = prediction
        st.line_chart(chart_df, height=300)
    st.caption(
        "The prediction reflects the health, mortality and development indicators entered above, "
        "not just historical trend — try changing an indicator to see its effect."
    )
else:
    st.info("Fill in the indicators above and click **Predict HALE_60** to get an estimate.")
