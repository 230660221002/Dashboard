import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="HR Attrition Dashboard - Jaya Jaya Maju", layout="wide")

sns.set_style("whitegrid")

# ---------------------------------------------------------------
# Load data & model artifacts
# ---------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("employee_data.csv")
    predicted = pd.read_csv("predicted_attrition.csv")
    return df, predicted

@st.cache_resource
def load_model_artifacts():
    model = joblib.load("model/model_attrition.joblib")
    scaler = joblib.load("model/scaler.joblib")
    encoders = joblib.load("model/encoders.joblib")
    feature_columns = joblib.load("model/feature_columns.joblib")
    return model, scaler, encoders, feature_columns

df, predicted = load_data()
model, scaler, encoders, feature_columns = load_model_artifacts()

labeled = df.dropna(subset=["Attrition"]).copy()
labeled["Attrition"] = labeled["Attrition"].astype(int)
labeled["Status"] = labeled["Attrition"].map({0: "Bertahan", 1: "Resign"})

# ---------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------
st.sidebar.title("Filter")
departments = st.sidebar.multiselect(
    "Department", options=sorted(labeled["Department"].unique()),
    default=sorted(labeled["Department"].unique())
)
job_roles = st.sidebar.multiselect(
    "Job Role", options=sorted(labeled["JobRole"].unique()),
    default=sorted(labeled["JobRole"].unique())
)

filtered = labeled[
    labeled["Department"].isin(departments) & labeled["JobRole"].isin(job_roles)
]

st.sidebar.markdown("---")
st.sidebar.caption(
    "Dashboard ini membantu departemen HR perusahaan Jaya Jaya Maju memantau "
    "tingkat attrition karyawan dan faktor-faktor yang mempengaruhinya."
)

# ---------------------------------------------------------------
# Header & KPI
# ---------------------------------------------------------------
st.title("📊 HR Attrition Dashboard — Jaya Jaya Maju")
st.caption("Monitoring tingkat attrition karyawan dan faktor-faktor pendorongnya")

col1, col2, col3, col4 = st.columns(4)
total_emp = len(filtered)
attr_rate = filtered["Attrition"].mean() * 100 if total_emp else 0
resign_count = int(filtered["Attrition"].sum())
avg_income = filtered["MonthlyIncome"].mean() if total_emp else 0

col1.metric("Total Karyawan (data historis)", f"{total_emp:,}")
col2.metric("Attrition Rate", f"{attr_rate:.1f}%")
col3.metric("Jumlah Resign", f"{resign_count:,}")
col4.metric("Rata-rata Monthly Income", f"${avg_income:,.0f}")

st.markdown("---")

# ---------------------------------------------------------------
# Row 1: Attrition by department & job role
# ---------------------------------------------------------------
c1, c2 = st.columns(2)

with c1:
    st.subheader("Attrition Rate per Department")
    dep_rate = filtered.groupby("Department")["Attrition"].mean().sort_values(ascending=False) * 100
    fig, ax = plt.subplots(figsize=(5, 3.5))
    sns.barplot(x=dep_rate.values, y=dep_rate.index, ax=ax, color="#4C72B0")
    ax.set_xlabel("Attrition Rate (%)")
    ax.set_ylabel("")
    st.pyplot(fig)

with c2:
    st.subheader("Attrition Rate per Job Role")
    role_rate = filtered.groupby("JobRole")["Attrition"].mean().sort_values(ascending=False) * 100
    fig, ax = plt.subplots(figsize=(5, 3.5))
    sns.barplot(x=role_rate.values, y=role_rate.index, ax=ax, color="#DD8452")
    ax.set_xlabel("Attrition Rate (%)")
    ax.set_ylabel("")
    st.pyplot(fig)

# ---------------------------------------------------------------
# Row 2: OverTime, MaritalStatus, BusinessTravel
# ---------------------------------------------------------------
c3, c4, c5 = st.columns(3)

with c3:
    st.subheader("OverTime")
    ot_rate = filtered.groupby("OverTime")["Attrition"].mean() * 100
    fig, ax = plt.subplots(figsize=(3.5, 3))
    sns.barplot(x=ot_rate.index, y=ot_rate.values, ax=ax, color="#55A868")
    ax.set_ylabel("Attrition Rate (%)")
    st.pyplot(fig)

with c4:
    st.subheader("Status Pernikahan")
    ms_rate = filtered.groupby("MaritalStatus")["Attrition"].mean() * 100
    fig, ax = plt.subplots(figsize=(3.5, 3))
    sns.barplot(x=ms_rate.index, y=ms_rate.values, ax=ax, color="#C44E52")
    ax.set_ylabel("Attrition Rate (%)")
    st.pyplot(fig)

with c5:
    st.subheader("Business Travel")
    bt_rate = filtered.groupby("BusinessTravel")["Attrition"].mean() * 100
    fig, ax = plt.subplots(figsize=(3.5, 3))
    sns.barplot(x=bt_rate.index, y=bt_rate.values, ax=ax, color="#8172B2")
    ax.set_ylabel("Attrition Rate (%)")
    ax.tick_params(axis='x', rotation=25)
    st.pyplot(fig)

# ---------------------------------------------------------------
# Row 3: Income & tenure distribution
# ---------------------------------------------------------------
c6, c7 = st.columns(2)

with c6:
    st.subheader("Monthly Income vs Attrition")
    fig, ax = plt.subplots(figsize=(5, 3.5))
    sns.boxplot(data=filtered, x="Status", y="MonthlyIncome", ax=ax)
    st.pyplot(fig)

with c7:
    st.subheader("Years At Company vs Attrition")
    fig, ax = plt.subplots(figsize=(5, 3.5))
    sns.boxplot(data=filtered, x="Status", y="YearsAtCompany", ax=ax)
    st.pyplot(fig)

# ---------------------------------------------------------------
# Row 4: Feature importance from the trained model
# ---------------------------------------------------------------
st.markdown("---")
st.subheader("🔎 Faktor Paling Berpengaruh terhadap Attrition (Model)")
importances = pd.Series(model.feature_importances_, index=feature_columns).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(9, 4))
sns.barplot(x=importances.head(12).values, y=importances.head(12).index, ax=ax, color="#4C72B0")
ax.set_xlabel("Feature Importance")
st.pyplot(fig)

# ---------------------------------------------------------------
# Row 5: Predicted attrition for unlabeled employees
# ---------------------------------------------------------------
st.markdown("---")
st.subheader("🎯 Prediksi Attrition — Karyawan Belum Berlabel")

pred_filtered = predicted[
    predicted["Department"].isin(departments) & predicted["JobRole"].isin(job_roles)
]

pc1, pc2 = st.columns([1, 2])
with pc1:
    st.metric("Total Karyawan Diprediksi", f"{len(pred_filtered):,}")
    st.metric("Diprediksi Resign", f"{int(pred_filtered['Attrition_Predicted'].sum()):,}")

with pc2:
    st.write("Karyawan dengan probabilitas resign tertinggi (prioritas retention):")
    top_risk = pred_filtered.sort_values("Attrition_Probability", ascending=False).head(10)
    st.dataframe(
        top_risk[["EmployeeId", "Department", "JobRole", "OverTime",
                   "MonthlyIncome", "Attrition_Probability"]]
        .style.format({"Attrition_Probability": "{:.1%}"}),
        use_container_width=True
    )

st.caption(
    "Dashboard dibangun dengan Streamlit. Data & model diambil dari hasil notebook analisis "
    "(employee_data.csv, model_attrition.joblib, scaler.joblib, encoders.joblib, "
    "feature_columns.joblib, predicted_attrition.csv)."
)
