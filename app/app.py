import os
import joblib
import pandas as pd
import streamlit as st

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model = joblib.load(os.path.join(BASE, "models", "best_model.pkl"))
features = joblib.load(os.path.join(BASE, "models", "feature_names.pkl"))
limits = joblib.load(os.path.join(BASE, "models", "clip_limits.pkl"))
activity_map = joblib.load(os.path.join(BASE, "models", "activity_map.pkl"))

def predict(df):
    df = df[features].clip(lower=limits["lower"], upper=limits["upper"], axis=1)
    pred = model.predict(df)
    proba = model.predict_proba(df)
    return pred, proba

st.set_page_config(page_title="Human Activity Recognition", page_icon="🏃")
st.title("🏃 Human Activity Recognition")
st.write("Predicts a person's activity (walking, sitting, etc.) from smartphone sensor features.")

mode = st.radio("Choose input method", ["Try a sample row", "Upload a CSV"])

if mode == "Try a sample row":
    sample = pd.read_csv(os.path.join(BASE, "app", "sample_data.csv"))
    idx = st.number_input("Sample row number", 0, len(sample) - 1, 0)
    row = sample.iloc[[idx]]
    st.write("Actual activity:", row["Actual"].values[0])
    if st.button("Predict"):
        pred, proba = predict(row)
        st.success(f"Predicted activity: {activity_map[pred[0]]}")
        st.bar_chart(pd.Series(proba[0], index=[activity_map[c] for c in model.classes_]))
else:
    file = st.file_uploader("Upload CSV with the 561+ feature columns", type="csv")
    if file is not None:
        data = pd.read_csv(file)
        if st.button("Predict"):
            try:
                pred, _ = predict(data)
                data["Predicted"] = [activity_map[p] for p in pred]
                st.dataframe(data[["Predicted"]])
            except KeyError:
                st.error("CSV is missing required feature columns.")