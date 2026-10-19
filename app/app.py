import os
import datetime
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def mpath(name):
    return os.path.join(BASE, "models", name)


def have(*names):
    return all(os.path.exists(mpath(n)) for n in names)


@st.cache_resource
def load(name):
    return joblib.load(mpath(name))


st.set_page_config(page_title="ML Capstone", page_icon="🤖", layout="wide")
st.title("🤖 ML Capstone: Classification, Regression & Clustering")
st.caption("23CSE301 Machine Learning - interactive demo of the three trained pipelines.")

tab_cls, tab_reg, tab_clu = st.tabs(
    ["🏃 Classification", "🚲 Regression", "👥 Clustering"]
)

# ---------------------------------------------------------------- CLASSIFICATION
with tab_cls:
    st.header("Human Activity Recognition")
    st.write("Predicts a person's activity (walking, sitting, etc.) from smartphone sensor features.")

    if not have("best_model.pkl", "feature_names.pkl", "clip_limits.pkl", "activity_map.pkl"):
        st.warning("Classification model files are missing in the models/ folder.")
    else:
        model = load("best_model.pkl")
        features = load("feature_names.pkl")
        limits = load("clip_limits.pkl")
        activity_map = load("activity_map.pkl")

        def predict_activity(df):
            df = df[features].clip(lower=limits["lower"], upper=limits["upper"], axis=1)
            return model.predict(df), model.predict_proba(df)

        mode = st.radio("Choose input method", ["Try a sample row", "Upload a CSV"], key="cls_mode")

        if mode == "Try a sample row":
            sample = pd.read_csv(os.path.join(BASE, "app", "sample_data.csv"))
            idx = st.number_input("Sample row number", 0, len(sample) - 1, 0, key="cls_idx")
            row = sample.iloc[[idx]]
            st.write("Actual activity:", row["Actual"].values[0])
            if st.button("Predict activity", key="cls_btn"):
                pred, proba = predict_activity(row)
                st.success(f"Predicted activity: {activity_map[pred[0]]}")
                st.bar_chart(
                    pd.Series(proba[0], index=[activity_map[c] for c in model.classes_])
                )
        else:
            file = st.file_uploader("Upload CSV with the sensor feature columns", type="csv", key="cls_file")
            if file is not None:
                data = pd.read_csv(file)
                if st.button("Predict activity", key="cls_btn2"):
                    try:
                        pred, _ = predict_activity(data)
                        data["Predicted"] = [activity_map[p] for p in pred]
                        st.dataframe(data[["Predicted"]])
                    except KeyError:
                        st.error("CSV is missing required feature columns.")

# ---------------------------------------------------------------- REGRESSION
with tab_reg:
    st.header("Seoul Bike Rental Demand")
    st.write("Predicts how many bikes will be rented in an hour from weather and calendar details.")

    if not have("reg_model.pkl", "reg_meta.pkl"):
        st.warning("Regression model files are missing in the models/ folder.")
    else:
        rmodel = load("reg_model.pkl")
        meta = load("reg_meta.pkl")
        cats = meta["categories"]
        st.caption(f"Model used: {meta.get('best_model_name', 'best regression model')}")

        c1, c2, c3 = st.columns(3)
        with c1:
            date = st.date_input(
                "Date",
                value=datetime.date(2018, 6, 15),
                min_value=datetime.date(2017, 12, 1),
                max_value=datetime.date(2018, 11, 30),
            )
            hour = st.slider("Hour of day", 0, 23, 12)
            temp = st.number_input("Temperature (°C)", -30.0, 50.0, 15.0)
            humidity = st.slider("Humidity (%)", 0, 100, 50)
        with c2:
            wind = st.number_input("Wind speed (m/s)", 0.0, 15.0, 1.7)
            vis = st.number_input("Visibility (10m)", 0, 2000, 1500)
            dew = st.number_input("Dew point temperature (°C)", -40.0, 40.0, 5.0)
            solar = st.number_input("Solar radiation (MJ/m2)", 0.0, 5.0, 0.5)
        with c3:
            rain = st.number_input("Rainfall (mm)", 0.0, 50.0, 0.0)
            snow = st.number_input("Snowfall (cm)", 0.0, 20.0, 0.0)
            season = st.selectbox("Seasons", cats["Seasons"])
            hol_opts = cats["Holiday"]
            hol_default = hol_opts.index("No Holiday") if "No Holiday" in hol_opts else 0
            holiday = st.selectbox("Holiday", hol_opts, index=hol_default)
            func_opts = cats["Functioning Day"]
            func_default = func_opts.index("Yes") if "Yes" in func_opts else 0
            func = st.selectbox("Functioning Day", func_opts, index=func_default)
            if func == "No":
                st.info("Functioning Day = No means the bike system is closed, so the model predicts 0 rentals.")

        if st.button("Predict bike demand", key="reg_btn"):
            d = pd.Timestamp(date)
            row = {
                "Hour": hour,
                "Temperature(°C)": temp,
                "Humidity(%)": humidity,
                "Wind speed (m/s)": wind,
                "Visibility (10m)": vis,
                "Dew point temperature(°C)": dew,
                "Solar Radiation (MJ/m2)": solar,
                "Rainfall(mm)": rain,
                "Snowfall (cm)": snow,
                "Seasons": season,
                "Holiday": holiday,
                "Functioning Day": func,
                "Year": d.year,
                "Month": d.month,
                "Day": d.day,
                "DayOfWeek": d.dayofweek,
                "Weekend": int(d.dayofweek >= 5),
                "PeakHour": int(hour in [7, 8, 9, 17, 18, 19]),
                "Temperature_Humidity": temp * humidity,
                "Hour_sin": np.sin(2 * np.pi * hour / 24),
                "Hour_cos": np.cos(2 * np.pi * hour / 24),
            }
            X = pd.DataFrame([row])[meta["columns"]]
            pred = float(rmodel.predict(X)[0])
            st.metric("Predicted rented bikes", f"{max(pred, 0):,.0f}")

# ---------------------------------------------------------------- CLUSTERING
with tab_clu:
    st.header("Customer Segmentation (RFM)")
    st.write(
        "Assigns a customer to a K-Means segment from Recency (days since last purchase), "
        "Frequency (number of orders) and Monetary (total spend)."
    )

    if not have("cluster_scaler.pkl", "cluster_kmeans.pkl", "cluster_pca.pkl", "cluster_rfm.csv"):
        st.warning("Clustering model files are missing in the models/ folder.")
    else:
        scaler = load("cluster_scaler.pkl")
        km = load("cluster_kmeans.pkl")
        pca = load("cluster_pca.pkl")
        rfm = pd.read_csv(mpath("cluster_rfm.csv"))
        cols = ["Recency", "Frequency", "Monetary"]

        st.subheader("Cluster profiles (average values)")
        profile = rfm.groupby("KMeans_Cluster")[cols].mean().round(1)
        profile["Customers"] = rfm.groupby("KMeans_Cluster").size()
        st.dataframe(profile)

        c1, c2, c3 = st.columns(3)
        r = c1.number_input("Recency (days)", 1, 400, 30)
        f = c2.number_input("Frequency (orders)", 1, 300, 5)
        m = c3.number_input("Monetary (total spend)", 1.0, 300000.0, 500.0)

        if st.button("Find my cluster", key="clu_btn"):
            x = np.log1p(pd.DataFrame([[r, f, m]], columns=cols))
            xs = pd.DataFrame(scaler.transform(x), columns=cols)
            cluster = int(km.predict(xs)[0])
            st.success(f"This customer belongs to cluster {cluster}")
            st.write("Average profile of this cluster:")
            st.dataframe(profile.loc[[cluster]])

            all_scaled = pd.DataFrame(scaler.transform(np.log1p(rfm[cols])), columns=cols)
            pts = pca.transform(all_scaled)
            new_pt = pca.transform(xs)
            fig, ax = plt.subplots(figsize=(7, 5))
            sc = ax.scatter(pts[:, 0], pts[:, 1], c=rfm["KMeans_Cluster"], cmap="tab10", alpha=0.4, s=10)
            ax.scatter(new_pt[:, 0], new_pt[:, 1], c="black", marker="*", s=300, label="This customer")
            ax.set_xlabel("Principal Component 1")
            ax.set_ylabel("Principal Component 2")
            ax.set_title("K-Means clusters (PCA 2D)")
            cluster_legend = ax.legend(*sc.legend_elements(), title="Cluster", loc="upper right")
            ax.add_artist(cluster_legend)
            ax.legend(loc="lower right")
            fig.tight_layout()
            st.pyplot(fig)