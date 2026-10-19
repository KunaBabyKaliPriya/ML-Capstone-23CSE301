import os, joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

df = pd.read_excel("data/online+retail/Online Retail.xlsx")
df = df.drop_duplicates()
df = df.dropna(subset=["CustomerID"])
df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
df["TotalAmount"] = df["Quantity"] * df["UnitPrice"]

ref = df["InvoiceDate"].max() + pd.Timedelta(days=1)
rfm = df.groupby("CustomerID").agg(
    Recency=("InvoiceDate", lambda x: (ref - x.max()).days),
    Frequency=("InvoiceNo", "nunique"),
    Monetary=("TotalAmount", "sum"),
).reset_index()

cols = ["Recency", "Frequency", "Monetary"]
rfm_log = np.log1p(rfm[cols])
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(rfm_log), columns=cols)

kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
rfm["KMeans_Cluster"] = kmeans.fit_predict(X_scaled)
pca = PCA(n_components=2).fit(X_scaled)

os.makedirs("models", exist_ok=True)
joblib.dump(scaler, "models/cluster_scaler.pkl")
joblib.dump(kmeans, "models/cluster_kmeans.pkl")
joblib.dump(pca, "models/cluster_pca.pkl")
rfm[cols + ["KMeans_Cluster"]].to_csv("models/cluster_rfm.csv", index=False)
print("Saved clustering files. Customers:", len(rfm))