# Machine Learning Capstone Project

**Course:** 23CSE301 Machine Learning, B.Tech. (CSE) III Year, Academic Year 2026-27

**Team members:** Kuna Baby Kali Priya , Reddy Srija , Sai Krishna

**Live app:** https://ml-capstone-23cse301.streamlit.app/

An end-to-end machine learning pipeline for three problem tracks: **Regression**, **Classification** and **Clustering**. Each track covers data loading, EDA, cleaning and feature engineering, model training and comparison, hyperparameter tuning and result visualisation. A Streamlit web app lets anyone try the trained models.

---

## 1. Problem statements and datasets

| Track | Problem | Dataset | Size |
|---|---|---|---|
| Regression | Predict the number of rented bikes per hour from weather and calendar details | [Seoul Bike Sharing Demand](https://archive.ics.uci.edu/dataset/560/seoul+bike+sharing+demand) (UCI) | 8,760 rows, 14 columns |
| Classification | Recognise a person's physical activity (walking, walking upstairs, walking downstairs, sitting, standing, laying) from smartphone sensor features | [Human Activity Recognition Using Smartphones](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones) (UCI) | About 10,300 samples, 561 features, 6 classes |
| Clustering | Segment online-retail customers using RFM (Recency, Frequency, Monetary) values | [Online Retail](https://archive.ics.uci.edu/dataset/352/online+retail) (UCI) | 4,338 customers after cleaning |

---

## 2. Results summary

### Regression (Review 1)
All ten algorithms were trained on the same 80:20 split. Metrics: R², RMSE, MAE.

| Rank | Model | R² | RMSE | MAE |
|---|---|---|---|---|
| 1 | Random Forest | 0.9374 | 162.47 | 94.62 |
| 2 | Gradient Boosting | 0.8746 | 229.98 | 156.26 |
| 3 | Decision Tree | 0.8595 | 243.38 | 135.28 |
| 4 | KNN | 0.8476 | 253.48 | 162.40 |
| 5 | Polynomial Regression | 0.8234 | 272.93 | 195.63 |
| 6 | Lasso Regression | 0.5523 | 434.49 | 322.48 |
| 7 | Ridge Regression | 0.5522 | 434.55 | 322.86 |
| 8 | Linear Regression | 0.5522 | 434.56 | 322.88 |
| 9 | ElasticNet | 0.4491 | 481.97 | 350.39 |
| 10 | SVR | 0.3003 | 543.20 | 371.57 |

**Tuning (GridSearchCV, 5-fold CV):** Random Forest (best: `max_depth=20, n_estimators=200, min_samples_split=2`, mean CV R² 0.9190) and Gradient Boosting (best: `learning_rate=0.1, max_depth=3, n_estimators=200`, mean CV R² 0.8958). Test results after tuning: Random Forest R² 0.9371, RMSE 162.89, MAE 95.25; Gradient Boosting R² 0.9048, RMSE 200.36, MAE 133.35. **Best regression model: tuned Random Forest.**

### Classification (Review 1 Part A and Review 2 Part B)
All ten algorithms were trained on the same stratified 80:20 split.

| Rank | Model | Accuracy | Precision | Recall | Weighted F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| 1 | Gradient Boosting | 0.9884 | 0.9884 | 0.9884 | 0.9884 | 0.9997 |
| 2 | Logistic Regression | 0.9854 | 0.9854 | 0.9854 | 0.9854 | 0.9996 |
| 3 | MLP | 0.9854 | 0.9855 | 0.9854 | 0.9854 | 0.9996 |
| 4 | SVM | 0.9825 | 0.9828 | 0.9825 | 0.9825 | 0.9994 |
| 5 | Random Forest | 0.9796 | 0.9796 | 0.9796 | 0.9796 | 0.9993 |
| 6 | Bagging | 0.9684 | 0.9687 | 0.9684 | 0.9685 | 0.9986 |
| 7 | KNN | 0.9617 | 0.9618 | 0.9617 | 0.9616 | 0.9970 |
| 8 | Decision Tree | 0.9398 | 0.9400 | 0.9398 | 0.9399 | 0.9630 |
| 9 | Gaussian Naive Bayes | 0.5961 | 0.7468 | 0.5961 | 0.5717 | 0.9453 |
| 10 | AdaBoost | 0.5408 | 0.3435 | 0.5408 | 0.4039 | 0.8740 |

**Final model: Gradient Boosting.** After RandomizedSearchCV (3-fold CV, weighted F1) the best parameters were `n_estimators=100, learning_rate=0.1, max_depth=3, min_samples_split=2, min_samples_leaf=2` (CV weighted F1 0.9815). Tuned test ROC-AUC rose slightly from 0.99972 to 0.99975, while accuracy stayed at 0.9884.

### Clustering (Review 2)
RFM features were log-transformed and standardised. Both algorithms use k = 5, chosen with the elbow curve.

| Algorithm | Silhouette Score | Davies-Bouldin Index | Calinski-Harabasz Index |
|---|---|---|---|
| K-Means | 0.3162 | 0.9878 | 3192.98 |
| Agglomerative Hierarchical (Ward) | 0.2805 | 1.0308 | 2754.95 |

K-Means scores better on all three metrics. The first two PCA components explain about 93.9% of the variance. Cluster plots use PCA (2D), with t-SNE for at least one algorithm.

---

## 3. Interactive GUI and deployment

The Streamlit app (`app/app.py`) has three tabs:

- **Classification:** pick a saved sample row, or upload a CSV of sensor features, and get the predicted activity with class probabilities.
- **Regression:** enter weather and calendar details to get the predicted number of rented bikes. Note that *Functioning Day = No* means the system is closed, so the prediction is 0.
- **Clustering:** enter Recency, Frequency and Monetary values to get the customer's K-Means segment, its average profile and a PCA plot.

Live app: _paste your Streamlit URL here_

---

## 4. Repository structure

```
.
├── README.md
├── requirements.txt
├── Classification_completed.ipynb
├── Regression (10).ipynb
├── Clustering (1).ipynb
├── save_clustering.py        # rebuilds and saves the clustering model for the app
├── data/
│   ├── UCI HAR Dataset/
│   ├── seoul+bike+sharing+demand/SeoulBikeData.csv
│   └── online+retail/Online Retail.xlsx
├── models/                   # saved models (.pkl via joblib)
└── app/
    ├── app.py                # Streamlit GUI
    └── sample_data.csv       # sample rows for the classification demo
```

---

## 5. Environment setup

Python 3.12 or newer is recommended.

```bash
git clone <your-repo-url>
cd <your-repo-folder>
python -m pip install -r requirements.txt
python -m pip install seaborn scipy openpyxl jupyter   # extra packages for the notebooks
```

## 6. How to run

**Notebooks:** open each notebook from the repository root folder and choose *Run All*. The notebooks use relative paths such as `data/UCI HAR Dataset`, so the working folder must be the repository root. Classification tuning is slow (it can take well over an hour on a laptop).

**Streamlit app:**
```bash
python -m streamlit run app/app.py
```
The app opens at http://localhost:8501.

**Regenerate the clustering model files (optional):**
```bash
python save_clustering.py
```

---

## 7. Notes

- All random seeds are fixed with `random_state=42`.
- Scalers and encoders are fitted on the training data only, to avoid data leakage.
- Datasets come from the UCI Machine Learning Repository.
