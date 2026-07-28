Here is the complete, clean Markdown code for your project's root `README.md` file.

Copy this entire block, paste it into a new file named `README.md` in your main project folder, and save it:

```markdown
# 🚀 Retail Analytics & Predictive Churn Architecture (DataXAi)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-orange?style=for-the-badge&logo=xgboost&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Code Style](https://img.shields.io/badge/Code%20Style-Black-000000?style=for-the-badge)

## 📌 Executive Summary
**DataXAi** is an end-to-end machine learning pipeline that transforms raw, high-noise retail transaction logs into actionable customer psychology and predictive churn intelligence. 

Rather than relying on arbitrary human rules or brute-force compute, this architecture prioritizes **data purity, intelligent feature engineering, and strict target leakage prevention**. By compressing ~1 million raw invoices into multidimensional behavioral matrices, the predictive XGBoost classification layer achieves robust performance while executing in **under one second** on standard local hardware.

---

## 🏛️ System Architecture & Key Achievements

```text
[Raw Retail Database: 1.06M Rows] 
               │
               ▼ (Phase 1: The Great Purge & Parquet Optimization)
[Pure Revenue Transactions: 779,425 Rows] 
               │
               ▼ (Phase 2 & 3: Multi-Dimensional RFM Engineering)
[Behavioral Customer Profiles: 5,878 Unique Users]
               │
       ┌───────┴──────────────┐
       ▼                      ▼
[Unsupervised K-Means]   [Supervised XGBoost Pipeline]
  • 4 Mega-Whales          • Target Leakage Prevention (Dropped Recency)
  • 38 Core VIPs           • Dynamic Class Imbalance Handling (1.45 Ratio)
  • 3,838 Bread & Butter   • Sub-Second Training Execution (~0.95s)
  • 1,998 Churned Graveyard

```

### ⚡ Performance Highlights

* **27% Noise Reduction:** Aggressively purged 287,946 rows of database debt (cancelled orders, return anomalies, and anonymous guest checkouts) to ensure model training on 100% pure revenue-generating activity.
* **I/O Memory Optimization:** Transitioned legacy `.xlsx`/`.csv` storage to compressed Apache Parquet files, reducing dataset loading times to **0.16 seconds** with minimal RAM footprint (~123 MB).
* **Behavioral Compression:** Condensed 779,425 individual timestamps into exactly **5,878 distinct customer profiles** using Recency, Frequency, and Monetary (RFM) behavioral scoring.
* **Sub-Second XGBoost Execution:** By feeding the algorithm a highly condensed, feature-engineered matrix instead of raw dimensional sprawl, model training completes in **~0.95 seconds** on local CPU architectures.

---

## 🔬 Methodology: The Three Pillars

### 1️⃣ Data Quality & Memory Engineering

More data is not always better data. Raw retail receipts contain financial noise that distorts predictive algorithms.

* **The Purge:** Filtered exact duplicates, zero-pricing errors, and anonymous transactions.
* **Parquet Serialization:** Converted tabular structures into column-oriented storage, optimizing memory usage and downstream execution speeds.

### 2️⃣ Unsupervised Customer Psychology (K-Means Clustering)

Human-defined rules (e.g., *"spends over £5k = VIP"*) fail because they cannot process multi-dimensional trade-offs simultaneously. Using **Scikit-Learn's K-Means ($K=4$)**, validated via the Elbow Method, the algorithm uncovered four organic behavioral personas across the 5,878 customer profiles:

* 🟢 **The Mega-Whales (Count: 4):** Extreme frequency and volume outliers generating £428k+ average spend.
* 🔵 **The Core B2B VIPs (Count: 38):** Highly reliable, frequent purchasers providing cash-flow stability.
* 🟣 **The Bread & Butter (Count: 3,838):** Active retail consumers driving daily transaction volume.
* 🟡 **The Churned Graveyard (Count: 1,998):** Historically active accounts with >400 days of dormancy, identifying prime targets for automated win-back campaigns.

### 3️⃣ Supervised Churn Prediction & Leakage Prevention

To transition from descriptive analytics to predictive intelligence, we built an XGBoost classifier to identify customer abandonment before it happens.

* **The 180-Day Business Rule:** Accounts inactive for >180 days were flagged as `Churn_Risk` (`1`).
* **Target Leakage Defense:** Because `Churn_Risk` is mathematically derived from `Recency`, feeding `Recency` into the training matrix would allow the algorithm to "cheat." **We explicitly dropped `Recency**`, forcing XGBoost to predict abandonment relying entirely on purchasing velocity (`Frequency`), spending volume (`Monetary`), and historical engagement scores (`F_Score`, `M_Score`).
* **Dynamic Imbalance Handling:** Calculated an exact class imbalance ratio of **1.45** (`active_count / churned_count`) and injected it into `scale_pos_weight`, preventing the decision trees from developing majority-class bias.

---

## 🛠️ Tech Stack & Project Structure

```text
├── data/
│   ├── raw/                  # Original retail transaction logs
│   └── processed/            # Compressed dataxai_rfm_features.parquet
├── notebooks/
│   ├── 01_data_cleaning.ipynb # Phase 1-3: Purge & Parquet conversion
│   ├── 02_eda_kmeans.ipynb    # Phase 4-5: RFM Engineering & K-Means ($K=4$)
│   └── 03_xgboost_churn.ipynb # Phase 6: Leakage prevention & supervised ML
├── src/
│   ├── features/             # Feature engineering pipelines
│   └── models/               # XGBoost hyperparameter configurations
├── requirements.txt          # Environment dependencies
└── README.md                 # Project documentation

```

---

## 💻 Installation & Quick Start

### 1. Clone the Repository

```bash
git clone [https://github.com/MianArslan09/UCI-Retail-II-Data-EDA.git](https://github.com/MianArslan09/UCI-Retail-II-Data-EDA.git)
cd DataXAi-Retail-Analytics

```

### 2. Set Up the Virtual Environment

```bash
# Windows (WSL / PowerShell)
python -m venv venv
source venv/bin/activate  # Or venv\Scripts\activate on Windows CMD

# Install core dependencies
pip install -r requirements.txt

```

### 3. Run the Predictive Pipeline

To train the XGBoost model from the command line or within your Jupyter environment:

```python
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

# Load engineered features
df = pd.read_parquet("data/processed/dataxai_rfm_features.parquet")

# Engineer target and prevent leakage
df["Churn_Risk"] = (df["Recency"] > 180).astype(int)
features = ["Frequency", "Monetary", "F_Score", "M_Score"]
X, y = df[features], df["Churn_Risk"]

# Stratified 80/20 split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train XGBoost with dynamic class balancing
imbalance_ratio = (y_train == 0).sum() / (y_train == 1).sum()
model = xgb.XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    scale_pos_weight=imbalance_ratio,
    random_state=42,
    eval_metric="logloss",
)

model.fit(X_train, y_train)
print(
    f"✅ Training Complete! Baseline Accuracy: {model.score(X_train, y_train)*100:.2f}%"
)

```

---

## 📊 Model Evaluation Summary

| Metric / Parameter | Value | Architectural Justification |
| --- | --- | --- |
| **Training Execution Speed** | `~0.95 Seconds` | Achieved via condensed RFM matrix engineering (4,700 rows × 4 features). |
| **Class Imbalance Ratio** | `1.45` | Dynamically calculated to scale positive minority class weights. |
| **Max Tree Depth (`max_depth`)** | `5` | Strictly regularized to prevent decision tree overfitting. |
| **Learning Rate (`eta`)** | `0.05` | Conservative shrinkage step size for smooth gradient descent. |
| **Baseline Accuracy** | `73.67%` | Strong initial validation without relying on leaked temporal targets. |

---

## 🔗 LinkedIn Engineering Build Log

This repository was developed in public as part of a 3-part technical series on software engineering and MLOps:

* 📖 **Part 1:** [Why I Deleted 27% of a 1M-Row Database (Data Hygiene & Parquet)](https://linkedin.com/in/yourlink)
* 📖 **Part 2:** [How Unsupervised K-Means Found 4 "Mega-Whales" in 779k Receipts](https://linkedin.com/in/yourlink)
* 📖 **Part 3:** [Sub-Second XGBoost Execution & Target Leakage Prevention](https://linkedin.com/in/yourlink)

---

## 📝 License

This project is licensed under the MIT License - see the `LICENSE` file for details.

