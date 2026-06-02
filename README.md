# 📚 Online Book Popularity Predictor

A machine learning project that predicts whether a book will be popular based on features like genre, ratings, price, page count, and more — with a Streamlit web app powered by Gemini AI suggestions.

---

## 🎯 Problem Statement

Publishers and authors struggle to predict whether a book will gain wide readership before or after publication. This project builds a binary classification model that predicts book popularity using publicly available book metadata — helping stakeholders make data-driven publishing decisions.

---

## ✅ Success Metric Definition

**Target Variable:** `is_popular` (binary: 1 = Popular, 0 = Not Popular)

**Popularity Score Formula:**
```
popularity_score = (avg_rating × 0.40)
                 + (ratings_norm × 0.35 × 10)
                 + (editions_norm × 0.25 × 10)
```

- `avg_rating` — quality signal (40%)
- `ratings_count` — reader engagement volume (35%)
- `edition_count` — long-term market demand (25%)

Books in the **top 30%** of popularity score are labeled **Popular (1)**.
Books in the **bottom 70%** are labeled **Not Popular (0)**.

**Model Evaluation Metrics:** Accuracy, Precision, Recall, F1-Score, ROC-AUC

---

## 📁 Repository Structure

```
book-popularity-predictor/
│
├── data/
│   ├── books_dataset.csv              # Raw scraped dataset (Week 1)
│   ├── books_week1_final.csv          # Cleaned dataset after EDA (Week 1)
│   └── books_week2_processed.csv      # Preprocessed dataset ready for ML (Week 2)
│
├── notebooks/
│   ├── week1_data_collection.ipynb    # Data collection via Open Library API
│   ├── week1_eda.ipynb                # EDA + feature engineering
│   ├── week2_preprocessing.ipynb      # Univariate, bivariate analysis + preprocessing
│   ├── week3_modeling.ipynb           # ML models — Logistic Regression, Random Forest, XGBoost
│   └── week4_streamlit.ipynb          # GenAI + Streamlit app
│
├── app/
│   └── app.py                         # Streamlit web application
│
├── models/
│   ├── best_model.pkl                 # Trained XGBoost model (F1: 99.47%)
│   ├── scaler.pkl                     # MinMaxScaler fitted on training data
│   ├── label_encoder.pkl              # Genre label encoder (35 genres)
│   ├── feature_columns.pkl            # Feature names in correct order
│   └── num_cols.pkl                   # Numerical columns list for scaling
│
├── reports/
│   ├── univariate_distributions.png   # Histogram plots
│   ├── univariate_boxplots.png        # Boxplot outlier detection
│   ├── univariate_categorical.png     # Target, price, genre distributions
│   ├── bivariate_vs_target.png        # Features vs is_popular
│   ├── bivariate_correlation.png      # Correlation heatmap
│   ├── bivariate_target_correlation.png  # Feature importance ranking
│   ├── bivariate_genre.png            # Genre vs popularity rate
│   ├── bivariate_price.png            # Price category vs popularity
│   ├── bivariate_scatter.png          # Scatter plots
│   ├── model_comparison.png           # All models metric comparison
│   ├── confusion_matrices.png         # Confusion matrix for each model
│   ├── roc_curves.png                 # ROC curves with AUC scores
│   └── feature_importance.png         # Top 15 features by importance
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 📊 Dataset Description

- **Source:** [Open Library API](https://openlibrary.org/developers/api) — free, public, no authentication required
- **Collection Method:** Python `requests` library, looping through 35 genres × 5 pages × 100 books
- **Raw Records Collected:** 17,118
- **After Deduplication:** 14,021 unique books
- **Final Columns (Week 1):** 20
- **Final Columns (Week 2, after preprocessing):** 22

### Raw Dataset Column Reference (Week 1)

| Column | Type | Description |
|---|---|---|
| title | text | Book title |
| authors | text | Author name(s) |
| genre | categorical | One of 35 genres searched |
| first_publish_year | numeric | Year first published |
| edition_count | numeric | Total editions ever published |
| subject_count | numeric | Number of topic tags on Open Library |
| has_cover | binary | Has a cover image (1=yes, 0=no) |
| is_available_online | binary | Free to read online (1=yes, 0=no) |
| avg_rating | numeric | Star rating (1–5) |
| ratings_count | numeric | Number of people who rated the book |
| page_count | numeric | Number of pages (genre-based imputation) |
| price_usd | numeric | Price in USD (domain-knowledge simulation) |
| review_length | numeric | Length of review/description text |
| book_age | numeric | 2025 minus first_publish_year |
| has_rating | binary | Had real ratings (1) or median-filled (0) |
| review_density | numeric | review_length / page_count |
| ratings_norm | numeric | ratings_count normalized 0–1 |
| editions_norm | numeric | edition_count normalized 0–1 |
| popularity_score | numeric | Weighted popularity score |
| is_popular | binary | **Target variable** (1=Popular, 0=Not Popular) |

### Class Distribution

| Class | Count | Percentage |
|---|---|---|
| Not Popular (0) | 9,807 | 70% |
| Popular (1) | 4,214 | 30% |

---

## 🔧 Week 2 — Preprocessing & Feature Engineering

### What Was Done

| Step | Action | Reason |
|---|---|---|
| Drop leakage columns | Removed `ratings_norm`, `editions_norm`, `popularity_score` | These were used to create `is_popular` — keeping them leaks the answer to the model |
| Drop text ID columns | Removed `title`, `authors` | Not usable as ML features directly |
| Outlier capping | Clipped `edition_count`, `ratings_count`, `subject_count`, `review_length`, `review_density` at 1st/99th percentile | Boxplots showed extreme right-skew with kurtosis > 290 in some columns |
| Review density | Recalculated: `review_length / (page_count + 1)` | Captures engagement per page after outlier treatment |
| Price categories | Binned `price_usd` → Budget / Mid-Range / Premium | Groups capture real popularity pattern (Budget: 42.7% popular vs Premium: 20.4%) |
| Genre encoding | Label encoding + One-Hot encoding | Converts text genre to numbers for ML models |
| Normalization | MinMaxScaler on all 9 numerical columns | Scales all values to 0–1 range for ML algorithms |

### Feature Engineering Summary

| New Feature | Created From | Purpose |
|---|---|---|
| `book_age` | `first_publish_year` | Age more meaningful than raw year for ML |
| `review_density` | `review_length / page_count` | Engagement per page |
| `has_rating` | `ratings_count > 0` | Flags real vs imputed ratings |
| `price_category` | `price_usd` binned | Meaningful price groups |
| `price_cat_encoded` | `price_category` | Numeric version for ML |
| `genre_encoded` | `genre` label encoded | For tree-based models |
| `genre_*` (35 cols) | `genre` one-hot encoded | For linear models |

### Key EDA Findings

**Univariate:**
- `ratings_count` and `edition_count` are **extremely right-skewed** (skewness > 12)
- `avg_rating` is **left-skewed** — most books rated 3.5–4.5
- `page_count` follows a **near-normal distribution** — healthiest feature
- `review_density` has **near-zero correlation** with target (r = -0.015) — weakest predictor

**Bivariate — Top predictors of popularity:**

| Feature | Correlation | Insight |
|---|---|---|
| has_rating | +0.538 | Books with real ratings are far more likely to be popular |
| avg_rating | +0.470 | Popular books rated higher (4.33 vs 3.83 avg) |
| subject_count | +0.291 | Popular books have more topic tags |
| edition_count | +0.285 | Popular books have 5× more editions (119 vs 23) |
| ratings_count | +0.261 | Popular books have 15× more raters (21.6 vs 1.4) |
| price_usd | -0.189 | Cheaper books tend to be more popular |
| review_density | -0.015 | Near-zero — not a useful predictor |

**Genre impact:**
- Highest: `fiction` (82.8%), `biography` (72.8%), `history` (72.1%)
- Lowest: `environment` (4.5%), `sports` (4.8%), `self_help` (6.0%)

**Price impact:**
- Budget (<$12.99): **42.7%** popular
- Mid-Range ($13–$19): **29.3%** popular
- Premium (>$19): **20.4%** popular

---

## 🤖 Week 3 — Model Development

### Models Trained

| Model | Purpose |
|---|---|
| Logistic Regression | Baseline — simple, interpretable, fast |
| Random Forest | Handles non-linearity, robust to outliers |
| XGBoost | Sequential error correction — best for tabular data |

### Train / Test Split
- Training set: **11,216 books (80%)**
- Test set: **2,805 books (20%)**
- Stratified split to preserve 70/30 class ratio

### Final Model Performance

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 93.58% | 89.05% | 89.68% | 89.36% | 98.24% |
| Random Forest | 99.18% | 98.12% | 99.17% | 98.64% | 99.97% |
| **XGBoost (Best)** | **99.68%** | **99.17%** | **99.76%** | **99.47%** | **99.99%** |

### Feature Importance (XGBoost)

| Rank | Feature | Importance |
|---|---|---|
| 1 | avg_rating | 35.12% |
| 2 | edition_count | 24.16% |
| 3 | ratings_count | 16.06% |
| 4 | has_rating | 12.31% |
| 5 | book_age | 3.02% |
| Last | review_density | 0.69% |

### Why F1 Score is the Primary Metric
The dataset has 70/30 class imbalance. Accuracy is misleading — a model predicting "Not Popular" always gets 70% accuracy without learning anything. F1 Score balances Precision and Recall and is the correct metric for imbalanced classification.

### Saved Model Artifacts

| File | Description |
|---|---|
| `best_model.pkl` | Trained XGBoost model |
| `scaler.pkl` | MinMaxScaler fitted on training data |
| `label_encoder.pkl` | Genre label encoder (35 genres) |
| `feature_columns.pkl` | 14 feature names in correct order |
| `num_cols.pkl` | 9 numerical columns for scaling |

---

## 🚀 Week 4 — Streamlit App + GenAI

### How to Run the App

**Option 1 — Local:**
```bash
pip install streamlit xgboost joblib requests
streamlit run app/app.py
```

**Option 2 — Google Colab:**
```python
!pip install streamlit xgboost joblib requests pyngrok -q

import subprocess, time
from pyngrok import ngrok

subprocess.Popen(['streamlit','run','app/app.py',
    '--server.port','8501','--server.headless','true'])
time.sleep(6)

ngrok.set_auth_token("YOUR_NGROK_TOKEN")
url = ngrok.connect(8501)
print(f"App URL: {url}")
```

### App Features

| Feature | Description |
|---|---|
| Book input form | Genre, rating, ratings count, pages, editions, subject tags, year, price, cover, online access |
| ML Prediction | XGBoost predicts Popular / Not Popular with probability score |
| Confidence level | High / Medium / Low based on distance from 0.5 threshold |
| Progress bar | Visual probability indicator |
| AI Suggestions | Gemini AI generates 5 personalized suggestions ranked by impact |
| Comparison table | Your book vs popular book average vs not-popular average (color coded) |
| Smart fallback | EDA-based personalized suggestions if AI is unavailable |

### GenAI Integration

The app calls the **Google Gemini API** with the book's full profile and the ML prediction result. Gemini responds as a publishing industry expert with 5 actionable suggestions, each with:
- A short title
- Detailed specific advice referencing the book's actual values
- Impact level: High 🔴 / Medium 🟡 / Low 🟢

If the API is unavailable, a smart fallback generates personalized suggestions based on EDA findings from the dataset — comparing the user's specific values against popular book averages.

### Preprocessing Pipeline in App

The app applies the **exact same preprocessing steps** as the training pipeline:
1. Compute `review_length` from `ratings_count`
2. Compute `review_density` from `review_length` and `page_count`
3. Compute `has_rating` binary flag
4. Encode `price_category` → 0/1/2
5. Encode `genre` using saved `label_encoder.pkl`
6. Scale numerical features using saved `scaler.pkl`
7. Order features using saved `feature_columns.pkl`
8. Pass to XGBoost model for prediction

---

## 🛠️ Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/book-popularity-predictor.git
cd book-popularity-predictor
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Week 1 — Data Collection
Open `notebooks/week1_data_collection.ipynb` in Google Colab and run all cells.
> Note: Data collection takes ~25 minutes due to API rate limiting.
> Output: `data/books_dataset.csv`

### 4. Run Week 1 — EDA
Open `notebooks/week1_eda.ipynb` and run all cells.
> Output: `data/books_week1_final.csv`

### 5. Run Week 2 — Preprocessing
Open `notebooks/week2_preprocessing.ipynb` and run all cells.
> Output: `data/books_week2_processed.csv`

### 6. Run Week 3 — Modeling
Open `notebooks/week3_modeling.ipynb` and run all cells.
> Output: all `.pkl` files in `models/` folder

### 7. Run Week 4 — Streamlit App
```bash
streamlit run app/app.py
```
> Requires Gemini API key from https://aistudio.google.com/app/apikey (free)

---

## 📦 Requirements

```
pandas
numpy
requests
matplotlib
seaborn
scikit-learn
xgboost
streamlit
joblib
jupyter
```

Install all at once:
```bash
pip install pandas numpy requests matplotlib seaborn scikit-learn xgboost streamlit joblib jupyter
```

---

## 📅 Project Timeline

| Week | Tasks | Status |
|---|---|---|
| Week 1 | Data Collection (Open Library API, 14,021 books) + EDA | ✅ Complete |
| Week 2 | Univariate & Bivariate Analysis + Preprocessing + Feature Engineering | ✅ Complete |
| Week 3 | ML Modeling — Logistic Regression, Random Forest, XGBoost (F1: 99.47%) | ✅ Complete |
| Week 4 | GenAI Integration (Gemini AI) + Streamlit Prediction App | ✅ Complete |

---

## 👥 Team

- **Person 1** — Data Collection, EDA, Preprocessing, Logistic Regression Baseline
- **Person 2** — Advanced Modeling (Random Forest, XGBoost), Model Evaluation, Streamlit App

---

## 📌 Data Source

- **Open Library API:** https://openlibrary.org/developers/api
- **License:** Open Library data is available under the [Open Data Commons Public Domain Dedication and License](https://opendatacommons.org/licenses/pddl/1-0/)
