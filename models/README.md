# 📚 Online Book Popularity Predictor

A machine learning project that predicts whether a book will be popular based on features like genre, ratings, price, page count, and more.

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
│   ├── week3_modeling.ipynb           # ML models (coming Week 3)
│   └── week4_genai.ipynb              # GenAI integration (coming Week 4)
│
├── app/
│   └── app.py                         # Flask web application (coming Week 4)
│
├── models/
│   └── (trained models saved here in Week 3)
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
│   └── bivariate_scatter.png          # Scatter plots
│
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

### Processed Dataset Column Reference (Week 2)

| Column | Type | Description |
|---|---|---|
| genre | categorical | Original genre text |
| first_publish_year | numeric | Year first published (normalized) |
| edition_count | numeric | Editions count (capped + normalized) |
| subject_count | numeric | Topic tags count (capped + normalized) |
| has_cover | binary | Has cover image |
| is_available_online | binary | Free to read online |
| avg_rating | numeric | Star rating (normalized) |
| ratings_count | numeric | Number of raters (capped + normalized) |
| page_count | numeric | Page count (normalized) |
| review_length | numeric | Review text length (capped + normalized) |
| price_usd | numeric | Price in USD (normalized) |
| book_age | numeric | Years since published (normalized) |
| has_rating | binary | Real rating flag |
| review_density | numeric | Review richness per page (normalized) |
| price_category | categorical | Budget / Mid-Range / Premium |
| price_cat_encoded | numeric | Price category as 0/1/2 |
| genre_encoded | numeric | Genre as label-encoded number |
| is_popular | binary | **Target variable** |

---

## 📈 Key EDA Findings (Week 2)

### Univariate Findings
- `ratings_count` and `edition_count` are **extremely right-skewed** (skewness > 12) — a few classic books dominate
- `avg_rating` is **left-skewed** — most books are rated 3.5–4.5, very few below 2 stars
- `page_count` follows a **near-normal distribution** — healthiest feature in the dataset
- `review_density` has **near-zero correlation** with target (r = -0.015) — weakest predictor

### Bivariate Findings

**Top predictors of popularity (correlation with `is_popular`):**

| Feature | Correlation | Direction |
|---|---|---|
| has_rating | +0.538 | Books with real ratings are far more likely to be popular |
| avg_rating | +0.470 | Popular books are genuinely better rated (4.33 vs 3.83) |
| subject_count | +0.291 | Popular books have more topic tags |
| edition_count | +0.285 | Popular books have 5× more editions (119 vs 23) |
| ratings_count | +0.261 | Popular books have 15× more raters (21.6 vs 1.4) |
| price_usd | -0.189 | Cheaper books tend to be more popular |
| review_density | -0.015 | Near-zero — not a useful predictor |

**Genre impact:**
- Highest popularity rate: `fiction` (82.8%), `biography` (72.8%), `history` (72.1%)
- Lowest popularity rate: `environment` (4.5%), `sports` (4.8%), `self_help` (6.0%)

**Price impact:**
- Budget books (<$12.99): **42.7%** popular
- Mid-Range ($13–$19): **29.3%** popular
- Premium (>$19): **20.4%** popular

**Multicollinearity warning:**
- `page_count` ↔ `review_density` correlation = **-0.635** (mathematical relationship)
- For linear models, one of these should be dropped

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
Open `notebooks/week1_data_collection.ipynb` in Google Colab or Jupyter and run all cells.
> Note: Data collection takes ~25 minutes due to API rate limiting (0.5s delay per request).

### 4. Run Week 1 — EDA
Open `notebooks/week1_eda.ipynb` and run all cells to reproduce the cleaned dataset and EDA charts.
> Output: `data/books_week1_final.csv`

### 5. Run Week 2 — Analysis & Preprocessing
Open `notebooks/week2_preprocessing.ipynb` and run all cells.
> This notebook covers univariate analysis, bivariate analysis, outlier treatment, encoding, and normalization.
> Output: `data/books_week2_processed.csv`

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
flask
jupyter
```

Install all at once:
```bash
pip install pandas numpy requests matplotlib seaborn scikit-learn xgboost flask jupyter
```

---

## 📅 Project Timeline

| Week | Tasks | Status |
|---|---|---|
| Week 1 | Data Collection (Open Library API, 14,021 books) + EDA | ✅ Complete |
| Week 2 | Univariate & Bivariate Analysis + Preprocessing + Feature Engineering | ✅ Complete |
| Week 3 | ML Modeling (Logistic Regression, Random Forest, XGBoost) | 🔄 Upcoming |
| Week 4 | GenAI Integration + Flask Prediction App | 🔄 Upcoming |

---



## 📌 Data Source

- **Open Library API:** https://openlibrary.org/developers/api
- **License:** Open Library data is available under the [Open Data Commons Public Domain Dedication and License](https://opendatacommons.org/licenses/pddl/1-0/)
