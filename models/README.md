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
│   ├── books_dataset.csv          # Raw scraped dataset
│   └── books_week1_final.csv      # Cleaned dataset after EDA
│
├── notebooks/
│   ├── week1_data_collection.ipynb   # Data collection via Open Library API
│   ├── week1_eda.ipynb               # EDA + feature engineering
│   ├── week2_preprocessing.ipynb     # Data preprocessing (coming Week 2)
│   ├── week3_modeling.ipynb          # ML models (coming Week 3)
│   └── week4_genai.ipynb             # GenAI integration (coming Week 4)
│
├── app/
│   └── app.py                     # Flask web application (coming Week 4)
│
├── models/
│   └── (trained models saved here in Week 3)
│
├── reports/
│   └── (charts and analysis outputs)
│
└── README.md
```

---

## 📊 Dataset Description

- **Source:** [Open Library API](https://openlibrary.org/developers/api) — free, public, no authentication required
- **Collection Method:** Python `requests` library, looping through 35 genres × 5 pages × 100 books
- **Raw Records Collected:** 17,118
- **After Deduplication:** 14,021 unique books
- **Final Columns:** 20

### Column Reference

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
| page_count | numeric | Number of pages |
| price_usd | numeric | Price in USD |
| review_length | numeric | Length of review/description text |
| book_age | numeric | Years since first published |
| has_rating | binary | Had real ratings (1) or median-filled (0) |
| review_density | numeric | review_length / page_count |
| ratings_norm | numeric | ratings_count normalized 0–1 |
| editions_norm | numeric | edition_count normalized 0–1 |
| popularity_score | numeric | Weighted popularity score |
| is_popular | binary | **Target variable** (1=Popular, 0=Not Popular) |

### Class Distribution
| Class | Count | Percentage |
|---|---|---|
| Not Popular (0) | 9,814 | 70% |
| Popular (1) | 4,207 | 30% |

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

### 3. Run Data Collection Notebook
Open `notebooks/week1_data_collection.ipynb` in Google Colab or Jupyter and run all cells.
> Note: Data collection takes ~25 minutes due to API rate limiting (0.5s delay per request).

### 4. Run EDA Notebook
Open `notebooks/week1_eda.ipynb` and run all cells to reproduce the cleaned dataset and EDA charts.

---

## 📦 Requirements

```
pandas
numpy
requests
matplotlib
seaborn
scikit-learn
flask
```

---

## 📅 Project Timeline

| Week | Tasks | Status |
|---|---|---|
| Week 1 | Data Collection + EDA | ✅ Complete |
| Week 2 | Preprocessing + Feature Engineering | 🔄 Upcoming |

---



---

## 📌 Data Source

- **Open Library API:** https://openlibrary.org/developers/api
- **License:** Open Library data is available under the [Open Data Commons Public Domain Dedication and License](https://opendatacommons.org/licenses/pddl/1-0/)
