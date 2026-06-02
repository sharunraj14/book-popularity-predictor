import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import json

# ── Page config ─────────────────────────────────────────────
st.set_page_config(
    page_title="Book Popularity Predictor",
    page_icon="📚",
    layout="wide"
)

# ── Load artifacts ───────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model        = joblib.load("best_model.pkl")
    scaler       = joblib.load("scaler.pkl")
    le           = joblib.load("label_encoder.pkl")
    feature_cols = joblib.load("feature_columns.pkl")
    num_cols     = joblib.load("num_cols.pkl")
    return model, scaler, le, feature_cols, num_cols

model, scaler, le, feature_cols, num_cols = load_artifacts()
GENRES = le.classes_.tolist()

# ── Paste your free Gemini API key here ─────────────────────
GEMINI_API_KEY = "AQ.Ab8RN6JimqDGUxxxxxxxx-r54GOUxYVWdXzhRhxxxxxTcA"

# ── GenAI function using Gemini ──────────────────────────────
def get_suggestions(book, pred, prob):

    prompt = f"""You are a publishing industry expert helping authors improve book popularity.

Book Details:
- Genre: {book['genre']}
- Average Rating: {book['avg_rating']}/5.0
- Number of Ratings: {book['ratings_count']}
- Edition Count: {book['edition_count']}
- Page Count: {book['page_count']}
- Price: ${book['price_usd']}
- Book Age: {book['book_age']} years old
- Available Online Free: {'Yes' if book['is_available_online'] else 'No'}
- Has Cover Image: {'Yes' if book['has_cover'] else 'No'}
- Number of Subject Tags: {book['subject_count']}

Machine Learning Prediction: {'POPULAR' if pred == 1 else 'NOT POPULAR'}
Model Confidence: {prob:.1%}

Based on these specific book details, give exactly 5 actionable suggestions
to improve this book's popularity and reach.
Make each suggestion specific to the actual values shown above.

Respond with ONLY a valid JSON array. No explanation, no markdown, no code blocks.
Format exactly like this:
[
  {{"title": "short title here", "suggestion": "detailed advice here", "impact": "High"}},
  {{"title": "short title here", "suggestion": "detailed advice here", "impact": "Medium"}},
  {{"title": "short title here", "suggestion": "detailed advice here", "impact": "Low"}},
  {{"title": "short title here", "suggestion": "detailed advice here", "impact": "High"}},
  {{"title": "short title here", "suggestion": "detailed advice here", "impact": "Medium"}}
]"""

    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

        response = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            json={
                "contents": [{
                    "parts": [{"text": prompt}]
                }],
                "generationConfig": {
                    "temperature": 0.7,
                    "maxOutputTokens": 1000
                }
            },
            timeout=30
        )

        if response.status_code != 200:
            raise Exception(f"API Error {response.status_code}: {response.text[:200]}")

        data     = response.json()
        raw_text = data["candidates"][0]["content"]["parts"][0]["text"].strip()

        # Clean markdown code blocks if present
        if "```" in raw_text:
            parts    = raw_text.split("```")
            raw_text = parts[1] if len(parts) > 1 else raw_text
            if raw_text.startswith("json"):
                raw_text = raw_text[4:]
        raw_text = raw_text.strip()

        suggestions = json.loads(raw_text)

        # Validate each suggestion has required fields
        for s in suggestions:
            if "title" not in s or "suggestion" not in s or "impact" not in s:
                raise Exception("Invalid response structure")

        return suggestions, True   # True = AI worked

    except Exception as e:
        return get_fallback_suggestions(book, pred, prob), False   # False = fallback used


def get_fallback_suggestions(book, pred, prob):
    """Personalized fallback suggestions based on EDA findings"""
    suggestions = []

    # Based on ratings count
    if book['ratings_count'] < 21:
        suggestions.append({
            "title"     : "Get More Ratings",
            "suggestion": f"Your book has only {book['ratings_count']} ratings. Popular books in our dataset average 21.6 ratings. Reach out to book clubs, bloggers, and readers to leave honest reviews on Goodreads and Open Library. Even reaching 20 ratings significantly boosts visibility.",
            "impact"    : "High"
        })
    else:
        suggestions.append({
            "title"     : "Maintain Rating Momentum",
            "suggestion": f"With {book['ratings_count']} ratings you are above the popular book average of 21.6. Keep engaging readers and encourage new readers to rate — consistent engagement maintains search visibility.",
            "impact"    : "Medium"
        })

    # Based on online availability
    if not book['is_available_online']:
        suggestions.append({
            "title"     : "Make Available Online",
            "suggestion": "Your book is not freely available online. Listing on Open Library or offering a free preview on Google Books significantly increases discoverability. Our data shows books available online have a much higher popularity rate.",
            "impact"    : "High"
        })
    else:
        suggestions.append({
            "title"     : "Promote Online Access",
            "suggestion": "Your book is available online — actively promote this on social media. Readers who discover free access are more likely to rate, review, and recommend to others, creating a viral loop.",
            "impact"    : "Medium"
        })

    # Based on price
    if book['price_usd'] > 18.99:
        suggestions.append({
            "title"     : "Reconsider Premium Pricing",
            "suggestion": f"At ${book['price_usd']:.2f} your book is Premium priced. Our dataset shows Budget books under $12.99 have a 42.7% popularity rate vs only 20.4% for Premium books. Consider a promotional price or separate ebook at lower cost.",
            "impact"    : "High"
        })
    elif book['price_usd'] <= 12.99:
        suggestions.append({
            "title"     : "Budget Pricing is Effective",
            "suggestion": f"At ${book['price_usd']:.2f} you are in the Budget tier which has the highest popularity rate of 42.7% in our dataset. Maintain this pricing while focusing on increasing ratings and subject tags.",
            "impact"    : "Low"
        })
    else:
        suggestions.append({
            "title"     : "Try a Price Promotion",
            "suggestion": f"At ${book['price_usd']:.2f} you are Mid-Range priced. A temporary discount below $12.99 can spike downloads and ratings which feeds back into long-term popularity. Budget books have a 42.7% popularity rate vs your tier's 29.3%.",
            "impact"    : "Medium"
        })

    # Based on subject tags
    if book['subject_count'] < 24:
        suggestions.append({
            "title"     : "Add More Subject Tags",
            "suggestion": f"Your book has {book['subject_count']} subject tags. Popular books in our dataset average 24.6 tags. Add more relevant genre, theme, and topic tags on Amazon, Goodreads, Open Library, and Google Books to improve search discoverability.",
            "impact"    : "Medium"
        })
    else:
        suggestions.append({
            "title"     : "Great Tag Coverage",
            "suggestion": f"With {book['subject_count']} subject tags you exceed the popular book average of 24.6. Ensure these tags are consistent across all platforms — Amazon, Goodreads, Open Library, and Google Books for maximum reach.",
            "impact"    : "Low"
        })

    # Based on cover
    if not book['has_cover']:
        suggestions.append({
            "title"     : "Add a Professional Cover",
            "suggestion": "Your book has no cover image — this is a critical gap. A professional cover is one of the most important factors in attracting readers. Books with strong covers get significantly more clicks both online and in stores.",
            "impact"    : "High"
        })
    else:
        suggestions.append({
            "title"     : "Leverage Your Cover in Marketing",
            "suggestion": "Your book has a cover image. Use it actively across social media, email campaigns, and your author website. Consistent visual branding reinforces recognition and makes promotional posts more shareable.",
            "impact"    : "Low"
        })

    return suggestions


# ── Preprocessing ────────────────────────────────────────────
def preprocess(inp):
    review_length  = 200 + min(inp['ratings_count'], 500) * 0.8
    review_density = review_length / (inp['page_count'] + 1)
    has_rating     = 1 if inp['ratings_count'] > 0 else 0
    price          = inp['price_usd']
    price_cat      = 0 if price <= 12.99 else (1 if price <= 18.99 else 2)
    genre_enc      = int(le.transform([inp['genre']])[0])

    raw = {
        "edition_count"      : inp['edition_count'],
        "subject_count"      : inp['subject_count'],
        "has_cover"          : inp['has_cover'],
        "is_available_online": inp['is_available_online'],
        "avg_rating"         : inp['avg_rating'],
        "ratings_count"      : inp['ratings_count'],
        "page_count"         : inp['page_count'],
        "review_length"      : review_length,
        "price_usd"          : price,
        "book_age"           : inp['book_age'],
        "has_rating"         : has_rating,
        "review_density"     : review_density,
        "price_cat_encoded"  : price_cat,
        "genre_encoded"      : genre_enc,
    }

    df_raw          = pd.DataFrame([raw])
    df_raw[num_cols] = scaler.transform(df_raw[num_cols])
    return df_raw[feature_cols].astype(float)


# ════════════════════════════════════════════════════════════
# UI
# ════════════════════════════════════════════════════════════

st.markdown("""
<div style='text-align:center; padding:20px 0 5px 0'>
    <h1 style='font-size:2.6rem'>📚 Book Popularity Predictor</h1>
    <p style='color:#666; font-size:1.05rem'>
        Enter your book details to predict popularity and get AI-powered suggestions to improve reach
    </p>
</div>
""", unsafe_allow_html=True)

st.divider()

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📊 Model Performance")
    st.metric("F1 Score",  "99.47%")
    st.metric("ROC-AUC",   "99.99%")
    st.metric("Accuracy",  "99.68%")
    st.markdown("---")
    st.markdown("**Model:** XGBoost (200 trees)")
    st.markdown("**Dataset:** 14,021 books | 35 genres")
    st.markdown("**Source:** Open Library API")
    st.markdown("---")
    st.markdown("**Top Predictive Features:**")
    st.markdown("1. avg_rating — 35.12%")
    st.markdown("2. edition_count — 24.16%")
    st.markdown("3. ratings_count — 16.06%")
    st.markdown("4. has_rating — 12.31%")
    st.markdown("---")
    st.markdown("**Class Distribution:**")
    st.markdown("- Popular (1): 4,214 (30%)")
    st.markdown("- Not Popular (0): 9,807 (70%)")

# ── Input Form ───────────────────────────────────────────────
st.markdown("## 📝 Enter Book Details")

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("#### 📖 Basic Info")
    genre         = st.selectbox("Genre", GENRES, index=GENRES.index("fiction"))
    avg_rating    = st.slider("Average Rating", 1.0, 5.0, 4.0, 0.1,
                              help="Current average star rating (1.0 to 5.0)")
    ratings_count = st.number_input("Number of Ratings", 0, 10000, 50, 10,
                                    help="How many people have rated this book")
    page_count    = st.number_input("Page Count", 50, 1200, 300, 10,
                                    help="Total number of pages in the book")

with c2:
    st.markdown("#### 📈 Publishing Details")
    edition_count = st.number_input("Edition Count", 1, 1000, 5, 1,
                                    help="How many editions/versions have been published")
    subject_count = st.number_input("Subject Tags Count", 1, 100, 10, 1,
                                    help="Number of topic/genre tags the book has")
    publish_year  = st.number_input("Publication Year", 1700, 2025, 2020, 1,
                                    help="Year the book was first published")
    book_age      = 2025 - publish_year
    st.caption(f"📅 Book Age: {book_age} years old")

with c3:
    st.markdown("#### 💰 Price & Availability")
    price_usd           = st.number_input("Price (USD)", 0.99, 99.99, 14.99, 0.50,
                                          help="Selling price in US dollars")
    has_cover           = st.selectbox("Has Cover Image?", [1, 0],
                                       format_func=lambda x: "✅ Yes" if x else "❌ No",
                                       help="Does the book have a cover image?")
    is_available_online = st.selectbox("Free to Read Online?", [0, 1],
                                       format_func=lambda x: "✅ Yes" if x else "❌ No",
                                       help="Is the book freely available online?")

    # Auto-computed display
    price_cat_label = "Budget 🟢" if price_usd <= 12.99 else ("Mid-Range 🟡" if price_usd <= 18.99 else "Premium 🔴")
    st.info(f"""
    **Price Category:** {price_cat_label}
    **Has Rating:** {"Yes ✅" if ratings_count > 0 else "No ❌"}
    **Book Age:** {book_age} years
    """)

# ── Predict Button ───────────────────────────────────────────
st.divider()
col_btn, _ = st.columns([1, 3])
with col_btn:
    predict_btn = st.button("🔮 Predict Popularity", type="primary", use_container_width=True)

# ── Results ──────────────────────────────────────────────────
if predict_btn:

    user_input = {
        "genre"              : genre,
        "avg_rating"         : avg_rating,
        "ratings_count"      : ratings_count,
        "page_count"         : page_count,
        "edition_count"      : edition_count,
        "subject_count"      : subject_count,
        "book_age"           : book_age,
        "price_usd"          : price_usd,
        "has_cover"          : has_cover,
        "is_available_online": is_available_online,
    }

    # Preprocess and predict
    X_input    = preprocess(user_input)
    prediction = model.predict(X_input)[0]
    prob       = model.predict_proba(X_input)[0][1]
    confidence = "High" if abs(prob - 0.5) > 0.3 else ("Medium" if abs(prob - 0.5) > 0.1 else "Low")

    st.divider()
    st.markdown("## 🎯 Prediction Result")

    r1, r2, r3 = st.columns(3)

    with r1:
        if prediction == 1:
            st.success("### ✅ POPULAR")
            st.markdown("This book is predicted to be **popular** based on its features.")
        else:
            st.error("### ❌ NOT POPULAR")
            st.markdown("This book is predicted to be **not popular** based on its features.")

    with r2:
        st.metric(
            label="Popularity Probability",
            value=f"{prob:.1%}",
            delta=f"{prob - 0.30:.1%} vs average (30%)"
        )

    with r3:
        conf_color = "🟢" if confidence == "High" else ("🟡" if confidence == "Medium" else "🔴")
        st.metric("Model Confidence", f"{conf_color} {confidence}")

    # Progress bar
    st.markdown(f"**Popularity Probability: {prob:.1%}**")
    st.progress(float(prob))
    st.caption("Popular threshold = top 30% of books | Random baseline = 30%")

    # ── AI Suggestions ────────────────────────────────────────
    st.divider()
    st.markdown("## 🤖 AI-Powered Suggestions to Improve Book Reach")

    with st.spinner("Generating personalized suggestions using Gemini AI..."):
        suggestions, ai_worked = get_suggestions(user_input, prediction, prob)

    if ai_worked:
        st.success("✅ Suggestions generated by Gemini AI")
    else:
        st.info("ℹ️ Showing suggestions based on dataset analysis (AI unavailable)")

    impact_icon = {"High": "🔴 High Impact", "Medium": "🟡 Medium Impact", "Low": "🟢 Low Impact"}

    for i, s in enumerate(suggestions, 1):
        impact = s.get("impact", "Medium")
        with st.expander(
            f"{impact_icon.get(impact, '🟡 Medium Impact')} — {i}. {s['title']}",
            expanded=(i <= 3)
        ):
            st.write(s["suggestion"])

    # ── Comparison Table ──────────────────────────────────────
    st.divider()
    st.markdown("## 📊 Your Book vs Dataset Averages")
    st.caption("Based on 14,021 books collected from Open Library API")

    comparison = pd.DataFrame({
        "Feature"         : ["avg_rating", "ratings_count", "edition_count",
                             "subject_count", "page_count", "price_usd"],
        "Your Book"       : [avg_rating, ratings_count, edition_count,
                             subject_count, page_count, price_usd],
        "Popular Avg"     : [4.33, 21.6, 119.0, 24.6, 302.3, 15.09],
        "Not Popular Avg" : [3.83, 1.4,  23.0,  13.8, 274.5, 16.77]
    })

    def highlight_row(row):
        your_val   = row["Your Book"]
        pop_val    = row["Popular Avg"]
        notpop_val = row["Not Popular Avg"]
        if abs(your_val - pop_val) < abs(your_val - notpop_val):
            return ["background-color: #e8f5e9"] * 4   # green — closer to popular
        else:
            return ["background-color: #fce4ec"] * 4   # red — closer to not popular

    styled = comparison.style.apply(highlight_row, axis=1).format({
        "Your Book"      : "{:.2f}",
        "Popular Avg"    : "{:.2f}",
        "Not Popular Avg": "{:.2f}"
    })

    st.dataframe(styled, use_container_width=True, hide_index=True)
    st.caption("🟢 Green rows = your value is closer to Popular average | 🔴 Red rows = closer to Not Popular average")

# ── Footer ───────────────────────────────────────────────────
st.divider()
st.markdown("""
<div style='text-align:center; color:#999; font-size:0.82rem'>
    Book Popularity Predictor | XGBoost Model (F1: 99.47%, AUC: 99.99%) |
    Dataset: 14,021 books from Open Library API | 35 Genres | Week 4 Deliverable
</div>
""", unsafe_allow_html=True)
