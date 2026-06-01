# 🍽️ UX Pain Point Detection from Restaurant Reviews

An NLP pipeline that analyzes Zomato user reviews to automatically detect UX and service issues — combining sentiment analysis with topic modeling.

---

## 🎯 Project Goal

Use machine learning to identify **what frustrates users most** in restaurant experiences, by analyzing negative reviews and grouping them into meaningful UX topics.

---

## 📁 Dataset

- **Source:** Zomato Reviews (`zomato_reviews.csv`)
- **Content:** User reviews with text and star ratings

---

## 🔄 Pipeline

1. **Text Cleaning** — Lowercase, remove URLs, special characters
2. **Sentiment Labeling** — Ratings ≥ 4 → Positive, < 4 → Negative
3. **Sentiment Classification** — Logistic Regression and Linear SVM via TF-IDF
4. **Topic Modeling** — LDA on negative reviews to extract 6 UX issue topics
5. **Severity Scoring** — Rank topics by frequency × dissatisfaction intensity

---

## 🤖 Models

| Model | Task |
|---|---|
| Logistic Regression | Sentiment classification |
| Linear SVM | Sentiment classification (comparison) |
| LDA | Topic modeling on negative reviews |

---

## 📈 Key Outputs

- Confusion matrix for both classifiers
- LDA topic keywords (`lda_topics_keywords.csv`)
- Negative reviews with assigned topics (`neg_reviews_with_topics.csv`)
- UX severity ranking by topic (`ux_topic_severity.csv`)

---

## 🛠️ Tech Stack

- Python 3
- Pandas, NumPy
- Scikit-learn (TF-IDF, Logistic Regression, SVM, LDA)
- Matplotlib, Seaborn

---

## 🚀 How to Run

1. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn
