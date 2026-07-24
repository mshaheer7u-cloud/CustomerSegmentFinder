# Customer Segment Finder 🎯

A machine learning web app that automatically discovers and predicts customer segments based on income and spending behavior — built with K-Means clustering.

**🔗 Live App:** https://customer-segment-finder.streamlit.app/
---

## 📌 Overview

Businesses often struggle to answer a simple but important question: *"Who are our customers, really?"* Instead of guessing, this project uses **unsupervised machine learning** to let the data speak for itself — automatically grouping customers into meaningful segments without any manual labeling.

Given a customer's **Annual Income** and **Spending Score**, the app predicts which of 5 data-driven segments they belong to, along with a plain-language explanation of what that segment means for marketing strategy.

## 🧠 How It Works

This project uses **K-Means Clustering**, an unsupervised learning algorithm that groups similar data points together — without being told the "correct" answer in advance.

1. **Data:** 200 real customer records (Annual Income, Spending Score)
2. **Optimal K selection:** Used the **Elbow Method** to determine that 5 clusters best represent the natural groupings in the data
3. **Training:** Fit a K-Means model (scikit-learn) on standardized features
4. **Segment naming:** Interpreted each cluster's characteristics and assigned business-friendly names
5. **Deployment:** Wrapped the trained model in an interactive Streamlit app

## 🧩 The 5 Segments

| Segment | Income | Spending | Business Meaning |
|---|---|---|---|
| **Premium Target Customers** | High | High | Most valuable segment — ideal for loyalty programs and premium offerings |
| **Careful Wealthy** | High | Low | High purchasing power, low engagement — the biggest untapped opportunity |
| **Impulsive Spenders** | Low | High | Highly engaged despite limited budget — responsive to promotions |
| **Budget Conscious** | Low | Low | Price-sensitive — best served with discount-driven offers |
| **Standard Customers** | Average | Average | General audience for broad marketing campaigns |

## 🛠️ Tech Stack

- **Python** — core language
- **scikit-learn** — K-Means clustering, StandardScaler
- **Pandas** — data handling
- **Plotly** — interactive visualization
- **Streamlit** — web app framework
- **joblib** — model serialization

## ⚠️ Known Limitations

- Trained on a small dataset (200 customers) with Income ranging $15k–$137k and Spending Score 1–99. Predictions for inputs far outside this range are less reliable, since the model has no data to learn from in those regions — the app surfaces a warning in this case rather than silently returning an unreliable result.
- Segments are based on only two features (Income, Spending Score). A production system would likely incorporate additional behavioral and demographic data for richer segmentation.

## 🚀 Running Locally

```bash
git clone https://github.com/mshaheer7u-cloud/CustomerSegmentFinder.git
cd CustomerSegmentFinder
pip install -r requirements.txt
streamlit run app.py
```

## 📂 Project Structure

```
CustomerSegmentFinder/
├── app.py                  # Streamlit application
├── requirements.txt        # Python dependencies
├── runtime.txt              # Python version (for deployment)
└── models/
    ├── kmeans_model.pkl     # Trained K-Means model
    ├── scaler.pkl           # Fitted StandardScaler
    └── cluster_names.json   # Cluster-to-segment-name mapping
```

---

*Part of a broader machine learning portfolio exploring regression, classification, imbalanced classification, and clustering across different real-world domains.*
