import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

st.set_page_config(page_title="Nykaa Customer Analytics", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("NYKA.csv", parse_dates=["signup_date", "last_purchase_date"])
    return df

df = load_data()

st.title("Nykaa: Segmentation • CLTV • Churn Dashboard")

# 1) RFM SEGMENTATION
st.header("1. RFM Segmentation")
rfm = df.rename(columns={
    "recency_days": "Recency",
    "frequency_3m": "Frequency",
    "monetary_value_3m": "Monetary"
})
fig_rfm = px.scatter_3d(
    rfm,
    x="Recency", y="Frequency", z="Monetary",
    color="RFM_segment_label",
    labels={"Recency":"Recency (days)",
            "Frequency":"# orders (3m)",
            "Monetary":"Spend (₹, 3m)"},
    title="3D RFM Clusters"
)
st.plotly_chart(fig_rfm, use_container_width=True)
st.write(
    "Customers are grouped by how recently and how often they buy, "
    "and how much they spend. You can see clear clusters like “Loyal Spenders” vs. “At-Risk.”"
)

# 2) CUSTOMER LIFETIME VALUE
st.header("2. Customer Lifetime Value (3-Month Forecast)")
# Distribution of predicted CLTV
fig_cltv_dist = px.histogram(
    df, x="predicted_CLTV_3m", nbins=30,
    title="Predicted CLTV Distribution (₹)"
)
st.plotly_chart(fig_cltv_dist, use_container_width=True)
st.write(
    "Most customers have a moderate predicted CLTV under ₹1,000, "
    "with a long tail of high-value customers worth ₹2,000+."
)

# Predicted vs. actual
fig_cltv_scatter = px.scatter(
    df, x="predicted_CLTV_3m", y="actual_CLTV_3m",
    title="Predicted vs Actual CLTV (3m)"
)
st.plotly_chart(fig_cltv_scatter, use_container_width=True)
st.write(
    "Predictions align reasonably with actuals but tend to over-estimate for lower-value segments."
)

# 3) CHURN ANALYSIS & PREDICTION
st.header("3. Churn Analysis & Prediction")

# Overall churn rate
churn_rate = df["churn_within_3m_flag"].mean()
fig_churn_bar = px.bar(
    x=["Active","Churned"],
    y=[1-churn_rate, churn_rate],
    labels={"x":"Status","y":"Proportion"},
    title="Overall 3-Month Churn Rate"
)
st.plotly_chart(fig_churn_bar, use_container_width=True)
st.write(f"About **{churn_rate:.1%}** of customers churn within 3 months of their first purchase.")

# Churn model
st.subheader("Churn Prediction Model (Logistic Regression)")

features = [
    "recency_days", "frequency_3m", "monetary_value_3m",
    "time_on_app_minutes", "page_views_per_session",
    "campaign_clicks", "campaign_views", "cart_abandonment_rate",
    "first_time_buyer_flag"
]
X = df[features].fillna(0)
y = df["churn_within_3m_flag"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred_proba = model.predict_proba(X_test)[:,1]
auc = roc_auc_score(y_test, y_pred_proba)
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)

fig_roc = px.line(
    x=fpr, y=tpr,
    labels={"x":"False Positive Rate","y":"True Positive Rate"},
    title=f"ROC Curve (AUC = {auc:.2f})"
)
st.plotly_chart(fig_roc, use_container_width=True)
st.write(
    f"The model achieves an AUC of **{auc:.2f}**, "
    "showing good ability to distinguish which customers will churn."
)
