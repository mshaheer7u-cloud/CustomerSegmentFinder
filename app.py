import streamlit as st
import joblib
import json
import plotly.graph_objects as go

st.set_page_config(page_title="Customer Segment Finder", layout="centered")

@st.cache_resource
def load_models():
    kmeans = joblib.load('models/kmeans_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    with open('models/cluster_names.json', 'r') as f:
        cluster_names = json.load(f)
    return kmeans, scaler, cluster_names

kmeans, scaler, cluster_names = load_models()

segment_descriptions = {
    "0": "Customers with average income and average spending. A steady, reliable base for generalized marketing campaigns.",
    "1": "High income and high spending. Ideal candidates for loyalty programs and premium product offerings.",
    "2": "Lower income but high spending. Highly responsive to sales, flash discounts, and limited-time offers.",
    "3": "High income but low spending. Harder to convert but highly valuable; needs quality-driven marketing.",
    "4": "Lower income and low spending. Value hunters who are highly price-sensitive."
}

st.title("Customer Segment Finder")
st.write("Enter a customer's income and spending score to find their segment.")

income = st.number_input("Annual Income (k$)", min_value=0, max_value=150, value=60)
spending = st.number_input("Spending Score (1-100)", min_value=0, max_value=100, value=50)

if st.button("Find Segment"):
    scaled_input = scaler.transform([[income, spending]])
    cluster = kmeans.predict(scaled_input)[0]
    segment_name = cluster_names[str(cluster)]

    st.success(f"Predicted Segment: **{segment_name}**")
    st.write(segment_descriptions[str(cluster)])

    centers = scaler.inverse_transform(kmeans.cluster_centers_)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=centers[:,0], y=centers[:,1], mode='markers',
                              marker=dict(size=15, color='gray'),
                              name='Segment Centers'))
    fig.add_trace(go.Scatter(x=[income], y=[spending], mode='markers',
                              marker=dict(size=18, color='#2952CC', symbol='star'),
                              name='Your Input'))
    fig.update_layout(xaxis_title='Annual Income (k$)', yaxis_title='Spending Score (1-100)')
    st.plotly_chart(fig)