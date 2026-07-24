import streamlit as st
import joblib
import json
import plotly.graph_objects as go

st.set_page_config(page_title="Customer Segment Finder", layout="centered")

# --- Custom CSS: clean, minimal, Inter font ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.main {
    background-color: #FAFAFA;
}

h1 {
    font-weight: 600;
    color: #1A1A1A;
}

.stButton>button {
    background-color: #2952CC;
    color: white;
    border-radius: 6px;
    border: none;
    padding: 0.5em 1.5em;
    font-weight: 500;
}

.stButton>button:hover {
    background-color: #1e3fa0;
    color: white;
}

div[data-testid="stMetricValue"] {
    color: #2952CC;
}
</style>
""", unsafe_allow_html=True)

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
st.markdown("<p style='color:#6B6B6B; font-size:16px;'>Enter a customer's income and spending score to find their marketing segment.</p>", unsafe_allow_html=True)

st.divider()
with st.expander("ℹ️ How this works"):
    st.markdown("""
    This tool uses **K-Means clustering** — a machine learning technique that 
    groups customers into segments based on patterns in their income and 
    spending behavior, without being told the "right answer" in advance.
    
    It was trained on 200 real customer records, and automatically discovered 
    **5 natural segments**:
    
    | Segment | Income | Spending | Meaning |
    |---|---|---|---|
    | Premium Target Customers | High | High | Best customers — prioritize for loyalty programs |
    | Careful Wealthy | High | Low | High potential — needs targeted convincing |
    | Impulsive Spenders | Low | High | Engaged despite limited budget |
    | Budget Conscious | Low | Low | Price-sensitive — respond to discounts |
    | Standard Customers | Average | Average | General audience |
    
    **How to read the chart:** the gray dots are the "center" of each segment 
    (based on training data). The blue star shows where your entered customer 
    falls relative to those segments — the closer to a gray dot, the more 
    strongly that customer matches that segment.
    
    **Note:** this model was trained on 200 customers with incomes ranging 
    from $15k–$137k and spending scores of 1–99. Predictions for values 
    far outside this range are less reliable, since the model has no data 
    to learn from in those regions.
    """)

col1, col2 = st.columns(2)
with col1:
    income = st.number_input("Annual Income (k$)", min_value=0, max_value=150, value=60)
with col2:
    spending = st.number_input("Spending Score (1-100)", min_value=0, max_value=100, value=50)

if st.button("Find Segment", use_container_width=True):
    scaled_input = scaler.transform([[income, spending]])
    cluster = kmeans.predict(scaled_input)[0]
    segment_name = cluster_names[str(cluster)]

    st.markdown(f"""
    <div style="background-color:#EEF2FF; border-left:4px solid #2952CC; padding:16px 20px; border-radius:6px; margin-top:20px;">
        <p style="color:#6B6B6B; margin:0; font-size:14px;">PREDICTED SEGMENT</p>
        <p style="color:#1A1A1A; font-size:24px; font-weight:600; margin:4px 0 8px 0;">{segment_name}</p>
        <p style="color:#4B4B4B; margin:0; font-size:15px;">{segment_descriptions[str(cluster)]}</p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Find Segment", use_container_width=True):
    
    # Out-of-range warning
    if income < 15 or income > 137 or spending < 1 or spending > 99:
        st.warning(
            "⚠️ This input is outside the range of the training data "
            "(Income: $15k–$137k, Spending Score: 1–99). The prediction "
            "below may be less reliable, since the model wasn't trained "
            "on customers with these characteristics."
        )
    
    scaled_input = scaler.transform([[income, spending]])
    # ... rest of the code same as before

    centers = scaler.inverse_transform(kmeans.cluster_centers_)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=centers[:,0], y=centers[:,1], mode='markers',
        marker=dict(size=16, color='#D1D5DB', line=dict(width=1, color='#9CA3AF')),
        name='Segment Centers'
    ))
    fig.add_trace(go.Scatter(
        x=[income], y=[spending], mode='markers',
        marker=dict(size=20, color='#2952CC', symbol='star', line=dict(width=1, color='#1e3fa0')),
        name='Your Input'
    ))
    fig.update_layout(
        xaxis_title='Annual Income (k$)',
        yaxis_title='Spending Score (1-100)',
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family='Inter', color='#1A1A1A'),
        margin=dict(t=30, b=30, l=30, r=30),
        legend=dict(bgcolor='rgba(0,0,0,0)')
    )
    st.plotly_chart(fig, use_container_width=True)
