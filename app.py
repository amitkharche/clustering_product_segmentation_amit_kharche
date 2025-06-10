"""
Streamlit app for product segmentation using KMeans clustering model.
"""

import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

REQUIRED_FEATURES = ["Avg_Purchase_Frequency", "Avg_Basket_Size", "Avg_Spend_Per_Purchase", "Return_Rate", "Discount_Availability"]

st.set_page_config(page_title="🛍️ Product Segmentation App", layout="wide")
st.title("🛍️ Product Segmentation Based on Purchase Behavior")

@st.cache_resource
def load_model():
    try:
        with open("model/kmeans_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("model/scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

def validate_data(df):
    missing = [col for col in REQUIRED_FEATURES if col not in df.columns]
    if missing:
        st.error(f"Missing required columns: {missing}")
        return False
    return True

def run_prediction(df, model, scaler):
    features = df[REQUIRED_FEATURES]
    scaled = scaler.transform(features)
    df["Cluster"] = model.predict(scaled)

    # PCA for visualization
    pca = PCA(n_components=2)
    components = pca.fit_transform(scaled)
    df["PCA1"], df["PCA2"] = components[:, 0], components[:, 1]

    # Silhouette Score
    score = silhouette_score(scaled, df["Cluster"])
    return df, score

def plot_clusters_interactive(df):
    fig = px.scatter(
        df, x="PCA1", y="PCA2", color="Cluster",
        hover_data=["ProductID", "Avg_Purchase_Frequency", "Avg_Basket_Size", "Avg_Spend_Per_Purchase", "Return_Rate", "Discount_Availability"],
        title="📊 Product Clusters (Interactive View)"
    )
    st.plotly_chart(fig, use_container_width=True)

def show_cluster_summary(df):
    st.subheader("📋 Cluster Profile Summary")
    summary = df.groupby("Cluster").agg({
        "Avg_Purchase_Frequency": "mean",
        "Avg_Basket_Size": "mean",
        "Avg_Spend_Per_Purchase": "mean",
        "Return_Rate": "mean",
        "Discount_Availability": "mean",
        "Cluster": "count"
    }).rename(columns={"Cluster": "Product_Count"})
    st.dataframe(summary.style.format("{:.2f}"))

def main():
    uploaded_file = st.file_uploader("📤 Upload Product Data (.csv)", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.subheader("📄 Uploaded Data Preview")
        st.write(df.head())

        if not validate_data(df):
            return

        model, scaler = load_model()
        if model and scaler:
            df, score = run_prediction(df, model, scaler)

            st.subheader("✅ Clustered Data Preview")
            st.dataframe(df.head())

            st.subheader("📈 Silhouette Score")
            st.metric("Score", f"{score:.3f}", help="Closer to 1 = better clustering")

            plot_clusters_interactive(df)
            show_cluster_summary(df)

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download Clustered Data", csv, "segmented_products.csv", "text/csv")

if __name__ == "__main__":
    main()
