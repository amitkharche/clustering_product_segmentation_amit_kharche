"""
Streamlit app for product segmentation using KMeans clustering model.
"""

import streamlit as st
import pandas as pd
import pickle
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

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

    pca = PCA(n_components=2)
    components = pca.fit_transform(scaled)
    df["PCA1"], df["PCA2"] = components[:, 0], components[:, 1]
    return df

def plot_clusters(df):
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x="PCA1", y="PCA2", hue="Cluster", palette="Set2", s=100, ax=ax)
    ax.set_title("Product Clusters")
    st.pyplot(fig)

def main():
    uploaded_file = st.file_uploader("Upload product data (.csv)", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.subheader("Uploaded Data")
        st.write(df.head())

        if not validate_data(df):
            return

        model, scaler = load_model()
        if model and scaler:
            df = run_prediction(df, model, scaler)
            st.subheader("Clustered Data")
            st.dataframe(df.head())

            plot_clusters(df)

            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Result as CSV", csv, "segmented_products.csv", "text/csv")

if __name__ == "__main__":
    main()
