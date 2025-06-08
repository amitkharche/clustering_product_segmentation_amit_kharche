"""
Model training script for product segmentation based on purchase behavior using clustering.
"""

import os
import logging
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Configuration
DATA_PATH = "data/retail_products.csv"
MODEL_DIR = "model"
MODEL_FILE = os.path.join(MODEL_DIR, "kmeans_model.pkl")
SCALER_FILE = os.path.join(MODEL_DIR, "scaler.pkl")
N_CLUSTERS = 5
RANDOM_STATE = 42


def load_data(path):
    """Load dataset from CSV file."""
    if not os.path.exists(path):
        logger.error("Data file not found.")
        raise FileNotFoundError("Dataset not found.")
    logger.info(f"Loading data from {path}")
    return pd.read_csv(path)


def preprocess_data(df):
    """Clean and preprocess data for clustering."""
    logger.info("Preprocessing data...")
    df_cleaned = df.drop(columns=["ProductID"], errors="ignore")
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(df_cleaned)
    return features_scaled, scaler


def train_model(X, n_clusters):
    """Train KMeans clustering model."""
    logger.info(f"Training KMeans with {n_clusters} clusters")
    model = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE)
    model.fit(X)
    return model


def evaluate_model(model, X):
    """Evaluate model using silhouette score."""
    logger.info("Evaluating model...")
    labels = model.predict(X)
    score = silhouette_score(X, labels)
    logger.info(f"Silhouette Score: {score:.4f}")


def save_object(obj, path):
    """Save model/scaler to file."""
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    logger.info(f"Saved object to {path}")


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    df = load_data(DATA_PATH)
    X, scaler = preprocess_data(df)
    model = train_model(X, N_CLUSTERS)
    evaluate_model(model, X)
    save_object(model, MODEL_FILE)
    save_object(scaler, SCALER_FILE)
    logger.info("Training pipeline completed successfully.")


if __name__ == "__main__":
    main()
