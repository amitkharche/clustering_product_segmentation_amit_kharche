# 🛍️ Product Segmentation via Clustering

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.22.0-red)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2.2-orange)

## 📌 Business Use Case

Retail businesses must understand product behavior to optimize inventory, marketing, and pricing. By segmenting products based on customer purchasing patterns, businesses can:

- Group similar products for promotions
- Identify high-return items
- Design tailored pricing strategies

## 🧠 Model Overview

We use **KMeans clustering** to segment products based on:

- Purchase frequency
- Basket size
- Spend per purchase
- Return rate
- Discount sensitivity

## ⚙️ Pipeline Overview

### 🛠 Training (`model_training.py`)
- Loads retail product data
- Cleans and standardizes features
- Trains KMeans clustering model
- Evaluates using silhouette score
- Saves model and scaler to `model/`

### 🌐 Deployment (`app.py`)
- Upload product data via Streamlit interface
- Runs model inference and clustering
- Shows PCA-based visualization
- Downloads segmented output as CSV

## 🚀 Running Instructions

### 1. Install requirements
```bash
pip install -r requirements.txt
```

### 2. Train the model
```bash
python model_training.py
```

### 3. Launch the Streamlit App
```bash
streamlit run app.py
```

> 📁 Ensure `model/` contains the trained model and scaler after training.

## 📂 Project Structure
```
product_segmentation_project/
├── data/                     # Input data (retail_products.csv)
├── model/                    # Saved model and scaler
├── notebooks/                # EDA and analysis
├── app.py                    # Streamlit app
├── model_training.py         # Training pipeline
├── requirements.txt          # Dependencies
├── README.md                 # Project documentation
└── .github/ISSUE_TEMPLATE/   # GitHub issue templates
```

## 📝 License

This project is licensed under the MIT License.

