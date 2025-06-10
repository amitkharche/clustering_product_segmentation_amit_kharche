
---

````markdown
# 🛍️ Product Segmentation with KMeans Clustering

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.22.0-red)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2.2-orange)

---

A practical machine learning solution to **segment retail products** based on customer purchasing patterns using **unsupervised clustering**.

🔍 This helps retail businesses **optimize inventory**, **target marketing**, and **improve pricing strategies** by grouping products with similar sales and behavioral traits.

---

##  Business Objective

In retail, understanding which products behave similarly can drive better:

-  Promotion targeting (e.g., offer bundles for similar products)
-  Inventory planning (e.g., forecast demand at the segment level)
-  Pricing decisions (e.g., identify discount-sensitive products)

By segmenting products into behavioral clusters, stakeholders can make more **data-driven decisions**.

---

##  Features Used for Clustering

| Feature                   | Description                              |
|--------------------------|------------------------------------------|
| Avg_Purchase_Frequency   | Frequency of purchase per product        |
| Avg_Basket_Size          | Avg. quantity of item per basket         |
| Avg_Spend_Per_Purchase   | Avg. customer spend on the product       |
| Return_Rate              | % of units returned                      |
| Discount_Availability    | Historical availability of discounts     |

---

##  Pipeline Breakdown

### 1. `model_training.py` – Model Training

- Loads product-level data from `data/retail_products.csv`
- Drops `ProductID` before modeling (but preserved in app for display)
- Standardizes numerical features using `StandardScaler`
- Applies **KMeans** clustering (`n_clusters=5`)
- Evaluates performance using **silhouette score**
- Saves the model and scaler in `model/`

---

### 2. `app.py` – Interactive Streamlit Application

- Upload a CSV containing product-level data
- Predicts cluster labels using the trained KMeans model
- Applies **PCA** to visualize clusters in 2D
- Displays:
  -  Interactive Plotly scatter chart
  -  Cluster summary table
  -  Silhouette score metric
- Allows CSV download of clustered products with labels

---

##  How to Run This Project

###  Step 1: Clone the Repo

```bash
git clone https://github.com/amitkharche/clustering_product_segmentation_amit_kharche.git
cd clustering_product_segmentation_amit_kharche
````

###  Step 2: Install Requirements

```bash
pip install -r requirements.txt
```

###  Step 3: Train the Model

```bash
python model_training.py
```

This will generate:

* `model/kmeans_model.pkl`
* `model/scaler.pkl`

###  Step 4: Launch the App

```bash
streamlit run app.py
```

Then upload your product CSV to view clusters and download results.

---

##  Project Structure

```
clustering_product_segmentation_amit_kharche/
├── data/
│   └── retail_products.csv           # Input product data
├── model/
│   ├── kmeans_model.pkl              # Trained KMeans model
│   └── scaler.pkl                    # StandardScaler object
├── app.py                            # Streamlit app for prediction
├── model_training.py                 # Clustering training pipeline
├── requirements.txt                  # Python dependencies
├── README.md                         # Project documentation
└── .github/
    └── ISSUE_TEMPLATE/               # GitHub issue templates
```

---

##  Sample Output

*  5 Product Segments visualized in 2D using PCA
*  Interactive color-coded clusters
*  Downloadable CSV with cluster labels
*  Cluster-wise average spend, return rate, etc.

---

##  Possible Enhancements

* Add dynamic cluster count (`k`) selection
* Use t-SNE or UMAP for nonlinear visualizations
* Integrate category-specific features (e.g., brand, size)
* Add SHAP explainability to understand cluster assignment

---

##  Contact

For collaboration or questions, connect with me:

* [LinkedIn – Amit Kharche](https://www.linkedin.com/in/amit-kharche)
* [Medium – @amitkharche14](https://medium.com/@amitkharche14)
* [GitHub – @amitkharche](https://github.com/amitkharche)

---

##  License

This project is licensed under the MIT License.

---
