# 🍷 Full Interactive KMeans Clustering + Prediction App

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Interactive KMeans Wine App", layout="wide")
st.title("🍷 Interactive KMeans Clustering & Prediction for Wine Dataset")

# Upload CSV
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # Numeric preprocessing
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    data = df[numeric_cols]
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # Number of clusters slider
    n_clusters = st.slider("Select number of clusters (KMeans)", 2, 10, 3)

    # Apply KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(data_scaled)
    df['Cluster'] = labels

    # Show cluster counts
    st.subheader("Cluster Labels Count")
    st.dataframe(df['Cluster'].value_counts().rename("Count"))

    # PCA 2D plot
    st.subheader("KMeans Clustering Plot (PCA 2D)")
    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(data_scaled)
    fig, ax = plt.subplots(figsize=(8,6))
    sns.scatterplot(x=data_pca[:,0], y=data_pca[:,1], hue=labels, palette='Set1', s=100, ax=ax)
    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    ax.set_title("KMeans Clustering of Wine Data")
    st.pyplot(fig)

    # Predict new wine data
    st.subheader("Predict Cluster for New Wine Data")
    new_input = {}
    for col in numeric_cols:
        new_input[col] = st.number_input(f"Enter value for {col}", value=float(df[col].mean()))

    if st.button("Predict Cluster"):
        new_df = pd.DataFrame([new_input])
        new_scaled = scaler.transform(new_df)
        predicted_cluster = kmeans.predict(new_scaled)[0]
        st.success(f"Predicted Cluster for new wine data: {predicted_cluster}")

    # Download clustered dataset
    st.subheader("Download Clustered Dataset")
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download CSV with Cluster Labels",
        data=csv,
        file_name='wine_kmeans_interactive.csv',
        mime='text/csv',
    )
else:
    st.info("Please upload a CSV file to get started.")
