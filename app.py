from tkinter import Menu

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="customer intelligence system", page_icon="📊", layout="wide")
st.title("Customer Intelligence using Unsupervised learning")
st.markdown("**Clustering and anoomaly detection PCA and Recommendation system**")

df = pd.read_csv("Mall_Customers.csv")

df["Genre"] = df["Genre"].map({"Male": 0, "Female": 1})

if Menu == "Dataset Overview":
    st.header("Dataset Overview")
    col1, col2, col3 = st.columns(3)
    col1.metrics("Total Customers", df.shape[0])
    col2.metrics("Average Income(k$)", df["Annual Income (k$)"].mean())
    col3.metric("Average Spending Score", round(df["Spending Score(1-100)"].mean(), 2))
    st.dataframe(df.head())

    #Preprocessing
    customer_df = df[["CustomerID", "Age", "Annual Income (k$)", "Spending Score(1-100)"]]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(customer_df[["Age", "Annual Income (k$)", "Spending Score(1-100)"]])

    if Menu == "Preprocessing":
        st.header("Preprocessing")
        st.write("The dataset has been preprocessed and scaled.")
        st.dataframe(customer_df.head())
    #kmeans clustering
    if menu == "Customer Segmentation":
        st.header("👥 Customer Segmentation using K-Means")

    k = st.slider("Select Number of Clusters", 2, 6, 4)

    kmeans = KMeans(n_clusters=k, random_state=42)
    customer_df['Cluster'] = kmeans.fit_predict(scaled_features)

    fig, ax = plt.subplots()
    ax.scatter(
        customer_df['Annual Income (k$)'],
        customer_df['Spending Score (1-100)'],
        c=customer_df['Cluster']
    )
    ax.set_xlabel("Annual Income (k$)")
    ax.set_ylabel("Spending Score")
    ax.set_title("Customer Segments")

    st.pyplot(fig)
    #Anomaly detection
    if menu == "Anomaly Detection":
        stheader("🚨 Anomaly Detection")

    detector = EllipticEnvelope(contamination=0.05)
    customer_df['Anomaly'] = detector.fit_predict(scaled_features)

    fig, ax = plt.subplots()
    ax.scatter(
        customer_df['Annual Income (k$)'],
        customer_df['Spending Score (1-100)'],
        c=customer_df['Anomaly']
    )
    ax.set_xlabel("Annual Income (k$)")
    ax.set_ylabel("Spending Score")
    ax.set_title("Anomalous Customers")

    st.pyplot(fig)

    st.info("🔴 -1 = Anomaly | 🔵 1 = Normal")
    #PCA 
    if Menu == "PCA Visualization":
       st.header("📉 PCA - Dimensionality Reduction")

    pca = PCA(n_components=2)
    pca_features = pca.fit_transform(scaled_features)

    st.write("Explained Variance Ratio:")
    st.write(pca.explained_variance_ratio_)

    fig, ax = plt.subplots()
    ax.scatter(
        pca_features[:, 0],
        pca_features[:, 1],
        c=customer_df['Cluster']
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA Projection")

    st.pyplot(fig)
    #Recommendation system
    if Menu == "Recommendation System":
        st.header("🎯 Similar Customer Recommendation")

    similarity = cosine_similarity(scaled_features)
    similarity_df = pd.DataFrame(
        similarity,
        index=customer_df['CustomerID'],
        columns=customer_df['CustomerID']
    )

    def recommend_similar_customers(customer_id, top_n=5):
        return similarity_df[customer_id].sort_values(ascending=False)[1:top_n+1]

    user_id = st.selectbox("Select Customer ID", customer_df['CustomerID'])
    st.subheader("Most Similar Customers")
    st.dataframe(recommend_similar_customers(user_id))
    st.info("Recommendations are based on cosine similarity of scaled features.")
    st.markdown("This system provides insights into customer segments, identifies anomalies, and recommends similar customers based on their profiles.")
    st.markdown("Feel free to explore the different sections to understand the customer intelligence system better!")
    st.markdown("Created by [Your Name] - Unsupervised Learning Project")
    st