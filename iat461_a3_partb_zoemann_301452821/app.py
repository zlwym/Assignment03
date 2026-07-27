# Imports

import streamlit as st
import pandas as pd
import geopandas as gpd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

print("running")

st.title("Vancouver Business Licences Explorer")

# B1

gdf = gpd.read_file("business-licences.geojson")

gdf = gdf[gdf["status"] == "Issued"].copy()

# Get rid of null values for local area and geometry
gdf = gdf.dropna(subset=["localarea"])
gdf = gdf.dropna(subset=["geometry"])

area_count = gdf["localarea"].value_counts()
print(area_count)

minimum_size = 100

areas = area_count[area_count >= minimum_size].index

gdf = gdf[gdf["localarea"].isin(areas)]

# Feature matrix

composition = (
    pd.crosstab(
        gdf["localarea"],
        gdf["businesstype"],
        normalize="index"
    ) * 100
)

# B2

k = st.slider(
    "Number of Clusters",
    2,
    10,
    4
)

# Running KMeans

kmeans = KMeans(
    n_clusters = k,
    random_state = 42,
    n_init = 10
)

clusters = kmeans.fit_predict(composition)

# PCA

pca = PCA(n_components = 2)

points = pca.fit_transform(composition)

pca_df = pd.DataFrame(
    points, 
    columns = ["PC1", "PC2"]
)

pca_df["cluster"] = clusters
pca_df["area"] = composition.index

fig = px.scatter(
    pca_df,
    x = "PC1",
    y = "PC2",
    color = pca_df["cluster"].astype(str),
    hover_name = "area",
    title = "Neighbourhood Similarity (PCA)"
)

st.plotly_chart(fig, width = "stretch")

# B3

gdf["longitude"] = gdf.geometry.x
gdf["latitude"] = gdf.geometry.y

centroid_df = (
    gdf.groupby("localarea")
        .agg(
            latitude = ("latitude", "mean"),
            longitude = ("longitude", "mean"),
            business_count = ("localarea", "size")
        )
        .reset_index()
)

centroid_df["cluster"] = clusters.astype(str)

fig_map = px.scatter_map(
    centroid_df,
    lat = "latitude",
    lon = "longitude",
    color = "cluster",
    size = "business_count",
    hover_name = "localarea",
    zoom = 10,
    height = 550,
    map_style = "carto-darkmatter"
)

st.plotly_chart(fig_map, width = "stretch")

# B4

cluster_df = pd.DataFrame({
    "Area": composition.index,
    "Cluster": clusters
})

for c in sorted(cluster_df["Cluster"].unique()):
    st.subheader(f"Cluster {c}")

    members = cluster_df[
        cluster_df["Cluster"] == c
    ]

    st.write(list(members["Area"]))