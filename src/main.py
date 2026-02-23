from src.data import load_data
from src.clustering import scale_data, train_kmeans
from src.visualize import elbow_plot, pca_plot
from src.features import get_features

df = load_data("data/app_user_behavior_dataset.csv")

X, feature_names = get_features(df)

X_scaled = scale_data(X)

elbow_plot(X_scaled)

clusters, model = train_kmeans(X_scaled, k=4)
df['cluster'] = clusters

# pca_plot(X_scaled, clusters)
# 
cluster_map = {
    0: 'High Engagement user',
    1: 'Moderate Engagement user',
    2: 'Low Engagement user',
    3: 'Inactive user'
}   

df['user_segment'] = df['cluster'].map(cluster_map)
# df['cluster'] = df.groupby('cluster')['cluster'].transform(lambda x: x.map(cluster_map))
df.to_csv("data/clustered_users.csv", index=False)

print("Clustering completed. Results saved to data/clustered_users.csv")


