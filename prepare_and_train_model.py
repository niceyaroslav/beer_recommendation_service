from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from data_preprocessor import DataPreprocessor

dp = DataPreprocessor()


def prepare_and_train_models():

    scaler = StandardScaler()
    df_clean = dp.df
    df_scaled = dp.scaled_df
    X = df_clean[dp.feature_cols].copy()
    scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=8, random_state=42, n_init=10)
    df_clean["cluster"] = kmeans.fit_predict(df_scaled)

    knn = NearestNeighbors(n_neighbors=10, metric="cosine")
    knn.fit(df_scaled)
    return kmeans, knn, scaler, df_clean
