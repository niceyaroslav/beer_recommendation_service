from sklearn.preprocessing import StandardScaler

from core_helpers import *


def recommend_from_input(user_input, df_clean, df_scaled, feature_cols, scaler, kmeans, knn, n=5, descriptors=None):

    if isinstance(user_input, str):
        user_input = text_to_model_input(user_input, descriptors, feature_cols, df_clean)
    else:
        user_input = slider_input_to_model_input(user_input, df_clean)

    user_vector = build_user_vector(user_input, df_clean, feature_cols)
    user_scaled = scaler.transform(user_vector)
    user_scaled_df = pd.DataFrame(
        user_scaled,
        columns=feature_cols
    )

    cluster = kmeans.predict(user_scaled_df)[0]

    distances, indices = knn.kneighbors(user_scaled_df, n_neighbors=len(df_clean))

    results = []

    for dist, idx in zip(distances[0], indices[0]):
        if df_clean.iloc[idx]["cluster"] == cluster:
            results.append((idx, dist))

        if len(results) >= n:
            break

    metric = knn.metric
    recs = df_clean.iloc[[idx for idx, _ in results]].copy()
    recs["distance"] = [dist for _, dist in results]
    recs["similarity"] = recs["distance"].apply(
        lambda d: distance_to_similarity(d, metric)
    )

    return cluster, recs[
        ["beer name full", "style", "brewery", "abv",
         "review overall", "number of reviews", "distance", "similarity"]
    ]


def distance_to_similarity(distance, metric):
    if metric == "cosine":
        # cosine distance is usually 0 to 2, often 0 to 1 in practice
        return 1 - distance

    if metric == "euclidean":
        # euclidean distance has no fixed upper bound
        return 1 / (1 + distance)

    return 1 / (1 + distance)


def recommend_similar_beer(beer_name, df_clean, X_scaled, knn, n=5):

    idx = df_clean[df_clean["beer name full"] == beer_name].index[0]
    row_pos = df_clean.index.get_loc(idx)

    distances, indices = knn.kneighbors([X_scaled[row_pos]], n_neighbors=n+1)

    recs = df_clean.iloc[indices[0][1:]].copy()
    recs["distance"] = distances[0][1:]

    return recs[
        ["beer name full", "style", "brewery", "abv",
         "review overall", "number of reviews", "distance"]
    ]


def rank_recommendations(recs):
    recs = recs.copy()

    rating_min = recs["review overall"].min()
    rating_max = recs["review overall"].max()

    if rating_max == rating_min:
        recs["rating norm"] = 1.0
    else:
        recs["rating norm"] = (
            recs["review overall"] - rating_min
        ) / (rating_max - rating_min)

    # normalize distance within returned results
    recs["similarity norm"] = recs["distance"].rank(ascending=True, pct=True)
    recs["similarity norm"] = 1 - recs["similarity norm"]

    # final score
    recs["score"] = 0.7 * recs["similarity norm"] + 0.3 * recs["rating norm"]

    return recs.sort_values("score", ascending=False)


if __name__ == '__main__':
    from data_preprocessor import DataPreprocessor
    from prepare_and_train_model import *

    data_preprocessor = DataPreprocessor()

    df_scaled = data_preprocessor.scaled_df
    feature_cols = data_preprocessor.feature_cols

    kmeans, knn, scaler, df_clean = prepare_and_train_models()

    user_input = {
        "bitter": 0.8,
        "hoppy": 0.9,
        "sweet": 0.2,
        "body": 0.5,
        "abv": 6.5
    }

    cluster, recs = recommend_from_input(
        user_input,
        df_clean,
        df_scaled,
        feature_cols,
        scaler,
        kmeans,
        knn
    )

    ranked_recs = rank_recommendations(recs)

    user_input_text = "fruity, hoppy, citra"

    cluster_from_text, recs_from_text = recommend_from_input(
        user_input_text,
        df_clean,
        df_scaled,
        feature_cols,
        scaler,
        kmeans,
        knn,
        descriptors=descriptors
    )

    ranked_recs_from_text = rank_recommendations(recs_from_text)