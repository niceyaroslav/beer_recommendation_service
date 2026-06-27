# This should be the main entrypoint
import streamlit as st
from sklearn.preprocessing import StandardScaler

from prepare_and_train_model import prepare_and_train_models
from recommender import (
    recommend_from_input,
    rank_recommendations,
    text_to_model_input, recommend_similar_beer,
)
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


from data_preprocessor import DataPreprocessor


st.set_page_config(
    page_title="Beer Recommender",
    page_icon="🍺",
    layout="wide"
)


@st.cache_resource
def get_artifacts():
    dp = DataPreprocessor()
    df_scaled = dp.scaled_df
    feature_cols = dp.feature_cols
    descriptors = dp.descriptors
    kmeans, knn, scaler, df_clean = prepare_and_train_models()
    return df_clean, df_scaled, feature_cols, descriptors, scaler, kmeans, knn


df_clean, df_scaled, feature_cols, descriptors, scaler, kmeans, knn = get_artifacts()


st.title("🍺 Beer Recommendation System")

st.write(
    "Recommend beers based on taste preferences using KMeans clustering "
    "and k-nearest neighbors similarity search."
)


tab_text, tab_sliders, tab_similar = st.tabs(
    ["Text input", "Slider input", "Similar beer"]
)


with tab_text:
    st.subheader("Describe what you want")

    user_text = st.text_input(
        "Example: very fruity, hoppy, light, not too sour",
        value="very hoppy, fruity"
    )

    n_recommendations = st.slider(
        "Number of recommendations",
        min_value=3,
        max_value=15,
        value=5
    )

    if st.button("Recommend from text"):
        extracted_user_input = text_to_model_input(
            user_text,
            descriptors,
            feature_cols,
            df_clean
        )

        st.write("Extracted preferences:")
        st.json(extracted_user_input)

        print(user_text)
        cluster, recs = recommend_from_input(
            user_text,
            df_clean,
            df_scaled,
            feature_cols,
            scaler,
            kmeans,
            knn,
            n=n_recommendations,
            descriptors=descriptors
        )

        ranked_recs = rank_recommendations(recs, df_clean)

        st.success(f"Predicted cluster: {cluster}")
        st.dataframe(ranked_recs, width="stretch")


with tab_sliders:
    st.subheader("Choose taste preferences")

    col1, col2, col3 = st.columns(3)

    with col1:
        bitter = st.slider("Bitter", 0.0, 1.0, 0.5)
        sweet = st.slider("Sweet", 0.0, 1.0, 0.5)
        sour = st.slider("Sour", 0.0, 1.0, 0.5)

    with col2:
        fruits = st.slider("Fruits", 0.0, 1.0, 0.5)
        hoppy = st.slider("Hoppy", 0.0, 1.0, 0.5)
        malty = st.slider("Malty", 0.0, 1.0, 0.5)

    with col3:
        body = st.slider("Body", 0.0, 1.0, 0.5)
        spices = st.slider("Spices", 0.0, 1.0, 0.5)
        abv = st.slider("ABV", 0.0, 15.0, 6.0)

    slider_raw_input = {
        "bitter": bitter,
        "sweet": sweet,
        "sour": sour,
        "fruits": fruits,
        "hoppy": hoppy,
        "malty": malty,
        "body": body,
        "spices": spices,
        "abv": abv,
    }

    if st.button("Recommend from sliders"):

        cluster, recs = recommend_from_input(
            slider_raw_input,
            df_clean,
            df_scaled,
            feature_cols,
            scaler,
            kmeans,
            knn,
            n=5
        )

        ranked_recs = rank_recommendations(recs, df_clean)

        st.success(f"Predicted cluster: {cluster}")
        st.dataframe(ranked_recs, width="stretch")


with tab_similar:
    st.subheader("Find beers similar to one you already know")

    beer_name = st.selectbox(
        "Select a beer",
        df_clean["beer name full"].sort_values().unique()
    )

    if st.button("Find similar beers"):
        # Use your existing recommend_similar_beer function here

        recs = recommend_similar_beer(beer_name, df_clean, df_scaled, knn, n=5)

        ranked_recs = rank_recommendations(recs, df_clean)

        st.dataframe(ranked_recs, width="stretch")