import re
from collections import defaultdict
import pandas as pd


def build_user_vector(user_input, df_clean, feature_cols):
    # default values = dataset median
    defaults = df_clean[feature_cols].median().to_dict()

    # override with user input
    defaults = df_clean[feature_cols].median().to_dict()

    for k, v in user_input.items():
        # weighted override (stronger than median)
        defaults[k] = 0.7 * v + 0.3 * defaults[k]

    # return ordered vector
    return pd.DataFrame([defaults])[feature_cols]


# Logic to transform text to features

def text_to_features(text, descriptors):

    forced_mapping = {
        "fruity": "fruits",
        "fruit": "fruits",
        "fruits": "fruits",
        "hoppy": "hoppy",
        "hop": "hoppy",
        "hops": "hoppy",
        "bitter": "bitter",
        "sweet": "sweet",
        "sour": "sour",
        "malty": "malty",
        "spicy": "spices",
        "spices": "spices",
        "light": "body",
        "strong": "alcohol"
    }
    text = text.lower()

    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    words = text.split()

    descriptor_lookup = (
        descriptors
        .drop_duplicates("word")
        .set_index("word")["category"]
        .to_dict()
    )

    descriptor_lookup.update(forced_mapping)

    weak_modifiers = {"slightly", "little", "light", "mild", "subtle"}
    strong_modifiers = {"very", "strong", "heavy", "intense", "high"}
    negations = {"not", "no", "without", "less"}

    features = defaultdict(float)

    for i, word in enumerate(words):
        if word not in descriptor_lookup:
            continue

        weight = 1.0
        context = words[max(0, i-2):i]

        if any(w in weak_modifiers for w in context):
            weight = 0.4

        if any(w in strong_modifiers for w in context):
            weight = 1.5

        if any(w in negations for w in context):
            weight = -0.5

        category = descriptor_lookup[word]
        features[category] += weight

    if "hoppy" in features and "bitter" not in features:
        features["bitter"] = features["hoppy"] * 0.7

    if "fruits" in features:
        features["fruits"] *= 0.8

    return dict(features)


def adjust_features(features):
    features = features.copy()

    # Hoppy implies bitterness
    if "hoppy" in features:
        features["bitter"] = max(
            features.get("bitter", 0),
            features["hoppy"] * 0.6
        )

    # Fruity often implies slight sweetness
    if "fruits" in features:
        features["sweet"] = max(
            features.get("sweet", 0),
            features["fruits"] * 0.3
        )

    # Light → low body
    if "body" in features and features["body"] < 0:
        features["body"] = 0
    return features


def text_to_model_input(text, descriptors, feature_cols, df_clean):
    print(f'user input as text is {text}')
    raw_features = text_to_features(text, descriptors)
    adjusted_features = adjust_features(raw_features)
    print(f'adjusted features are {adjusted_features}')
    user_input = {}

    for feature, value in adjusted_features.items():
        if feature in feature_cols:
            value = max(value, 0)

            if value == 0:
                user_input[feature] = float(df_clean[feature].quantile(0.10))
            elif value < 0.75:
                user_input[feature] = float(df_clean[feature].quantile(0.40))
            elif value < 1.25:
                user_input[feature] = float(df_clean[feature].quantile(0.65))
            else:
                user_input[feature] = float(df_clean[feature].quantile(0.85))

    return user_input


def slider_input_to_model_input(slider_input, df_clean):
    user_input = {}

    for feature, value in slider_input.items():
        if feature == "abv":
            user_input[feature] = value
        else:
            user_input[feature] = float(df_clean[feature].quantile(value))

    return user_input


if __name__ == "__main__":
    # For test
    from data_preprocessor import DataPreprocessor
    data_preprocessor = DataPreprocessor()
    descriptors = data_preprocessor.descriptors

    print(text_to_features(
        "I want something very hoppy, slightly bitter, a little fruity, not sour",
        descriptors
    ))