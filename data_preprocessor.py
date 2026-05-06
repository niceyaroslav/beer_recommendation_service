import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np


class DataPreprocessor:

    feature_cols = [
        "abv", "min ibu", "max ibu",
        "astringency", "body", "alcohol",
        "bitter", "sweet", "sour", "salty",
        "fruits", "hoppy", "spices", "malty"
    ]

    rating_cols = [
        "review aroma", "review appearance", "review palate",
        "review taste", "review overall", "number of reviews"
    ]

    def __init__(self):
        self.main_dataset_path = 'data/beer_profile_and_ratings.csv'
        self.descriptors_path = 'data/Beer Descriptors Simplified.xlsx'
        self.df = self.import_main_dataset()
        self.descriptors = self.import_and_process_descriptors()
        self.descriptors_lookup = self.return_descriptors_lookup()
        self.scaled_df = self.scale_df()

    def import_main_dataset(self):
        df = pd.read_csv(self.main_dataset_path)
        df.columns = (
            df.columns
            .str.strip()
            .str.lower()
            .str.replace(" ", "_")
            .str.replace(r"[()]", "", regex=True)
        )
        df = self.clean_up_main_df(df)
        return df

    @staticmethod
    def reshape_descriptor_sheet(df):
        descriptors = []

        cols = df.columns.tolist()

        # iterate over column pairs (word column + impact column)
        for i in range(0, len(cols), 2):
            category = cols[i]  # e.g. "Astringent", "Bitter", etc.
            words = df[cols[i]].dropna()

            for word in words:
                descriptors.append({
                    "word": str(word).lower().strip(),
                    "category": category.strip()
                })

        return pd.DataFrame(descriptors)

    def import_and_process_descriptors(self):
        mouthfeel = pd.read_excel(self.descriptors_path, sheet_name='Mouthfeel', engine='openpyxl')
        taste = pd.read_excel(self.descriptors_path, sheet_name='Taste', engine='openpyxl')
        flavor = pd.read_excel(self.descriptors_path, sheet_name='Flavor And Aroma', engine='openpyxl')

        df_mouthfeel = self.reshape_descriptor_sheet(mouthfeel)
        df_taste = self.reshape_descriptor_sheet(taste)
        df_flavor = self.reshape_descriptor_sheet(flavor)

        descriptors = pd.concat(
            [df_mouthfeel, df_taste, df_flavor],
            ignore_index=True
        )
        descriptors = descriptors.drop_duplicates()

        descriptors["word"] = descriptors["word"].str.lower().str.strip()
        descriptors["category"] = descriptors["category"].str.lower().str.strip()
        return descriptors

    @classmethod
    def clean_up_main_df(cls, df):
        df_clean = df.copy()

        # Standardize column names
        df_clean.columns = (
            df_clean.columns
            .str.strip()
            .str.replace("_", " ")
            .str.replace(r"[()]", "", regex=True)
        )

        text_cols = ["name", "style", "brewery", "beer name full", "description"]
        for col in text_cols:
            df_clean[col] = df_clean[col].astype(str).str.strip()

        # Remove "Notes:" prefix from description
        df_clean["description"] = (
            df_clean["description"]
            .str.replace("Notes:", "", regex=False)
            .str.strip()
        )

        # Replace empty descriptions with NaN
        df_clean["description"] = df_clean["description"].replace("", np.nan)

        df_clean = cls.fill_na_on_main_df(df_clean)
        return df_clean

    @classmethod
    def fill_na_on_main_df(cls, df_clean):

        id_cols = [
            "name", "style", "brewery", "beer name full", "description"
        ]

        df_clean[cls.rating_cols] = df_clean[cls.rating_cols].fillna(
            df_clean[cls.rating_cols].median()
        )

        df_clean[cls.feature_cols] = df_clean[cls.feature_cols].fillna(
            df_clean[cls.feature_cols].median()
        )
        return df_clean

    @staticmethod
    def align_descriptors_with_categories(descriptors):
        category_alignment = {
            "astringent": "astringency",
            "alcoholic": "alcohol",
            "fruity": "fruits",
            "body": "body",
            "bitter": "bitter",
            "sweet": "sweet",
            "sour": "sour",
            "salty": "salty",
            "hoppy": "hoppy",
            "spices": "spices",
            "malty": "malty"
        }

        descriptors["category"] = (
            descriptors["category"]
            .astype(str)
            .str.strip()
            .map(category_alignment)
        )

        # Remove rows where category did not map correctly
        descriptors = descriptors.dropna(subset=["category"])

        # Clean words
        descriptors["word"] = (
            descriptors["word"]
            .astype(str)
            .str.lower()
            .str.strip()
        )

        descriptors = descriptors.drop_duplicates()
        return descriptors

    def return_descriptors_lookup(self):

        descriptor_lookup = (
            self.descriptors
            .drop_duplicates("word")  # keep first occurrence only
            .set_index("word")["category"]
            .to_dict()
        )

        return descriptor_lookup

    def scale_df(self):
        scaler = StandardScaler()
        x = self.df[self.feature_cols].copy()
        x_scaled = scaler.fit_transform(x)

        df_scaled = pd.DataFrame(
            x_scaled,
            columns=self.feature_cols,
            index=self.df.index
        )

        return df_scaled


if __name__ == '__main__':
    data_class = DataPreprocessor()