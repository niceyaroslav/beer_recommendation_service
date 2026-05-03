from pathlib import Path

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from data_preprocessor import DataPreprocessor
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import numpy as np


data_preprocessor = DataPreprocessor()
scaler = StandardScaler()

df_clean = data_preprocessor.df
X = df_clean[data_preprocessor.feature_cols].copy()
X_scaled = scaler.fit_transform(X)
pca_2d = PCA(n_components=2)
X_pca = pca_2d.fit_transform(X_scaled)


def _train_kmeans_and_get_clusters():
    kmeans = KMeans(n_clusters=8, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    df_clean["cluster"] = clusters

    return kmeans, clusters, df_clean


def _get_dominant_beer_by_styles(df_clean):
    dominant_style_per_cluster = (
        df_clean
        .groupby("cluster")["style"]
        .value_counts(normalize=True)
        .rename("share")
        .reset_index()
        .sort_values(["cluster", "share"], ascending=[True, False])
        .groupby("cluster")
        .head(1)
    )

    dominant_style_per_cluster["share"] = (
            dominant_style_per_cluster["share"] * 100
    ).round(1)

    return dominant_style_per_cluster


def _save_or_show(fig, save_path=None, show=False, dpi=300):
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def ibu_and_abv_distribution(
    save_path='plots/ibu_and_abv_distribution.png',
    show=False,
    figsize=(12, 5),
    dpi=300
):
    """
    ABV is the alcohol content by volume.
    IBU is the International Bitterness Unit.
    """

    fig, ax = plt.subplots(1, 2, figsize=figsize)

    sns.histplot(df_clean["abv"], bins=30, ax=ax[0])
    ax[0].set_title("Distribution of ABV")
    ax[0].set_xlabel("ABV")
    ax[0].set_ylabel("Count")

    sns.histplot(df_clean["max ibu"], bins=30, ax=ax[1])
    ax[1].set_title("Distribution of Max IBU")
    ax[1].set_xlabel("Max IBU")
    ax[1].set_ylabel("Count")

    fig.tight_layout()

    _save_or_show(fig, save_path, show, dpi)

    return fig, ax


def taste_profile_distribution(
    save_path='plots/taste_profile_distribution.png',
    show=False,
    figsize=(8, 5),
    dpi=300
):
    """
    Taste profile distributions.
    """
    taste_cols = ["bitter", "sweet", "sour", "salty"]

    taste_scaled = df_clean[taste_cols].mean()
    taste_scaled = taste_scaled / taste_scaled.sum()

    fig, ax = plt.subplots(figsize=figsize)

    taste_scaled.plot(kind="bar", ax=ax)

    ax.set_title("Relative Contribution of Taste Features")
    ax.set_ylabel("Proportion")
    ax.set_xlabel("Taste Feature")
    ax.tick_params(axis="x", rotation=45)

    fig.tight_layout()

    _save_or_show(fig, save_path, show, dpi)

    return fig, ax


def flavor_profile_comparison(
    save_path='plots/flavor_profile_comparison.png',
    show=False,
    figsize=(8, 5),
    dpi=300
):
    """
    Flavor profile comparison.
    """
    flavor_cols = ["fruits", "hoppy", "spices", "malty"]

    fig, ax = plt.subplots(figsize=figsize)

    df_clean[flavor_cols].mean().plot(kind="bar", ax=ax)

    ax.set_title("Average Flavor Profile")
    ax.set_ylabel("Average Intensity")
    ax.set_xlabel("Aroma Feature")
    ax.tick_params(axis="x", rotation=45)

    fig.tight_layout()

    _save_or_show(fig, save_path, show, dpi)

    return fig, ax


def correlation_matrix_of_taste_descriptors(
    save_path='plots/correlation_matrix_of_taste_descriptors.png',
    show=False,
    figsize=(12, 8),
    dpi=300
):
    """
    Correlation matrix of taste descriptors.
    """

    corr_cols = [
        "astringency", "body", "alcohol",
        "bitter", "sweet", "sour", "salty",
        "fruits", "hoppy", "spices", "malty"
    ]

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        df_clean[corr_cols].corr(),
        cmap="coolwarm",
        center=0,
        annot=False,
        ax=ax
    )

    ax.set_title("Correlation Between Beer Features")

    fig.tight_layout()

    _save_or_show(fig, save_path, show, dpi)

    return fig, ax


def rating_distribution_vs_number_of_reviews(
    save_path='plots/rating_distribution_vs_number_of_reviews.png',
    show=False,
    figsize=(8, 5),
    dpi=300
):
    """
    Rating distribution: quality vs popularity.
    """
    df_clean = data_preprocessor.df

    fig, ax = plt.subplots(figsize=figsize)

    sns.scatterplot(
        x="number of reviews",
        y="review overall",
        data=df_clean,
        alpha=0.5,
        ax=ax
    )

    ax.set_xscale("log")
    ax.set_title("Rating vs Number of Reviews")
    ax.set_xlabel("Number of Reviews")
    ax.set_ylabel("Review Overall")

    fig.tight_layout()

    _save_or_show(fig, save_path, show, dpi)

    return fig, ax


def selection_of_clustering_basis_by_silhouette_score(
    save_path='plots/selection_of_clustering_basis_by_silhouette_score.png',
    show=False,
    figsize=(8, 5),
    dpi=300
):
    """
    Selection of clustering basis by silhouette score.
    """

    scores = []
    k_range = range(2, 15)
    df_scaled = data_preprocessor.scaled_df

    for n in k_range:
        calculated_k_means = KMeans(
            n_clusters=n,
            random_state=42,
            n_init=10
        )
        calculated_clusters = calculated_k_means.fit_predict(df_scaled)
        score = silhouette_score(df_scaled, calculated_clusters)
        scores.append(score)

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(k_range, scores, marker="o")
    ax.set_title("Silhouette Score vs Number of Clusters (k)")
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Silhouette Score")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    _save_or_show(fig, save_path, show, dpi)

    return fig, ax


def pca_component_amount_vs_explained_variance(
        save_path='plots/pca_component_amount_vs_explained_variance.png',
        show=False,
        figsize=(8, 5),
        dpi=300
):
    """
    Selection of clustering basis by silhouette score
    """

    pca_full = PCA().fit(X_scaled)
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(np.cumsum(pca_full.explained_variance_ratio_))
    ax.set_xlabel("Number of Components")
    ax.set_ylabel("Cumulative Explained Variance")
    ax.set_title("PCA Explained Variance")
    fig.tight_layout()

    _save_or_show(fig, save_path, show, dpi)

    return fig, ax


def pca_plot_colored_by_clusters(
    save_path='plots/pca_plot_colored_by_clusters.png',
    show=True,
    figsize=(10, 6),
    dpi=300
):
    """
    PCA plot colored by KMeans clusters.
    Cluster points have no legend; centroids are labeled by dominant style.
    """
    kmeans, clusters, df_clean = _train_kmeans_and_get_clusters()

    centroids_pca = pca_2d.transform(kmeans.cluster_centers_)

    dominant_style_per_cluster = _get_dominant_beer_by_styles(df_clean)

    dominant_style_map = dict(
        zip(
            dominant_style_per_cluster["cluster"],
            dominant_style_per_cluster["style"]
        )
    )

    fig, ax = plt.subplots(figsize=figsize)

    palette = sns.color_palette("tab10", n_colors=kmeans.n_clusters)

    sns.scatterplot(
        x=X_pca[:, 0],
        y=X_pca[:, 1],
        hue=df_clean["cluster"],
        palette=palette,
        alpha=0.45,
        s=35,
        legend=False,
        ax=ax
    )

    for cluster_id in range(kmeans.n_clusters):
        label = dominant_style_map.get(cluster_id, f"Cluster {cluster_id}")

        ax.scatter(
            centroids_pca[cluster_id, 0],
            centroids_pca[cluster_id, 1],
            marker="X",
            s=260,
            color=palette[cluster_id],
            edgecolor="black",
            linewidth=1.2,
            label=f"{cluster_id}: {label}"
        )

    ax.set_title("KMeans Clusters with Dominant Style Centroids")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")

    ax.legend(
        title="Dominant Style by Cluster",
        bbox_to_anchor=(1.05, 1),
        loc="upper left"
    )

    fig.tight_layout()

    _save_or_show(fig, save_path, show, dpi)

    return fig, ax


def dominant_style_by_cluster(
    save_path='plots/dominant_style_by_cluster.png',
    show=True,
    figsize=(10, 5),
    dpi=300
):
    """
    Plot the most dominant beer style in each cluster.
    """
    kmeans, clusters, df_clean = _train_kmeans_and_get_clusters()

    dominant_style_per_cluster = _get_dominant_beer_by_styles(df_clean)

    fig, ax = plt.subplots(figsize=figsize)

    sns.barplot(
        data=dominant_style_per_cluster,
        x="cluster",
        y="share",
        hue="style",
        ax=ax
    )

    ax.set_title("Dominant Beer Style per Cluster")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Share within Cluster (%)")

    ax.legend(
        title="Dominant Style",
        bbox_to_anchor=(0.5, -0.25),
        loc="upper center",
        ncol=4
    )

    fig.tight_layout()

    _save_or_show(fig, save_path, show, dpi)

    return fig, ax, dominant_style_per_cluster