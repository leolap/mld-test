import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


def normalize_data(embeddings: np.ndarray) -> np.ndarray:
    scaler = StandardScaler()
    return scaler.fit_transform(embeddings)


def perform_tsne(
    embeddings: np.ndarray,
    *,
    n_components: int = 2,
    random_state: int = 42,
    perplexity: int = 30,
    n_iter: int = 1000,
) -> np.ndarray:
    tsne = TSNE(
        n_components=n_components,
        random_state=random_state,
        perplexity=perplexity,
        n_iter=n_iter,
        early_exaggeration=4,
    )
    return tsne.fit_transform(embeddings)


def generate_tsne_plot(syndrome_ids: np.ndarray, embeddings_tsne: np.ndarray):
    plt.figure(figsize=(10, 8))

    for i, syndrome_id in enumerate(np.unique(syndrome_ids)):
        mask = syndrome_ids == syndrome_id
        plt.scatter(
            embeddings_tsne[mask, 0], embeddings_tsne[mask, 1], label=f"Syndrome {i}"
        )
    plt.title("tSNE of the inputs")
    plt.legend()
    plt.show()


def generate_tsne_plot_3d(syndrome_ids: np.ndarray, embeddings_tsne: np.ndarray):
    num_labels = len(np.unique(syndrome_ids))
    colors = cm.rainbow(np.linspace(0, 1, num_labels))

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    for label, color in zip(np.unique(syndrome_ids), colors):
        mask = syndrome_ids == label
        ax.scatter(
            embeddings_tsne[mask, 0],
            embeddings_tsne[mask, 1],
            embeddings_tsne[mask, 2],
            color=color.reshape(1, -1),
            label=f"Syndrome {label}",
        )

    ax.set_title("tSNE of the inputs (3D)")

    custom_legend = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=c,
            markersize=10,
            label=f"Syndrome {i}",
        )
        for i, c in enumerate(colors)
    ]
    ax.legend(handles=custom_legend, title="Syndrome IDs", loc="upper right")

    plt.show()
