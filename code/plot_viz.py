# plot metamap (meta tsne) for all vizs of a dataset

import math
import joblib

# import numpy as np
from itertools import product
import matplotlib.pyplot as plt
from common.dataset import dataset
from run_viz import run_umap, run_tsne


def plot_2_labels(Z, labels, other_labels, out_name):
    _, [ax0, ax1] = plt.subplots(1, 2, figsize=(12, 6))
    ax0.scatter(Z[:, 0], Z[:, 1], c=labels, cmap="Spectral", alpha=0.5)
    ax1.scatter(Z[:, 0], Z[:, 1], c=other_labels, cmap="Spectral", alpha=0.5)
    plt.savefig(out_name)
    plt.close()


def plot_test_vis(
    X, dataset_name, plot_dir="", embedding_dir="", labels=None, other_labels=None, debug=False
):
    other_labels, des = dataset.load_additional_labels(dataset_name, label_name="class_matcat")
    print(des)

    list_min_dist = [0.001, 0.01, 0.1, 0.5, 1.0]
    list_n_neighbors = [5, 10, 15, 20, 30, 50, 100]

    if debug:
        list_params = [(0.01, 45), (0.17, 18)]
    else:
        list_params = product(list_min_dist, list_n_neighbors)

    for min_dist, n_neighbors in list_params:
        Z = run_umap(
            X,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            seed=42,
            embedding_dir=embedding_dir,
            check_log=True,
        )
        out_name = ""  # f"{plot_dir}/test_umap{n_neighbors}_{min_dist:.4f}.png"
        plot_2_labels(Z, labels, other_labels, out_name)

    for perp in [10, 15, 30, 60, 150]:  # list_n_neighbors:
        Z = run_tsne(X, perp, seed=42, check_log=True, embedding_dir=embedding_dir)
        out_name = f"{plot_dir}/test_tsne{perp}.png"
        plot_2_labels(Z, labels, None, out_name)


def _simple_scatter(ax, Z, labels=None, title="", axis_off=True):
    ax.scatter(Z[:, 0], Z[:, 1], c=labels, alpha=0.7, cmap="Spectral", s=6)
    ax.set_title(title)
    if axis_off:
        ax.axis("off")


def show_viz_grid(
    dataset_name, method_name, labels=None, plot_dir="", embedding_dir="", list_params=[]
):
    n_viz = len(list_params)
    n_rows, n_cols = math.ceil(n_viz / 3), 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))

    for i, (params, ax) in enumerate(zip(list_params, axes.ravel())):
        if method_name == "umap":
            param_key = f"{params[0]}_{params[1]:.4f}"
            param_explanation = f"n_neighbors={params[0]}, min_dist={params[1]}"
        else:
            param_key = str(params[0])
            param_explanation = f"perplexity={params[0]}"
        comment = f"{params[-1]}, {param_explanation}"
        print(i, comment)
        Z = joblib.load(f"{embedding_dir}/{param_key}.z")
        _simple_scatter(ax, Z, labels, title=comment)

    fig.tight_layout()
    fig.savefig(f"{plot_dir}/show.png")
    plt.close()


def get_params_to_show(dataset_name, method_name):
    return {
        "20NEWS5": {
            "umap": [
                (15, 0.2154, "++RNX"),
                (134, 0.001, "++qij"),
                (147, 0.01, "predict"),
                (126, 0.1, "default"),
                (86, 0.0464, "+qij, =RNX"),
                (7, 0.1, "++RNX, --qij"),
            ],
            "tsne": {
                (114, "++qij, +bic, -rnx"),
                (89, "+qij, +bic, =rnx"),
                (250, "+qij, =bic, -rnx"),
                (25, "++rnx, -qij, =bic"),
                (11, "+rnx, --qij, =bic"),
                (156, "+qij --rnx, ++bic"),
            },
        },
        "DIGITS": {
            "tsne": [
                (50, "++qij, ++bic, predict"),
                (14, "++rnx, -qij, -bic"),
                (22, "+qij, +rnx, -bic"),
                (76, "+qij, -rnx, +bic"),
                (5, "--qij, --rnx, --bic"),
                (270, "--qij, --rnx, --bic"),
            ],
            "umap": {
                (5, 0.001, "++rnx, -qij"),
                (11, 0.01, "++qij, +rnx, predict"),
                (401, 0.0464, "--qij, +rnx"),
            },
        },
        "COIL20": {
            "tsne": [(36, "++qij, +rnx +bic, predict"), (5, "---"), (142, "+rnx, --")],
            "umap": [
                (4, 0.4642, "++rnx, --qij"),
                (9, 0.0022, "++qij, =rnx"),
                (13, 0.001, "predict, +qij, =rnx"),
                (150, 0.01, "--qij, =rnx"),
                (20, 0.0464, "=qij, =rnx"),
                (3, 0.0179, "---"),
            ],
        },
        "NEURON_1K": {
            "tsne": [
                (72, "++qij, -rnx, ++bic, prediction"),
                # (13, "++rnx , --qij, -bic"),
                # (40, "+rnx, +qij, +bic"),
                (106, "+qij, +bic, -rnx"),
                (150, "---"),
            ],
            "umap": [
                (16, 0.001, "prediction, ++qij, +rnx"),
                (5, 0.1, "++rnx, +qij"),
                (150, 0.0464, "-qij, -rnx"),
            ],
        },
        "FASHION_MOBILENET": {
            "tsne": [
                (82, "++qij, ++bic, +rnx, prediction"),
                (12, "--qij, +rnx, --bic"),
                (164, "+qij, +bic, --rnx"),
            ],
            "umap": [
                (19, 0.0022, "prediction, +"),
                (9, 0.0046, "++rnx, +qij"),
                (310, 0.1, "="),
            ],
        },
    }[dataset_name][method_name]


if __name__ == "__main__":
    import argparse
    import sys

    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset_name", default="")
    ap.add_argument("-m", "--method_name", default="umap", help="['tsne', 'umap', 'largevis']")
    ap.add_argument("-s", "--seed", default=42, type=int)
    ap.add_argument("--use_other_label", default=None)
    ap.add_argument("--plot_test_vis", action="store_true")
    ap.add_argument("--show_viz_grid", action="store_true")
    args = ap.parse_args()

    dataset.set_data_home("./data")
    dataset_name = args.dataset_name
    method_name = args.method_name

    X_origin, X, labels = dataset.load_dataset(dataset_name, preprocessing_method="auto")

    other_label_name = args.use_other_label
    if other_label_name is not None:
        other_labels, des = dataset.load_additional_labels(dataset_name, other_label_name)
        if labels is None:
            raise ValueError("Fail to load additional labels: " + des)
        print("Using additional labels: ", other_label_name)
    else:
        other_labels = None

    embedding_dir = f"./embeddings/{dataset_name}/{method_name}"
    plot_dir = f"./plots/{dataset_name}/{method_name}"

    if args.plot_test_vis:
        plot_test_vis(X, dataset_name, plot_dir, embedding_dir, labels, other_labels)
        sys.exit(0)

    if args.show_viz_grid:
        list_params = get_params_to_show(dataset_name, method_name)
        show_viz_grid(dataset_name, method_name, labels, plot_dir, embedding_dir, list_params)
        sys.exit(0)
