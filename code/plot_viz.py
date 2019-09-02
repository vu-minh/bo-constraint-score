# plot metamap (meta tsne) for all vizs of a dataset

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


if __name__ == "__main__":
    import argparse
    import sys

    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset_name", default="")
    ap.add_argument("-m", "--method_name", default="umap", help="['tsne', 'umap', 'largevis']")
    ap.add_argument("-s", "--seed", default=42, type=int)
    ap.add_argument("--use_other_label", default=None)
    ap.add_argument("--plot_test_vis", action="store_true")
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
