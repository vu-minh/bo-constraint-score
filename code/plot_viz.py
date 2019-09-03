# plot metamap (meta tsne) for all vizs of a dataset

import os
import math
import joblib
from itertools import product
from collections import namedtuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from common.dataset import dataset
from run_viz import run_umap, run_tsne
from sklearn.preprocessing import StandardScaler
from MulticoreTSNE import MulticoreTSNE

from utils import get_scores


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


def _simple_scatter_with_colorbar(
    ax, Z, labels, title="", cmap_name="viridis", highlight_range=()
):
    cmap = cm.get_cmap(cmap_name, 10)
    scatter = ax.scatter(
        Z[:, 0], Z[:, 1], c=labels, alpha=0.8, cmap=cmap, s=80, edgecolor="black"
    )
    ax.text(
        x=0.5, y=-0.2, s=title, transform=ax.transAxes, va="bottom", ha="center", fontsize=18
    )
    ax.axis("off")
    plt.colorbar(scatter, ax=ax, orientation="horizontal")


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


def get_all_embeddings(embedding_dir, ignore_new_files=False):
    found = False
    if ignore_new_files:
        # if os.path.exists(f"{embedding_dir}/all_updated.z"):
        #     all_embeddings = joblib.load(f"{embedding_dir}/all_updated.z")
        #     found = True
        if os.path.exists(f"{embedding_dir}/all.z"):
            all_embeddings = joblib.load(f"{embedding_dir}/all.z")
            found = True

    if not found:
        print(f"Walk {embedding_dir}")
        all_embeddings = {}
        for dirname, _, filenames in os.walk(embedding_dir):
            for filename in filenames:
                if filename.startswith(("all", "meta")):
                    continue
                elif filename.endswith(".z"):
                    Z = joblib.load(os.path.join(dirname, filename))
                    all_embeddings[filename[:-2]] = Z
        joblib.dump(all_embeddings, f"{embedding_dir}/all_updated.z")
    return all_embeddings


def meta_tsne(X, meta_perplexity=30):
    return MulticoreTSNE(
        perplexity=meta_perplexity,
        n_iter=1500,
        n_jobs=-1,
        random_state=42,
        n_iter_without_progress=1500,
        min_grad_norm=1e-32,
    ).fit_transform(X)


def create_metamap(method_name, meta_perplexity, embedding_dir, ignore_new_files):
    all_embeddings = get_all_embeddings(embedding_dir, ignore_new_files)
    print("All embeddings: ", len(all_embeddings))

    # convert the dict of embeddings hashed by param into numpy array
    X = np.array(list(map(np.ravel, all_embeddings.values())))
    print(X.shape)
    Z = meta_tsne(X, meta_perplexity)

    # extract params values from key names and use them as labels
    if method_name == "umap":
        labels1, labels2 = zip(*[k.split("_") for k in all_embeddings.keys()])
    else:
        labels1, labels2 = list(all_embeddings.keys()), None
    labels1 = np.array(list(map(float, labels1)))
    if labels2 is not None:
        labels2 = np.array(list(map(lambda s: float(s) * 100, labels2)))
    res = {"Z": Z, "labels1": labels1, "labels2": labels2}
    joblib.dump(res, f"{embedding_dir}/metamap{meta_perplexity}.z")
    return res


def plot_metamap(
    dataset_name,
    method_name,
    plot_dir="",
    embeddinging_dir="",
    meta_perplexity=50,
    ignore_new_files=False,
):
    # run tsne to create metamap
    meta_name = f"{embedding_dir}/metamap{meta_perplexity}.z"
    if ignore_new_files and os.path.exists(meta_name):
        res = joblib.load(meta_name)
    else:
        res = create_metamap(
            method_name, meta_perplexity, embedding_dir, ignore_new_files=True
        )
    Z, labels1, labels2 = res.values()

    # plot metamap
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    scatter = ax.scatter(Z[:, 0], Z[:, 1], c=labels1, s=labels2, cmap="PuBu", alpha=0.8)
    fig.colorbar(scatter)
    fig.savefig(f"{plot_dir}/metamap_{meta_perplexity}.png")


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


def plot_metamap_with_scores_tsne(
    dataset_name,
    plot_dir,
    embedding_dir,
    score_dir,
    meta_perplexity=30,
    n_labels_each_class=10,
):
    ScoreConfig = namedtuple("ScoreConfig", ["score_name", "score_title", "score_cmap"])
    score_config = [
        ScoreConfig("bic", "BIC score", "Purples_r"),
        ScoreConfig("rnx", "$AUC_{log}RNX$", "Blues"),
        ScoreConfig("qij", "Constraint score", "Greens"),
        ScoreConfig("perplexity", "Perplexity", "Greys"),
    ]

    # if title == "BIC":  # need to find the min
    #     threshold = 1.0 + (1.0 - threshold)
    #     pivot = threshold * min(score_data)
    #     (best_indices,) = np.where(score_data < pivot)
    #     param_best = list_params[np.argmin(score_data)]
    # else:  # need to find the max
    #     pivot = threshold * score_data.max()
    #     (best_indices,) = np.where(score_data > pivot)
    #     param_best = list_params[np.argmax(score_data)]
    # param_min = list_params[best_indices.min()]
    # param_max = list_params[best_indices.max()]

    perp_values = []
    all_scores = []
    for config in score_config:
        if config.score_name != "perplexity":
            perp_values, scores = get_scores(config.score_name, score_dir)
        else:
            scores = perp_values
        all_scores.append(scores)

    all_embeddings = joblib.load(f"{embedding_dir}/all.z")
    X = [Z for key, Z in all_embeddings.items() if int(key) in perp_values]
    X = np.array(list(map(np.ravel, X)))
    print(X.shape)
    X = StandardScaler().fit_transform(X)
    Z = meta_tsne(X, meta_perplexity)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for config, scores, ax in zip(score_config, all_scores, axes.ravel()):
        _, score_title, score_cmap = config
        _simple_scatter_with_colorbar(
            ax, Z, labels=scores, title=score_title, cmap_name=score_cmap
        )

    fig.tight_layout()
    fig.savefig(f"{plot_dir}/metamap_scores_{meta_perplexity}.png")


if __name__ == "__main__":
    # plt.rcParams.update({"font.size": 20})
    import argparse
    import sys

    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset_name", default="")
    ap.add_argument("-m", "--method_name", default="umap", help="['tsne', 'umap', 'largevis']")
    ap.add_argument("-s", "--seed", default=42, type=int)
    ap.add_argument("--use_other_label", default=None)
    ap.add_argument("--plot_test_vis", action="store_true")
    ap.add_argument("--show_viz_grid", action="store_true")
    ap.add_argument("--plot_metamap", action="store_true")
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
    score_dir = f"./scores/{dataset_name}/{method_name}"

    if args.plot_test_vis:
        plot_test_vis(X, dataset_name, plot_dir, embedding_dir, labels, other_labels)
        sys.exit(0)

    if args.show_viz_grid:
        list_params = get_params_to_show(dataset_name, method_name)
        show_viz_grid(dataset_name, method_name, labels, plot_dir, embedding_dir, list_params)
        sys.exit(0)

    if args.plot_metamap:
        # plot_metamap(dataset_name, method_name, plot_dir, embedding_dir)
        plot_metamap_with_scores_tsne(dataset_name, plot_dir, embedding_dir, score_dir)
        sys.exit(0)
