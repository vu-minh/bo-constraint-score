# plot metamap (meta tsne) for all vizs of a dataset

import re
import math
import joblib
from itertools import product
from collections import namedtuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# from matplotlib import rc
from matplotlib import cm
from matplotlib import ticker
from common.dataset import dataset
from run_viz import run_umap, run_tsne
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from MulticoreTSNE import MulticoreTSNE
from umap import UMAP

from utils import change_border
from utils import get_scores_tsne, get_config_labels_for_score_flexibility


def plot_2_labels(Z, labels, other_labels, out_name, title1="", title2=""):
    _, [ax0, ax1] = plt.subplots(1, 2, figsize=(12, 6))
    ax0.set_title(title1)
    ax0.scatter(Z[:, 0], Z[:, 1], c=labels, cmap="Spectral", alpha=0.5)
    ax1.set_title(title2)
    ax1.scatter(Z[:, 0], Z[:, 1], c=other_labels, cmap="Spectral", alpha=0.5)
    plt.savefig(out_name)
    plt.close()


def plot_test_vis(
    X, dataset_name, plot_dir="", embedding_dir="", labels=None, other_labels=None, debug=False,
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


def _simple_scatter(ax, Z, labels=None, title="", comment="", axis_off=False):
    ax.scatter(Z[:, 0], Z[:, 1], c=labels, alpha=0.7, cmap="Spectral", s=6)
    ax.set_title(title, loc="center")
    ax.text(
        x=1.0, y=0.0, s=comment, transform=ax.transAxes, ha="right", va="bottom", fontsize=14,
    )
    if axis_off:
        ax.axis("off")
    else:
        change_border(ax, width=0.25, color="black")


def _simple_scatter_with_colorbar(
    ax,
    Z,
    labels,
    title="",
    cmap_name="viridis",
    best_indices=None,
    best_idx=None,
    debug_label=False,
):
    ax.axis("off")
    plot_for_score_values = best_indices is not None
    marker_size = 80
    c_min, c_max = labels.min(), labels.max()
    norm = plt.Normalize(c_min, c_max)
    cmap = cm.get_cmap(cmap_name, 20)

    if best_indices is not None:
        ax.scatter(
            Z[best_indices][:, 0],
            Z[best_indices][:, 1],
            c=labels[best_indices],
            cmap=cmap,
            edgecolor="orange",
            marker="s",
            s=marker_size + 30,
            linewidths=2,
            zorder=99,
            alpha=0.8,
            norm=norm,
        )

    if best_idx is not None:
        ax.scatter(*Z[best_idx], c="red", marker="X", s=2 * marker_size, zorder=100)

    # should custom colorbar for metaplot colored by perplexity values
    scatter = ax.scatter(
        Z[:, 0],
        Z[:, 1],
        c=labels,
        alpha=0.5 if plot_for_score_values else 0.8,
        cmap=cmap,
        s=marker_size,
        edgecolor="black",
        norm=norm,
    )
    ax.text(
        x=0.5, y=-0.2, s=title, transform=ax.transAxes, va="bottom", ha="center", fontsize=18,
    )

    cb = plt.colorbar(scatter, ax=ax, orientation="horizontal")
    if best_indices is None:
        nbins = math.floor(max(labels))
        print("nbins: ", nbins)
        tick_locator = ticker.MaxNLocator(nbins=nbins)
        cb.locator = tick_locator
        cb.update_ticks()
        cb.ax.set_xticklabels([math.ceil(math.exp(i)) for i in range(nbins + 1)])

    if debug_label and not plot_for_score_values:
        # debug show param in the metmap
        for s, (x, y) in zip(labels, Z):
            # ax.text(x=x, y=y, s=str(int(math.exp(s))), fontsize=10)
            pass


def _scatter_with_colorbar_and_legend_size(
    ax,
    Z,
    labels,
    sizes=None,
    title="",
    cmap_name="viridis",
    best_indices=None,
    best_idx=None,
    show_legend=True,
    debug_label=False,
):
    ax.axis("off")
    plot_for_score_values = best_indices is not None
    c_min, c_max = labels.min(), labels.max()
    norm = plt.Normalize(c_min, c_max)
    cmap = cm.get_cmap(cmap_name, 20)

    if best_indices is not None:
        Z_highlight = Z[best_indices]
        ax.scatter(
            Z_highlight[:, 0],
            Z_highlight[:, 1],
            s=sizes[best_indices],
            c=labels[best_indices],
            cmap=cmap,
            edgecolor="orange",
            marker="s",
            zorder=99,
            norm=norm,
            alpha=0.9,
        )
    if best_idx is not None:
        ax.scatter(*Z[best_idx], c="red", marker="X", s=2 * sizes[best_idx], zorder=100)

    scatter = ax.scatter(
        Z[:, 0],
        Z[:, 1],
        c=labels,
        s=sizes,
        alpha=0.35 if plot_for_score_values else 0.8,
        cmap=cmap,
        norm=norm,
        edgecolor="black",
    )
    ax.text(
        x=0.5, y=-0.2, s=title, transform=ax.transAxes, va="bottom", ha="center", fontsize=18,
    )
    cb = plt.colorbar(scatter, ax=ax, orientation="horizontal")

    if not plot_for_score_values:  # plot metamap by n_neighbors and min_dist values
        # create "free" legend showing size of points by min_dist values
        min_dist_vals = [1e-3, 0.1, 0.5, 1.0]
        min_dist_shows = MinMaxScaler((30, 90)).fit_transform(
            np.array(min_dist_vals).reshape(-1, 1)
        )

        for min_dist_val, min_dist_show in zip(min_dist_vals, min_dist_shows.ravel()):
            ax.scatter([], [], c="k", alpha=0.5, s=min_dist_show, label=str(min_dist_val))
        ax.legend(
            scatterpoints=1,
            frameon=False,
            labelspacing=0.25,
            title="min_dist",
            loc="lower center",
            bbox_to_anchor=(0.5, -0.15),
            ncol=len(min_dist_vals),
            fontsize=18,
            title_fontsize=18,
        )

        # should custom colorbar to show n_neighbors in log scale
        nbins = math.floor(max(labels))
        tick_locator = ticker.MaxNLocator(nbins=nbins)
        cb.locator = tick_locator
        cb.update_ticks()
        cb.ax.set_xticklabels([math.ceil(math.exp(i)) for i in range(nbins + 1)])

    if debug_label and not plot_for_score_values:
        # debug show param in the metmap
        for i, (p1, p2, (x, y)) in enumerate(zip(labels, sizes, Z)):
            if i % 72 == 0:
                # ax.text(x=x, y=y, s=str(i), fontsize=15, color="blue")
                pass
        # test_idx = [170]
        # print(labels[test_idx], sizes[test_idx])


def show_viz_grid(
    dataset_name, method_name, labels=None, plot_dir="", embedding_dir="", list_params=[],
):
    n_viz = len(list_params)
    n_rows, n_cols = math.ceil(n_viz / 4), 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4.75))

    for i, (params, ax) in enumerate(zip(list_params, axes.ravel())):
        if method_name == "umap":
            param_key = f"{params[0]}_{params[1]:.4f}"
            param_explanation = f"n_neighbors={params[0]}, min_dist={params[1]}"
        else:
            param_key = str(params[0])
            param_explanation = f"perplexity={params[0]}"
        comment = f"{params[-1]}"
        print(i, "\n", comment, "\n", param_explanation)
        Z = joblib.load(f"{embedding_dir}/{param_key}.z")
        _simple_scatter(ax, Z, labels, title=param_explanation, comment=comment)

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.05, left=0.01, right=0.99)
    fig.savefig(f"{plot_dir}/show.png")
    plt.close()


def meta_tsne(X, meta_perplexity=30, cache=False, embedding_dir=""):
    if cache:
        Z = joblib.load(f"{embedding_dir}/metamap{meta_perplexity}.z")
    else:
        Z = MulticoreTSNE(
            perplexity=meta_perplexity,
            n_iter=1500,
            n_jobs=-1,
            random_state=42,
            n_iter_without_progress=1500,
            min_grad_norm=1e-32,
        ).fit_transform(X)
        joblib.dump(Z, f"{embedding_dir}/metamap{meta_perplexity}.z")
    return Z


def meta_umap(X, meta_n_neighbors=15, cache=False, embedding_dir=""):
    if cache:
        Z = joblib.load(f"{embedding_dir}/metamap{meta_n_neighbors}.z")
    else:
        Z = UMAP(n_neighbors=meta_n_neighbors, min_dist=1.0, random_state=30).fit_transform(X)
        joblib.dump(Z, f"{embedding_dir}/metamap{meta_n_neighbors}.z")
    return Z


def plot_metamap_with_scores_tsne(
    dataset_name,
    plot_dir,
    embedding_dir,
    score_dir,
    meta_n_neighbors=50,
    n_labels_each_class=10,
    threshold=0.96,
    use_cache=False,
):
    ScoreConfig = namedtuple("ScoreConfig", ["score_name", "score_title", "score_cmap"])
    score_config = [
        ScoreConfig("qij", "Constraint score", "Greens"),
        ScoreConfig("bic", "BIC score", "Purples_r"),
        ScoreConfig("rnx", "$AUC_{log}RNX$", "Blues"),
        ScoreConfig("perplexity", "Perplexity in log-scale", "bone"),
    ]  # perplexity should be in the last of this list, since we have to get list_params first

    perp_values = []
    all_scores = []
    for config in score_config:
        if config.score_name != "perplexity":
            perp_values, scores = get_scores_tsne(config.score_name, score_dir)
            perp_values = np.array(perp_values)
        else:
            scores = np.log(perp_values)
        all_scores.append(np.array(scores))

    all_embeddings = joblib.load(f"{embedding_dir}/all.z")
    X = [Z for key, Z in all_embeddings.items() if int(key) in perp_values]
    X = np.array(list(map(np.ravel, X)))
    print("Metamap input data: ", X.shape)
    X = StandardScaler().fit_transform(X)
    Z = meta_umap(X, meta_n_neighbors, cache=use_cache, embedding_dir=embedding_dir)

    fig, [ax0, ax1, ax2, ax3] = plt.subplots(1, 4, figsize=(20, 6))
    # note: roll axes to make subfigure for perplexity being moved from last to first
    for config, scores, ax in zip(score_config, all_scores, [ax1, ax2, ax3, ax0]):
        score_name, score_title, score_cmap = config

        if score_name == "perplexity":
            best_indices, best_idx = None, None
        else:
            if score_name == "bic":
                pivot = (1.0 + (1.0 - threshold)) * min(scores)
                (best_indices,) = np.where(scores < pivot)
                best_idx = np.argmin(scores)
            else:
                pivot = threshold * scores.max()
                (best_indices,) = np.where(scores > pivot)
                best_idx = np.argmax(scores)

        _simple_scatter_with_colorbar(
            ax,
            Z,
            labels=scores,
            title=score_title,
            cmap_name=score_cmap,
            best_indices=best_indices,
            best_idx=best_idx,
            debug_label=use_cache,
        )

    # ax0 show metamap colored by perplexity values.
    # now show a list of selected perplexities.
    list_selected_params = get_params_to_show(dataset_name, method_name="tsne")
    list_annotations = []
    for param in list_selected_params:
        perplexity = param[0]
        if perplexity in perp_values:
            pos = Z[np.where(perp_values == perplexity)]
            list_annotations.append((perplexity, *pos[0]))
    annotate_selected_params_tsne(ax0, list_annotations)

    fig.tight_layout()
    plt.subplots_adjust(wspace=0.075, bottom=-0.05, top=1.0, left=0.01)
    fig.savefig(f"{plot_dir}/metamap_scores_{meta_n_neighbors}.png", dpi=300)


def plot_metamap_with_scores_umap(
    dataset_name,
    plot_dir,
    embedding_dir,
    score_dir,
    meta_n_neighbors=100,
    n_labels_each_class=10,
    threshold=0.96,
    use_cache=False,
):
    df_qij = pd.read_csv(f"{score_dir}/qij/umap_scores.csv")
    qij_scores = df_qij["qij_score"].to_numpy()
    n_params = df_qij.shape[0]
    print("Number of qij scores: ", n_params)
    list_n_neighbors = df_qij["n_neighbors"].to_numpy()
    list_min_dist = df_qij["min_dist"].to_numpy()

    df_metrics = pd.read_csv(f"{score_dir}/qij/umap_metrics.csv")
    print("Number of metric scores: ", df_metrics.shape)
    assert n_params == df_metrics.shape[0]
    rnx_scores = df_metrics["auc_rnx"].to_numpy()

    all_embeddings = joblib.load(f"{embedding_dir}/all.z")
    print("Number of embeddings: ", len(all_embeddings))
    assert n_params == len(all_embeddings)

    X = np.array(list(map(np.ravel, all_embeddings.values())))
    print("Metamap input data: ", X.shape)
    X = StandardScaler().fit_transform(X)
    Z = meta_umap(X, meta_n_neighbors, cache=use_cache, embedding_dir=embedding_dir)

    ScoreConfig = namedtuple(
        "ScoreConfig", ["score_name", "score_title", "score_cmap", "score_values"]
    )
    score_config = [
        ScoreConfig(
            "n_neighbors", "n_neighbors in log-scale", "bone", np.log(list_n_neighbors)
        ),
        ScoreConfig("qij", "Constraint score in log-scale", "Greens", qij_scores),
        ScoreConfig("rnx", "$AUC_{log}RNX$ in log-scale", "Blues", rnx_scores),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(21, 9))
    for config, ax in zip(score_config, axes.ravel()):
        score_name, score_title, score_cmap, score_values = config

        if score_name == "n_neighbors":
            best_indices, best_idx = None, None
            show_legend = True
        else:
            pivot = threshold * score_values.max()
            (best_indices,) = np.where(score_values > pivot)
            best_idx = np.argmax(score_values)
            show_legend = False

        _scatter_with_colorbar_and_legend_size(
            ax,
            Z,
            labels=score_values,
            sizes=MinMaxScaler((30, 90)).fit_transform(list_min_dist.reshape(-1, 1)),
            title=score_title,
            cmap_name=score_cmap,
            best_indices=best_indices,
            best_idx=best_idx,
            show_legend=show_legend,
            debug_label=use_cache,
        )

    # ax0 show metamap colored by perplexity values.
    # now show a list of selected perplexities.
    list_selected_params = get_params_to_show(dataset_name, method_name="umap")
    list_annotations = []
    all_keys = list(all_embeddings.keys())
    for param in list_selected_params:
        n_neighbors, min_dist = param[0], param[1]
        key = f"{n_neighbors}_{min_dist:.4f}"
        if key in all_keys:
            pos = Z[all_keys.index(key)]
            list_annotations.append((n_neighbors, min_dist, *pos))
        else:
            print("Debug: params to show but not found: ", key)
    annotate_selected_params_umap(axes[0], list_annotations)

    fig.tight_layout()
    plt.subplots_adjust(bottom=-0.075, top=1.0, left=0.01, right=0.99, wspace=0.05)
    fig.savefig(f"{plot_dir}/metamap_scores_{meta_n_neighbors}.png", dpi=300)
    plt.close(fig)


def annotate_selected_params_tsne(ax, list_annotations):
    print(list_annotations)
    offset = 1.0 / len(list_annotations)
    # sort by x coordinate
    for i, (perp_val, pos_x, pos_y) in enumerate(sorted(list_annotations, key=lambda p: p[1])):
        ax.scatter(pos_x, pos_y, marker="X", color="orange", s=80)
        # ax.annotate(str(perp_val), (pos_x, pos_y), fontsize=10)
        ax.annotate(
            str(perp_val),
            xy=(pos_x, pos_y),
            xycoords="data",
            xytext=(i * offset, -0.075),
            textcoords="axes fraction",
            arrowprops=dict(
                arrowstyle="->",
                linestyle="--",
                color="#0047BB",
                connectionstyle="angle,angleA=0,angleB=90,rad=10",
            ),
            fontsize=18,
        )


def annotate_selected_params_umap(ax, list_annotations):
    print(list_annotations)
    # sort by y coordinate
    offset = 1.0 / len(list_annotations)
    for i, (n_neighbors, min_dist, pos_x, pos_y) in enumerate(
        sorted(list_annotations, key=lambda p: p[2])
    ):
        ax.scatter(pos_x, pos_y, marker="X", color="orange")
        txt = f"{n_neighbors:>6},\n{min_dist:>5}"
        # ax.annotate(txt, (pos_x, pos_y), fontsize=12)
        ax.annotate(
            txt,
            xy=(pos_x, pos_y),
            xycoords="data",
            xytext=(i * offset, 0),
            textcoords="axes fraction",
            arrowprops=dict(
                arrowstyle="->",
                linestyle="--",
                color="#0047BB",
                connectionstyle="angle,angleA=0,angleB=90,rad=10",
            ),
            fontsize=14,
        )


def get_params_to_show(dataset_name, method_name):
    symbol_map = {
        "++": " ⇧⇧",
        "+": "   ⇧",
        "=": "   ▯",
        "-": "   ⇩",
        "--": " ⇩⇩",
        "~": "   ☆",
        "": "     ",
    }

    method_text_map = {
        "qij": "$f_{score}$",
        "rnx": "$AUC_{log}RNX$",
        "bic": "BIC",
        "prediction": "Good prediction",
        "all": "All scores",
        "": "",
    }

    def transform_text(text):
        res = []
        for s in text.split():
            s = s.strip()
            m = re.compile("([+-=~]*)([a-z]*)")
            g = m.match(s)
            if g:
                symbol, method_text = g.groups()
                res.append(f"{method_text_map[method_text]} {symbol_map[symbol]:<2}")
            else:
                print("[Deubg] Invalid selected param: ", s, text)
        return "\n".join(res)

    def transform_list_items(list_items):
        transformed_list = []
        for *p, text in list_items:
            print(*p, text)
            transformed_list.append((*p, transform_text(text)))
        return transformed_list

    config_params = {
        "20NEWS5": {
            "tsne": {
                (130, "++qij, ++bic, --rnx, ~prediction"),
                (89, "++qij, ++bic, +rnx"),
                (25, "--qij, --bic, ++rnx"),
                (512, "--all"),
            },
            "umap": [
                (15, 0.2154, "++rnx"),
                (134, 0.001, "++qij"),
                (147, 0.01, "~prediction"),
                (86, 0.0464, "+qij, =rnx"),
                (7, 0.1, "++rnx, --qij"),
            ],
        },
        "DIGITS": {
            "tsne": [
                (50, "++qij, ++bic, +rnx, ~prediction"),
                (14, "+qij, -bic, ++rnx"),
                (90, "+qij, ++bic, --rnx,"),
                (260, "--all"),
            ],
            "umap": {
                (5, 0.001, "+qij, ++rnx"),
                (11, 0.0022, "++qij, +rnx, ~prediction"),
                (19, 0.0046, "++qij, +rnx"),
                (401, 0.0464, "--qij, +rnx"),
            },
        },
        "COIL20": {
            "tsne": [
                (40, "++qij, ++bic, ++rnx, ~prediction"),
                (28, "++qij, ++bic, ++rnx"),
                (124, "--qij, --bic, +rnx,"),
                (247, "--all"),
            ],
            "umap": [
                (11, 0.01, "++qij, -rnx"),
                (5, 0.4642, "-qij, ++rnx"),
                (102, 0.1, "--qij, +rnx"),
                # (8, 0.01, "~prediction, ++qij, =rnx"),
                # # bo_constraint.py  --seed 2019 -d COIL20 -m umap -u ei -x 0.1 -nr 40 --run
                # # (5, 0.01, "+qij, +rnx"),
                # (4, 0.4642, "--qij, ++rnx"),
                # (56, 0.1, "--qij, +rnx"),
                (300, 0.4642, "--all"),
            ],
        },
        "NEURON_1K": {
            "tsne": [
                (72, "++qij, ++bic, --rnx, ~prediction"),
                (13, " --qij, -bic, ++rnx"),
                (40, "=qij, +rnx,  ++bic"),
                (150, "--all"),
            ],
            "umap": [
                (16, 0.001, "~prediction, ++qij, +rnx"),
                (5, 0.1, "++rnx, +qij"),
                (150, 0.0464, "-qij, -rnx"),
            ],
        },
        "FASHION_MOBILENET": {
            "tsne": [
                (82, "++qij, ++bic, +rnx, ~prediction"),
                (12, "-qij, --bic, ++rnx"),
                (35, "++qij, --bic, ++rnx"),
                (151, "++qij, ++bic, --rnx"),
            ],
            "umap": [
                (19, 0.0022, "~prediction, +"),
                (9, 0.0046, "++rnx, +qij"),
                (310, 0.1, "="),
            ],
        },
        "FASHION1000": {
            "tsne": [
                (36, "++qij, +bic, ++rnx, ~prediction"),
                (14, "+qij, --bic, ++rnx, "),
                (69, "+qij, ++bic, =rnx"),
                (220, "--all"),
            ],
            "umap": [
                # (4, 0.01, "++qij, +rnx"),
                (4, 0.2154, "++qij, ++rnx"),
                (6, 0.01, "~prediction"),
                # bo_constraint.py  --seed 2018 -d FASHION1000 -m umap -u ei -x 0.1 -nr 50
                (50, 0.1, "+qij, +rnx"),
                (150, 0.4642, "-qij, +rnx"),
            ],
        },
    }

    params = config_params[dataset_name][method_name]
    return transform_list_items(params)


def plot_samples(dataset_name, data, plot_dir="", n_samples=4, transpose=False):
    img_size = int(math.sqrt(data.shape[1]))
    fig, axes = plt.subplots(1, n_samples, figsize=(n_samples, 1.0))
    for i, ax in enumerate(axes.ravel()):
        X = data[i].reshape(img_size, img_size)
        if transpose:
            X = X.transpose()
        ax.imshow(X, cmap="gray")
        ax.axis("off")

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.05, bottom=0.1, top=0.9, left=0.05, right=0.95)
    fig.savefig(f"{plot_dir}/{dataset_name}_samples.pdf")
    plt.close()


def plot_viz_with_score_flexibility(
    dataset_name, method_name, config_label1, config_label2, plot_dir="", embedding_dir="",
):
    label_name1, title1, best_param1, labels1, label_names1 = config_label1
    label_name2, title2, best_param2, labels2, label_names2 = config_label2

    fig, [[ax0, ax1], [ax2, ax3]] = plt.subplots(2, 2, figsize=(8, 9.5))
    sub_text = [
        [f"Best perplexity {best_param1}", f"Best perplexity {best_param1}"],
        [f"Best perplexity {best_param2}", f"Best perplexity {best_param2}"],
    ]

    for [ax_lbl1, ax_lbl2], best_param, [t1, t2] in zip(
        [[ax0, ax2], [ax1, ax3]], [best_param1, best_param2], sub_text
    ):
        Z = joblib.load(f"{embedding_dir}/{best_param}.z")

        # ax_lbl1.set_title(title1)
        s1 = ax_lbl1.scatter(Z[:, 0], Z[:, 1], s=6, c=labels1, alpha=0.7, cmap="Spectral")
        ax_lbl1.text(0.98, 0.02, t1, transform=ax_lbl1.transAxes, fontsize=16, ha="right")

        # ax_lbl2.set_title(title2)
        s2 = ax_lbl2.scatter(Z[:, 0], Z[:, 1], s=6, c=labels2, alpha=0.7, cmap="Paired")
        ax_lbl2.text(0.98, 0.02, t2, transform=ax_lbl2.transAxes, fontsize=16, ha="right")

    for ax in [ax0, ax1, ax2, ax3]:
        change_border(ax, width=0.25, color="black")
        # ax.set_aspect("equal")

    # legend
    if label_names1 is None:
        label_names1 = range(len(np.unique(labels1)))
    handles1, _ = s1.legend_elements()
    legend1 = ax0.legend(
        handles1,
        label_names1,
        fancybox=True,
        loc="upper left",
        bbox_to_anchor=(0.0, 1.3),
        borderaxespad=0.1,
        ncol=min(6, len(label_names1)),  # note to change according to dataset
        title=title1,
        fontsize="small",
    )

    if label_names2 is None:
        label_names2 = range(len(np.unique(labels2)))
    handles2, _ = s2.legend_elements()
    legend1 = ax2.legend(
        handles2,
        label_names2,
        fancybox=True,
        loc="lower left",
        bbox_to_anchor=(0, -0.2),
        borderaxespad=0.1,
        ncol=min(4, len(label_names2)),
        title=title2,
        fontsize="small",
    )

    fig.tight_layout()  # (pad=0.4, w_pad=0.2, h_pad=1.0)
    # This works for NEURON_1K
    # plt.subplots_adjust(wspace=0.02, hspace=0.02, left=0.01, right=0.99)
    plt.subplots_adjust(wspace=0.05, hspace=0.02, left=0.02, right=0.98)
    fig.savefig(f"{plot_dir}/score_flexibility.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    import argparse

    # import sys

    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset_name", default="")
    ap.add_argument("-m", "--method_name", default="umap", help="['tsne', 'umap', 'largevis']")
    # ap.add_argument("-s", "--seed", default=42, type=int)
    ap.add_argument("--use_other_label", default=None)
    ap.add_argument("--plot_test_vis", action="store_true")
    ap.add_argument("--show_viz_grid", action="store_true")
    ap.add_argument("--plot_metamap", action="store_true")
    ap.add_argument("--plot_samples", action="store_true")
    ap.add_argument("--plot_score_flexibility", action="store_true")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    dataset.set_data_home("./data")
    dataset_name = args.dataset_name
    method_name = args.method_name

    # print(get_params_to_show(dataset_name, method_name))
    # sys.exit(0)

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

    # note to make big font size for plots in the paper
    plt.rcParams.update({"font.size": 12})

    if args.plot_test_vis:
        plot_test_vis(X, dataset_name, plot_dir, embedding_dir, labels, other_labels)

    if args.show_viz_grid:
        list_params = get_params_to_show(dataset_name, method_name)
        show_viz_grid(dataset_name, method_name, labels, plot_dir, embedding_dir, list_params)

    if args.plot_metamap:
        # plot_metamap(dataset_name, method_name, plot_dir, embedding_dir)
        plot_metamap_func = {
            "tsne": plot_metamap_with_scores_tsne,
            "umap": plot_metamap_with_scores_umap,
            "largevis": lambda a, b, c, d: None,
        }[method_name]
        plot_metamap_func(
            dataset_name, plot_dir, embedding_dir, score_dir, use_cache=args.debug
        )

    if args.plot_samples:
        n_samples = 4
        transpose = True if dataset_name == "COIL20" else False
        data = X_origin[np.random.choice(X_origin.shape[0], size=n_samples)]
        print("Sampled data: ", data.shape)
        plot_samples(
            dataset_name,
            data,
            plot_dir=f"./plots/{dataset_name}",
            n_samples=n_samples,
            transpose=transpose,
        )

    if args.plot_score_flexibility:
        config_labels = get_config_labels_for_score_flexibility()
        config = config_labels[dataset_name][method_name]
        config_label1 = config["label1"]
        config_label2 = config["label2"]
        labels1, label_names1 = dataset.load_additional_labels(
            dataset_name, label_name=config_label1[0]
        )
        labels2, label_names2 = dataset.load_additional_labels(
            dataset_name, label_name=config_label2[0]
        )

        # hard-coded to fix custom labels of NEURON_1K dataset
        if dataset_name == "NEURON_1K":
            label_names2 = config["label2_correct"]

        config_label1 += [labels1, label_names1]
        config_label2 += [labels2, label_names2]

        plt.rcParams.update({"font.size": 16})
        plot_viz_with_score_flexibility(
            dataset_name,
            method_name,
            config_label1,
            config_label2,
            plot_dir=plot_dir,
            embedding_dir=embedding_dir,
        )
