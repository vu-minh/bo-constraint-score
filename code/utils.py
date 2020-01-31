import math
import json
from functools import partial

import numpy as np

import constraint_score
from common.dataset import constraint
from common.metric.dr_metrics import DRMetric


def generate_constraints(
    constraint_strategy, score_name, labels, n_constraints=50, n_labels_each_class=5, seed=None
):
    if constraint_strategy == "partial_labels":
        sim_links, dis_links = constraint.generate_constraints_from_partial_labels(
            labels, n_labels_each_class=n_labels_each_class, seed=seed
        )
        constraints = {"sim_links": sim_links, "dis_links": dis_links}
        print(
            f"[Debug]: From {n_labels_each_class} =>"
            f"(sim-links: {len(sim_links)}, dis-links: {len(dis_links)})"
        )
    else:
        constraints = {
            "qij": {
                "sim_links": constraint.gen_similar_links(
                    labels, n_constraints, include_link_type=True, seed=seed
                ),
                "dis_links": constraint.gen_dissimilar_links(
                    labels, n_constraints, include_link_type=True, seed=seed
                ),
            },
            "contrastive": {
                "contrastive_constraints": constraint.generate_contrastive_constraints(
                    labels, n_constraints, seed=seed
                )
            },
        }[score_name]
    return constraints


def _contrastive_score(Z, contrastive_constraints):
    return constraint_score.contrastive_score(Z, contrastive_constraints)


def _qij_score(Z, sim_links, dis_links, degrees_of_freedom=1.0):
    Q = constraint_score.calculate_Q(Z, degrees_of_freedom)
    final_score, sim_scores, dis_scores = constraint_score.qij_based_scores(
        Q, sim_links, dis_links, normalized=False
    )
    return final_score


def score_embedding(Z, score_name, constraints, degrees_of_freedom=1.0):
    score_func = {
        "contrastive": partial(_contrastive_score, **constraints),
        "qij": partial(_qij_score, degrees_of_freedom=degrees_of_freedom, **constraints),
    }[score_name]
    return score_func(Z)


def calculate_all_metrics(X, Z):
    results = {}
    dr_metrics = DRMetric(X, Z)
    for metric_name in DRMetric.metrics_names:
        metric_method = getattr(dr_metrics, metric_name)
        results[metric_name] = metric_method()
    return results


def generate_value_range(min_val=2, max_val=1000, num=150, range_type="log", dtype=int):
    return {
        "log": partial(_generate_log_range, base=math.e),
        "log2": partial(_generate_log_range, base=2),
        "log10": partial(_generate_log_range, base=10),
        "linear": _generate_linear_range,
    }[range_type](min_val, max_val, num=num, dtype=dtype)


def _generate_linear_range(min_val=2, max_val=1000, num=100, dtype=int):
    return np.unique(np.linspace(min_val, max_val, num=num, dtype=dtype))


def _generate_log_range(min_val=2, max_val=1000, num=150, dtype=int, base=math.e):
    log_func = {math.e: math.log, 2: math.log2, 10: math.log10}[base]
    min_exp = log_func(min_val)
    max_exp = log_func(max_val)
    range_values = np.logspace(min_exp, max_exp, num=num, base=base, dtype=dtype)
    return np.unique(range_values[np.where(range_values >= min_val)])


def get_scores_tsne(score_name, score_dir):
    def _get_bic_score(score_dir):
        with open(f"{score_dir}/bic/BIC.txt", "r") as in_file_bic:
            bic_data = json.load(in_file_bic)
            list_params = list(map(int, bic_data["list_params"]))
            bic_score = np.array(bic_data["bic"])
        return list_params, bic_score

    def _get_rnx_score(score_dir):
        with open(f"{score_dir}/metrics/metrics.txt", "r") as in_file_metric:
            metrics_data = json.load(in_file_metric)
            list_params = list(map(int, metrics_data["list_params"]))
            rnx_score = np.array(metrics_data["auc_rnx"])
        return list_params, rnx_score

    def _get_qij_score(score_dir, n_labels_each_class=10):
        with open(f"{score_dir}/qij/dof1.0_all.txt", "r") as in_file_qij:
            qij_score_data = json.load(in_file_qij)
            list_params = list(map(int, qij_score_data["list_params"]))
            all_scores = qij_score_data["all_scores"]
            qij_score = np.mean(all_scores[str(n_labels_each_class)], axis=0)
        return list_params, qij_score

    return {
        "qij": partial(_get_qij_score, score_dir=score_dir, n_labels_each_class=10),
        "rnx": partial(_get_rnx_score, score_dir=score_dir),
        "bic": partial(_get_bic_score, score_dir=score_dir),
    }[score_name]()


def get_dataset_display_name(dataset_name):
    return {
        "DIGITS": "DIGITS",
        "COIL20": "COIL20",
        "FASHION1000": "FASHION_1K",
        "FASHION_MOBILENET": "FASH_MOBI",
        "20NEWS5": "5NEWS",
        "NEURON_1K": "NEURON_1K",
    }.get(dataset_name, dataset_name)


def get_method_display_name(method_name):
    return {
        "tsne": "t-SNE",
        "largevis": "LargeVis",
        "umap": "UMAP",
        "umap1": "UMAP\n$_{(min\_dist\!=\!0.1)}$",
    }.get(method_name, method_name)


def get_param_display_name(method_name):
    return {
        "tsne": "perplexity",
        "largevis": "perplexity",
        "umap": "(n_neighbors, perplexity)",
        "umap1": "n_neighbors",
    }.get(method_name, method_name)


def get_score_display_name(score_name):
    return {
        "qij": "$f_{score}$",
        "qij_score": "$f_{score}$",
        "f_score": "$f_{score}$",
        "auc_rnx": "$AUC_{log}RNX$",
        "bic": "BIC-based score",
    }.get(score_name, score_name)


def get_config_labels_for_score_flexibility():
    return {
        "NEURON_1K": {
            "tsne": {
                "label1": [
                    "graph_based_cluster",  # label name
                    "Points colored by graph-based cluster indices",  # title/description
                    68,  # best param
                    # list labels of each point will be added
                    # list names for each label will be added
                ],
                "label2": ["umi", "Points colored by UMI count", 144],
                "label2_correct": ["less than 6.5K", "from 6.5K to 12.5K", "more than 12.5K",],
            },
            "umap": {
                "label1": [
                    "graph_based_cluster",  # label name
                    "Points colored by graph-based cluster indices",  # title/description
                    "10_0.0038",  # best param
                ],
                "label2": ["umi", "Points colored by UMI count", "8_0.0185"],
                "label2_correct": ["less than 6.5K", "from 6.5K to 12.5K", "more than 12.5K",],
            },
        },
        "20NEWS5": {
            "tsne": {
                "label1": ["cat", "Group sub-categories", 130],
                "label2": ["matcat", "Semantic master-categories", 44,],
            },
            "umap": {
                "label1": ["cat", "Group sub-categories", "169_0.0010"],
                "label2": ["matcat", "Semantic master-categories", "161_0.0636",],
            },
        },
        "FASHION_MOBILENET": {
            "tsne": {
                "label1": ["class_subcat", "Group sub-categories", 77],
                "label2": ["class_matcat", "Hierarchical master-categories", 113],
            },
            "umap": {
                "label1": ["class_subcat", "Group sub-categories", "20_0.0046"],
                "label2": ["class_matcat", "Hierarchical master-categories", "20_0.0255"],
            },
        },
    }


def change_border(ax, width=0.25, color="black"):
    ax.axes.get_yaxis().set_visible(False)
    ax.axes.get_xaxis().set_visible(False)
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(width)
        ax.spines[axis].set_edgecolor(color)


if __name__ == "__main__":
    values = generate_value_range(
        min_val=2, max_val=1796 // 3, num=150, range_type="log", dtype=int
    )
    print(len(values))
    print(values)
