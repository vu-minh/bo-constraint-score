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


if __name__ == "__main__":
    values = generate_value_range(
        min_val=2, max_val=1796 // 3, num=150, range_type="log", dtype=int
    )
    print(len(values))
    print(values)
