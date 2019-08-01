from functools import partial

import math
import numpy as np
from common.dataset import constraint
import constraint_score


def generate_constraints(constraint_strategy, score_name, labels,
                         n_constraints=50, n_labels_each_class=5, seed=None):
    if constraint_strategy == "partial_labels":
        sim_links, dis_links = constraint.generate_constraints_from_partial_labels(
            labels, n_labels_each_class=n_labels_each_class, seed=seed)
        constraints = {'sim_links': sim_links, 'dis_links': dis_links}
        print(f"[Debug]: From {n_labels_each_class} =>"
              f"(sim-links: {len(sim_links)}, dis-links: {len(dis_links)})")
    else:
        constraints = {
            "qij": {
                "sim_links": constraint.gen_similar_links(
                    labels, n_constraints, include_link_type=True, seed=seed),
                "dis_links": constraint.gen_dissimilar_links(
                    labels, n_constraints, include_link_type=True, seed=seed)
            },
            "contrastive": {
                "contrastive_constraints": constraint.generate_contrastive_constraints(
                    labels, n_constraints, seed=seed)
            }
        }[score_name]
    return constraints


def _contrastive_score(Z, contrastive_constraints):
    return constraint_score.contrastive_score(Z, contrastive_constraints)


def _qij_score(Z, sim_links, dis_links, degrees_of_freedom=0.5):
    Q = constraint_score.calculate_Q(Z, degrees_of_freedom)
    final_score, sim_scores, dis_scores = constraint_score.qij_based_scores(
        Q, sim_links, dis_links, normalized=False
    )
    return final_score


def score_embedding(Z, score_name, constraints, degrees_of_freedom=0.5):
    score_func = {
        "contrastive": partial(_contrastive_score, **constraints),
        "qij": partial(_qij_score, degrees_of_freedom=degrees_of_freedom, **constraints)
    }[score_name]
    return score_func(Z)


def generate_value_range(range_type='linear', min_val=2, max_val=1000, num=100, dtype=int):
    return {
        'log2': _generate_log2_range,
        'log10': _generate_log10_range,
        'linear': _generate_linear_range,
    }[range_type](min_val, max_val, num, dtype)


def _generate_linear_range(min_val=2, max_val=1000, num=100, dtype=int):
    return np.unique(np.linspace(min_val, max_val, num=num, dtype=dtype))


def _generate_log2_range(min_val=2, max_val=1000, num=100, dtype=int):
    min_exp = int(math.log2(min_val))
    max_exp = int(math.log2(max_val))
    return np.unique(np.logspace(min_exp, max_exp, num=num, base=2, dtype=dtype))


def _generate_log10_range(min_val=1, max_val=1000, num=100, dtype=int):
    min_exp = int(math.log10(min_val))
    max_exp = int(math.log10(max_val))
    return np.unique(np.logspace(min_exp, max_exp, num=num, base=10, dtype=dtype))
