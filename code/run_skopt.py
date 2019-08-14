import joblib
from functools import partial

import numpy as np
from matplotlib import pyplot as plt

from skopt.space import Real, Integer
from skopt import gp_minimize, forest_minimize
from skopt.utils import use_named_args
from skopt.plots import plot_convergence, plot_objective

from utils import generate_constraints, score_embedding
from common.dataset import dataset
from run_viz import run_tsne, run_largevis, run_umap


def hyperopt_workflow(X, constraints,
                      method_name: str, score_name: str,
                      seed: int=42, embedding_dir: str="",
                      n_total_runs=20, acq_func="EI", kappa=5.0, xi=0.01):
    if method_name in ["tsne", "largevis"]:
        space = [Real(2, X.shape[0]//3, "log-uniform", name="perplexity")]
    elif method_name in ["umap"]:
        space = [
            Real(2, X.shape[0]//3, "log-uniform", name="n_neighbors"),
            Real(0.001, 1.0, "log-uniform", name="min_dist")
        ]
    else:
        raise ValueError(f"Invalid method_name: {method_name}")
    print(space)
    
    @use_named_args(space)
    def objective(**params):
        embedding_function = partial({
            'tsne': run_tsne,
            'umap': run_umap,
            'largevis': run_largevis
        }[method_name], X=X, seed=42, check_log=True, embedding_dir=embedding_dir)
        Z = embedding_function(**params)
        return -1.0 * score_embedding(Z, score_name, constraints)

    from numpy.random import RandomState
    res_gp = gp_minimize(objective, space,
                         n_random_starts=min(5, int(0.1 * n_total_runs)), n_calls=n_total_runs,
                         random_state=RandomState(seed=42),
                         acq_func=acq_func, kappa=kappa, xi=xi)
    # res_gp = forest_minimize(objective, space, n_calls=20, n_random_starts=5, random_state=42)
    best_score = res_gp.fun
    best_param = res_gp.x[0]

    print(best_param, best_score)

    plot_convergence(res_gp)
    plt.savefig(f"{plot_dir}/convergence.png")

    plot_objective(res_gp)
    plt.savefig(f"{plot_dir}/objective.png")


if __name__ == "__main__":
    import argparse
    import os

    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset_name", default="")
    ap.add_argument("-m", "--method_name", default="tsne",
                    help=" in ['tsne', 'umap']")
    ap.add_argument("-sc", "--score_name", default="qij",
                    help=" in ['qij', 'contrastive', 'cosine', 'cosine_ratio']")
    ap.add_argument("-st", "--strategy", default="partial_labels",
                    help="strategy: using partial labels or auto-generated constraints")
    ap.add_argument("-nr", "--n_total_runs", default=15, type=int,
                    help="number of evaluated points run by BayOpt")
    ap.add_argument("-nc", "--n_constraints", default=50, type=int,
                    help="number of constraints each type")
    ap.add_argument("-nl", "--n_labels_each_class", default=5, type=int,
                    help="number of labelled points selected for each class")
    ap.add_argument("--seed", default=42, type=int)
    ap.add_argument("-cp", "--constraint_proportion", default=1.0, type=float,
                    help="target_function = cp * user_constraint + (1-cp)* John's metric")
    ap.add_argument("-u", "--utility_function", default="ucb",
                    help="in ['ucb', 'ei', 'poi']")
    ap.add_argument("-k", "--kappa", default=5.0, type=float,
                    help="For UCB, small ->exploitation, large ->exploration, default 5.0")
    ap.add_argument("-x", "--xi", default=0.025, type=float,
                    help="For EI/POI, small ->exploitation, large ->exploration, default 0.025")
    args = ap.parse_args()

    # mlflow.set_experiment('HyperOpt-Constraint-Scores-skopt')
    # for arg_key, arg_value in vars(args).items():
    #     mlflow.log_param(arg_key, arg_value)

    dataset.set_data_home("./data")
    dataset_name = args.dataset_name
    method_name = args.method_name
    score_name = args.score_name

    # custom preprocessing method for each dataset
    preprocessing_method = {
        'COIL20': None,
        'WINE': 'standardizer',
    }.get(dataset_name, 'unitScale')  # default for image dataset

    X_origin, X, labels = dataset.load_dataset(dataset_name, preprocessing_method)
    embedding_dir = f"./embeddings/{dataset_name}/{method_name}"
    plot_dir = f"./plots/{dataset_name}/{method_name}"
    for dir_path in [embedding_dir, plot_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    
    constraints = generate_constraints(
        args.strategy, score_name, labels,
        n_constraints=args.n_constraints,
        n_labels_each_class=args.n_labels_each_class
    )

    hyperopt_workflow(X, constraints, method_name, score_name,
                      seed=args.seed, embedding_dir=embedding_dir)
