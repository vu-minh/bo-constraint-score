import joblib
from functools import partial

import numpy as np
from matplotlib import pyplot as plt

from skopt.space import Real, Integer
from skopt import gp_minimize
from skopt.utils import use_named_args
from skopt.plots import plot_convergence

from utils import generate_constraints, score_embedding
from common.dataset import dataset
from run_viz import run_tsne, run_largevis, run_umap


def hyperopt_workflow(X, constraints,
                      method_name: str, score_name: str,
                      seed: int=42, embedding_dir: str=""):
    if method_name in ["tsne", "largevis"]:
        space = [Real(2, 100, "log-uniform", name="perplexity")]
    elif method_name in ["umap"]:
        space = [Real(2, 100, "log-uniform", name="n_neighbors"),
                 Real(0.001, 1.0, "log-uniform", name="min_dist")]
    else:
        raise ValueError(f"Invalid method_name: {method_name}")
    print(space)
    
    @use_named_args(space)
    def objective(**params):
        embedding_function = partial({
            'tsne': run_tsne,
            'umap': run_umap,
            'largevis': run_largevis
        }[method_name], X=X, seed=seed, check_log=True, embedding_dir=embedding_dir)

        Z = embedding_function(**params)
        return -1.0 * score_embedding(Z, score_name, constraints)

    res_gp = gp_minimize(objective, space, n_calls=30, random_state=seed)
    best_score = res_gp.fun
    best_param = res_gp.x[0]

    print(best_param, best_score)
    plot_convergence(res_gp)
    plt.show()
    

if __name__ == "__main__":
    import argparse
    import mlflow
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
    ap.add_argument("-rs", "--random_seed", default=None)
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
    rnd_seed = int(args.random_seed)
    
    # custom preprocessing method for each dataset
    preprocessing_method = {
        'COIL20': None
    }.get(dataset_name, 'unitScale')  # default for image dataset

    X_origin, X, labels = dataset.load_dataset(dataset_name, preprocessing_method)
    embedding_dir = f"./embeddings/{dataset_name}/{method_name}"
    plot_dir = f"./plots/{dataset_name}/{method_name}"
    for dir_path in [embedding_dir, plot_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    
    constraints = generate_constraints(
        args.strategy, score_name, labels, seed=rnd_seed,
        n_constraints=args.n_constraints,
        n_labels_each_class=args.n_labels_each_class
    )

    hyperopt_workflow(X, constraints, method_name, score_name,
                      seed=rnd_seed, embedding_dir=embedding_dir)