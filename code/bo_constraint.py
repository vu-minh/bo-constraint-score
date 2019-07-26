import joblib
import numpy as np
from functools import partial
from bayes_opt import BayesianOptimization

import utils
from common.dataset import dataset
from bo_plot import plot_gp_one_param

from MulticoreTSNE import MulticoreTSNE
import umap


def target_function(method_name, score_name, constraints, seed, p):
    method = {
        "tsne": MulticoreTSNE(perplexity=p, n_iter=1000, random_state=seed, n_jobs=3,
                              n_iter_without_progress=1000, min_grad_norm=1e-32),
        "umap": umap.UMAP(n_neighbors=int(p), random_state=seed)
    }[method_name]
    Z = method.fit_transform(X)
    score = utils.score_embedding(Z, score_name, constraints)
    return score


def run_bo(target_func,
           n_total_runs=15, n_random_inits=5,
           kappa=5, xi=0.025, util_func="ucb", seed=None):
    perp_range = np.array(list(range(2, X.shape[0] // 3)))
    true_target_values = None

    # create BO object to find max of `target_func` in the domain of param `p`
    optimizer = BayesianOptimization(
        target_func,
        {"p": (2, X.shape[0] // 3)},
        random_state=seed,
    )

    # using `util_func`, evaluate the target function at some randomly initial points
    optimizer.maximize(acq=util_func, init_points=n_random_inits, n_iter=0, kappa=kappa, xi=xi)
    plot_gp_one_param(optimizer, x=perp_range.reshape(-1, 1), y=true_target_values,
                      util_func=util_func, kappa=kappa, xi=xi,
                      plot_dir=plot_dir, dataset_name=dataset_name)

    # then predict the next best param to evaluate
    for i in range(n_total_runs - n_random_inits):
        optimizer.maximize(acq=util_func, init_points=0, n_iter=1, kappa=kappa, xi=xi)
        # print("Current max: ", optimizer.max)
        plot_gp_one_param(optimizer, x=perp_range.reshape(-1, 1), y=true_target_values,
                          util_func=util_func, kappa=kappa, xi=xi,
                          plot_dir=plot_dir, dataset_name=dataset_name)

    # log the internal calculated scores (which is the target_function value)
    for res in optimizer.res:
        with mlflow.start_run(nested=True):
            mlflow.log_param("p", res["params"]["p"])
            mlflow.log_metric("target_func", res["target"])

    # log optimizer for further plotting
    log_obj = locals()
    log_obj['optimizer'] = optimizer
    log_name = f"{log_dir}/{util_func}-kappa{kappa}-xi{xi}-n{n_total_runs}.z"
    joblib.dump(log_obj, log_name)
    mlflow.log_artifact(log_name)

    return optimizer.max


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

    mlflow.set_experiment('BO-with-Constraint-Scores')
    for arg_key, arg_value in vars(args).items():
        mlflow.log_param(arg_key, arg_value)

    dataset.set_data_home("./data")
    dataset_name = args.dataset_name
    _, X, labels = dataset.load_dataset(dataset_name)  # COIL20 normalized
    # X /= 255.0  # for DIGITS, not for COIL20

    method_name = args.method_name
    score_name = args.score_name
    rnd_seed = int(args.random_seed)

    dir_name_pattern = f"{dataset_name}/{method_name}/{score_name}"
    plot_dir = f"./plots/{dir_name_pattern}"
    log_dir = f"./logs/{dir_name_pattern}"
    for dir_path in [plot_dir, log_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    constraints = utils.generate_constraints(
        args.strategy, score_name, labels, seed=rnd_seed,
        n_constraints=args.n_constraints,
        n_labels_each_class=args.n_labels_each_class
    )

    target_function_wrapper = partial(
        target_function, method_name, score_name, constraints, rnd_seed)
    best_result = run_bo(target_func=target_function_wrapper, n_total_runs=args.n_total_runs)
    print(best_result)
