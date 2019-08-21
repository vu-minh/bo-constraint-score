import math
import joblib
from functools import partial
import numpy as np

from bayes_opt import BayesianOptimization
from bayes_opt.observer import JSONLogger
from bayes_opt.event import Events

import utils
from common.dataset import dataset
from run_viz import run_tsne, run_largevis, run_umap
from bo_plot import plot_bo_one_param_summary


# transformation rules to transform the params in log scale to linear scale
transformation_rules = {
    # 'param_name': (cast_type, transformation_func),
    'perplexity': (int, math.exp),
    'n_neighbors': (int, math.exp),
    'min_dist': (float, math.exp)
}


def _target_func(method_name, X, check_log, seed, embedding_dir, **params_in_log_scale):
    ''' The score function, calls viz method to obtain `Z` and calculates score on `Z`
    The `params_in_log_scale` should be:
        + log(`perplexity`) for tsne and largevis,
        + log(`n_neighbors`, `min_dist`) for umap
    '''
    embedding_function = {
        'tsne': run_tsne,
        'umap': run_umap,
        'largevis': run_largevis
    }[method_name]

    # transform and cast type the param in log scale into original linear scale
    for param_name, (cast_type, transformation_func) in transformation_rules.items():
        if param_name in params_in_log_scale:
            params_in_log_scale[param_name] = cast_type(transformation_func(
                params_in_log_scale[param_name]
            ))

    # get the embedding and calculate constraint score
    seed = 42
    Z = embedding_function(X=X, seed=seed, check_log=True, embedding_dir=embedding_dir,
                           **params_in_log_scale)
    return utils.score_embedding(Z, score_name, constraints)


def bayopt_workflow(X, constraints,
                    method_name: str, score_name: str, seed: int=42,
                    embedding_dir: str="", score_dir="", plot_dir="",
                    bayopt_params={}):

    # value range of params in original linear scale
    min_perp, max_perp = 2, int(X.shape[0] // 3)
    start_min_dist, stop_min_dist = 0.001, 1.0

    # value range of param in log scale
    perp_range = (math.log(min_perp), math.log(max_perp))
    min_dist_range = (math.log(start_min_dist), math.log(stop_min_dist))

    # create 'logspace' for the parameters
    params_space = {
        'tsne': {'perplexity': perp_range},
        'largevis': {'perplexity': perp_range},
        'umap': {'n_neighbors': perp_range, 'min_dist': min_dist_range},
    }[method_name]

    # define the target function which will be maximize by BayOpt
    target_func = partial(_target_func, method_name=method_name,
                          X=X, seed=seed, check_log=True, embedding_dir=embedding_dir)

    # run bayopt
    optimizer = run_bo(target_func, params_space, **bayopt_params)

    # transform the best param to original linear scale
    result = optimizer.max
    for param_name, param_value in result['params'].items():
        cast_type, transformation_func = transformation_rules[param_name]
        result['params'][param_name] = cast_type(transformation_func(param_value))    
    return result, optimizer


def run_bo(target_func, params_space, seed=42, log_dir="",
           n_total_runs=15, n_random_inits=5, kappa=5, xi=0.025, util_func="ucb"):

    # create BO object to find max of `target_func` in the domain of param `p`
    optimizer = BayesianOptimization(
        f=target_func,
        pbounds=params_space,
        random_state=seed,
    )

    # log the progress for plotting
    log_name = f"{util_func}_k{kappa}_xi{xi}_n{n_total_runs}"
    logger = JSONLogger(path=f"{log_dir}/{log_name}.json")
    optimizer.subscribe(Events.OPTMIZATION_STEP, logger)

    # specific params for GPs in BO: alpha controls noise level, default to 1e-10 (noise-free).
    optimizer_params = dict(alpha=1e-5, n_restarts_optimizer=3, random_state=seed)

    optimizer.maximize(acq=util_func, init_points=n_random_inits,
                       n_iter=(n_total_runs - n_random_inits),
                       kappa=kappa, xi=xi, **optimizer_params)

    joblib.dump(optimizer, f"{log_dir}/{log_name}.z")
    return optimizer


def _posterior(optimizer, x_obs, y_obs, param_range):
    ''' Predict the mean and variance for each param value using the observed points'''
    # from the observed points, update the GP model
    optimizer._gp.fit(x_obs, y_obs)
    # make predict for all value in `param_range`
    mu, sigma = optimizer._gp.predict(param_range, return_std=True)
    return {'pred_mu': mu, 'pred_sigma': sigma}


def _calculate_score(method_name, list_perp_in_log_scale,
                     default_min_dist=0.1, degrees_of_freedom=1.0, embedding_dir=""):
    all_embeddings = joblib.load(f"{embedding_dir}/all.z")
    scores = []
    for perp in list_perp_in_log_scale:
        if method_name in ['umap']:
            key_name = f"{perp}_{default_min_dist:.4f}"
        else:
            key_name = str(perp)
        embedding = all_embeddings[key_name]
        score = utils.score_embedding(embedding, score_name, constraints,
                                      degrees_of_freedom=degrees_of_freedom)
        scores.append(score)
    return scores


def plot_bo(optimizer, list_perp_in_log_scale, true_score,
            threshold=0.95, bayopt_params={}, plot_dir=""):
    # test in case 1D, only one param `perplexity`
    # note that the observations are in logscale
    print("List params for plotting (int real value: ")
    print(min(list_perp_in_log_scale), max(list_perp_in_log_scale))
    
    x_obs = np.array([[res["params"]["perplexity"]] for res in optimizer.res])
    y_obs = np.array([res["target"] for res in optimizer.res])
    observation = {'x_obs': x_obs, 'y_obs': y_obs}

    # convert `list_perp_in_log_scale` into exponential value by passing it to np.log
    list_params = np.log(list_perp_in_log_scale).reshape(-1, 1)
    print("List param in log scale: ", min(list_params), max(list_params))
    
    true_target = {'list_params': list_params, 'true_score': true_score}
    prediction = _posterior(optimizer, param_range=list_params, **observation)

    plot_bo_one_param_summary(optimizer, plot_dir=plot_dir, threshold=threshold,
                              **observation, **true_target, **prediction, **bayopt_params)


def test_skopt_plot(optimizer):
    from matplotlib import pyplot as plt
    from bo_utils import bayes2skopt
    from skopt.plots import plot_convergence, plot_objective
    
    optimizer = bayes2skopt(optimizer)
    plot_objective(optimizer)
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
    ap.add_argument("-nl", "--n_labels_each_class", default=10, type=int,
                    help="number of labelled points selected for each class")
    ap.add_argument("--seed", default=42, type=int)
    ap.add_argument("-u", "--utility_function", default="ucb",
                    help="in ['ucb', 'ei', 'poi']")
    ap.add_argument("-k", "--kappa", default=5.0, type=float,
                    help=("For UCB, small(1.0)->exploitation, " +
                          "large(10.0)->exploration, default 5.0"))
    ap.add_argument("-x", "--xi", default=0.025, type=float,
                    help=("For EI/POI, small(1e-4)->exploitation, " +
                          "large(1e-1)->exploration, default 0.025"))
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    # setup mlflow to trace the experiment
    mlflow.set_experiment('BO-with-Constraint-Scores-v3')
    for arg_key, arg_value in vars(args).items():
        mlflow.log_param(arg_key, arg_value)

    # extract input params
    dataset_name = args.dataset_name
    method_name = args.method_name
    score_name = args.score_name
    seed = args.seed

    # prepare directory for data, log and plot
    dataset.set_data_home("./data")
    embedding_dir = f"./embeddings/{dataset_name}/{method_name}"

    dir_pattern = f"{dataset_name}/{method_name}/{score_name}"
    plot_dir, log_dir, score_dir = [f"./{dir_name}/{dir_pattern}"
                                    for dir_name in ["plots", "logs", "scores"]]
    for dir_path in [plot_dir, log_dir, embedding_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    # prepare params for bayopt
    bayopt_params = {
        'log_dir': log_dir,
        'util_func': args.utility_function,
        'kappa': args.kappa,
        'xi': args.xi,
        'n_total_runs': args.n_total_runs,
        'n_random_inits': 5,
        'seed': args.seed
    }

    # custom preprocessing method for each dataset
    preprocessing_method = {
        'COIL20': None
    }.get(dataset_name, 'unitScale')  # default for image dataset
    X_origin, X, labels = dataset.load_dataset(dataset_name, preprocessing_method)

    # generate constraints according to the choosen strategy
    constraints = utils.generate_constraints(
        args.strategy, score_name, labels, seed=seed,
        n_constraints=args.n_constraints,
        n_labels_each_class=args.n_labels_each_class
    )

    if args.run:
        # run bayopt workflow
        best_result, optimizer = bayopt_workflow(X, constraints, method_name, score_name,
                                                 seed=seed, embedding_dir=embedding_dir,
                                                 bayopt_params=bayopt_params)
        mlflow.log_metric("score_func", best_result["target"])
        for param_name, param_value in best_result["params"].items():
            mlflow.log_metric(f"best_{param_name}", param_value)
            print("Final result: ", best_result)

    if args.plot:
        # plot true score values
        # note that the true target is not in logscale, test convert by call np.log
        list_perp_in_log_scale = utils.generate_value_range(
            min_val=2, max_val=X.shape[0]//3, range_type="log", num=200, dtype=int)
        true_score = _calculate_score(method_name, list_perp_in_log_scale,
                                      embedding_dir=embedding_dir)

        # prepare threshold value to filter the top highest scores
        threshold = {'tsne': 0.95, 'umap': 0.95, 'largevis': 0.90}[method_name]

        # load optimizer object
        log_name = f"{args.utility_function}_k{args.kappa}_xi{args.xi}_n{args.n_total_runs}"
        optimizer = joblib.load(f"{log_dir}/{log_name}.z")
        plot_bo(optimizer, list_perp_in_log_scale, true_score, plot_dir=plot_dir,
                threshold=threshold,
                bayopt_params={'util_func': args.utility_function,
                               'kappa': args.kappa, 'xi': args.xi})
        # test_skopt_plot(optimizer)

    # python bo_constraint.py -d COIL20 -m umap -nr 100 -nl 5 --seed 2029

    # python bo_constraint.py  -d COIL20 -m tsne --seed 1024 -k 5 (best = 36)

    # python bo_constraint.py  -d DIGITS -m umap --seed 42 -k 5.0 -nr 35
    # {'target': 2.1383999006985643, 'params': {'min_dist': 0.004450459827546657, 'n_neighbors': 11}}
