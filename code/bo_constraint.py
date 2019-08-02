import math
from functools import partial

from bayes_opt import BayesianOptimization
from bayes_opt.observer import JSONLogger
from bayes_opt.event import Events

import utils
from common.dataset import dataset
from run_viz import run_tsne, run_largevis, run_umap


def bayopt_workflow(X, constraints,
                    method_name: str, score_name: str,
                    seed: int=42, embedding_dir: str="",
                    bayopt_params={}):

    # value range of params in original linear scale
    min_perp, max_perp = 2, int(X.shape[0] // 3)
    start_min_dist, stop_min_dist = 0.001, 1.0

    # value range of param in log scale
    perp_range = (math.log(min_perp), math.log(max_perp))
    min_dist_range = (math.log(start_min_dist), math.log(stop_min_dist))

    # transformation rules to transform the params in log scale to linear scale
    transformation_rules = {
        # 'param_name': (cast_type, transformation_func),
        'perplexity': (int, math.exp),
        'n_neighbors': (int, math.exp),
        'min_dist': (float, math.exp)
    }

    # create 'logspace' for the parameters
    params_space = {
        'tsne': {'perplexity': perp_range},
        'largevis': {'perplexity': perp_range},
        'umap': {'n_neighbors': perp_range, 'min_dist': min_dist_range},
    }[method_name]

    # define the target function which will be maximize by BayOpt
    def target_func(**params_in_log_scale):
        '''The `params_in_log_scale` should be:
        + log(`perplexity`) for tsne and largevis,
        + log(`n_neighbors`, `min_dist`) for umap
        '''
        embedding_function = partial({
            'tsne': run_tsne,
            'umap': run_umap,
            'largevis': run_largevis
        }[method_name], X=X, seed=seed, check_log=True, embedding_dir=embedding_dir)

        # transform and cast type the param in log scale into original linear scale
        for param_name, (cast_type, transformation_func) in transformation_rules.items():
            if param_name in params_in_log_scale:
                params_in_log_scale[param_name] = cast_type(transformation_func(
                    params_in_log_scale[param_name]
                ))

        # get the embedding and calculate constraint score
        Z = embedding_function(**params_in_log_scale)
        return utils.score_embedding(Z, score_name, constraints)

    # run bayopt and transform the best param to original linear scale
    result = run_bo(target_func, params_space, **bayopt_params)
    for param_name, param_value in result['params'].items():
        cast_type, transformation_func = transformation_rules[param_name]
        result['params'][param_name] = cast_type(transformation_func(param_value))
    return result


def run_bo(target_func, params_space, seed=42, log_dir="",
           n_total_runs=15, n_random_inits=5,
           kappa=5, xi=0.025, util_func="ucb"):

    # create BO object to find max of `target_func` in the domain of param `p`
    optimizer = BayesianOptimization(
        target_func,
        params_space,
        random_state=seed,
    )

    # log the progress for plotting
    log_name = f"{util_func}_k{kappa}_xi{xi}_n{n_total_runs}"
    logger = JSONLogger(path=f"{log_dir}/{log_name}.json")
    optimizer.subscribe(Events.OPTMIZATION_STEP, logger)
    optimizer_params = dict(alpha=1e-3, n_restarts_optimizer=5, random_state=seed)

    # using `util_func`, evaluate the target function at some randomly initial points
    optimizer.maximize(acq=util_func, init_points=n_random_inits, n_iter=0, kappa=kappa, xi=xi,
                       **optimizer_params)

    # then predict the next best param to evaluate
    for i in range(n_total_runs - n_random_inits):
        optimizer.maximize(acq=util_func, init_points=0, n_iter=1, kappa=kappa, xi=xi,
                           **optimizer_params)

    # log the internal calculated scores (which is the target_function value)
    for res in optimizer.res:
        with mlflow.start_run(nested=True):
            mlflow.log_metric("target_func", res["target"])
            for param_name, param_value in res["params"].items():
                mlflow.log_param(param_name, param_value)

    # TODO PLOT
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
    ap.add_argument("-nl", "--n_labels_each_class", default=10, type=int,
                    help="number of labelled points selected for each class")
    ap.add_argument("--seed", default=2019, type=int)
    ap.add_argument("-u", "--utility_function", default="ucb",
                    help="in ['ucb', 'ei', 'poi']")
    ap.add_argument("-k", "--kappa", default=5.0, type=float,
                    help=("For UCB, small(1.0)->exploitation, " +
                          "large(10.0)->exploration, default 5.0"))
    ap.add_argument("-x", "--xi", default=0.025, type=float,
                    help=("For EI/POI, small(1e-4)->exploitation, " +
                          "large(1e-1)->exploration, default 0.025"))
    args = ap.parse_args()

    # setup mlflow to trace the experiment
    mlflow.set_experiment('BO-with-Constraint-Scores-v2')
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
    plot_dir = f"./plots/{dataset_name}/{method_name}/{score_name}"
    log_dir = f"./logs/{dataset_name}/{method_name}/{score_name}"
    for dir_path in [plot_dir, log_dir, embedding_dir]:
        print(dir_path)
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

    # run bayopt workflow
    best_result = bayopt_workflow(X, constraints, method_name, score_name,
                                  seed=seed, embedding_dir=embedding_dir,
                                  bayopt_params=bayopt_params)
    print(best_result)
    # python bo_constraint.py -d COIL20 -m umap -nr 100 -nl 5 --seed 2029
