# import joblib
import datetime
import math
from functools import partial

import numpy as np
# import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import gridspec

from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction

from common.dataset import dataset
from common.dataset import constraint
# from common.metric.dr_metrics import DRMetric
# from constraint_app import data_filter

from sklearn import preprocessing
from MulticoreTSNE import MulticoreTSNE
import umap

import importlib
constraint_score = importlib.import_module("constraint-score")


def generate_constraints(score_name, n_constraints):
    return {
        "qij": {
            "sim_links": constraint.gen_similar_links(
                labels, n_constraints, include_link_type=True, seed=rnd_seed),
            "dis_links": constraint.gen_dissimilar_links(
                labels, n_constraints, include_link_type=True, seed=rnd_seed)
        },
        "contrastive": {
            "contrastive_constraints": constraint.generate_contrastive_constraints(
                labels, n_constraints, seed=rnd_seed)
        }
    }[score_name]


def contrastive_score(y, contrastive_constraints):
    eps = 1e-32
    y = preprocessing.normalize(y)

    score = 0.0
    for fx, fx_positive, fx_negative in y[contrastive_constraints]:
        numerator = math.exp(np.dot(fx, fx_positive))
        denominator = math.exp(np.dot(fx, fx_positive)) + math.exp(np.dot(fx, fx_negative))
        score += math.log(numerator / (denominator + eps))
    return score / len(contrastive_constraints)


def qij_score(Z, sim_links, dis_links, degrees_of_freedom=0.5):
    Q = constraint_score.calculate_Q(Z, degrees_of_freedom)
    final_score, sim_scores, dis_scores = constraint_score.qij_based_scores(
        Q, sim_links, dis_links, normalized=False
    )
    return final_score


def posterior(optimizer, x_obs, y_obs, grid):
    optimizer._gp.fit(x_obs, y_obs)

    mu, sigma = optimizer._gp.predict(grid, return_std=True)
    return mu, sigma


def plot_gp(optimizer, x, y, util_func="ucb", kappa=5, xi=0.01):
    plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
    axis = plt.subplot(gs[0])
    acq = plt.subplot(gs[1])

    x_obs = np.array([[res["params"]["p"]] for res in optimizer.res])
    y_obs = np.array([res["target"] for res in optimizer.res])

    steps = len(optimizer.space)
    current_max_target_function = optimizer.max["target"]
    current_best_param = optimizer.max["params"]["p"]

    mu, sigma = posterior(optimizer, x_obs, y_obs, x)
    # axis.plot(x, y, linewidth=3, label="Target")
    axis.plot(x_obs.flatten(), y_obs, "D", markersize=8, label="Observations", color="r")
    axis.plot(x, mu, "--", color="k", label="Prediction")

    axis.fill(
        np.concatenate([x, x[::-1]]),
        np.concatenate([mu - 1.9600 * sigma, (mu + 1.9600 * sigma)[::-1]]),
        alpha=0.25,
        fc="c",
        ec="None",
        label="95% confidence interval",
    )

    # axis.set_xlim((x_obs.min(), x_obs.max()))
    # TODO 20190614 if the score is negative, the current ylim will trim out some range.
    # a workaround is to add sign of the value, but not tested
    axis.set_ylim((0.85 * y_obs.min() * np.sign(y_obs.min()),
                   1.15 * y_obs.max() * np.sign(y_obs.max())))
    axis.set_ylabel("tsne_with_metric_and_constraint", fontdict={"size": 16})

    utility_function = UtilityFunction(kind=util_func, kappa=kappa, xi=xi)
    utility = utility_function.utility(x, optimizer._gp, y_max=current_max_target_function)

    acq.plot(x, utility, label=f"Utility Function ({util_func})", color="purple")
    acq.plot(
        x[np.argmax(utility)],
        np.max(utility),
        "*",
        markersize=15,
        label="Next Best Guess",
        markerfacecolor="gold",
        markeredgecolor="k",
        markeredgewidth=1,
    )
    # acq.set_xlim((x_obs.min(), x_obs.max()))
    # acq.set_ylim((0, np.max(utility) + 0.5))
    acq.set_ylabel(f"Utility ({util_func})", fontdict={"size": 16})
    acq.set_xlabel("perplexity", fontdict={"size": 16})

    axis.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.0)
    acq.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.0)

    # debug next best guess
    next_best_guess_param = x[np.argmax(utility)]
    acq.set_title(f"Next best guess param: {next_best_guess_param}", fontdict={"size": 16})

    # draw indicator vline @ the next perplexity
    acq.axvline(next_best_guess_param, color='g', linestyle='--', alpha=0.4)
    axis.axvline(next_best_guess_param, color='g', linestyle='--', alpha=0.4)
    # draw indicator hline @ the current  max value of the  target function
    axis.axhline([current_max_target_function], color='r', linestyle='--', alpha=0.4)

    debug_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    debug_method_name = {
        "ucb": f"ucb_kappa{kappa}",
        "ei": f"ei_xi{xi}",
        "poi": f"poi_xi{xi}"
    }[util_func]

    axis.set_title(f"Figure created @ {debug_time}", size=12)
    plt.suptitle(
        f"GP ({debug_method_name} utility function) after {steps} steps"
        f" with best predicted perlexity = {current_best_param:.2f}", size=20)
    fig_name = (f"./{plot_dir}/{debug_method_name}"
                f"_constraint{constraint_proportion}"
                f"_{dataset_name}_step{steps}.png")
    plt.savefig(fig_name, bbox_inches="tight")
    mlflow.log_artifact(fig_name)


def score_embedding(Z, score_name, constraints):
    score_func = {
        "contrastive": partial(contrastive_score, **constraints),
        "qij": partial(qij_score, **constraints)
    }[score_name]
    return score_func(Z)


def target_function(method_name, score_name, constraints, p):
    method = {
        "tsne": MulticoreTSNE(perplexity=p, n_iter=1000, random_state=2019, n_jobs=3,
                              n_iter_without_progress=1000, min_grad_norm=1e-32),
        "umap": umap.UMAP(n_neighbors=int(p))
    }[method_name]
    Z = method.fit_transform(X)
    score = score_embedding(Z, score_name, constraints)
    return score


def run_bo(target_func,
           n_total_runs=15, n_random_inits=5,
           kappa=5, xi=0.025, util_func="ucb"):
    perp_range = np.array(list(range(2, X.shape[0] // 3)))
    true_target_values = None

    # create BO object to find max of `target_func` in the domain of param `p`
    optimizer = BayesianOptimization(
        target_func,
        {"p": (2, X.shape[0] // 3)},
        random_state=rnd_seed,
    )

    # using `util_func`, evaluate the target function at some randomly initial points
    optimizer.maximize(acq=util_func, init_points=n_random_inits, n_iter=0,
                       kappa=kappa, xi=xi)
    plot_gp(optimizer, x=perp_range.reshape(-1, 1), y=true_target_values,
            util_func=util_func,
            kappa=kappa, xi=xi)

    # then predict the next best param to evaluate
    for i in range(n_total_runs - n_random_inits):
        optimizer.maximize(acq=util_func, init_points=0, n_iter=1,
                           kappa=kappa, xi=xi)
        print("Current max: ", optimizer.max)
        plot_gp(optimizer, x=perp_range.reshape(-1, 1), y=true_target_values,
                util_func=util_func,
                kappa=kappa, xi=xi)

    # log the internal calculated scores (which is the target_function value)
    for res in optimizer.res:
        with mlflow.start_run(nested=True):
            mlflow.log_param("p", res["params"]["p"])
            mlflow.log_metric("target_func", res["target"])

    return optimizer.max


if __name__ == "__main__":
    import argparse
    import mlflow
    import os

    ap = argparse.ArgumentParser()
    # ap.add_argument("-x", "--tracking", action="store_true", help="Tracking with MLFlow?")
    ap.add_argument("-d", "--dataset_name", default="")
    ap.add_argument("-m", "--method_name", default="tsne",
                    help=" in ['tsne', 'umap']")
    ap.add_argument("-s", "--score_name", default="qij",
                    help=" in ['qij', 'contrastive', 'cosine', 'cosine_ratio']")
    ap.add_argument("-n", "--n_constraints", default=50, type=int)
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
    X, _, labels = dataset.load_dataset(dataset_name)
    X /= 255.0

    # add preprocessing with PCA
    # from sklearn.decomposition import PCA
    # X = PCA(n_components=100).fit_transform(X)
    print(X.shape, labels.shape)

    method_name = args.method_name
    score_name = args.score_name
    n_constraints = args.n_constraints
    rnd_seed = int(args.random_seed)
    constraint_proportion = args.constraint_proportion

    plot_dir = f"./plots/{dataset_name}/{method_name}/{score_name}"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    constraints = generate_constraints(score_name, n_constraints)
    target_function_wrapper = partial(target_function,
                                      method_name, score_name, constraints)
    best_result = run_bo(target_func=target_function_wrapper)
    print(best_result)
