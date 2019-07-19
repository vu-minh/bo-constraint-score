import numpy as np
from functools import partial
from bayes_opt import BayesianOptimization


from common.dataset import dataset
from common.dataset import constraint
import constraint_score
from bo_plot import plot_gp_one_param

from MulticoreTSNE import MulticoreTSNE
import umap


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


def contrastive_score(Z, contrastive_constraints):
    return constraint_score.contrastive_score(Z, contrastive_constraints)


def qij_score(Z, sim_links, dis_links, degrees_of_freedom=0.5):
    Q = constraint_score.calculate_Q(Z, degrees_of_freedom)
    final_score, sim_scores, dis_scores = constraint_score.qij_based_scores(
        Q, sim_links, dis_links, normalized=False
    )
    return final_score


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

    return optimizer.max


if __name__ == "__main__":
    import argparse
    import mlflow
    import os

    ap = argparse.ArgumentParser()
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
