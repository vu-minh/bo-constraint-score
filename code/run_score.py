# run score function for all pre-calculate embeddings

import os
import json
import math
import joblib
import collections
import matplotlib.pyplot as plt
from tqdm import tqdm

import utils
from plot_score import plot_scores, plot_quality_metrics
from common.dataset import dataset


# to profile with line_profiler, add @profile decoration to the target function
# and run kernprof -l script.py -with --params

# @profile
def run_qij_score(method_name, list_n_labels_values,
                  seed=42, n_repeat=1, degrees_of_freedom=1.0,
                  list_accepted_perp=None, default_min_dist=0.1,
                  embedding_dir="", score_dir=""):
    all_embeddings = joblib.load(f"{embedding_dir}/all.z")

    for i in range(n_repeat):
        print("[Debug] n_run: ", i + 1)
        scores = collections.defaultdict(dict)
        run_index = seed + i

        for n_labels_each_class in tqdm(list_n_labels_values, desc="n_labels per class"):
            constraints = utils.generate_constraints(
                constraint_strategy="partial_labels", score_name=score_name,
                labels=labels, seed=run_index,
                n_labels_each_class=n_labels_each_class)

            if list_accepted_perp is None:
                print("# TODO deal with this case latter")
                pass

            for perp in tqdm(list_accepted_perp, desc="perplexity"):
                if method_name in ['umap']:
                    key_name = f"{perp}_{default_min_dist:.4f}"
                else:
                    key_name = str(perp)
                embedding = all_embeddings[key_name]
                score = utils.score_embedding(embedding, score_name, constraints,
                                              degrees_of_freedom=degrees_of_freedom)
                scores[n_labels_each_class][str(perp)] = score

        out_name = f"{score_dir}/dof{degrees_of_freedom}_{run_index}.txt"
        with open(out_name, "w") as out_file:
            json.dump(scores, out_file)

    merge_all_score_files(list_n_labels_values, seed=seed, n_repeat=n_repeat,
                          degrees_of_freedom=degrees_of_freedom, score_dir=score_dir)


def run_all_quality_metric(X, list_perps, default_min_dist=0.1,
                           embedding_dir="", score_dir=""):
    all_embeddings = joblib.load(f"{embedding_dir}/all.z")

    # store the list of metric results for each metric name
    all_metrics = collections.defaultdict(list)
    all_metrics['list_params'] = list_perps

    for perp in tqdm(list_perps, desc="perplexity"):
        if method_name in ['umap']:
            key_name = f"{perp}_{default_min_dist:.4f}"
        else:
            key_name = str(perp)
        embedding = all_embeddings[key_name]

        metric_result = utils.calculate_all_metrics(X, embedding)
        for metric_name, metric_value in metric_result.items():
            all_metrics[metric_name].append(metric_value)

    with open(f"{score_dir}/metrics.txt", 'w') as in_file:
            json.dump(all_metrics, in_file)


def run_BIC_score(X, list_perps, score_dir="", seed=42):
    """Must re-run tsne for calculate BIC score exactly"""
    from MulticoreTSNE import MulticoreTSNE

    N = X.shape[0]
    scores = {
        'list_params': list_perps,
        'tsne_loss': [],
        'bic': []
    }

    for perp in tqdm(list_perps, desc="perplexity"):
        tsne = MulticoreTSNE(perplexity=perp, n_iter=1500, n_jobs=-1, random_state=seed,
                             n_iter_without_progress=1500, min_grad_norm=1e-32)
        tsne.fit_transform(X)
        loss = float(tsne.kl_divergence_)
        bic = 2 * loss + math.log(N) * perp / N
        scores['bic'].append(bic)
        scores['tsne_loss'].append(loss)

    with open(f"{score_dir}/BIC.txt", 'w') as in_file:
        json.dump(scores, in_file)


def merge_all_score_files(list_n_labels_values, seed=42, n_repeat=1,
                          degrees_of_freedom=1.0, score_dir=""):
    all_scores = collections.defaultdict(list)
    list_params = None

    for i in range(n_repeat):
        run_index = seed + i
        in_name = f"{score_dir}/dof{degrees_of_freedom}_{run_index}.txt"
        with open(in_name, "r") as in_file:
            scores_run_i = json.load(in_file)

        for n_labels_each_class, scores_i_j in scores_run_i.items():
            if list_params is None:
                list_params = list(scores_i_j.keys())
            elif list_params != list(scores_i_j.keys()):
                raise ValueError(f"Inconsistant list params in score files: {in_name}")

            all_scores[n_labels_each_class].append(list(scores_i_j.values()))

    for n_labels_each_class, scores_i_j in all_scores.items():
        print("[Debug]#score values:", n_labels_each_class, len(scores_i_j), len(scores_i_j[0]))

    with open(f"{score_dir}/dof{degrees_of_freedom}_all.txt", "w") as out_file:
        json.dump({
            'list_params': list_params,
            'all_scores': all_scores
        }, out_file)


if __name__ == "__main__":
    """Calculate scores for different embeddings w.r.t only one hyperparam `perplexity`.
    For umap, the default `min_dist` is set to 0.1.
    Note to repeat the score calculation (e.g. 10 times) to get the mean score and variance.
    """

    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset_name", default="")
    ap.add_argument("-m", "--method_name", default="umap",
                    help="['tsne', 'umap', 'largevis']")
    ap.add_argument("-sc", "--score_name", default="qij",
                    help=" in ['qij', 'contrastive', 'cosine', 'metrics', 'bic']")
    ap.add_argument("-nr", "--n_repeat", default=1, type=int,
                    help="number of times to repeat the score calculation")
    ap.add_argument("-nl", "--n_labels_each_class", default=5, type=int,
                    help="number of labelled points selected for each class")
    ap.add_argument("-dof", "--degrees_of_freedom", default=1.0, type=float,
                    help="degrees_of_freedom for qij_based score")
    ap.add_argument("--seed", default=2019, type=int)
    ap.add_argument("--use_log_scale", action="store_true",
                    help="use the param values in log scale")
    ap.add_argument("--debug", action="store_true", help="run score for debugging")
    ap.add_argument("--run", action="store_true", help="run score function")
    ap.add_argument("--plot", action="store_true", help="plot score function")
    args = ap.parse_args()

    dataset.set_data_home("./data")
    dataset_name = args.dataset_name
    method_name = args.method_name
    score_name = args.score_name
    X_origin, X, labels = dataset.load_dataset(dataset_name)

    embedding_dir = f"./embeddings/{dataset_name}/{method_name}"
    plot_dir = f"./plots/{dataset_name}/{method_name}"
    score_dir = f"./scores/{dataset_name}/{method_name}/{score_name}"

    for dir_path in [embedding_dir, plot_dir, score_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    default_min_dist = 1.0
    # ['0.0010', '0.0022', '0.0046', '0.0100', '0.0215', '0.0464', '0.1000', '0.2154', '0.4642', '1.0000']

    default_param_name = {
        'tsne': 'perplexity', 'largevis': 'perplexity', 'umap': 'n_neighbors'
    }[method_name]
    list_n_labels_values = [3, 5, 10, 15] if args.debug else range(2, 16)
    list_perp_in_log_scale = utils.generate_value_range(
        min_val=2, max_val=X.shape[0]//3, range_type="log", num=200, dtype=int).tolist()
    # hardcoded num of params to 200 for being consistant with default n_perp in run_viz
    print(list_perp_in_log_scale)

    if args.run:
        if score_name == "qij":
            run_qij_score(
                method_name, list_n_labels_values, seed=args.seed,
                n_repeat=args.n_repeat, degrees_of_freedom=args.degrees_of_freedom,
                list_accepted_perp=list_perp_in_log_scale if args.use_log_scale else None,
                embedding_dir=embedding_dir, score_dir=score_dir,
                default_min_dist=default_min_dist)
        elif score_name == "metrics":
            run_all_quality_metric(X, list_perps=list_perp_in_log_scale,
                                   embedding_dir=embedding_dir, score_dir=score_dir,
                                   default_min_dist=default_min_dist)
        elif score_name == "bic":
            run_BIC_score(X, list_perp_in_log_scale, score_dir, seed=args.seed)
        else:
            raise ValueError(f"Invalid score name {score_name}, should be ['qij', 'metrics']")

    if args.plot:
        # note to make big font size for plots
        plt.rcParams.update({'font.size': 24})

        if score_name == "qij":
            plot_scores(dataset_name, method_name, score_name, list_n_labels_values,
                        param_name=default_param_name, score_dir=score_dir, plot_dir=plot_dir,
                        compare_with_rnx=False)
        elif score_name == "metrics":
            plot_quality_metrics(dataset_name, method_name, param_name=default_param_name,
                                 score_dir=score_dir, plot_dir=plot_dir)
        else:
            raise ValueError(f"Invalid score name {score_name}, should be ['qij', 'metrics']")

    # TODO : change --user_log_scale to --disable_log_scale
