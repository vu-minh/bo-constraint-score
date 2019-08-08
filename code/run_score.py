# run score function for all pre-calculate embeddings

import os
import json
import joblib
import collections
import matplotlib.pyplot as plt

import utils
from plot_score import plot_scores_with_std
from common.dataset import dataset
from common.metric.dr_metrics import DRMetric


# to profile with line_profiler, add @profile decoration to the target function
# and run kernprof -l script.py -with --params

# @profile
def run_qij_score(method_name, list_n_labels_values,
                  seed=42, n_repeat=1, degrees_of_freedom=1.0,
                  list_accepted_perp=None, default_min_dist=0.1, score_dir=""):
    all_embeddings = joblib.load(f"{embedding_dir}/all.z")

    for i in range(n_repeat):
        print("[Debug] n_run: ", i + 1)
        scores = collections.defaultdict(dict)
        run_index = seed + i

        for n_labels_each_class in list_n_labels_values:
            constraints = utils.generate_constraints(
                constraint_strategy="partial_labels", score_name=score_name,
                labels=labels, seed=run_index,
                n_labels_each_class=n_labels_each_class)

            if list_accepted_perp is None:
                print("# TODO deal with this case latter")
                pass

            for perp in list_accepted_perp:
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

    merge_all_score_files(method_name, list_n_labels_values, seed=seed,
                          n_repeat=n_repeat, degrees_of_freedom=degrees_of_freedom,
                          score_dir=score_dir)


def run_rnx_metric(method_name, X, score_dir=""):
    # run all other metric
    pass


def run_all_quality_metric(method_name, X, score_dir=""):
    all_embeddings = joblib.load(f"{embedding_dir}/all.z")

    pass


def merge_all_score_files(method_name, list_n_labels_values,
                          seed=42, n_repeat=1, degrees_of_freedom=1.0, score_dir=""):
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
                    help=" in ['qij', 'contrastive', 'cosine', 'rnx'], 'rnx' is John's metric")
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

    default_param_name = {
        'tsne': 'perplexity', 'largevis': 'perplexity', 'umap': 'n_neighbors'
    }[method_name]
    list_n_labels_values = [3, 5, 10, 15] if args.debug else range(2, 16)
    list_perp_in_log_scale = utils.generate_value_range(
        min_val=2, max_val=X.shape[0]//3, range_type="log", num=200, dtype=int)
    # hardcoded num of params to 200 for being consistant with default n_perp in run_viz
    print(list_perp_in_log_scale)

    if args.run:
        if score_name == "qij":
            run_qij_score(
                method_name, list_n_labels_values, seed=args.seed,
                n_repeat=args.n_repeat, degrees_of_freedom=args.degrees_of_freedom,
                list_accepted_perp=list_perp_in_log_scale if args.use_log_scale else None,
                score_dir=score_dir)
        elif score_name == "rnx":
            run_rnx_metric()
        else:
            raise ValueError(f"Invalid score name {score_name}, should be in ['qij', 'rnx']")

    if args.plot:
        # note to make big font size for plots
        plt.rcParams.update({'font.size': 20})

        plot_scores_with_std(dataset_name, method_name, score_name, list_n_labels_values,
                             param_name=default_param_name,
                             degrees_of_freedom=args.degrees_of_freedom,
                             score_dir=score_dir, plot_dir=plot_dir)

    # reproduce by running
    # python run_score.py  -d DIGITS -m umap --seed 2019 \
    #                      --use_log_scale --debug \
    #                      --run -nr 10 --plot
