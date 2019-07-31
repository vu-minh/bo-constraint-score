# run score function for all pre-calculate embeddings

import os
import json
import joblib
import collections
import numpy as np
import matplotlib.pyplot as plt
from common.dataset import dataset
import utils


# to profile with line_profiler, add @profile decoration to the target function
# and run kernprof -l script.py -with --params

# @profile
def run_score(method_name, score_name, list_n_labels_values,
              seed=42, n_repeat=1, degrees_of_freedom=1.0):
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

            for param_value, embedding in enumerate(all_embeddings):
                if embedding is not None:
                    score = utils.score_embedding(embedding, score_name, constraints,
                                                  degrees_of_freedom=degrees_of_freedom)
                    scores[n_labels_each_class][param_value] = score

        out_name = f"{score_dir}/dof{degrees_of_freedom}_{run_index}.txt"
        with open(out_name, "w") as out_file:
            json.dump(scores, out_file)

    merge_all_score_files(method_name, score_name, list_n_labels_values, seed=seed,
                          n_repeat=n_repeat, degrees_of_freedom=degrees_of_freedom)


def merge_all_score_files(method_name, score_name, list_n_labels_values,
                          seed=42, n_repeat=1, degrees_of_freedom=1.0):
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

    print(len(list_params))
    for n_labels_each_class, scores_i_j in all_scores.items():
        print(n_labels_each_class, len(scores_i_j), len(scores_i_j[0]))

    with open(f"{score_dir}/dof{degrees_of_freedom}_all.txt", "w") as out_file:
        json.dump({
            'list_params': list_params,
            'all_scores': all_scores
        }, out_file)


def plot_scores_with_std(method_name, score_name, list_n_labels_values,
                         seed=42, n_repeat=1, degrees_of_freedom=1.0):
    with open(f"{score_dir}/dof{degrees_of_freedom}_all.txt", "r") as in_file:
        json_data = json.load(in_file)

    list_params = json_data['list_params']
    all_scores = json_data['all_scores']

    def _plot_scores_with_variance(ax, scores):
        mean = np.mean(scores, axis=0)
        sigma = np.std(scores, axis=0)
        ax.plot(mean)
        ax.fill_between(np.array(list_params), mean + 2 * sigma, mean - 2 * sigma,
                        fc="#CCDAF1", alpha=0.5)
        ax.axvline(np.argmax(mean), color='g', linestyle='--', alpha=0.5)

    n_rows = len(all_scores)
    _, axes = plt.subplots(n_rows, 1, figsize=(16, 4*n_rows))
    for ax, (n_labels_each_class, scores) in zip(np.array(axes).ravel(), all_scores.items()):
        print(n_labels_each_class, len(scores))
        _plot_scores_with_variance(ax, np.array(scores))
        # ax.set_title(f"{n_labels_each_class} labels each class")

    plt.tight_layout()
    plt.savefig(f"{plot_dir}/scores_with_std_dof{degrees_of_freedom}.png")
    plt.close()


def plot_scores(method_name, score_name, list_n_labels_values,
                seed=42, n_repeat=1, degrees_of_freedom=1.0):
    with open(f"{score_dir}/dof{degrees_of_freedom}_all.txt", "r") as in_file:
        json_data = json.load(in_file)

    list_params = json_data['list_params']
    all_scores = json_data['all_scores']

    def _plot_scores_with_variance(ax, scores):
        mean = np.mean(scores, axis=0)
        sigma = np.std(scores, axis=0)
        ax.plot(mean)
        ax.fill_between(np.array(list_params), mean + 2 * sigma, mean - 2 * sigma,
                        fc="#CCDAF1", alpha=0.5)
        ax.axvline(np.argmax(mean), color='g', linestyle='--', alpha=0.5)

    _, ax = plt.subplots(1, 1, figsize=(16, 8))
    for n_labels_each_class, scores in all_scores.items():
        _plot_scores_with_variance(ax, np.array(scores))

    plt.tight_layout()
    plt.savefig(f"{plot_dir}/all_scores_dof{degrees_of_freedom}.png")
    plt.close()


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset_name", default="")
    ap.add_argument("-m", "--method_name", default="umap",
                    help="['tsne', 'umap', 'largevis', 'TODO-umap-n-params']")
    ap.add_argument("-sc", "--score_name", default="qij",
                    help=" in ['qij', 'contrastive', 'cosine']")
    ap.add_argument("-nr", "--n_repeat", default=1, type=int,
                    help="number of times to repeat the score calculation")
    ap.add_argument("-nl", "--n_labels_each_class", default=5, type=int,
                    help="number of labelled points selected for each class")
    ap.add_argument("-dof", "--degrees_of_freedom", default=1.0, type=float,
                    help="degrees_of_freedom for qij_based score")
    ap.add_argument("--seed", default=2019, type=int)
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
        print(dir_path)

    list_n_labels_values = [5] if args.debug else range(2, 16)

    if args.run:
        run_score(method_name, score_name, list_n_labels_values, seed=args.seed,
                  n_repeat=args.n_repeat, degrees_of_freedom=args.degrees_of_freedom)
        merge_all_score_files(method_name, score_name, list_n_labels_values, seed=args.seed,
                              n_repeat=args.n_repeat,
                              degrees_of_freedom=args.degrees_of_freedom)

    if args.plot:
        plot_scores_with_std(method_name, score_name, list_n_labels_values, seed=args.seed,
                             n_repeat=args.n_repeat,
                             degrees_of_freedom=args.degrees_of_freedom)
