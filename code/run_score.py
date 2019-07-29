# run score function for all pre-calculate embeddings

import os
import json
import joblib
import collections
import numpy as np
import matplotlib.pyplot as plt
from common.dataset import dataset
import utils

# nested_dict = lambda: collections.defaultdict(nested_dict)


def run_score(method_name, score_name, list_n_labels_values, seed=None, degrees_of_freedom=1.0):
    all_embeddings = joblib.load(f"{embedding_dir}/all.z")
    scores = collections.defaultdict(dict)

    for n_labels_each_class in list_n_labels_values:
        constraints = utils.generate_constraints(
            constraint_strategy="partial_labels", score_name=score_name,
            labels=labels, seed=seed,
            n_labels_each_class=n_labels_each_class)

        for param_value, embedding in enumerate(all_embeddings):
            if embedding is not None:
                score = utils.score_embedding(embedding, score_name, constraints,
                                              degrees_of_freedom=degrees_of_freedom)
                scores[n_labels_each_class][param_value] = score

    with open(f"{score_dir}/raw_scores_dof{degrees_of_freedom}.txt", "w") as out_file:
        json.dump(scores, out_file)


def plot_score(method_name, score_name, list_n_labels_values, degrees_of_freedom=1.0):
    with open(f"{score_dir}/raw_scores_dof{degrees_of_freedom}.txt", "r") as in_file:
        all_scores = json.load(in_file)

    plt.figure(figsize=(10, 5))
    for n_labels_each_class, scores in all_scores.items():
        plt.plot(list(scores.values()), alpha=0.5,
                 label=f"n_labels_each_class={n_labels_each_class}")
        best_param = np.argmax(list(scores.values()))
        print(f"With n_labels_each_class={n_labels_each_class}, best_param={best_param}")
        plt.axvline(x=best_param, linestyle="--", alpha=0.4)

    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/all_scores_dof{degrees_of_freedom}.png")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset_name", default="")
    ap.add_argument("-m", "--method_name", default="umap",
                    help="['tsne', 'umap', 'largevis', 'TODO-umap-n-params']")
    ap.add_argument("-sc", "--score_name", default="qij",
                    help=" in ['qij', 'contrastive', 'cosine']")
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

    list_n_labels_values = [5] if args.debug else range(2, 16)

    if args.run:
        run_score(method_name, score_name, list_n_labels_values,
                  seed=args.seed, degrees_of_freedom=args.degrees_of_freedom)

    if args.plot:
        plot_score(method_name, score_dir, list_n_labels_values,
                   degrees_of_freedom=args.degrees_of_freedom)
