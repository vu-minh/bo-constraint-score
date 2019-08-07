# run score function for all pre-calculate embeddings

import os
import json
import joblib
import collections
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from common.dataset import dataset
import utils


# to profile with line_profiler, add @profile decoration to the target function
# and run kernprof -l script.py -with --params

# @profile
def run_score(method_name, score_name, list_n_labels_values,
              seed=42, n_repeat=1, degrees_of_freedom=1.0,
              list_accepted_perp=None, default_min_dist=0.1):
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

    for n_labels_each_class, scores_i_j in all_scores.items():
        print("[Debug]#score values:", n_labels_each_class, len(scores_i_j), len(scores_i_j[0]))

    with open(f"{score_dir}/dof{degrees_of_freedom}_all.txt", "w") as out_file:
        json.dump({
            'list_params': list_params,
            'all_scores': all_scores
        }, out_file)


def _plot_line_with_variance(ax, score_mean, score_sigma, list_params):
    # plot line of mean score
    ax.plot(list_params, score_mean)

    # fill the variance (mean +- sigma)
    ax.fill_between(np.array(list_params),
                    score_mean + score_sigma, score_mean - score_sigma,
                    fc="#CCDAF1", alpha=0.5)

    # custom xaxis in log scale
    ax.set_xscale("log", basex=np.e)
    ax.set_xlim(left=min(list_params), right=max(list_params))

    # force list params to show on axias
    list_params_to_show = utils.generate_value_range(
        min_val=1.1, max_val=max(list_params), num=9, range_type="log", dtype=int)
    ax.set_xticks(list_params_to_show)
    ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())

    # custom yaxis to show only 3 score values
    ax.locator_params(axis='y', nbins=3)


def _plot_best_param(ax, best_param_value, text_y_pos=0.0, text_align="left"):
    ax.axvline(best_param_value, color='g', linestyle='--', alpha=0.5,
               marker="^", markersize=16, clip_on=False,
               markeredgecolor="#FF8200", markerfacecolor="#FF8200", markevery=100)
    ax.text(x=best_param_value, y=text_y_pos, s=str(best_param_value), ha=text_align)


def _plot_best_range(ax, param_min, param_max, param_name="", text_y_pos=0.0):
    ax.axvspan(param_min, param_max, alpha=0.12, color="orange")

    # vertical line on the left
    ax.axvline(param_min, color="orange", linestyle='--', alpha=0.4,
               marker=">", markersize=14, clip_on=False,
               markeredgecolor="orange", markerfacecolor="orange", markevery=100)
    ax.text(x=param_min, y=text_y_pos, s=str(param_min), ha="right")

    # vertical line on the right
    ax.axvline(param_max, color="orange", linestyle='--', alpha=0.4,
               marker="<", markersize=14, clip_on=False,
               markeredgecolor="orange", markerfacecolor="orange", markevery=100)
    ax.text(x=param_max, y=text_y_pos, s=str(param_max), ha="left")

    # add text to show the best range
    ax.text(x=1.0, y=1.025, transform=ax.transAxes, ha="right", va="bottom",
            s=f" best {param_name} range: [{param_min}, {param_max}]")


def plot_scores_with_std(dataset_name, method_name, score_name, list_n_labels_values,
                         degrees_of_freedom=1.0, param_name="param"):
    # prepare score data
    with open(f"{score_dir}/dof{degrees_of_freedom}_all.txt", "r") as in_file:
        json_data = json.load(in_file)

    list_params = list(map(int, json_data['list_params']))
    list_params = np.array(list_params)
    all_scores = json_data['all_scores']

    # prepare threshold value to filter the top highest scores
    threshold = {'tsne': 0.96, 'umap': 0.96, 'largevis': 0.9}[method_name]

    # prepare subplots
    n_rows = len(list_n_labels_values)
    _, axes = plt.subplots(n_rows, 1, figsize=(12, 4*n_rows))

    for ax, n_labels_each_class in zip(np.array(axes).ravel(), sorted(list_n_labels_values)):
        first_plot = n_labels_each_class == min(list_n_labels_values)
        last_plot = n_labels_each_class == max(list_n_labels_values)

        ax.set_title(f"{n_labels_each_class} labels per class", loc="left")
        scores = all_scores[str(n_labels_each_class)]
        score_mean, score_sigma = np.mean(scores, axis=0), np.std(scores, axis=0)
        text_y_pos = min(score_mean)  # y-position to show text, e.g., value of best param

        # determine best param range and best param value
        pivot = threshold * max(score_mean)
        (best_indices, ) = np.where(score_mean > pivot)
        param_min = list_params[best_indices.min()]
        param_max = list_params[best_indices.max()]
        param_best = list_params[np.argmax(score_mean)]

        # horizontal line to indicate top 96% highest score
        ax.axhline(pivot, linestyle="--", alpha=0.4)

        # main line of mean_score with variance
        _plot_line_with_variance(ax, score_mean, score_sigma, list_params)

        # plot bestparam in the same customized xaxis
        text_align = ("right" if (param_best - param_min) > (param_max - param_best)
                      else "left")
        _plot_best_param(ax, param_best, text_y_pos, text_align)

        # plot also the best param range (the top 96% scores)
        _plot_best_range(ax, param_min, param_max, param_name, text_y_pos)

        # show label for params only for the last plot
        if last_plot:
            ax.set_xlabel(f"{param_name} in log-scale")
        ax.set_ylabel('constraint score')

        # additional annotation for first plot
        if first_plot:
            # hint text for the top hightest scores horizontal line
            ax.text(x=min(list_params), y=pivot, ha="left", va="bottom", alpha=0.7,
                    s=u"\u2199" + f"{threshold} max(score)", fontsize=18, color="#0047BB")
            # hint the dataset and method name
            ax.text(x=0.985, y=0.875, s=f"{dataset_name}, {method_name}",
                    transform=ax.transAxes, ha="right",
                    bbox=dict(edgecolor='b', facecolor='w'))

    plt.tight_layout()
    plt.savefig(f"{plot_dir}/scores_with_std_dof{degrees_of_freedom}.png")
    plt.close()


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
                    help=" in ['qij', 'contrastive', 'cosine']")
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
        print(dir_path)

    default_param_name = {
        'tsne': 'perplexity', 'largevis': 'perplexity', 'umap': 'n_neighbors'
    }[method_name]
    list_n_labels_values = [3, 5, 10, 15] if args.debug else range(2, 16)
    list_perp_in_log_scale = utils.generate_value_range(
        min_val=2, max_val=X.shape[0]//3, range_type="log", num=200, dtype=int)
    # hardcoded num of params to 200 for being consistant with default n_perp in run_viz
    print(list_perp_in_log_scale)

    # note to make big font size for plots
    plt.rcParams.update({'font.size': 20})

    if args.run:
        run_score(method_name, score_name, list_n_labels_values, seed=args.seed,
                  n_repeat=args.n_repeat, degrees_of_freedom=args.degrees_of_freedom,
                  list_accepted_perp=list_perp_in_log_scale if args.use_log_scale else None)
    if args.plot:
        plot_scores_with_std(dataset_name, method_name, score_name, list_n_labels_values,
                             param_name=default_param_name,
                             degrees_of_freedom=args.degrees_of_freedom)

    # reproduce by running
    # python run_score.py  -d DIGITS -m umap --seed 2019 \
    #                      --use_log_scale --debug \
    #                      --run -nr 10 --plot
