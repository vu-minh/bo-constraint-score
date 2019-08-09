import json
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import utils
from common.metric.dr_metrics import DRMetric


def _plot_line_with_variance(ax, list_params, score_mean, score_sigma=None):
    ax.plot(list_params, score_mean)
    if score_sigma is not None:
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


def _plot_best_param(ax, best_param_value, text_y_pos=0.0, text_align=None):
    ax.axvline(best_param_value, color='g', linestyle='--', alpha=0.5,
               marker="^", markersize=16, clip_on=False,
               markeredgecolor="#FF8200", markerfacecolor="#FF8200", markevery=100)
    if text_align is not None:
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


def _plot_score_with_best_param_and_range(ax, param_name, list_params, pivot,
                                          param_best, param_min, param_max,
                                          score_mean, score_sigma=None, text_y_pos=0):
    # horizontal line to indicate top 96% highest score
    ax.axhline(pivot, linestyle="--", alpha=0.4)

    # main line of mean_score with variance
    _plot_line_with_variance(ax, list_params, score_mean, score_sigma=score_sigma)

    # plot bestparam in the same customized xaxis
    # but do not show text if the best param is overlapped with the best range
    if param_best == param_min or param_best == param_max:
        text_align = None
    else:
        text_align = ("right" if (param_best - param_min) > (param_max - param_best)
                      else "left")
    _plot_best_param(ax, param_best, text_y_pos, text_align)

    # plot also the best param range (the top 96% scores)
    _plot_best_range(ax, param_min, param_max, param_name, text_y_pos)


def plot_scores(dataset_name, method_name, score_name, list_n_labels_values,
                param_name="", score_dir="", plot_dir="", compare_with_rnx=True):
    # prepare qij score data
    with open(f"{score_dir}/dof1.0_all.txt", "r") as in_file:
        qij_score_data = json.load(in_file)
        list_params = list(map(int, qij_score_data['list_params']))
        all_scores = qij_score_data['all_scores']

    # verify if the AUC_log_RNX score is valid
    if compare_with_rnx:
        with open(f"{score_dir}/../metrics/metrics.txt", "r") as in_file:
            metric_data = json.load(in_file)
        assert list_params == list(map(int, metric_data['list_params']))       
        rnx_score = metric_data["auc_rnx"]

    # prepare threshold value to filter the top highest scores
    threshold = {'tsne': 0.96, 'umap': 0.96, 'largevis': 0.9}[method_name]

    # prepare subplots
    n_rows = len(list_n_labels_values) + (1 if compare_with_rnx else 0)
    _, axes = plt.subplots(n_rows, 1, figsize=(9.5, 4*n_rows))

    for ax, n_labels_each_class in zip(np.array(axes).ravel(),
                                       sorted(list_n_labels_values) + ["rnx"]):
        if n_labels_each_class == "rnx":  # AUC_log_RNX score, not the qij score
            first_plot, last_plot = False, True
            title, ylabel = "$AUC_{log}RNX$ score", "metric score"
            score_mean, score_sigma = np.array(rnx_score), None
        else:
            first_plot = n_labels_each_class == min(list_n_labels_values)
            last_plot = n_labels_each_class == max(list_n_labels_values)
            title, ylabel = f"{n_labels_each_class} labels per class", "constraint score"
            scores = all_scores[str(n_labels_each_class)]
            score_mean, score_sigma = np.mean(scores, axis=0), np.std(scores, axis=0)

        ax.set_title(title, loc="left")
        ax.set_ylabel(ylabel)
        text_y_pos = min(score_mean)  # y-position to show text, e.g., value of best param

        # determine best param range and best param value
        pivot = threshold * max(score_mean)
        (best_indices, ) = np.where(score_mean > pivot)
        param_min = list_params[best_indices.min()]
        param_max = list_params[best_indices.max()]
        param_best = list_params[np.argmax(score_mean)]

        _plot_score_with_best_param_and_range(
            ax, param_name, list_params, pivot, param_best, param_min, param_max,
            score_mean=score_mean, score_sigma=score_sigma, text_y_pos=text_y_pos)

        # show label for params only for the last plot
        if last_plot:
            ax.set_xlabel(f"{param_name} in log-scale")

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
    plt.savefig(f"{plot_dir}/scores.png")
    plt.close()


def plot_quality_metrics(dataset_name, method_name, param_name="", score_dir="", plot_dir=""):
    with open(f"{score_dir}/metrics.txt", "r") as in_file:
        metrics_data = json.load(in_file)
        list_params = list(map(int, metrics_data['list_params']))

    for metric_name, metric_display_name in DRMetric.metrics_names.items():
        _, ax = plt.subplots(1, 1, figsize=(9, 4.5))
        _plot_line_with_variance(ax, list_params, score_sigma=None,
                                 score_mean=metrics_data[metric_name])
        ax.set_title(metric_display_name)
        ax.text(x=0.985, y=0.875, s=f"{dataset_name}, {method_name}",
                transform=ax.transAxes, ha="right",
                bbox=dict(edgecolor='b', facecolor='w'))
        ax.set_xlabel(f"{param_name} in log-scale")
        # ax.set_ylabel(f"{metric_display_name} score")
        ax.grid(which='major', axis='y', linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.savefig(f"{plot_dir}/{metric_name}.png")
        plt.close()    

