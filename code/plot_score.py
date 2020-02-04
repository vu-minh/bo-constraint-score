import json
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import utils
from common.metric.dr_metrics import DRMetric
from common.plot.plot_utils import draw_seperator_between_subplots


def _plot_line_with_variance(
    ax,
    list_params,
    score_mean,
    score_sigma=None,
    nbins_x=6,
    nbins_y=3,
    linewidth=1.5,
    color="#1f77b4",
):
    ax.plot(list_params, score_mean, linewidth=linewidth, color=color)
    if score_sigma is not None:
        # fill the variance (mean +- sigma)
        ax.fill_between(
            np.array(list_params),
            score_mean + score_sigma,
            score_mean - score_sigma,
            facecolor="#CCDAF1",
            alpha=0.5,
        )

    # custom xaxis in log scale
    ax.set_xscale("log", basex=np.e)
    ax.set_xlim(left=min(list_params), right=max(list_params))

    # force list params to show on axias
    list_params_to_show = utils.generate_value_range(
        min_val=1.1, max_val=max(list_params), num=nbins_x, range_type="log", dtype=int
    )
    ax.set_xticks(list_params_to_show)
    ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())

    # custom yaxis to show only 3 score values
    ax.locator_params(axis="y", nbins=nbins_y)


def _plot_best_param(ax, best_param_value, text_y_pos=0.0, text_align=None):
    ax.axvline(
        best_param_value,
        color="g",
        linestyle="--",
        alpha=0.5,
        marker="^",
        markersize=16,
        clip_on=False,
        markeredgecolor="#FF8200",
        markerfacecolor="#FF8200",
        markevery=100,
    )
    if text_align is not None:
        ax.text(x=best_param_value, y=text_y_pos, s=str(best_param_value), ha=text_align)


def _plot_best_range(ax, param_min, param_max, text_y_pos=0.0, best_param=None):
    ax.axvspan(param_min, param_max, alpha=0.12, color="orange")

    # vertical line on the left
    ax.axvline(
        param_min,
        color="orange",
        linestyle="--",
        alpha=0.6,
        marker=">",
        markersize=14,
        clip_on=False,
        markeredgecolor="orange",
        markerfacecolor="orange",
        markevery=100,
    )
    ax.text(x=param_min, y=text_y_pos, s=str(param_min), ha="right")

    # vertical line on the right
    ax.axvline(
        param_max,
        color="orange",
        linestyle="--",
        alpha=0.6,
        marker="<",
        markersize=14,
        clip_on=False,
        markeredgecolor="orange",
        markerfacecolor="orange",
        markevery=100,
    )
    ax.text(x=param_max, y=text_y_pos, s=str(param_max), ha="left")

    # add text to show the best range on top right of figure
    ax.text(
        x=1.0,
        y=1.065,
        transform=ax.transAxes,
        ha="right",
        va="center",
        s=f"[{param_min}, {param_max}]" + ("" if best_param is None else f", ({best_param})"),
        color="#0047BB",
    )


def _plot_score_with_best_param_and_range(
    ax,
    list_params,
    pivot,
    param_best,
    param_min,
    param_max,
    score_mean,
    score_sigma=None,
    text_y_pos=0,
):
    # horizontal line to indicate top 96% highest score
    ax.axhline(pivot, linestyle="--", alpha=0.4)

    # main line of mean_score with variance
    _plot_line_with_variance(ax, list_params, score_mean, score_sigma=score_sigma)

    # plot bestparam in the same customized xaxis
    # but do not show text if the best param is overlapped with the best range
    if param_best == param_min or param_best == param_max:
        text_align = None
    else:
        text_align = "right" if (param_best - param_min) > (param_max - param_best) else "left"
    _plot_best_param(ax, param_best, text_y_pos, text_align)

    # # plot also the best param range (the top 96% scores)
    # show_best_param_on_top = text_y_pos < 10
    # _plot_best_range(
    #     ax,
    #     param_min,
    #     param_max,
    #     text_y_pos,
    #     best_param=None if not show_best_param_on_top else param_best,
    # )


def plot_scores(
    dataset_name,
    method_name,
    score_name,
    list_n_labels_values,
    param_name="",
    score_dir="",
    plot_dir="",
    compare_with_rnx=False,
    show_best_range=False,
):
    # prepare qij score data
    with open(f"{score_dir}/dof1.0_all.txt", "r") as in_file:
        qij_score_data = json.load(in_file)
        list_params = list(map(int, qij_score_data["list_params"]))
        all_scores = qij_score_data["all_scores"]

    # verify if the AUC_log_RNX score is valid
    if compare_with_rnx:
        with open(f"{score_dir}/../metrics/metrics.txt", "r") as in_file:
            metric_data = json.load(in_file)
        assert list_params == list(map(int, metric_data["list_params"]))
        rnx_score = metric_data["auc_rnx"]

    # prepare threshold value to filter the top highest scores
    threshold = {"tsne": 0.96, "umap": 0.96, "largevis": 0.90}[method_name]

    # prepare subplots
    n_rows = len(list_n_labels_values) + (1 if compare_with_rnx else 0)
    fig, axes = plt.subplots(n_rows, 1, figsize=(7.5, 4.75 * n_rows))

    for ax, n_labels_each_class in zip(
        np.array(axes).ravel(), sorted(list_n_labels_values) + [99]
    ):
        first_plot = n_labels_each_class == min(list_n_labels_values)
        last_plot = n_labels_each_class == (
            99 if compare_with_rnx else max(list_n_labels_values)
        )
        if n_labels_each_class == 99:  # AUC_log_RNX score, not the qij score
            title, ylabel = "$AUC_{log}RNX$ score", "metric score"
            score_mean, score_sigma = np.array(rnx_score), None
            text_y_pos = min(rnx_score)
        else:
            title, ylabel = (
                f"{n_labels_each_class} labels per class",
                "constraint score",
            )
            scores = all_scores[str(n_labels_each_class)]
            score_mean, score_sigma = np.mean(scores, axis=0), np.std(scores, axis=0)
            text_y_pos = min(score_mean - score_sigma)

        ax.set_title(title, loc="left")
        # ax.set_ylabel(ylabel) # e.g. ylabel = "constraint score"
        # ax.set_ylim(top=1.15*max(score_mean), bottom=0.8*min(score_mean))

        if show_best_range:
            # determine best param range and best param value
            pivot = threshold * max(score_mean)
            (best_indices,) = np.where(score_mean > pivot)
            param_min = list_params[best_indices.min()]
            param_max = list_params[best_indices.max()]
            param_best = list_params[np.argmax(score_mean)]

            _plot_score_with_best_param_and_range(
                ax,
                list_params,
                pivot,
                param_best,
                param_min,
                param_max,
                score_mean=score_mean,
                score_sigma=score_sigma,
                text_y_pos=text_y_pos,
            )
        else:
            # main line of mean_score with variance
            _plot_line_with_variance(ax, list_params, score_mean, score_sigma)

        # additional annotation for the first plot and the last plot
        if first_plot:
            # hint the dataset and method name
            ax.text(
                x=0.985,
                y=0.865,
                # s=f"{dataset_name}, {method_name}",
                s=dataset_name,
                transform=ax.transAxes,
                ha="right",
                bbox=dict(edgecolor="b", facecolor="w"),
            )
            if show_best_range:
                # hint text for the top hightest scores horizontal line
                ax.text(
                    x=min(list_params),
                    y=pivot * 1.01,
                    ha="left",
                    va="bottom",
                    s="\u2199" + f"{threshold} max(score)",
                    color="#0047BB",
                )
        if last_plot:
            # hint xlabel param in log-scale
            ax.set_xlabel(f"{param_name} in log-scale")
            if show_best_range:
                # hint text for the top hightest scores horizontal line
                ax.text(
                    x=max(list_params),
                    y=pivot * 1.01,
                    ha="right",
                    va="bottom",
                    s=f"{threshold} max(score)" + "\u2198",
                    color="#0047BB",
                )
        # make the border grey
        utils.change_border(ax, width=0.1, color="0.5", hide_axis=False)
    fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    fig.savefig(f"{plot_dir}/scores.pdf")
    plt.close(fig)


def plot_quality_metrics(dataset_name, method_name, param_name="", score_dir="", plot_dir=""):
    with open(f"{score_dir}/metrics.txt", "r") as in_file:
        metrics_data = json.load(in_file)
        list_params = list(map(int, metrics_data["list_params"]))

    for metric_name, metric_display_name in DRMetric.metrics_names.items():
        _, ax = plt.subplots(1, 1, figsize=(9, 4.5))
        _plot_line_with_variance(
            ax, list_params, score_sigma=None, score_mean=metrics_data[metric_name]
        )
        ax.set_title(metric_display_name)
        ax.text(
            x=0.985,
            y=0.875,
            s=f"{dataset_name}, {method_name}",
            transform=ax.transAxes,
            ha="right",
            bbox=dict(edgecolor="b", facecolor="w"),
        )
        ax.set_xlabel(f"{param_name} in log-scale")
        # ax.set_ylabel(f"{metric_display_name} score")
        ax.grid(which="major", axis="y", linestyle="--", alpha=0.5)

        plt.tight_layout()
        plt.savefig(f"{plot_dir}/{metric_name}.png")
        plt.close()


def plot_bic_scores(dataset_name, method_name, param_name="", score_dir="", plot_dir=""):
    with open(f"{score_dir}/BIC.txt", "r") as in_file:
        bic_data = json.load(in_file)
        list_params = list(map(int, bic_data["list_params"]))

    for score_key_name, score_data in bic_data.items():
        if score_key_name == "list_params":
            continue

        _, ax = plt.subplots(1, 1, figsize=(9, 4.5))

        # determine best param range and best param value
        score_data = np.array(score_data)
        # filter the point above the min value with a margin of 0.04
        threshold = 1.04

        pivot = threshold * min(score_data)
        (best_indices,) = np.where(score_data < pivot)
        param_min = list_params[best_indices.min()]
        param_max = list_params[best_indices.max()]
        param_best = list_params[np.argmin(score_data)]

        _plot_score_with_best_param_and_range(
            ax,
            list_params,
            pivot,
            param_best,
            param_min,
            param_max,
            score_mean=score_data,
            score_sigma=None,
            text_y_pos=0,
        )
        ax.set_title(score_key_name)

        plt.tight_layout()
        plt.savefig(f"{plot_dir}/{score_key_name}.png")
        plt.close()


def plot_compare_qij_rnx_bic(
    dataset_name,
    n_labels_each_class=10,
    threshold=0.96,
    param_name="",
    score_dir="",
    plot_dir="",
    list_score_names=["Constraint score", "$AUC_{log}RNX$", "BIC"],
    show_range=True,
):
    # prepare subplots
    n_rows = len(list_score_names)
    fig, axes = plt.subplots(n_rows, 1, figsize=(6, 4 * n_rows))

    for i, (ax, title) in enumerate(zip(axes.ravel(), list_score_names)):
        ax.set_title(title, loc="left")

        # prepare score data
        if title == "BIC":
            with open(f"{score_dir}/../bic/BIC.txt", "r") as in_file_bic:
                bic_data = json.load(in_file_bic)
                list_params = list(map(int, bic_data["list_params"]))
                score_data = np.array(bic_data["bic"])

        if title == "$AUC_{log}RNX$":
            with open(f"{score_dir}/../metrics/metrics.txt", "r") as in_file_metric:
                metrics_data = json.load(in_file_metric)
                list_params = list(map(int, metrics_data["list_params"]))
                score_data = np.array(metrics_data["auc_rnx"])

        if title == "$f_{score}$":
            # get qij score for 10 labeled points per class
            with open(f"{score_dir}/../qij/dof1.0_all.txt", "r") as in_file_qij:
                qij_score_data = json.load(in_file_qij)
                list_params = list(map(int, qij_score_data["list_params"]))
                all_scores = qij_score_data["all_scores"]
                score_data = np.mean(all_scores[str(n_labels_each_class)], axis=0)

        if show_range:
            # find best param range
            if title == "BIC":  # need to find the min
                pivot = (1.0 + (1.0 - threshold)) * min(score_data)
                (best_indices,) = np.where(score_data < pivot)
                param_best = list_params[np.argmin(score_data)]
            else:  # need to find the max
                pivot = threshold * score_data.max()
                (best_indices,) = np.where(score_data > pivot)
                param_best = list_params[np.argmax(score_data)]
            param_min = list_params[best_indices.min()]
            param_max = list_params[best_indices.max()]

            _plot_score_with_best_param_and_range(
                ax,
                list_params,
                pivot,
                param_best,
                param_min,
                param_max,
                score_mean=score_data,
                score_sigma=None,
                text_y_pos=-10,
            )
        else:
            _plot_line_with_variance(ax, list_params, score_data, score_sigma=None)

        # show title for last subplot
        if i == len(list_score_names) - 1:
            ax.set_xlabel("perplexity in log-scale")

        # make the border grey
        utils.change_border(ax, width=0.1, color="0.5", hide_axis=False)

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.4, top=0.96, bottom=0.075, left=0.15, right=0.94)
    fig.savefig(f"{plot_dir}/plot_compare.pdf")
    plt.close(fig)


def plot_all_score_all_method_all_dataset(
    list_datasets=[],
    list_methods=[],
    score_root_dir="",
    plot_root_dir="",
    n_labels_each_class=10,
):
    """Plot f_score for 6 datasets and 3 methods.
        | D1 | .... | D6 |
    tsne| f1 | .....| f6 |
    umap| ...            |
    lvis| ...            |
    """
    # re-set small fontsize (for tick labels)
    # plt.rcParams.update({"font.size": 18})

    n_rows = len(list_methods)  # + 1
    n_cols = len(list_datasets)  # + 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5.5, n_rows * 3.5))

    # deco line color for different methods
    line_colors = ["#ff7f0e", "#2ca02c", "#1f77b4"]

    # for each dataset and each method, load the corresponding score file
    for col_idx, dataset_name in enumerate(list_datasets):
        for row_idx, method_name in enumerate(list_methods):
            # load the score file
            score_file_name = f"{score_root_dir}/{dataset_name}/{method_name}/qij/dof1.0_42.txt"
            try:
                with open(score_file_name) as in_file:
                    score_data = json.load(in_file).get(str(n_labels_each_class), None)
                    if score_data is None:
                        continue
                    list_params = list(map(int, score_data.keys()))
                    score_values = list(map(float, score_data.values()))
            except FileNotFoundError as fnf_error:
                print(fnf_error)
                list_params, score_values = [1], [1]

            # hardcoded method_name "umap" to "umap1"
            if method_name == "umap":
                method_name = "umap1"

            # get axis by index
            ax = axes[row_idx, col_idx]
            ax.grid(axis="x", linestyle=":", linewidth=0.5)

            # do plot the score values
            _plot_line_with_variance(
                ax,
                list_params,
                score_mean=score_values,
                score_sigma=None,
                nbins_x=5,
                nbins_y=2,
                color=line_colors[row_idx],
                linewidth=1.5,
            )

            # show x-axis label in the middle of the plot
            if col_idx == len(list_datasets) // 2 - 1:
                ax.set_xlabel(
                    f"{utils.get_param_display_name(method_name)} in log-scale", fontsize=28
                )

            # show dataset name, method name
            if row_idx == 0:
                ax.set_title(utils.get_dataset_display_name(dataset_name), fontsize=28)
            if col_idx == 0:
                ax.set_ylabel(
                    utils.get_method_display_name(method_name),
                    {"color": line_colors[row_idx]},
                    fontsize=28,
                    rotation="horizontal",
                    ha="right",
                )
                ax.text(0.05, 0.85, "$f_{score}$", transform=ax.transAxes, fontsize=22)

            # make the border grey
            utils.change_border(ax, width=0.1, color="0.5", hide_axis=False)

    fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    draw_seperator_between_subplots(fig, axes, color="gray", linestyle="--", linewidth=1.0)
    fig.subplots_adjust(hspace=0.6)
    fig.savefig(f"{plot_root_dir}/all_scores_all_methods.pdf")
    plt.close(fig)


def plot_kl_loss(list_datasets=[], score_root_dir="", plot_root_dir=""):
    from matplotlib.ticker import FuncFormatter

    def log_e_format(x, pos):
        """The two args are the value and tick position.
        Label ticks with the product of the exponentiation"""
        return int(x) + 1

    fig, ax = plt.subplots(figsize=(10, 6))
    markers = ["*", "s", "o", "p", "d", "^"]
    linestyles = ["-", "--", "-.", "--", "-.", "-"]

    for i, dataset_name in enumerate(list_datasets):
        score_dir = f"{score_root_dir}/{dataset_name}/tsne/bic"
        with open(f"{score_dir}/BIC.txt", "r") as in_file:
            raw_data = json.load(in_file)
            list_params = list(map(int, raw_data.get("list_params")))
            losses = raw_data.get("tsne_loss")

        ax.semilogx(
            list_params,
            losses,
            label=utils.get_dataset_display_name(dataset_name),
            basex=np.e,
            marker=markers[i],
            linestyle=linestyles[i],
            markevery=[2],
            markersize=8,
        )
        ax.xaxis.set_major_formatter(FuncFormatter(log_e_format))
        ax.set_xlabel("perplexity in log-scale")
        ax.set_ylabel("KL loss")

        # make the border grey
        utils.change_border(ax, width=0.1, color="0.5", hide_axis=False)

    ax.set_xlim(left=2, right=1100)  # [2, N//3]
    ax.yaxis.grid(linestyle="--")
    plt.legend(loc="upper right", fontsize=18)
    fig.tight_layout()
    fig.savefig(f"{plot_root_dir}/all_kl_loss.pdf")
