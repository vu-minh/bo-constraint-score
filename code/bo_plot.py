from itertools import product

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.ticker import FuncFormatter

from bayes_opt import UtilityFunction
from utils import generate_value_range


def _plot_acq_func(ax, util_func, list_params, utility_values, next_best_guess_param):
    ax.plot(list_params, utility_values, color="green", label=f"{util_func.upper()} function")
    ax.axvline(
        next_best_guess_param,
        color="green",
        linestyle="--",
        alpha=0.5,
        marker="^",
        markersize=16,
        clip_on=False,
        markeredgecolor="#FF8200",
        markerfacecolor="#FF8200",
        markevery=100,
    )
    ax.set_ylabel(f"Utility ({util_func})")
    ax.set_xlabel("param")

    ax.text(
        x=next_best_guess_param,
        y=min(utility_values),
        s=f"Next best guess param: {int(np.exp(next_best_guess_param))}",
    )


def _plot_true_target_values(ax, list_params, true_score, threshold=0.95):
    ax.plot(
        list_params,
        true_score,
        label="True target",
        marker="s",
        markersize=3,
        color="#FF8200",
        alpha=0.75,
        linewidth=1.25,
    )
    ax.set_ylabel(f"constraint score")

    # determine true best range
    pivot = threshold * max(true_score)
    (best_indices,) = np.where(true_score > pivot)
    param_min = list_params[best_indices.min()][0]
    param_max = list_params[best_indices.max()][0]
    # plt.arrow(x=param_min, y=ax.get_ylim()[0], dx=param_max-param_min, dy=0,
    #     width=0.005, color="orange", length_includes_head=True)

    # add text to show the best range
    value_min = int(np.exp(param_min))
    value_max = int(np.exp(param_max))
    ax.text(
        x=0.0,
        y=1.065,
        transform=ax.transAxes,
        ha="left",
        va="center",
        s=f"True best range: [{value_min}, {value_max}]",
        fontsize=18,
    )


def _plot_observed_points(ax, x_obs, y_obs):
    ax.plot(x_obs.flatten(), y_obs, "o", markersize=7, label="Observations", color="#1B365D")


def _plot_gp_predicted_values(ax, pred_mu, pred_sigma, list_params, threshold=0.95):
    list_params = list_params.ravel()
    ax.plot(list_params, pred_mu, color="#0047BB", linestyle="--", label="Prediction")
    ax.fill_between(
        x=list_params,
        y1=pred_mu + 1.96 * pred_sigma,
        y2=pred_mu - 1.96 * pred_sigma,
        alpha=0.75,
        facecolor="#CCDAF1",
        edgecolor="None",
        label="95% confidence",
    )

    # determine best param range and best param value
    pivot = threshold * max(pred_mu)
    (best_indices,) = np.where(pred_mu > pivot)
    param_min = list_params[best_indices.min()]
    param_max = list_params[best_indices.max()]
    # note best param here is max(pred_mu), should take into account
    # the uncertainty in this prediction
    # param_best = int(np.exp(list_params[np.argmax(pred_mu)]))
    # print("Debug best param: ", param_best, np.max(pred_mu))

    # plot best predicted range
    ax.axhline(pivot, linestyle="--", alpha=0.4)
    _plot_best_range(ax, param_min, param_max)


def plot_bo_one_param_detail(
    optimizer,
    x_obs,
    y_obs,  # the Observations
    list_params,
    true_score,  # the true target values
    pred_mu,
    pred_sigma,  # the predicted values
    util_func="ucb",
    kappa=5,
    xi=0.01,
    plot_dir="",
):
    """ Plot the prediction of BayOpt with GP model.
    Note that all values of `list_params` are in log space (the real param in logscale)
    """
    # note to make big font size for plots in the paper
    plt.rcParams.update({"font.size": 22})

    plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
    ax0 = plt.subplot(gs[0])  # plot gp prediction values
    ax1 = plt.subplot(gs[1])  # plot acquision function values

    # set limit value for yaxis in gp prediction chart
    ax0.set_ylim((0.92 * y_obs.min(), 1.08 * y_obs.max()))

    if true_score is not None:
        _plot_true_target_values(ax0, list_params, true_score)
    _plot_observed_points(ax0, x_obs, y_obs)
    _plot_gp_predicted_values(ax0, pred_mu, pred_sigma, list_params)

    current_max_target_function = optimizer.max["target"]
    current_best_param = optimizer.max["params"]["perplexity"]

    # draw indicator hline at the current  max value of the  target function
    # ax0.axhline([current_max_target_function], color='#A1AFF8', linestyle='--', alpha=0.7)

    # calculate values of utility function for each param
    utility_function = UtilityFunction(kind=util_func, kappa=kappa, xi=xi)
    utility_values = utility_function.utility(
        list_params, optimizer._gp, y_max=current_max_target_function
    )
    next_best_guess_param = list_params[np.argmax(utility_values)][0]

    # plot acquision function
    _plot_acq_func(ax1, util_func, list_params, utility_values, next_best_guess_param)

    # plot in common for two axes
    for ax in [ax0, ax1]:
        # shift legend to the right cornner
        ax.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.0)
        # draw indicator vline @ the next query point
        ax.axvline(next_best_guess_param, color="g", linestyle="--", alpha=0.5)

    # set title and save figure
    acq_method_name = {
        "ucb": f"UCB($\\kappa$={kappa})",
        "ei": f"EI($\\xi$={xi})",
        "poi": f"POI($\\xi$={xi})",
    }[util_func]
    plt.suptitle(
        f"{acq_method_name} after {len(optimizer.space)} steps"
        f" with best predicted param = {int(np.exp(current_best_param))}"
    )
    plt.savefig(f"{plot_dir}/bo_detail.png", bbox_inches="tight")
    plt.close()


def _plot_best_range(ax, param_min, param_max):
    ax.axvspan(param_min, param_max, alpha=0.12, color="#CCDAF1")

    value_min = int(np.exp(param_min))
    value_max = int(np.exp(param_max))
    text_y_pos = ax.get_ylim()[0] * 1.1

    # vertical line on the left
    ax.axvline(
        param_min,
        color="#CCDAF1",
        linestyle="--",
        alpha=0.6,
        marker=">",
        markersize=14,
        clip_on=False,
        markeredgecolor="#CCDAF1",
        markerfacecolor="#CCDAF1",
        markevery=100,
    )
    ax.text(x=param_min, y=text_y_pos, s=str(value_min), ha="right")

    # vertical line on the right
    ax.axvline(
        param_max,
        color="#CCDAF1",
        linestyle="--",
        alpha=0.6,
        marker="<",
        markersize=14,
        clip_on=False,
        markeredgecolor="#CCDAF1",
        markerfacecolor="#CCDAF1",
        markevery=100,
    )
    ax.text(x=param_max, y=text_y_pos, s=str(value_max), ha="left")

    # add text to show the best range
    ax.text(
        x=1.0,
        y=1.065,
        transform=ax.transAxes,
        ha="right",
        va="center",
        s=f"Predicted best range: [{value_min}, {value_max}]",
        color="#0047BB",
        fontsize=18,
    )


def plot_bo_one_param_summary(
    optimizer,
    x_obs,
    y_obs,  # the Observations
    list_params,
    true_score,  # the true target values
    pred_mu,
    pred_sigma,  # the predicted values
    param_name="perplexity",
    threshold=0.95,
    util_func="ucb",
    kappa=5,
    xi=0.01,
    plot_dir="",
):
    """ Plot the prediction of BayOpt with GP model.
    Note that all values of `list_params` are in log space (the real GP params are in logscale)
    """
    # note to make big font size for plots in the paper
    plt.rcParams.update({"font.size": 22})
    _, ax = plt.subplots(1, 1, figsize=(11, 5))

    ax.set_xlim(left=list_params.min(), right=list_params.max())
    if true_score is not None:
        ax.set_ylim(top=1.1 * max(true_score), bottom=0.9 * min(true_score))
        _plot_true_target_values(ax, list_params, true_score, threshold=threshold)

    _plot_observed_points(ax, x_obs, y_obs)
    _plot_gp_predicted_values(ax, pred_mu, pred_sigma, list_params, threshold=threshold)

    # draw indicator hline at the current  max value of the  target function
    # ax.axhline([current_max_target_function], color='#A1AFF8', linestyle='--', alpha=0.7)

    # draw indicator vline @ the best param
    best_param = optimizer.max["params"][param_name]
    ax.axvline(
        best_param,
        color="green",
        linestyle="--",
        alpha=0.5,
        marker="^",
        markersize=16,
        clip_on=False,
        markeredgecolor="#FF8200",
        markerfacecolor="#FF8200",
        markevery=100,
    )

    # set limit value for yaxis in gp prediction chart
    # ax.set_ylim((0.95 * y_obs.min(), 1.12 * y_obs.max()))  # make place for legend
    ax.locator_params(axis="y", nbins=4)
    # ax.yaxis.grid(linestyle="--")
    # ax.xaxis.grid(linestyle="--", alpha=0.3)

    # show param in log scale in xaxis
    list_params_to_show = generate_value_range(
        min_val=min(list_params), max_val=max(list_params), num=9, range_type="log", dtype=int
    )
    ax.set_xlim(left=min(list_params), right=max(list_params))
    ax.set_xticks(list_params_to_show)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(round(np.exp(x)))}"))
    ax.set_xlabel(f"{param_name} in log-scale")

    # plot text for best param
    ax.text(x=best_param, y=1.1 * ax.get_ylim()[0], ha="center", s=f"{int(np.exp(best_param))}")

    # hint text for the top hightest scores horizontal line
    pivot = threshold * max(pred_mu)
    ax.text(
        x=min(list_params),
        y=pivot * 1.01,
        ha="left",
        va="bottom",
        fontsize=18,
        s="\u2199" + f"{threshold} max(score)",
        color="#0047BB",
    )

    # set title and save figure
    plt.legend(loc="upper center", ncol=4, prop={"size": 14})
    plt.savefig(f"{plot_dir}/bo_summary.png", bbox_inches="tight")
    plt.close()


def plot_density_2D(
    input_score_name: str = "umap_scores",
    target_key_name: str = "qij_score",
    title: str = "",
    log_dir: str = "",
    score_dir: str = "",
    plot_dir: str = "",
):
    # plot contour/contourf guide:
    # https://matplotlib.org/3.1.1/gallery/images_contours_and_fields/irregulardatagrid.html#sphx-glr-gallery-images-contours-and-fields-irregulardatagrid-py

    df = pd.read_csv(f"{score_dir}/{input_score_name}.csv")
    print(df)

    Z = df.pivot(index="min_dist", columns="n_neighbors", values=target_key_name)
    print(Z)

    min_dist_values = Z.index.to_numpy()
    n_neighbors_values = Z.columns.to_numpy()
    Z = Z.to_numpy()
    print(Z.shape, min_dist_values.shape, n_neighbors_values.shape)

    # get the row of max value
    best_param = df.loc[df[target_key_name].idxmax()]
    print("Print: best params: ", best_param)
    best_n_neighbors = best_param["n_neighbors"]
    best_min_dist = best_param["min_dist"]
    best_score = best_param[target_key_name]
    print(best_n_neighbors, best_min_dist, best_score)

    X, Y = np.meshgrid(n_neighbors_values, min_dist_values)
    print(X.shape, Y.shape)

    fig, ax = plt.subplots(1, 1, figsize=(8, 3))
    ax.set_title(f"[{dataset_name}] {title} for UMAP embeddings")

    # contour for score
    ax.contour(X, Y, Z, levels=10, linewidths=0.5, colors="k")
    cntr1 = ax.contourf(X, Y, Z, levels=10, cmap="YlGn")
    fig.colorbar(cntr1, ax=ax)

    # grid of sampled points
    ax.plot(df.n_neighbors, df.min_dist, ".w", ms=1)

    # custom axes
    ax.set_xscale("log", basex=np.e)
    ax.set_xlabel("n_neighbors in log-scale")
    ax.set_yscale("log", basey=np.e)
    ax.set_ylabel("min_dist in log-scale")

    # show ticks values in log scale
    ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.set_yticks([0.001, 0.01, 0.1, 1.0] + [best_min_dist] * 2)
    ax.set_xticks(np.append(ax.get_xticks(), [best_n_neighbors] * 2))

    # plot best param
    ax.plot(best_n_neighbors, best_min_dist, "s", c="orange")

    plt.tight_layout()
    plt.savefig(f"{plot_dir}/2D/{target_key_name}.png")


def plot_prediction_density_2D(
    optimizer,
    list_n_neigbors,
    list_min_dist,
    title: str = "",
    threshold=0.96,
    log_dir: str = "",
    score_dir: str = "",
    plot_dir: str = "",
):
    # plot contour/contourf guide:
    # https://matplotlib.org/3.1.1/gallery/images_contours_and_fields/irregulardatagrid.html#sphx-glr-gallery-images-contours-and-fields-irregulardatagrid-py

    # workflow for plot the density of the prediction

    # sample gird of points for two input params
    # the grid in linear space
    # that mean the plot must be in linear space
    print(
        f"Input params: list_n_neigbors {len(list_n_neigbors)},"
        f" list_min_dist {len(list_min_dist)}"
    )
    X, Y = np.meshgrid(list_n_neigbors, list_min_dist)
    print("Grid size: ", X.shape, Y.shape)

    # get observation from optimizer and retrain the GP model in the optimizer
    # note that, optimizer._gp always work on log-scale
    X_obs = np.array(
        [[obs["params"]["min_dist"], obs["params"]["n_neighbors"]] for obs in optimizer.res]
    )
    y_obs = np.array([obs["target"] for obs in optimizer.res])
    gp = optimizer._gp
    gp.fit(X_obs, y_obs)

    # feed the list of pairs of 2 input into GP of optimizer model
    # and get the predicted mean (and sigma) score for each pair
    # since the input_grid in linear-space, it must be transformed to log-space
    input_grid = np.log(list(product(list_min_dist, list_n_neigbors)))
    predicted = gp.predict(input_grid)
    Z = predicted.reshape(X.shape)

    # plot contour and contourf for the sampled grid
    fig = plt.figure(figsize=(6 + 3, 2.5 + 1.5))
    gs = gridspec.GridSpec(
        2, 2, height_ratios=[1, 2], width_ratios=[3, 1], wspace=0.22, hspace=0.22
    )
    ax = plt.subplot(gs[1, 0])

    # grid of sampled points in the grid
    ax.plot(X, Y, ".w", ms=0.5)

    # contour for the prediction (in linear space)
    ax.contour(X, Y, Z, levels=10, linewidths=0.5, colors="k")
    cntr1 = ax.contourf(X, Y, Z, levels=10, cmap="YlGn")

    # scatter plot the selected points in optimizer
    # encode their score values as color
    # the observations selected by optimizer._gp are in log space
    # they must be transformed to linear space by np.exp
    ax.scatter(
        np.exp(X_obs[:, 1]), np.exp(X_obs[:, 0]), c=y_obs, cmap="YlGn", edgecolors="white"
    )

    # evaluate the correctness of the viz by evaluating the corresponding color
    # of the sampled points (OK)

    # get the row of max value
    best_param = optimizer.max["params"]
    print("Print: best params: ", best_param)
    best_n_neighbors = np.exp(best_param["n_neighbors"])
    best_min_dist = np.exp(best_param["min_dist"])

    # plot best param
    ax.plot(best_n_neighbors, best_min_dist, marker="s", color="orange", markersize=8)

    # custom axes: show ticks values in log scale, add best param indicator
    ax.set_xlabel("n_neighbors in log-scale")
    ax.set_xscale("log", basex=np.e)
    ax.set_xticks(np.append(ax.get_xticks(), [best_n_neighbors] * 2))
    ax.set_xlim(list_n_neigbors.min(), list_n_neigbors.max())
    ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())

    ax.set_ylabel("min_dist in log-scale")
    ax.set_yscale("log", basey=np.e)
    ax.set_yticks([0.001, 0.01, 0.1, 1.0] + [best_min_dist] * 2)
    ax.set_ylim(list_min_dist.min(), list_min_dist.max())
    ax.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter())

    # show xaxis on top and yaxis on the right
    ax.xaxis.tick_top()
    ax.yaxis.tick_right()

    # partial dependence: plot score by n_neighbors
    ax0 = plt.subplot(gs[0, 0], sharex=ax)
    ax0.set_title("Score by n_neighbors")
    score_by_n_neighbors = Z.mean(axis=0)
    plt_score_by_n_neighbors, = ax0.plot(list_n_neigbors, score_by_n_neighbors)
    plt.setp(ax0.get_xticklabels(), visible=False)

    # determine best param range and best param value
    pivot = threshold * max(score_by_n_neighbors)
    (best_indices,) = np.where(score_by_n_neighbors > pivot)
    param_min0 = list_n_neigbors[best_indices.min()]
    param_max0 = list_n_neigbors[best_indices.max()]
    ax0.axvspan(xmin=param_min0, xmax=param_max0, alpha=0.1, ls="--", lw=2, edgecolor="blue")
    ax0.axvline(best_n_neighbors, ls="--", c="orange")

    # partial dependence: plot score by min_dist
    ax1 = plt.subplot(gs[1, 1], sharey=ax)
    ax1.set_title("Score by min_dist")
    score_by_min_dist = Z.mean(axis=1)
    plt_score_by_min_dist, = ax1.plot(score_by_min_dist, list_min_dist)
    plt.setp(ax1.get_yticklabels(), visible=False)

    # determine best param range and best param value
    pivot = threshold * max(score_by_min_dist)
    (best_indices,) = np.where(score_by_min_dist > pivot)
    param_min1 = list_min_dist[best_indices.min()]
    param_max1 = list_min_dist[best_indices.max()]
    ax1.axhspan(ymin=param_min1, ymax=param_max1, alpha=0.1, ls="--", lw=2, edgecolor="blue")
    ax1.axhline(best_min_dist, ls="--", c="orange")

    # for plotting colorbar and legend
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    ax2 = plt.subplot(gs[0, 1])
    ax2.axis("off")
    cax2 = inset_axes(ax2, width="100%", height="20%", loc="upper center")
    cbar = fig.colorbar(cntr1, cax=cax2, orientation="horizontal")
    cbar.ax.set_title("Score value")

    # ax2.legend((plt_score_by_min_dist, plt_score_by_n_neighbors),
    #           ("Score by min_dis", "Score by n_neighbors"), loc="lower center")

    plt.savefig(f"{plot_dir}/predicted_score.png", bbox_inches="tight")


if __name__ == "__main__":
    dataset_name = "NEURON_1K"
    method_name = "umap"
    score_name = "qij"
    log_dir = f"./logs/{dataset_name}/{method_name}/{score_name}"
    score_dir = f"./scores/{dataset_name}/{method_name}/{score_name}"
    plot_dir = f"./plots/{dataset_name}/{method_name}"

    plot_density_2D(
        input_score_name="umap_scores",
        target_key_name="qij_score",
        title="Constraint preserving score",
        log_dir=log_dir,
        score_dir=score_dir,
        plot_dir=plot_dir,
    )

    plot_density_2D(
        input_score_name="umap_metrics",
        target_key_name="auc_rnx",
        title="$AUC_{log}RNX$ score",
        log_dir=log_dir,
        score_dir=score_dir,
        plot_dir=plot_dir,
    )
