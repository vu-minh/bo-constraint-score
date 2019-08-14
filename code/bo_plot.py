import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import FuncFormatter
from matplotlib import gridspec

from bayes_opt import UtilityFunction
from utils import generate_value_range


def _plot_acq_func(ax, util_func, list_params, utility_values, next_best_guess_param):
    ax.plot(list_params, utility_values, color="green", label=f"{util_func.upper()} function")
    ax.axvline(next_best_guess_param, color="green", linestyle='--', alpha=0.5,
               marker="^", markersize=16, clip_on=False,
               markeredgecolor="#FF8200", markerfacecolor="#FF8200", markevery=100)
    ax.set_ylabel(f"Utility ({util_func})")
    ax.set_xlabel("param")

    ax.text(x=next_best_guess_param, y=min(utility_values),
            s=f"Next best guess param: {int(np.exp(next_best_guess_param))}")


def _plot_true_target_values(ax, list_params, true_score):
    ax.plot(list_params, true_score, label="True target", marker='s', markersize=3,
            color="#FF8200", alpha=0.75, linewidth=1.25)
    ax.set_ylabel(f"constraint score")


def _plot_observed_points(ax, x_obs, y_obs):
    for x in x_obs:
        print(x, np.exp(x))
    ax.plot(x_obs.flatten(), y_obs, "o", markersize=7, label="Observations", color="#1B365D")


def _plot_gp_predicted_values(ax, pred_mu, pred_sigma, list_params):
    list_params = list_params.ravel()
    ax.plot(list_params, pred_mu, color="#0047BB", linestyle="--", label="Prediction")
    ax.fill_between(
        x=list_params,
        y1=pred_mu + 1.96*pred_sigma,
        y2=pred_mu - 1.96*pred_sigma,
        alpha=0.75,
        fc="#CCDAF1",
        ec="None",
        label="95% confidence",
    )


def plot_bo_one_param_detail(optimizer,
                             x_obs, y_obs,  # the Observations
                             list_params, true_score,  # the true target values
                             pred_mu, pred_sigma,  # the predicted values
                             util_func="ucb", kappa=5, xi=0.01, plot_dir=""):
    ''' Plot the prediction of BayOpt with GP model.
    Note that all values of `list_params` are in log space (the real param in logscale)
    '''
    # note to make big font size for plots in the paper
    plt.rcParams.update({'font.size': 22})

    plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
    ax0 = plt.subplot(gs[0])  # plot gp prediction values
    ax1 = plt.subplot(gs[1])  # plot acquision function values

    # set limit value for yaxis in gp prediction chart
    ax0.set_ylim((0.92 * y_obs.min(), 1.08 * y_obs.max()))

    _plot_true_target_values(ax0, list_params, true_score)
    _plot_observed_points(ax0, x_obs, y_obs)
    _plot_gp_predicted_values(ax0, pred_mu, pred_sigma, list_params)

    current_max_target_function = optimizer.max["target"]
    current_best_param = optimizer.max["params"]["perplexity"]

    # draw indicator hline at the current  max value of the  target function
    # ax0.axhline([current_max_target_function], color='#A1AFF8', linestyle='--', alpha=0.7)

    # calculate values of utility function for each param
    utility_function = UtilityFunction(kind=util_func, kappa=kappa, xi=xi)
    utility_values = utility_function.utility(list_params, optimizer._gp,
                                              y_max=current_max_target_function)
    next_best_guess_param = list_params[np.argmax(utility_values)][0]

    # plot acquision function
    _plot_acq_func(ax1, util_func, list_params, utility_values, next_best_guess_param)

    # plot in common for two axes
    for ax in [ax0, ax1]:
        # shift legend to the right cornner
        ax.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.0)
        # draw indicator vline @ the next query point
        ax.axvline(next_best_guess_param, color='g', linestyle='--', alpha=0.5)

    # set title and save figure
    acq_method_name = {
        "ucb": f"UCB($\\kappa$={kappa})", "ei": f"EI($\\xi$={xi})", "poi": f"POI($\\xi$={xi})"
    }[util_func]
    plt.suptitle(
        f"{acq_method_name} after {len(optimizer.space)} steps"
        f" with best predicted param = {int(np.exp(current_best_param))}")
    plt.savefig(f"{plot_dir}/bo_detail.png", bbox_inches="tight")
    plt.close()


def plot_bo_one_param_summary(optimizer,
                              x_obs, y_obs,  # the Observations
                              list_params, true_score,  # the true target values
                              pred_mu, pred_sigma,  # the predicted values
                              param_name="perplexity",
                              util_func="ucb", kappa=5, xi=0.01, plot_dir=""):
    ''' Plot the prediction of BayOpt with GP model.
    Note that all values of `list_params` are in log space (the real GP params are in logscale)
    '''   
    # note to make big font size for plots in the paper
    plt.rcParams.update({'font.size': 22})
    _, ax = plt.subplots(1, 1, figsize=(11, 5))

    ax.set_xlim(left=list_params.min(), right=list_params.max())
    ax.set_ylim(top=1.1*max(true_score), bottom=0.9*min(true_score))

    _plot_true_target_values(ax, list_params, true_score)
    _plot_observed_points(ax, x_obs, y_obs)
    _plot_gp_predicted_values(ax, pred_mu, pred_sigma, list_params)

    # draw indicator hline at the current  max value of the  target function
    # ax.axhline([current_max_target_function], color='#A1AFF8', linestyle='--', alpha=0.7)

    # draw indicator vline @ the best param
    best_param = optimizer.max["params"][param_name]
    ax.axvline(best_param, color="green", linestyle='--', alpha=0.5,
               marker="^", markersize=16, clip_on=False,
               markeredgecolor="#FF8200", markerfacecolor="#FF8200", markevery=100)
    ax.text(x=best_param, y=1.1*min(true_score),
            s=f"best {param_name}: {int(round(np.exp(best_param)))}")

    # set limit value for yaxis in gp prediction chart
    # ax.set_ylim((0.95 * y_obs.min(), 1.12 * y_obs.max()))  # make place for legend
    ax.locator_params(axis='y', nbins=4)
    ax.yaxis.grid(linestyle="--")
    ax.xaxis.grid(linestyle="--", alpha=0.4)

    # force list params to show on axias
#    list_params_to_show = generate_value_range(
#        min_val=1.1, max_val=max(list_params), num=6, range_type="log", dtype=int)
#    ax.set_xticks(list_params_to_show)
#    ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())

    # show param in log scale in xaxis
    list_params_to_show = generate_value_range(
        min_val=min(list_params), max_val=max(list_params), num=8, range_type="log", dtype=int)
    ax.set_xlim(left=min(list_params), right=max(list_params))
    ax.set_xticks(list_params_to_show)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{np.exp(x):.2f}"))
    ax.set_xlabel(f"{param_name} in log-scale")

    # set title and save figure
    plt.legend(loc="upper center", ncol=4, prop={'size': 14})
    plt.savefig(f"{plot_dir}/bo_summary.png", bbox_inches="tight")
    plt.close()
