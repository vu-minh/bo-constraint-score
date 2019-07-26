import datetime
import mlflow
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

from bayes_opt import UtilityFunction


def _posterior(optimizer, x_obs, y_obs, grid):
    optimizer._gp.fit(x_obs, y_obs)

    mu, sigma = optimizer._gp.predict(grid, return_std=True)
    return mu, sigma


def plot_gp_one_param(optimizer, x, y, util_func="ucb", kappa=5, xi=0.01,
                      plot_dir="./", dataset_name=""):
    plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
    axis = plt.subplot(gs[0])
    acq = plt.subplot(gs[1])

    x_obs = np.array([[res["params"]["p"]] for res in optimizer.res])
    y_obs = np.array([res["target"] for res in optimizer.res])

    steps = len(optimizer.space)
    current_max_target_function = optimizer.max["target"]
    current_best_param = optimizer.max["params"]["p"]

    mu, sigma = _posterior(optimizer, x_obs, y_obs, x)
    # axis.plot(x, y, marker="--", linewidth=1, color="#FF8200", label="True Target")
    axis.plot(x_obs.flatten(), y_obs, "o", markersize=8, label="Observations", color="#1B365D")
    axis.plot(x, mu, color="#0047BB", label="Prediction")

    axis.fill(
        np.concatenate([x, x[::-1]]),
        np.concatenate([mu - 1.9600 * sigma, (mu + 1.9600 * sigma)[::-1]]),
        alpha=0.75,
        fc="#CCDAF1",
        ec="None",
        label="95% confidence interval",
    )

    # axis.set_xlim((x_obs.min(), x_obs.max()))
    # TODO 20190614 if the score is negative, the current ylim will trim out some range.
    # a workaround is to add sign of the value, but not tested
    axis.set_ylim((0.85 * y_obs.min() * np.sign(y_obs.min()),
                   1.15 * y_obs.max() * np.sign(y_obs.max())))
    utility_function = UtilityFunction(kind=util_func, kappa=kappa, xi=xi)
    utility = utility_function.utility(x, optimizer._gp, y_max=current_max_target_function)

    acq.plot(x, utility, label=f"Utility Function ({util_func})", color="green")
    acq.plot(
        x[np.argmax(utility)],
        np.max(utility),
        "^",
        markersize=11,
        label="Next Best Guess",
        markerfacecolor="#FF8200",
        markeredgecolor="k",
        markeredgewidth=1,
    )
    # acq.set_xlim((x_obs.min(), x_obs.max()))
    # acq.set_ylim((0, np.max(utility) + 0.5))
    acq.set_ylabel(f"Utility ({util_func})", fontdict={"size": 16})
    acq.set_xlabel("param", fontdict={"size": 16})

    axis.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.0)
    acq.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.0)

    # debug next best guess
    next_best_guess_param = x[np.argmax(utility)]
    acq.set_title(f"Next best guess param: {next_best_guess_param}", fontdict={"size": 16})

    # draw indicator vline @ the next query point
    acq.axvline(next_best_guess_param, color='g', linestyle='--', alpha=0.5)
    axis.axvline(next_best_guess_param, color='g', linestyle='--', alpha=0.5)
    # draw indicator hline @ the current  max value of the  target function
    axis.axhline([current_max_target_function], color='#A1AFF8', linestyle='--', alpha=0.7)

    debug_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    debug_method_name = {
        "ucb": f"ucb_kappa{kappa}",
        "ei": f"ei_xi{xi}",
        "poi": f"poi_xi{xi}"
    }[util_func]

    axis.set_title(f"Figure created @ {debug_time}", size=12)
    plt.suptitle(
        f"GP ({debug_method_name} utility function) after {steps} steps"
        f" with best predicted param = {current_best_param:.2f}", size=20)
    fig_name = (f"./{plot_dir}/{debug_method_name}"
                f"_{dataset_name}_step{steps}.png")
    plt.savefig(fig_name, bbox_inches="tight")
    plt.close()
    mlflow.log_artifact(fig_name)
