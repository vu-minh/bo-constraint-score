# run umap for a grid of its params

import os
import joblib
from itertools import product
from subprocess import check_call

import numpy as np
import matplotlib.pyplot as plt
from MulticoreTSNE import MulticoreTSNE
import umap

from common.dataset import dataset
import utils


def run_viz(
    method_name, X, seed=42, embedding_dir="", perplexity_range=[], min_dist_range=[0.1]
):

    if method_name == "tsne":
        for perp in perplexity_range:
            run_tsne(X, perplexity=perp, seed=seed, check_log=True, embedding_dir=embedding_dir)

    if method_name == "largevis":
        for perp in perplexity_range:
            run_largevis(
                X, perplexity=perp, seed=seed, check_log=True, embedding_dir=embedding_dir
            )

    if method_name == "umap":
        for perp, min_dist in product(perplexity_range, min_dist_range):
            run_umap(
                X,
                n_neighbors=perp,
                min_dist=min_dist,
                seed=seed,
                check_log=True,
                embedding_dir=embedding_dir,
            )


def run_largevis(
    X, perplexity: int = 30, seed: int = 42, check_log: bool = True, embedding_dir: str = ""
):
    # https://github.com/lferry007/LargeVis
    perplexity = int(round(perplexity))
    print(f"[Debug] LargeVis(perplexity={perplexity}, seed={seed}")

    embedded_file_name = f"{embedding_dir}/{perplexity}.z"
    if check_log and os.path.exists(embedded_file_name):
        print(f"[Debug] Reuse {embedded_file_name}")
        return joblib.load(embedded_file_name)

    # perpare input file
    current_path = os.path.dirname(os.path.realpath(__file__))
    input_file_name = f"{current_path}/temp_largevis_input.txt"
    input_header = f"{X.shape[0]} {X.shape[1]}"
    np.savetxt(input_file_name, X, header=input_header, comments="")  # disable comment header

    # prepare params to run largevis program
    output_file_name = f"{current_path}/temp_largevis_output.txt"
    largevis_exe = "/opt/LargeVis/LargeVis_run.py"

    check_call(
        [
            "python2",
            largevis_exe,
            "-input",
            input_file_name,
            "-output",
            output_file_name,
            "-perp",
            f"{perplexity}",
            "-samples",
            f"{int(X.shape[0]/100)}",
            "-threads",
            "32",  # update number of threads for running on dev server
        ]
    )

    # loaf from output file to numpy array
    Z = np.loadtxt(output_file_name, skiprows=1)
    # save numpy array to z file
    joblib.dump(Z, embedded_file_name)
    return Z


def run_tsne(
    X, perplexity: int = 30, seed: int = 42, check_log: bool = True, embedding_dir: str = ""
):
    perplexity = int(round(perplexity))
    print(f"[Debug] MulticoreTSNE(perplexity={perplexity}, seed={seed})")

    embedded_file_name = f"{embedding_dir}/{perplexity}.z"
    if check_log and os.path.exists(embedded_file_name):
        print(f"[Debug] Reuse {embedded_file_name}")
        return joblib.load(embedded_file_name)

    Z = MulticoreTSNE(
        perplexity=perplexity,
        n_iter=1500,
        n_jobs=-1,
        random_state=seed,
        n_iter_without_progress=1500,
        min_grad_norm=1e-32,
    ).fit_transform(X)
    joblib.dump(Z, embedded_file_name)
    return Z


def run_umap(
    X,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    seed: int = 42,
    check_log: bool = True,
    embedding_dir: str = "",
):
    n_neighbors = int(round(n_neighbors))
    print(f"[Debug] UMAP(n_neighbors={n_neighbors}, min_dist={min_dist:.4f}, seed={seed})")

    embedded_file_name = f"{embedding_dir}/{n_neighbors}_{min_dist:.4f}.z"
    if check_log and os.path.exists(embedded_file_name):
        print(f"[Debug] Reuse {embedded_file_name}")
        return joblib.load(embedded_file_name)

    Z = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=seed).fit_transform(
        X
    )
    joblib.dump(Z, embedded_file_name)
    return Z


def merge_embeddings(method_name, perplexity_range=[], min_dist_range=[0.1]):
    """Merge all embeddings in .z file into a big .z file.

    + tSNE and LargeVis use `perplexity`,
    + UMAP use `n_neighbors` and `min_dist`

    Returns:
        dict of all embeddings indexed by its corresponding param value(s)
        E.g.: {50: []} or {50_0.1: []}
    """

    # build list of embedding indices that will be used as file name
    embedding_indices = []
    if method_name in ["tsne", "largevis"]:
        for perplexity in perplexity_range:
            embedding_indices.append(str(perplexity))
    if method_name in ["umap"]:
        for n_neighbors in perplexity_range:
            for min_dist in min_dist_range:
                embedding_indices.append(f"{n_neighbors}_{min_dist:.4f}")

    # load all embeddings and add to final dict
    all_embeddings = {}
    for embedding_index in embedding_indices:
        embedding_file_name = f"{embedding_dir}/{embedding_index}.z"
        if os.path.exists(embedding_file_name):
            all_embeddings[embedding_index] = joblib.load(embedding_file_name)

    # store the array of all embeddings into new file
    joblib.dump(all_embeddings, f"{embedding_dir}/all.z")


def test_load_from_all_embeddings(method_name, param1="30", param2="0.0010"):
    all_embeddings = joblib.load(f"{embedding_dir}/all.z")
    print(f"Embedded keys: {str(list(all_embeddings.keys()))[:100]} ... ]")

    if method_name in ["tsne", "largevis"]:
        embedding_index = str(param1)
    if method_name in ["umap"]:
        embedding_index = f"{param1}_{param2}"
    Z = all_embeddings.get(embedding_index, None)
    if Z is None:
        raise ValueError(f"Invalid param: {embedding_index}")

    plt.figure(figsize=(8, 8))
    plt.title(f"[Debug] Test load {dataset_name} {method_name} {embedding_index}")
    plt.scatter(Z[:, 0], Z[:, 1], c=labels, alpha=0.4, s=14, cmap="Spectral")
    plt.savefig(f"{plot_dir}/test_load_{embedding_index}.png")
    plt.close()


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset_name", default="")
    ap.add_argument("-m", "--method_name", default="umap", help="['tsne', 'umap', 'largevis']")
    ap.add_argument("-s", "--seed", default=42, type=int)
    ap.add_argument(
        "--perp_scale",
        default="log",
        help="perplexity scale, in ['log', 'linear', 'hardcoded']",
    ),
    ap.add_argument(
        "--min_dist_scale",
        default="log",
        help="min_dist scale, in ['log', 'linear', 'hardcoded']",
    )
    ap.add_argument(
        "--n_perp", default=200, type=int, help="approximated number of perplexity to evaluate"
    )
    ap.add_argument(
        "--n_min_dist",
        default=10,
        type=int,
        help="approximated number of min_dist to evaluate",
    )
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--run", action="store_true", help="run method for all params")
    ap.add_argument("--plot", action="store_true", help="generate plot for all params")
    args = ap.parse_args()

    dataset.set_data_home("./data")
    dataset_name = args.dataset_name
    method_name = args.method_name

    X_origin, X, labels = dataset.load_dataset(dataset_name, preprocessing_method="auto")

    embedding_dir = f"./embeddings/{dataset_name}/{method_name}"
    plot_dir = f"./plots/{dataset_name}/{method_name}"
    for dir_path in [embedding_dir, plot_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    if args.perp_scale == "hardcoded":
        perplexity_range = [2, 3, 4, 10, 15, 20, 30, 50, 100, 200, 350, 500]
    else:
        min_perp, max_perp = 2, int(X.shape[0] // 3)
        perplexity_range = utils.generate_value_range(
            min_perp, max_perp, range_type=args.perp_scale, num=args.n_perp, dtype=int
        )
        print(perplexity_range)

    if args.debug:
        min_dist_range = [0.1]
    elif args.min_dist_scale == "hardcoded":
        min_dist_range = [0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
    else:
        start_min_dist, stop_min_dist = 0.001, 1.0
        min_dist_range = utils.generate_value_range(
            start_min_dist,
            stop_min_dist,
            range_type=args.min_dist_scale,
            num=args.n_min_dist,
            dtype=float,
        )
        print(list(map("{:.4f}".format, min_dist_range)))

    if args.run:
        run_viz(
            method_name,
            X,
            seed=args.seed,
            embedding_dir=embedding_dir,
            perplexity_range=perplexity_range,
            min_dist_range=min_dist_range,
        )
        merge_embeddings(
            method_name, perplexity_range=perplexity_range, min_dist_range=min_dist_range
        )

    if args.debug and args.plot:
        print("You wanna gen some plot for debug purpose?" "Consider using `plot_viz.py`.")
