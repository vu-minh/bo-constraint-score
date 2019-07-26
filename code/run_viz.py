# run umap for all a grid of its params

import os
import joblib
import matplotlib.pyplot as plt
from common.dataset import dataset
import umap
from MulticoreTSNE import MulticoreTSNE


def run_viz(method_name, X, param_ranges, seed=42):
    {
        'tsne': lambda: run_tsne(X, perplexity_range=param_ranges, seed=seed),
        'umap': lambda: run_umap(X, n_neighbors_range=param_ranges, seed=seed),
        'umap_nneighbors': lambda: (),  # TODO: run umap with 1 param
        'umap_nneighbors_mindist': lambda: (),  # TODO: run umap with 2 params
    }[method_name]()


def run_tsne(X, perplexity_range, seed=42):
    for perp in perplexity_range:
        print("Compute tsne with perp: ", perp)
        Z = MulticoreTSNE(
            perplexity=perp, n_iter=1500, n_jobs=2, random_state=seed,
            n_iter_without_progress=1500, min_grad_norm=1e-32,
        ).fit_transform(X)
        joblib.dump(Z, f"{embedding_dir}/{perp}.z")


def run_umap(X, n_neighbors_range, seed=42):
    for n_neighbors in n_neighbors_range:
        print("Compute umap with n_neighbors: ", n_neighbors)
        Z = umap.UMAP(
            n_neighbors=n_neighbors, random_state=seed
        ).fit_transform(X)
        joblib.dump(Z, f"{embedding_dir}/{n_neighbors}.z")


def merge_embeddings(n_embeddings):
    all_embeddings = []
    # load all pre-calculated embeddings
    for i in range(n_embeddings):
        log_name = f"{embedding_dir}/{i}.z"
        all_embeddings.append(None if not os.path.exists(log_name)
                              else joblib.load(log_name))
    # store the array of all embeddings into new file
    joblib.dump(all_embeddings, f"{embedding_dir}/all.z")


def test_plot(method_name, n_neighbors_range):
    for n_neighbors in n_neighbors_range:
        in_name = f"{embedding_dir}/{n_neighbors}.z"
        Z = joblib.load(in_name)
        plt.figure(figsize=(8, 8))
        plt.title(f"{dataset_name} {method_name} (n_neighbors={n_neighbors})")
        plt.scatter(Z[:, 0], Z[:, 1], c=labels, alpha=0.4, cmap="Spectral")
        plt.savefig(f"{plot_dir}/{n_neighbors}.png")


def test_load_all_embeddings(method_name, list_n_neighbors=[2, 5, 20, 50]):
    all_embeddings = joblib.load(f"{embedding_dir}/all.z")
    for n_neighbors in list_n_neighbors:
        Z = all_embeddings[n_neighbors]
        plt.figure(figsize=(8, 8))
        plt.title(f"{dataset_name} {method_name} (n_neighbors={n_neighbors})")
        plt.scatter(Z[:, 0], Z[:, 1], c=labels, alpha=0.4, cmap="Spectral")
        plt.savefig(f"{plot_dir}/test_load_{n_neighbors}.png")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset_name", default="")
    ap.add_argument("-m", "--method_name", default="umap",
                    help="['tsne', 'umap', 'largevis', 'TODO-umap-n-params']")
    ap.add_argument("-s", "--seed", default=2019, type=int)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--run", action="store_true", help="run method for all params")
    ap.add_argument("--plot", action="store_true", help="generate plot for all params")
    args = ap.parse_args()

    dataset.set_data_home("./data")
    dataset_name = args.dataset_name
    method_name = args.method_name
    X_origin, X, labels = dataset.load_dataset(dataset_name)
    embedding_dir = f"./embeddings/{dataset_name}/{method_name}"
    plot_dir = f"./plots/{dataset_name}/{method_name}"
    for dir_path in [embedding_dir, plot_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    if args.debug:
        n_neighbors_range = [2, 3, 5, 10, 15, 20, 30, 50]
    else:
        n_neighbors_range = range(2, X.shape[0] // 3)

    if args.run:
        run_viz(args.method, X, n_neighbors_range, seed=args.seed)
        merge_embeddings(n_embeddings=(X.shape[0]//3))

    if args.plot:
        test_plot(method_name, n_neighbors_range)
        test_load_all_embeddings(method_name)
