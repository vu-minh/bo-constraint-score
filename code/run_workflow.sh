#!/bin/bash -x

# -x for debugging: show command to be run

# workflow for reproducing
# NOTE TO USE --seed 42

DEFAULT_SEED=42


function run_viz {
    DATASET_NAME=$1
    METHOD=$2

    # run viz, merge all viz with selected params (in log scale) to one .z file
    echo "RUN VIZ for ${DATASET_NAME} with ${METHOD}"

    # for umap, run in debug mode to set min_dist to 0.1
    # for tsne and largevis, run viz for all params:
    # + set n_perp to a very large number
    # + set perp_scale to "linear"
    python run_viz.py \
	   --seed $DEFAULT_SEED \
	   -d $DATASET_NAME \
	   -m $METHOD \
	   --n_perp 200 \
	   --run \
	   # --debug # only for umap
	   # --perp_scale linear \
}


function run_score {
    # caluclate score for all embeddings, plot stability of the score

    # debug number of generated constraints:
    # DIGITS
    # [Debug]: From 3 =>(sim-links: 30, dis-links: 405)
    # [Debug]: From 5 =>(sim-links: 100, dis-links: 1125)
    # [Debug]: From 10 =>(sim-links: 450, dis-links: 4500)
    # [Debug]: From 15 =>(sim-links: 1050, dis-links: 10125)

    # COIL20
    # [Debug]: From 3 =>(sim-links: 60, dis-links: 1710)
    # [Debug]: From 5 =>(sim-links: 200, dis-links: 4750)
    # [Debug]: From 10 =>(sim-links: 900, dis-links: 19000)
    # [Debug]: From 15 =>(sim-links: 2100, dis-links: 42750)

    DATASET_NAME=$1
    METHOD=$2
    echo "RUN SCORE for ${DATASET_NAME} with ${METHOD}"

    # always run in debug mode for fixed number of labels per class (3, 5, 10, 15)
    python run_score.py \
	   --seed $DEFAULT_SEED \
	   -d $DATASET_NAME \
	   -m $METHOD \
	   -sc qij \
	   --debug \
	   --use_log_scale \
	   --plot \
	   --run -nr 10 # comment out this param to make the plot
}


function run_score_umap_2D {
    DATASET_NAME=$1
    echo "RUN SCORE for ${DATASET_NAME} with UMAP (2 params)"

    # always run in debug mode for fixed number of labels per class (3, 5, 10, 15)
    python run_score.py \
       --seed $DEFAULT_SEED \
       -d $DATASET_NAME \
       -m umap \
       -sc qij \
       --run_score_umap
}


function run_metric  {
    DATASET_NAME=$1
    METHOD=$2
    echo "RUN METRIC for $DATASET_NAME with $METHOD"

    python run_score.py \
	   --seed $DEFAULT_SEED \
	   -d $DATASET_NAME \
	   -m $METHOD \
	   -sc metrics \
	   --use_log_scale \
	   --plot \
	   --run
}

function run_bic {
    DATASET_NAME=$1
    echo "RUN METRIC for $DATASET_NAME (only support tsne)"

    python run_score.py \
	   --seed $DEFAULT_SEED \
	   -d $DATASET_NAME \
	   -m tsne \
	   -sc bic \
	   --use_log_scale \
	   --plot \
	   --run
}


function plot_compare_scores {
    DATASET_NAME=$1
    METHOD=$2
    echo "PLOT COMPARE SCORES for $DATASET_NAME with $METHOD"

    python run_score.py \
        --seed $DEFAULT_SEED \
        -d $DATASET_NAME \
        -m $METHOD \
        --plot_compare \
        --use_log_scale
}


function gen_metamap {
    python plot_viz.py -d $1 -m $2 --plot_metamap
}

function gen_grid_viz_demo {
    python plot_viz.py -d $1 -m $2 --show_viz_grid
}


############################################################################
RUN_ALL=false

if [ $RUN_ALL = true ]; then
    declare -a LIST_DATASETS=("FASHION1000" "DIGITS" "COIL20")
    declare -a LIST_METHODS=("tsne" "umap" "largevis")
else
    declare -a LIST_DATASETS=("DIGITS" "COIL20" "FASHION1000" "FASHION_MOBILENET" "NEURON_1K" "20NEWS5")
    declare -a LIST_METHODS=("tsne" "umap")
fi

for DATASET_NAME in "${LIST_DATASETS[@]}"; do
    for METHOD in "${LIST_METHODS[@]}"; do
	    # echo $DATASET_NAME $METHOD
     #    if [ $METHOD == "tsne" ]; then
     #        run_bic $DATASET_NAME # run BIC first to generate tsne embeddings
     #    fi
     #    run_viz $DATASET_NAME $METHOD
	    # run_score $DATASET_NAME $METHOD
     #    run_metric $DATASET_NAME $METHOD

     #    if [ $METHOD == "umap" ]; then
	    #    run_score_umap_2D $DATASET_NAME
     #    fi

        # plot_compare_scores $DATASET_NAME $METHOD
        gen_metamap $DATASET_NAME $METHOD
        gen_grid_viz_demo $DATASET_NAME $METHOD
    done
done

if [ $RUN_ALL = true ]; then
    echo "Copy figures for latex"
    bash copy_figures_for_latex.sh
fi