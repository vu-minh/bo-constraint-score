#!/bin/bash -x

# -x for debugging: show command to be run

# workflow for reproducing
# NOTE TO USE --seed 42


DEFAULT_SEED=42
DATASET_NAME=DIGITS
METHOD=tsne

function run_viz {
    # run viz, merge all viz with selected params (in log scale) to one .z file
    echo "RUN VIZ for ${DATASET_NAME}"

    # for tsne and largevis, run viz for all params:
    # + set n_perp to a very large number
    # + set perp_scale to "linear"
    python run_viz.py \
	   --seed $DEFAULT_SEED \
	   -d $DATASET_NAME \
	   -m $METHOD
	   --n_perp 5000 \
	   --perp_scale linear \
	   --run

    # for umap, run in debug mode to set min_dist to 0.1
}


function run_score {
    # caluclate score for all embeddings, plot stability of the score

    # debug number of generated constraints:
    # DIGITS
    # [Debug]: From 3 =>(sim-links: 30, dis-links: 405)
    # [Debug]: From 5 =>(sim-links: 100, dis-links: 1125)
    # [Debug]: From 10 =>(sim-links: 450, dis-links: 4500)
    # [Debug]: From 15 =>(sim-links: 1050, dis-links: 10125)

    # always run in debug mode for fixed number of labels per class (3, 5, 10, 15)
    python run_score.py \
	   --seed $DEFAULT_SEED \
	   -d $DATASET_NAME \
	   -m $METHOD \
	   --debug \
	   --use_log_scale \
	   --run -nr 10 \
	   --plot
}


# run_viz
run_score
