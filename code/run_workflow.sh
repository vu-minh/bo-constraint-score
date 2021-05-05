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
       --min_dist_scale hardcoded
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
       --run_score_umap \
       --debug
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
    declare -a LIST_DATASETS=("DIGITS" "COIL20" "FASHION1000" "FASHION_MOBILENET" "20NEWS5" "NEURON_1K")
    declare -a LIST_METHODS=("tsne")
fi

for DATASET_NAME in "${LIST_DATASETS[@]}"; do
    for METHOD in "${LIST_METHODS[@]}"; do
	   echo $DATASET_NAME $METHOD

        # if [ $METHOD == "tsne" ]; then
        #     run_bic $DATASET_NAME # run BIC first to generate tsne embeddings
        # fi
        # 
    	# run_viz $DATASET_NAME $METHOD
    	# run_score $DATASET_NAME $METHOD
        # run_metric $DATASET_NAME $METHOD

        # if [ $METHOD == "umap" ]; then
        #    run_score_umap_2D $DATASET_NAME
        # fi

        plot_compare_scores $DATASET_NAME $METHOD
        # gen_metamap $DATASET_NAME $METHOD
        # gen_grid_viz_demo $DATASET_NAME $METHOD
    done
done

if [ $RUN_ALL = true ]; then
    echo "Copy figures for latex"
    bash copy_figures_for_latex.sh
fi


# V3- TAI

###############################################################################
# PLOT OVERVIEW ALL SCORE + KL LOSS
# Fig.1 (kl loss) + Fig.3 (all_scores_all_methods.pdf)
#       python3 run_score.py --plot_all_score


###############################################################################
# PLOT COMPARE F-SCORE, AUC[RNX], BIC for TSNE (Sec.6)
# Fig.6 compare 3 scores for tsne (Sec.6.1)
# Repeat for other datasets. Edit run_score.py L497 for show_range
#       python3 run_score.py -d DIGITS -m tsne --plot-compare


###############################################################################
# PLOT METAMAP + SAMPLE VIZ (Sec.6)
# Fig.8+9: Metamap Neuron_1K + COIL20
#       python3 plot_viz.py -d COIL20 -m umap --plot_metamap
#       python3 plot_viz.py -d NEURON_1K -m tsne --plot_metamap
# Fig.9: Sample viz (with metamap) for COIL20
#       python3 plot_viz.py -d COIL20 -m umap --show_viz_grid
# Fig.8: Sample viz
#       python3 plot_viz.py -d NEURON_1K -m tsne --show_viz_grid


###############################################################################
# PLOT SCORE FLEXIBILITY
# Fig.5: Score flexibility for 3 datasets: NEURON_1K, FASHION_MOBILENET, 20NEWS5
# Only use method_name (-m) tsne
#       python3 plot_viz.py -d NEURON_1K -m tsne --plot_score_flexibility


###############################################################################
# PLOT COMPARISION (Sec.6)
# Fig.7 Compare f-score + AUC[RNX] for umap embeddings.
#       python3 bo_plot.py

###############################################################################
# PLOT CHARACTERISTICS of F-SCORE (Sec.5)
# Fig.4 Stability of f-score for COIL20 dataset
#       python3 run_score.py -d COIL20 -m tsne --plot --sc qij --debug
#       python3 run_score.py -d COIL20 -m largevis --plot --sc qij --debug
#       python3 run_score.py -d COIL20 -m umap --plot --sc qij --debug


###############################################################################
# PLOT BO PREDICTION (Sec.7): Using Exploration (EI) with large xi (0.25)
# Fig.11 BO prediction for umap (50 runs)
#       python3 bo_constraint.py --seed 42 -d DIGITS -m umap -u ei -x 0.25 -nr 50 --plot
#       python3 bo_constraint.py --seed 42 -d COIL20 -m umap -u ei -x 0.25 -nr 50 --plot
#       python3 bo_constraint.py --seed 42 -d FASHION1000 -m umap -u ei -x 0.25 -nr 50 --plot
#       python3 bo_constraint.py --seed 42 -d FASHION_MOBILENET -m umap -u ei -x 0.25 -nr 50 --plot
#       python3 bo_constraint.py --seed 42 -d 20NEWS5 -m umap -u ei -x 0.25 -nr 50 --plot
#       python3 bo_constraint.py --seed 42 -d NEURON_1K -m umap -u ei -x 0.25 -nr 50 --plot

# Fig.10 same thing for tsne (15 runs)



