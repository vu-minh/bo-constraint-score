#!/bin/bash

# script to reproduce the figure in the paper

# Version V2: 22/01/2020, 03/02/2020


# Target folder for latex figures
TARGET_DIR="../tex/figures_pdf"
EXT="pdf"
COPY=true


################################################################################
# Show all scores for all dataset (tsne, largevis, umap with mind_dist 0.1)
# See: plot_score::plot_all_score_all_method_all_dataset
# V2-Fig4 : all f_score
# V2-Fig2 : all kl loss

# python run_score.py --plot_all_score
# if $COPY; then
# 	cp plots/all_scores_all_methods.pdf $TARGET_DIR/all_scores_all_methods.pdf
# fi

################################################################################
# Figure showing score stability for COIL20
# See: plot_score::plot_scores
# V2-Fig5

# python run_score.py -d COIL20 -m tsne --use_log_scale --debug --plot
# python run_score.py -d COIL20 -m largevis --use_log_scale --debug --plot
# python run_score.py -d COIL20 -m umap --use_log_scale --debug --plot


################################################################################
# (2D contour UMAP) Compare f_score and AUC_logRNX for all 6 datasets

# python bo_plot.py


################################################################################
# Score flexibility
# See: plot_viz::plot_viz_with_score_flexibility
# V2-Fig6

# Run BayOpt to find best params
# python bo_constraint.py --seed 42 -d FASHION_MOBILENET --use_other_label class_matcat -m tsne -u ei -x 0.1 --plot --run -nr 15
# note to `change ncol=min(3, len(label_names1))`
# python plot_viz.py -d FASHION_MOBILENET -m tsne --plot_score_flexibility

# python bo_constraint.py --seed 42 -d 20NEWS5 --use_other_label matcat -m tsne -u ei -x 0.1 --plot --run -nr 15
# note to `change ncol=min(4, len(label_names1))`
# python plot_viz.py -d 20NEWS5 -m tsne --plot_score_flexibility

# python bo_constraint.py --seed 42 -d NEURON_1K --use_other_label umi -m tsne -u ei -x 0.1 --plot --run -nr 15
# note to `change ncol=min(6, len(label_names1))`
# python plot_viz.py -d NEURON_1K -m tsne --plot_score_flexibility


################################################################################
# Compare f_score, ACU_RNX, BIC
# V2: Fig8
# python run_score.py -d DIGITS -m tsne --plot_compare
# python run_score.py -d FASHION1000 -m tsne --plot_compare
# python run_score.py -d FASHION_MOBILENET -m tsne --plot_compare
# python run_score.py -d 20NEWS5 -m tsne --plot_compare
# python run_score.py -d NEURON_1K -m tsne --plot_compare
# python run_score.py -d COIL20 -m tsne --plot_compare


################################################################################
# Metamap for tsne embeddings and selected visualizations
# See: plot_viz::plot_metamap_with_scores_tsne, plot_viz::show_viz_grid

# V2: Fig7 (1st row: metamap, 2nd row: grid of viz)
# python plot_viz.py -d NEURON_1K -m tsne --plot_metamap --debug
# python plot_viz.py -d NEURON_1K -m tsne --show_viz_grid

# V2: Fig9
# python plot_viz.py -d COIL20 -m umap --plot_metamap --debug
# python plot_viz.py -d COIL20 -m umap --show_viz_grid

if $COPY; then
	echo "Copy V2:Fig7, V2:Fig9"
	# cp plots/NEURON_1K/tsne/metamap_scores_50.$EXT ${TARGET_DIR}/NEURON_1K_tsne_metamap.$EXT
	# cp plots/NEURON_1K/tsne/show.$EXT $TARGET_DIR/NEURON_1K_tsne_show.$EXT
fi

################################################################################
# Run BayOpt for tsne 1D
# python bo_constraint.py -d DIGITS -m tsne -nr 15 -u ei -x 0.05 --plot --seed 2018
# python bo_constraint.py -d COIL20 -m tsne -nr 15 -u ei -x 0.05  --plot --seed 2018
# python bo_constraint.py -d FASHION1000 -m tsne -nr 15 -u ei -x 0.05 --plot --seed 2018
# python bo_constraint.py -d FASHION_MOBILENET -m tsne -nr 15 -u ei -x 0.05 --plot --seed 1024
# python bo_constraint.py -d 20NEWS5 -m tsne -nr 15 -u ei -x 0.1 --plot --seed 1024
# python bo_constraint.py -d NEURON_1K -m tsne -nr 15 -u ei -x 0.05 --plot --seed 1024


################################################################################
# Run BayOpt for umap 2D
# python bo_constraint.py -d DIGITS -m umap -nr 40 -u ei --run --plot
# python bo_constraint.py -d COIL20 -m umap -nr 40 -u ei --run --plot
# python bo_constraint.py -d FASHION1000 -m umap -nr 40 -u ei --run --plot
# python bo_constraint.py -d FASHION_MOBILENET -m umap -nr 40 -u ei --run --plot
# python bo_constraint.py -d 20NEWS5 -m umap -nr 40 -u ei --run --plot
# python bo_constraint.py -d NEURON_1K -m umap -nr 40 -u ei --run --plot
