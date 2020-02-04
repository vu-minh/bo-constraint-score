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

if $COPY; then
	echo "Copy V2-Fig2, V2-Fig4"
	# cp plots/all_scores_all_methods.$EXT $TARGET_DIR/all_scores_all_methods.$EXT
	# cp plots/all_kl_loss.$EXT $TARGET_DIR/all_kl_loss.$EXT
fi

################################################################################
# Figure showing score stability for COIL20
# See: plot_score::plot_scores
# V2-Fig5

# python run_score.py -d COIL20 -m tsne --use_log_scale --debug --plot
# python run_score.py -d COIL20 -m largevis --use_log_scale --debug --plot
# python run_score.py -d COIL20 -m umap --use_log_scale --debug --plot

if $COPY; then
	echo "Copy V2-Fig5"
	# cp plots/COIL20/umap/scores.$EXT $TARGET_DIR/COIL20_umap_scores.$EXT
	# cp plots/COIL20/tsne/scores.$EXT $TARGET_DIR/COIL20_tsne_scores.$EXT
	# cp plots/COIL20/largevis/scores.$EXT $TARGET_DIR/COIL20_largevis_scores.$EXT
fi


################################################################################
# (2D contour UMAP) Compare f_score and AUC_logRNX for all 6 datasets
# See: bo_plot::plot_density_for_all_datasets
# V2-Fig10

# python bo_plot.py

if $COPY; then
	echo "Copy V2-Fig10"
	cp plots/umap2D_compare.$EXT $TARGET_DIR/umap2D_compare.$EXT
fi


################################################################################
# Score flexibility
# See: plot_viz::plot_viz_with_score_flexibility
# V2-Fig6

# Run BayOpt to find best params
# python bo_constraint.py --seed 42 -d FASHION_MOBILENET --use_other_label class_matcat -m tsne -u ei -x 0.1 --plot --run -nr 15
# python plot_viz.py -d FASHION_MOBILENET -m tsne --plot_score_flexibility

# python bo_constraint.py --seed 42 -d 20NEWS5 --use_other_label matcat -m tsne -u ei -x 0.1 --plot --run -nr 15
# python plot_viz.py -d 20NEWS5 -m tsne --plot_score_flexibility

# python bo_constraint.py --seed 42 -d NEURON_1K --use_other_label umi -m tsne -u ei -x 0.1 --plot --run -nr 15
# python plot_viz.py -d NEURON_1K -m tsne --plot_score_flexibility

if $COPY; then
	echo "Copy V2-Fig6"
	# cp plots/FASHION_MOBILENET/tsne/score_flexibility.$EXT $TARGET_DIR/FASHION_MOBILENET_score_flexibility.$EXT
	# cp plots/20NEWS5/tsne/score_flexibility.$EXT $TARGET_DIR/20NEWS5_score_flexibility.$EXT
	# cp plots/NEURON_1K/tsne/score_flexibility.$EXT $TARGET_DIR/NEURON_1K_score_flexibility.$EXT
fi



################################################################################
# Compare f_score, ACU_RNX, BIC
# See plot_score::plot_compare_qij_rnx_bic
# V2: Fig8

# python run_score.py -d DIGITS -m tsne --plot_compare
# python run_score.py -d COIL20 -m tsne --plot_compare
# python run_score.py -d FASHION1000 -m tsne --plot_compare
# python run_score.py -d FASHION_MOBILENET -m tsne --plot_compare
# python run_score.py -d 20NEWS5 -m tsne --plot_compare
# python run_score.py -d NEURON_1K -m tsne --plot_compare

if $COPY; then
	echo "Copy V2-Fig8"
	# cp plots/DIGITS/tsne/plot_compare.$EXT ${TARGET_DIR}/DIGITS_tsne_compare_scores.$EXT
	# cp plots/COIL20/tsne/plot_compare.$EXT ${TARGET_DIR}/COIL20_tsne_compare_scores.$EXT
	# cp plots/FASHION1000/tsne/plot_compare.$EXT ${TARGET_DIR}/FASHION1000_tsne_compare_scores.$EXT
	# cp plots/FASHION_MOBILENET/tsne/plot_compare.$EXT ${TARGET_DIR}/FASHION_MOBILENET_tsne_compare_scores.$EXT
	# cp plots/20NEWS5/tsne/plot_compare.$EXT ${TARGET_DIR}/20NEWS5_tsne_compare_scores.$EXT
	# cp plots/NEURON_1K/tsne/plot_compare.$EXT ${TARGET_DIR}/NEURON_1K_tsne_compare_scores.$EXT
fi


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

	# cp plots/COIL20/umap/metamap_scores_100.$EXT ${TARGET_DIR}/COIL20_umap_metamap.$EXT
	# cp plots/COIL20/umap/show.$EXT $TARGET_DIR/COIL20_umap_show.$EXT
fi

################################################################################
# Run BayOpt for tsne 1D
# add --run to re-run
# V2: Fig11

# python bo_constraint.py -d DIGITS 			-m tsne -nr 15 -u ei -x 0.1 --plot --seed 2018 
# python bo_constraint.py -d COIL20 				-m tsne -nr 15 -u ei -x 0.1 --plot --seed 2018 --run
# python bo_constraint.py -d FASHION1000 			-m tsne -nr 15 -u ei -x 0.1 --plot --seed 2018 --run
# python bo_constraint.py -d FASHION_MOBILENET 	-m tsne -nr 15 -u ei -x 0.1 --plot --seed 2018 --run 
# python bo_constraint.py -d 20NEWS5 				-m tsne -nr 15 -u ei -x 0.1 --plot --seed 2018 --run 
# python bo_constraint.py -d NEURON_1K 			-m tsne -nr 15 -u ei -x 0.1 --plot --seed 24 --run

if $COPY; then
	echo "Copy V2-Fig11"
	# cp plots/DIGITS/tsne/qij/bo_summary.$EXT ${TARGET_DIR}/DIGITS_tsne_bo.$EXT
	# cp plots/COIL20/tsne/qij/bo_summary.$EXT ${TARGET_DIR}/COIL20_tsne_bo.$EXT
	# cp plots/FASHION1000/tsne/qij/bo_summary.$EXT ${TARGET_DIR}/FASHION1000_tsne_bo.$EXT
	# cp plots/FASHION_MOBILENET/tsne/qij/bo_summary.$EXT ${TARGET_DIR}/FASHION_MOBILENET_tsne_bo.$EXT
	# cp plots/20NEWS5/tsne/qij/bo_summary.$EXT ${TARGET_DIR}/20NEWS5_tsne_bo.$EXT
	# cp plots/NEURON_1K/tsne/qij/bo_summary.$EXT ${TARGET_DIR}/NEURON_1K_tsne_bo.$EXT
fi



################################################################################
# Run BayOpt for umap 2D
# add --run to re-run
# See: bo_plot::plot_prediction_density_2D
# Note: run COIL20, now disable pca when loading dataset (pca=None)
# V2: Fig12

# python bo_constraint.py -d DIGITS 				-m umap -nr 40 -u ei -x 0.1 --plot --seed 2018 --run
# python bo_constraint.py -d COIL20 				-m umap	-nr 40 -u ei -x 0.1 --plot --seed 2018 --run
# python bo_constraint.py -d FASHION1000 			-m umap -nr 40 -u ei -x 0.1 --plot --seed 2018 --run
# python bo_constraint.py -d FASHION_MOBILENET 	-m umap -nr 40 -u ei -x 0.1 --plot --seed 2018 --run
# python bo_constraint.py -d 20NEWS5 				-m umap -nr 40 -u ei -x 0.1 --plot --seed 2018 --run
# python bo_constraint.py -d NEURON_1K 			-m umap -nr 40 -u ei -x 0.1 --plot --seed 2018 --run

if $COPY; then
	echo "Copy V2-Fig12"
	# cp plots/DIGITS/umap/qij/predicted_score.$EXT ${TARGET_DIR}/DIGITS_umap_predicted_score.$EXT
	# cp plots/COIL20/umap/qij/predicted_score.$EXT ${TARGET_DIR}/COIL20_umap_predicted_score.$EXT
	# cp plots/FASHION1000/umap/qij/predicted_score.$EXT ${TARGET_DIR}/FASHION1000_umap_predicted_score.$EXT
	# cp plots/FASHION_MOBILENET/umap/qij/predicted_score.$EXT ${TARGET_DIR}/FASHION_MOBILENET_umap_predicted_score.$EXT
	# cp plots/20NEWS5/umap/qij/predicted_score.$EXT ${TARGET_DIR}/20NEWS5_umap_predicted_score.$EXT
	# cp plots/NEURON_1K/umap/qij/predicted_score.$EXT ${TARGET_DIR}/NEURON_1K_umap_predicted_score.$EXT
fi

################################################################################
# Plot samples for table dataset
# See plot_viz::plot_samples

# python plot_viz.py -d DIGITS --plot_samples
# python plot_viz.py -d COIL20 --plot_samples
# python plot_viz.py -d FASHION1000 --plot_samples

if $COPY; then
	echo "Copy samples for table datasets"
	# cp plots/DIGITS/DIGITS_samples.$EXT $TARGET_DIR/DIGITS_samples.$EXT
	# cp plots/COIL20/COIL20_samples.$EXT $TARGET_DIR/COIL20_samples.$EXT
	# cp plots/FASHION1000/FASHION1000_samples.$EXT $TARGET_DIR/FASHION1000_samples.$EXT
fi
