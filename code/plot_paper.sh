# script to reproduce the figure in the paper

# Version V2: 22/01/2020

################################################################################
# Show all scores for all dataset (tsne, largevis, umap with mind_dist 0.1)
# See: plot_score::plot_all_score_all_method_all_dataset
# V2-Fig4 : all f_score
# V2-Fig2 : all kl loss

python run_score.py --plot_all_score


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
# Run BayOpt for tsne 1D
# python bo_constraint.py -d DIGITS -m tsne -nr 15 -u ei -x 0.05 --run --plot --seed 2018
# python bo_constraint.py -d COIL20 -m tsne -nr 15 -u ei -x 0.05 --run --plot --seed 2018
# python bo_constraint.py -d FASHION1000 -m tsne -nr 15 -u ei -x 0.05 --run --plot --seed 2018
# python bo_constraint.py -d FASHION_MOBILENET -m tsne -nr 15 -u ei -x 0.05 --run --plot --seed 1024
# python bo_constraint.py -d 20NEW5 -m tsne -nr 15 -u ei -x 0.05 --run --plot --seed 1024
# python bo_constraint.py -d NEURON_1K -m tsne -nr 15 -u ei -x 0.05 --run --plot --seed 1024


################################################################################
# Run BayOpt for umap 2D
# python bo_constraint.py -d DIGITS -m umap -nr 40 -u ei --run --plot
# python bo_constraint.py -d COIL20 -m umap -nr 40 -u ei --run --plot
# python bo_constraint.py -d FASHION1000 -m umap -nr 40 -u ei --run --plot
# python bo_constraint.py -d FASHION_MOBILENET -m umap -nr 40 -u ei --run --plot
# python bo_constraint.py -d 20NEW5 -m umap -nr 40 -u ei --run --plot
# python bo_constraint.py -d NEURON_1K -m umap -nr 40 -u ei --run --plot
