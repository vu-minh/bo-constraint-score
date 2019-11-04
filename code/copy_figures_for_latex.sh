#!/bin/bash -x

# Generate all figures,
# then copy them from code folder to tex/figures folder

LIST_DATASETS=("DIGITS" "COIL20" "FASHION1000" "FASHION_MOBILENET" "NEURON_1K" "20NEWS5")
LIST_METHODS=("tsne" "umap")
TARGET_DIR="../tex/figures"

# # (1) figures to show the stability of the scores

cp plots/COIL20/umap/scores.png ../tex/figures/COIL20_umap_scores.png
cp plots/COIL20/tsne/scores.png ../tex/figures/COIL20_tsne_scores.png
cp plots/COIL20/largevis/scores.png ../tex/figures/COIL20_largevis_scores.png

cp plots/DIGITS/umap/scores.png ../tex/figures/DIGITS_umap_scores.png
cp plots/DIGITS/tsne/scores.png ../tex/figures/DIGITS_tsne_scores.png
cp plots/DIGITS/largevis/scores.png ../tex/figures/DIGITS_largevis_scores.png


# # (2) figures contour score for UMAP 2D
cp plots/DIGITS/umap/2D/qij_score.png ../tex/figures/DIGITS_umap_qij_score.png
cp plots/DIGITS/umap/2D/auc_rnx.png ../tex/figures/DIGITS_umap_auc_rnx.png

cp plots/COIL20/umap/2D/qij_score.png ../tex/figures/COIL20_umap_qij_score.png
cp plots/COIL20/umap/2D/auc_rnx.png ../tex/figures/COIL20_umap_auc_rnx.png

cp plots/FASHION1000/umap/2D/qij_score.png ../tex/figures/FASHION1000_umap_qij_score.png
cp plots/FASHION1000/umap/2D/auc_rnx.png ../tex/figures/FASHION1000_umap_auc_rnx.png


# # (3) figures contour predicted score from BO
cp plots/DIGITS/umap/qij/predicted_score.png ../tex/figures/DIGITS_umap_predicted_score.png
cp plots/COIL20/umap/qij/predicted_score.png ../tex/figures/COIL20_umap_predicted_score.png
cp plots/FASHION1000/umap/qij/predicted_score.png ../tex/figures/FASHION1000_umap_predicted_score.png


for DATASET_NAME in "${LIST_DATASETS[@]}"; do
    for METHOD in "${LIST_METHODS[@]}"; do

        # # (4) figures comparing BIC, AUC_log_RUX and qij_score

        echo "COPY SCORE COMPARE: " $DATASET_NAME $METHOD
        cp plots/${DATASET_NAME}/${METHOD}/plot_compare.png \
            ${TARGET_DIR}/${DATASET_NAME}_${METHOD}_compare_scores.png

        # # (6) figures grid of selected vizs
        cp plots/${DATASET_NAME}/${METHOD}/show.png \
            ${TARGET_DIR}/${DATASET_NAME}_${METHOD}_show.png

        # # (1.b) Score stability for tsne
        cp plots/${DATASET_NAME}/${METHOD}/scores.png \
            ${TARGET_DIR}/${DATASET_NAME}_${METHOD}_scores.png

    done

    # # (5) figures metamap
    echo "COPY METAMAP: " $DATASET_NAME "tsne"
    cp plots/${DATASET_NAME}/tsne/metamap_scores_50.png \
       ${TARGET_DIR}/${DATASET_NAME}_tsne_metamap.png

    echo "COPY METAMAP: " $DATASET_NAME "umap"
    cp plots/${DATASET_NAME}/umap/metamap_scores_100.png \
       ${TARGET_DIR}/${DATASET_NAME}_umap_metamap.png

    # # (8) Plots BayOpt in action for tsne
    echo "COPY BayOpt prediction for tsne:"
    cp plots/${DATASET_NAME}/tsne/qij/bo_summary.png \
        ${TARGET_DIR}/${DATASET_NAME}_tsne_bo.png
done

# # (7) Score flexibility
cp plots/FASHION_MOBILENET/tsne/score_flexibility.png ../tex/figures/FASHION_MOBILENET_score_flexibility.png
cp plots/20NEWS5/tsne/score_flexibility.png ../tex/figures/20NEWS5_score_flexibility.png
cp plots/NEURON_1K/tsne/score_flexibility.png ../tex/figures/NEURON_1K_score_flexibility.png

# # (8) Copy plot for all score
cp plots/all_scores_all_methods.png ../tex/figures/all_scores_all_methods.png


################################################################################################
# Command for reproducing figures/data

# # Run score with UMAP(n_neighbors, min_dist)
# python run_score.py --seed 1024 -d QPCR -m umap --run_score_umap -nl 10 -nr 10 --plot --run

# # Plot Bic score
# python run_score.py --seed 1024 -d DIGITS -m tsne -sc bic --plot --use_log_scale

# # (4) plot compare scores: 3 scores for tsne and 2 scores for umap
# python run_score.py --seed 1024 -d DIGITS -m tsne --plot_compare --use_log_scale

# # BO
# # 20NEWS5
# python bo_constraint.py --seed 42 -d 20NEWS5 -m umap -u ei -x 0.1 --plot --run -nr 40
# python bo_constraint.py --seed 42 -d 20NEWS5 -m tsne -u ei -x 0.1 --plot --run -nr 15

# # (5) Gen metamap
# python plot_viz.py  --seed 42 -d COIL20 -m tsne --plot_metamap

# # (6) Show grid of selected vizs
# python plot_viz.py --seed 42 -d COIL20 -m tsne --show_viz_grid
