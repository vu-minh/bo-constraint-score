# Generate all figures,
# then copy them from code folder to tex/figures folder

# (1) figures to show the stability of the scores

cp plots/COIL20/umap/scores.png ../tex/figures/COIL20_umap_scores.png
cp plots/COIL20/tsne/scores.png ../tex/figures/COIL20_tsne_scores.png
cp plots/COIL20/largevis/scores.png ../tex/figures/COIL20_largevis_scores.png

cp plots/DIGITS/umap/scores.png ../tex/figures/DIGITS_umap_scores.png
cp plots/DIGITS/tsne/scores.png ../tex/figures/DIGITS_tsne_scores.png
cp plots/DIGITS/largevis/scores.png ../tex/figures/DIGITS_largevis_scores.png


# (2) figures contour score for UMAP 2D
cp plots/DIGITS/umap/2D/qij_score.png ../tex/figures/DIGITS_umap_qij_score.png
cp plots/DIGITS/umap/2D/auc_rnx.png ../tex/figures/DIGITS_umap_auc_rnx.png

cp plots/COIL20/umap/2D/qij_score.png ../tex/figures/COIL20_umap_qij_score.png
cp plots/COIL20/umap/2D/auc_rnx.png ../tex/figures/COIL20_umap_auc_rnx.png


# (3) figures contour predicted score from BO
cp plots/DIGITS/umap/qij/predicted_score.png ../tex/figures/DIGITS_umap_predicted_score.png
cp plots/COIL20/umap/qij/predicted_score.png ../tex/figures/COIL20_umap_predicted_score.png


# (4) figures comparing BIC, AUC_log_RUX and qij_score
cp plots/DIGITS/tsne/plot_compare.png ../tex/figures/DIGITS_compare_scores.png
cp plots/COIL20/tsne/plot_compare.png ../tex/figures/COIL20_compare_scores.png