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

