# Generate all figures,
# then copy them from code folder to tex/figures folder

# (1) figures to show the stability of the scores

# copy for latex
cp plots/COIL20/umap/scores_with_std_dof1.0.png ../tex/figures/COIL20_umap_scores.png
cp plots/COIL20/tsne/scores_with_std_dof1.0.png ../tex/figures/COIL20_tsne_scores.png
cp plots/COIL20/largevis/scores_with_std_dof1.0.png ../tex/figures/COIL20_largevis_scores.png

cp plots/DIGITS/umap/scores_with_std_dof1.0.png ../tex/figures/DIGITS_umap_scores.png
cp plots/DIGITS/tsne/scores_with_std_dof1.0.png ../tex/figures/DIGITS_tsne_scores.png
cp plots/DIGITS/largevis/scores_with_std_dof1.0.png ../tex/figures/DIGITS_largevis_scores.png

