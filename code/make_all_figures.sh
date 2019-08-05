# Generate all figures,
# then copy them from code folder to tex/figures folder

# (1) figures to show the stability of the scores
python run_score.py  -d DIGITS -m umap --seed 2019 --plot --debug --use_log_scale
python run_score.py  -d DIGITS -m tsne --seed 2019 --plot --debug --use_log_scale
python run_score.py  -d COIL20 -m umap --seed 2019 --plot --debug --use_log_scale
python run_score.py  -d COIL20 -m tsne --seed 2019 --plot --debug --use_log_scale

# copy for latex
cp plots/COIL20/umap/scores_with_std_dof1.0.png ../tex/figures/COIL20_umap_scores.png
cp plots/COIL20/tsne/scores_with_std_dof1.0.png ../tex/figures/COIL20_tsne_scores.png
cp plots/DIGITS/umap/scores_with_std_dof1.0.png ../tex/figures/DIGITS_umap_scores.png
cp plots/DIGITS/tsne/scores_with_std_dof1.0.png ../tex/figures/DIGITS_tsne_scores.png
