# run BayOpt for each dataset

# update 20/09/2020, re-run umap, make sure the range of min_dist is [0.01, 1.0]

declare -a LIST_DATASETS=("DIGITS" "COIL20" "FASHION1000" "FASHION_MOBILENET" "20NEWS5" "NEURON_1K")

for DATASET_NAME in "${LIST_DATASETS[@]}"; do
    python bo_constraint.py --seed 2020 -d $DATASET_NAME -m umap \
        -u ei -x 0.25 --plot --run -nr 50
done

