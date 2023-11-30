#!/bin/bash
datasets="esol freesolv "
modes="rdkit "
source activate gem
for dataset in $datasets; do
    for mode in $modes; do
          python split_test.py --dataset_name=$dataset --split_type=scaffold \
          --data_path="./chemrl_downstream_datasets/$dataset" --cached_data_path="./Pre-Processed MoleculeNet/$dataset/$mode/"
    done
done
