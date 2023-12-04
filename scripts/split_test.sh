#!/bin/bash
datasets="esol freesolv lipophilicity qm7 qm8 qm9 bbbp bace clintox tox21 toxcast sider hiv pcba muv "
modes="rdkit "
source activate gem
for dataset in $datasets; do
    for mode in $modes; do
          python split_test.py --dataset_name=$dataset --split_type=scaffold \
          --data_path="./chemrl_downstream_datasets/$dataset" --cached_data_path="./Pre-Processed_TestSplit_MoleculeNet/$dataset/$mode/"
    done
done
