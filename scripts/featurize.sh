#!/bin/bash
datasets="esol freesolv lipophilicity qm7 qm8 qm9 bbbp bace clintox tox21 toxcast sider hiv muv pcba"
modes="rdkit "
batch_size=128
source activate gem
for dataset in $datasets; do
    for mode in $modes; do
        {
          python feature_extractor.py \
                  --batch_size=$batch_size --dataset_name=$dataset \
                  --split_type=scaffold --data_path="./Raw_MoleculeNet/$dataset" \
                  --cached_data_path="./Pre-Processed_TestSplit_MoleculeNet/$dataset/$mode/" --compound_encoder_config=model_configs/geognn_l8.json \
                  --model_config=model_configs/down_mlp2.json --init_model="./pretrain_models-chemrl_gem/regr.pdparams" \
                  --model_dir="./output/finetune/$dataset/" #> "./$dataset-$mode-result$num.txt"
        }
    done
done
