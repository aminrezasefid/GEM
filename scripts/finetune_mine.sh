#!/bin/bash
datasets="bace bbbp tox21 toxcast clintox sider hiv muv pcba "
modes="rdkit "
epoch=1
batch_size=64
source activate gem
for dataset in $datasets; do
    for mode in $modes; do
        for num in $(seq 1 1); do
        {
          python finetune_class.py \
                  --batch_size=$batch_size --max_epoch=$epoch --dataset_name=$dataset \
                  --split_type=scaffold --data_path="./Raw_MoleculeNet/$dataset" \
                  --seed=$num \
                  --cached_data_path="./Pre-Processed_TestSplit_MoleculeNet/$dataset/$mode/TrainValid/" --compound_encoder_config=model_configs/geognn_l8.json \
                  --model_config=model_configs/down_mlp2.json --init_model="./pretrain_models-chemrl_gem/regr.pdparams" \
                  --model_dir="./output/finetune/$dataset/$mode/seed=$num" --encoder_lr=1e-3 --head_lr=4e-3 \
                  --dropout_rate=0.1 > "./$dataset-$mode-result$num.txt"
        }
        done
    done
done
