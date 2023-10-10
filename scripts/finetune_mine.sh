#!/bin/bash
datasets="esol freesolv qm7 "
modes="rdkit geomol mmffless"
epoch=100
batch_size=128
source activate gem
for dataset in $datasets; do
    for mode in $modes; do
        for num in $(seq 1 10); do
        {
          batch_size=128
          if [ $dataset = "bbbp" ] || [ $dataset = "bace" ]; then
            model="class"
          elif [ "$dataset" == "freesolv" ]; then
            batch_size=30
            model="class"
          else
            model="regr"
          fi
          python finetune_$model.py \
                  --batch_size=$batch_size --max_epoch=$epoch --dataset_name=$dataset \
                  --split_type=scaffold --data_path="./chemrl_downstream_datasets/$dataset" \
                  --cached_data_path="./cached_data/$dataset/$mode" --compound_encoder_config=model_configs/geognn_l8.json \
                  --model_config=model_configs/down_mlp2.json --init_model="./pretrain_models-chemrl_gem/$model.pdparams" \
                  --model_dir="./output/chemrl_gem/finetune/$dataset" --encoder_lr=1e-3 --head_lr=4e-3 \
                  --dropout_rate=0.1 > "./$dataset-$mode-result$num.txt"
        }
        done
    done
done
