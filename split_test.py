import os
from os.path import join, exists, basename
import argparse
import numpy as np

import paddle
paddle.seed(0)
import paddle.nn as nn
import pgl

from pahelix.model_zoo.gem_model import GeoGNNModel
from pahelix.utils import load_json_config
from pahelix.datasets.inmemory_dataset import InMemoryDataset

from src.model import DownstreamModel
from src.featurizer import DownstreamTransformFn, DownstreamCollateFn
from src.utils import get_dataset, create_splitter, get_downstream_task_names, \
        calc_rocauc_score, exempt_parameters

def main(args):
    

    
    ### load data
    # featurizer:
    #     Gen features according to the raw data and return the graph data.
    #     Collate features about the graph data and return the feed dictionary.
    # splitter:
    #     split type of the dataset:random,scaffold,random with scaffold. Here is randomsplit.
    #     `ScaffoldSplitter` will firstly order the compounds according to Bemis-Murcko scaffold, 
    #     then take the first `frac_train` proportion as the train set, the next `frac_valid` proportion as the valid set 
    #     and the rest as the test set. `ScaffoldSplitter` can better evaluate the generalization ability of the model on 
    #     out-of-distribution samples. Note that other splitters like `RandomSplitter`, `RandomScaffoldSplitter` 
    #     and `IndexSplitter` is also available."
    task_names=get_downstream_task_names(args.dataset_name, args.data_path)
    if args.task == 'data':
        print('Preprocessing data...')
        pickfile="cached_data/bbbp/bbbp.pkl"
        pickfile=None
        dataset = get_dataset(args.dataset_name, args.data_path, task_names)
        transform_fn = DownstreamTransformFn(pos_file=pickfile,mode=args.mode)
        dataset.transform(transform_fn, num_workers=args.num_workers)
        dataset.save_data(args.cached_data_path)
    else:
        if args.cached_data_path is None or args.cached_data_path == "":
            dataset = get_dataset(args.dataset_name, args.data_path, task_names)
            dataset.transform(DownstreamTransformFn(), num_workers=args.num_workers)
        else:
            dataset = InMemoryDataset(npz_data_path=args.cached_data_path)

    splitter = create_splitter(args.split_type)
    train_dataset, valid_dataset, test_dataset = splitter.split(dataset,0.8,0.1,0.1)
    dataset._save_npz_data(test_dataset, args.cached_data_path+"/TestDataset")
    dataset._save_npz_data(train_dataset, args.cached_data_path+"/TrainDataset")
    dataset._save_npz_data(valid_dataset, args.cached_data_path+"/ValidDataset")
    print(f"{args.dataset_name}:Test dataset splitted.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=['train', 'data'], default='train')
    parser.add_argument("--mode",choices=['rdkit', 'mmffless','geomol','graph'],default='rdkit')

    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--dataset_name",type=str)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--cached_data_path", type=str, default=None)
    parser.add_argument("--split_type", 
            choices=['random', 'scaffold', 'random_scaffold', 'index','saved'])
    parser.add_argument("--compound_encoder_config", type=str)
    parser.add_argument("--model_config", type=str)
    parser.add_argument("--init_model", type=str)
    parser.add_argument("--exp_id", type=int, help='used for identification only')
    args = parser.parse_args()
    
    main(args)
