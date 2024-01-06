import os
from os.path import join, exists, basename
import argparse
import numpy as np
import paddle.fluid as fluid
import paddle
paddle.seed(0)
import random
np.random.seed(42) 
random.seed(42)
fluid.default_startup_program().random_seed = 42
fluid.default_main_program().random_seed = 42
import paddle.nn as nn
import pgl
from tqdm import tqdm
from pahelix.model_zoo.gem_model import GeoGNNModel
from pahelix.utils import load_json_config
from pahelix.datasets.inmemory_dataset import InMemoryDataset

from src.model import DownstreamModel
from src.featurizer import DownstreamTransformFn, DownstreamCollateFn
from src.utils import get_dataset, create_splitter, get_downstream_task_names, get_dataset_stat, \
        calc_rocauc_score, calc_rmse, calc_mae, exempt_parameters


def feature_extract_class(args, model,dataset, collate_fn,datasetType):
    data_gen = dataset.get_data_loader(
            batch_size=args.batch_size, 
            num_workers=args.num_workers, 
            shuffle=False,
            collate_fn=collate_fn)
    total_features=[]
    total_label = []
    smiles_list=[]
    model.eval()
    total_valid = []
    model.eval()
    for atom_bond_graphs, bond_angle_graphs, valids, labels,smiles,atom_poses in tqdm(data_gen,desc="Progress.."):
        atom_bond_graphs = atom_bond_graphs.tensor()
        bond_angle_graphs = bond_angle_graphs.tensor()
        labels = paddle.to_tensor(labels, 'float32')
        valids = paddle.to_tensor(valids, 'float32')
        features = model(atom_bond_graphs, bond_angle_graphs)
        smiles_list.extend(smiles)
        total_features.append(features.numpy())
        total_valid.append(valids.numpy())
        total_label.append(labels.numpy())
    total_features = np.concatenate(total_features, 0)
    total_label = np.concatenate(total_label, 0)
    total_valid = np.concatenate(total_valid, 0)
    final_dic={"smiles":[],"features":[]}
    task_names = get_downstream_task_names(args.dataset_name, args.data_path)
    for task_name in task_names:
        final_dic["lbl_"+task_name]=[]
    for i in tqdm(range(total_label.shape[0]),desc="Organaizing:"):
        final_dic["smiles"].append(smiles_list[i])
        final_dic["features"].append(total_features[i])
        for j in range(len(total_label[i])):
            valid=total_valid[i][j].item()
            if valid:
                final_dic["lbl_"+task_names[j]].append(total_label[i][j].item())
            else:
                final_dic["lbl_"+task_names[j]].append(None)
    import pickle
    if  not os.path.exists(f'./{args.model_dir}'):
        os.mkdir(f'./{args.model_dir}')
    with open(f'./{args.model_dir}/{args.dataset_name}-{datasetType}-{args.mode}.pickle', 'wb') as handle:
        pickle.dump(final_dic, handle, protocol=pickle.HIGHEST_PROTOCOL)
def feature_extract_regr(args, model,dataset, collate_fn,datasetType):
    data_gen = dataset.get_data_loader(
            batch_size=args.batch_size, 
            num_workers=args.num_workers, 
            shuffle=False,
            collate_fn=collate_fn)
    total_features=[]
    total_label = []
    smiles_list=[]
    model.eval()
    for atom_bond_graphs, bond_angle_graphs, labels,smiles,atom_poses in tqdm(data_gen,desc="Progress.."):
        atom_bond_graphs = atom_bond_graphs.tensor()
        bond_angle_graphs = bond_angle_graphs.tensor()
        labels = paddle.to_tensor(labels, 'float32')
        features = model(atom_bond_graphs, bond_angle_graphs)
        smiles_list.extend(smiles)
        total_features.append(features.numpy())
        total_label.append(labels.numpy())
    total_features = np.concatenate(total_features, 0)
    total_label = np.concatenate(total_label, 0)
    final_dic={"smiles":[],"features":[]}
    task_names = get_downstream_task_names(args.dataset_name, args.data_path)
    for task_name in task_names:
        final_dic["lbl_"+task_name]=[]
    for i in tqdm(range(total_label.shape[0]),desc="Organaizing:"):
        final_dic["smiles"].append(smiles_list[i])
        final_dic["features"].append(total_features[i])
        for j in range(len(total_label[i])):
            final_dic["lbl_"+task_names[j]].append(total_label[i][j].item())
    import pickle
    if  not os.path.exists(f'{args.model_dir}'):
        os.mkdir(f'{args.model_dir}')
    with open(f'./{args.model_dir}/{args.dataset_name}-{datasetType}-{args.mode}.pickle', 'wb') as handle:
        pickle.dump(final_dic, handle, protocol=pickle.HIGHEST_PROTOCOL)
def main(args):
    """
    Call the configuration function of the model, build the model and load data, then start training.
    model_config:
        a json file  with the hyperparameters,such as dropout rate ,learning rate,num tasks and so on;
    num_tasks:
        it means the number of task that each dataset contains, it's related to the dataset;
    """
    ### config for the body
    compound_encoder_config = load_json_config(args.compound_encoder_config)

    
    task_type = args.task
    task_names = get_downstream_task_names(args.dataset_name, args.data_path)
    
    model_config = load_json_config(args.model_config)

    model_config['task_type'] = task_type
    model_config['num_tasks'] = len(task_names)
    print('model_config:')
    print(model_config)

    ### build model
    compound_encoder = GeoGNNModel(compound_encoder_config)
    model = DownstreamModel(model_config, compound_encoder,featurize=True)
    encoder_params = compound_encoder.parameters()
    head_params = exempt_parameters(model.parameters(), encoder_params)

    print('Total param num: %s' % (len(model.parameters())))
    print('Encoder param num: %s' % (len(encoder_params)))
    print('Head param num: %s' % (len(head_params)))
    for i, param in enumerate(model.named_parameters()):
        print(i, param[0], param[1].name)

    if not args.init_model is None and not args.init_model == "":
        model.set_state_dict(paddle.load(args.init_model))
        print('Load state_dict from %s' % args.init_model)

    print('Read preprocessing data...')
    train_dataset = InMemoryDatas et(npz_data_path=args.cached_data_path+"/TrainDataset")
    valid_dataset = InMemoryDataset(npz_data_path=args.cached_data_path+"/ValidDataset")
    test_dataset = InMemoryDataset(npz_data_path=args.cached_data_path+"/TestDataset")

    collate_fn = DownstreamCollateFn(
            atom_names=compound_encoder_config['atom_names'], 
            bond_names=compound_encoder_config['bond_names'],
            bond_float_names=compound_encoder_config['bond_float_names'],
            bond_angle_float_names=compound_encoder_config['bond_angle_float_names'],
            task_type=task_type,is_inference=True)
    if args.task=="class":
        feat_func=feature_extract_class
    else:
        feat_func=feature_extract_regr

    feat_func(
            args, model, 
            train_dataset, collate_fn,"TrainDataset")
    feat_func(
            args, model, 
            valid_dataset, collate_fn,"ValidDataset")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=['class','regr'], default='train')
    parser.add_argument("--mode",choices=['rdkit', 'mmffless','geomol','graph'],default='rdkit')
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--dataset_name",type=str)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--cached_data_path", type=str, default=None)
    parser.add_argument("--split_type", 
            choices=['random', 'scaffold', 'random_scaffold', 'index','saved'])

    parser.add_argument("--compound_encoder_config", type=str)
    parser.add_argument("--model_config", type=str)
    parser.add_argument("--init_model", type=str)
    parser.add_argument("--model_dir", type=str)
    args = parser.parse_args()
    if args.dataset_name not in ['esol', 'freesolv', 'lipophilicity', 
                'qm7', 'qm8', 'qm9', 'qm9_gdb']:
        args.task="class"
    else:
        args.task="regr"
    main(args)