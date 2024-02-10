from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Dataset
from gensim.models.doc2vec import Doc2Vec
from cfg_generation import *

import pandas as pd
import numpy as np
import os
import pickle

import argparse
import gensim
import torch
import torch_geometric

# Please modify your project dir
project_dir = "/media/ubuntu/My_Passport/PonziDetector"


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--split", type=str, default="split1")
    args = parser.parse_args()
    return args


class CFGDataset(Dataset):
    def __init__(self, root, addr_list, label_list, transform=None, pre_transform=None, pre_filter=None):
        self.addr_list = addr_list
        self.label_list = label_list
        self.root = root
        self.transform = transform
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter
        super(CFGDataset, self).__init__(root, transform, pre_transform, pre_filter)
        
        self.data_list = torch.load(os.path.join(self.processed_dir, self.processed_file_names[0]))
        
    @property
    def raw_dir(self):
        return os.path.join(self.root, 'raw')
    
    @property
    def processed_dir(self):
        return os.path.join(self.root, 'processed')
    
    @property
    def raw_file_names(self):
        return []
    
    @property
    def processed_file_names(self):
        return ["cfg_dataset.pt"]

    def process(self):
        data_list = []
        
        for k, addr in enumerate(self.addr_list):
            if not os.path.exists(project_dir + "/data/.temp/" + addr + "/"):
                data_list.append(None)
                continue

            print(k, addr)
            b = construct_ir_table(addr)
            find_state_var_dependency(b, addr)
            G = contract_graph(b, addr)
            
            block2idx = np.load(project_dir + "/data/facts/" + addr + "/block2idx.npy", allow_pickle=True).item()
            idx2block = np.load(project_dir + "/data/facts/" + addr + "/idx2block.npy", allow_pickle=True).item()

            x = torch.zeros((G.vcount(), args.dim), dtype=torch.float32)
            y = torch.tensor(self.label_list[k], dtype=torch.long)
            edge_index = torch.zeros((2, 2 * G.ecount()), dtype=torch.long)
            
            for i in range(G.vcount()):
                bk = idx2block[i]
                opcode = (b[b["blockname"] == bk].reset_index(drop=True))["op"].values.tolist()
                node_embedding = doc2vec_model.infer_vector(opcode)
                x[i, :] = torch.from_numpy(node_embedding)
            
            for i, e in enumerate(G.es()):
                src, trg = e.source, e.target
                edge_index[0, 2 * i] = src
                edge_index[1, 2 * i] = trg
                edge_index[0, 2 * i + 1] = trg
                edge_index[1, 2 * i + 1] = src
                
            d = Data(x=x, edge_index=edge_index, y=y, num_nodes=G.vcount())
            data_list.append(d)
            print()

        torch.save(data_list, os.path.join(self.processed_dir, self.processed_file_names[0]))
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        if self.data_list[idx] != None:
            self.data_list[idx]["address"] = self.addr_list[idx]
        return self.data_list[idx]


def get_split_index(split):
    df = pd.read_csv(project_dir + "/data/PonziDataset_20230906.csv")
    addr_list = df["address"].values.tolist()
    label_list = df["label"].values.tolist()
    
    train_idx = pickle.load(open(project_dir + "/data/split/random/train_idx_{}.pkl".format(split), "rb"))
    val_idx = pickle.load(open(project_dir + "/data/split/random/val_idx_{}.pkl".format(split), "rb"))
    test_idx = pickle.load(open(project_dir + "/data/split/random/test_idx_{}.pkl".format(split), "rb"))

    addr_list_error = sorted(list(set(addr_list) - set(os.listdir(project_dir + "/data/.temp/"))))
    addr_idx_error = [addr_list.index(addr) for addr in addr_list_error]
    
    train_idx = sorted(list(set(train_idx) - set(addr_idx_error)))
    val_idx = sorted(list(set(val_idx) - set(addr_idx_error)))
    test_idx = sorted(list(set(test_idx) - set(addr_idx_error)))
    
    return train_idx, val_idx, test_idx, addr_list, label_list, addr_list_error, addr_idx_error



if __name__ == "__main__":
    args = get_args()
    train_idx, _, _, addr_list, label_list, _, _ = get_split_index(args.split)

    doc2vec_model = Doc2Vec.load(project_dir + "/algorithm/model_files/doc2vec_{}dim".format(args.dim))
    dataset = CFGDataset(
        root = project_dir + "/algorithm/dataset_files/dataset_{}dim/".format(args.dim),
        addr_list = addr_list,
        label_list = label_list
    )
    print(dataset)
    print(dataset[0])
