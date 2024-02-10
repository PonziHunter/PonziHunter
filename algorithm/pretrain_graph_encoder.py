from graph_encoder import GraphEncoder, GCLModel
from dataset import CFGDataset, get_split_index
from sampler import ImbalancedSampler

import pandas as pd
import numpy as np
import time
import os
import pickle
import random
import argparse

from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Dataset, Batch
from torch_geometric.utils import dropout_adj
from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F

# Please modify your project dir
project_dir = "/media/ubuntu/My_Passport/PonziDetector"


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--split", type=str, default="split1")
    parser.add_argument("--device", type=int, default=0)
    
    parser.add_argument("--base_model", type=str, default="SAGEConv")
    parser.add_argument("--activation", type=str, default="relu")
    parser.add_argument("--norm", type=str, default="BatchNorm")
    parser.add_argument("--pooling", type=str, default="TopKPooling")
    parser.add_argument("--use_pool", type=bool, default=True)
    parser.add_argument("--use_skip", type=bool, default=False)
    parser.add_argument("--num_layers", type=int, default=3)
    
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--proj_dim", type=int, default=32)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--biased", type=bool, default=True)
    parser.add_argument("--ablation", type=str, default="feature")
    
    parser.add_argument("--pn", type=float, default=0.6)
    parser.add_argument("--pe", type=float, default=0.6)
    
    args = parser.parse_args()
    return args


def mask_node_features_unbiased(batch_data, p_mask=0.1):
    batch_data_list = batch_data.to_data_list()
    
    for k in range(len(batch_data_list)):
        mask = torch.empty((batch_data_list[k].x.size(1),), dtype=torch.float32).uniform_(0, 1) < p_mask
        mask = mask.to(device)
        batch_data_list[k][:, mask] = 0
    
    batch_data = Batch.from_data_list(batch_data_list)
    return batch_data


def mask_node_features_biased(batch_data, p_node=0.6, p_mask=0.1):
    batch_data_list = batch_data.to_data_list()
    
    for k in range(len(batch_data_list)):
        addr = batch_data_list[k]["address"]
        idx2impt = np.load(project_dir + "/data/facts/" + addr + "/idx2impt.npy", allow_pickle=True).item()  # 节点重要性
        S = list(idx2impt.values())

        if all(s == S[0] for s in S):
            for i in range(batch_data_list[k].x.size(0)):
                mask = torch.empty((batch_data_list[k].x.size(1),), dtype=torch.float32).uniform_(0, 1) < p_mask
                mask = mask.to(device)
                batch_data_list[k].x[i, mask] = 0

        else:
            S_max, S_mean = np.max(S), np.mean(S)
            p_node_threshold = 0.5 * p_node

            p = [
                min(p_node_threshold, (S_max - idx2impt[i])/(S_max - S_mean) * p_node)
                for i in range(len(S))
            ]
            
            n = torch.empty((batch_data_list[k].x.size(0),), dtype=torch.float32).uniform_(0, 1) < torch.tensor(p)
            n = n.to(device)
            mask = torch.empty((batch_data_list[k].x.size(1),), dtype=torch.float32).uniform_(0, 1) < p_mask
            mask = mask.to(device)
            
            for i in range(len(n)):
                if n[i]: batch_data_list[k].x[i, mask] = 0
    
    batch_data = Batch.from_data_list(batch_data_list)
    return batch_data


def drop_edges_unbiased(batch_data, p_edge=0.3):
    batch_data_list = batch_data.to_data_list()
    
    for k in range(len(batch_data_list)):
        edge_index_deleted, _ = dropout_adj(batch_data_list[k].edge_index, p=p_edge)
        batch_data_list[k].edge_index = edge_index_deleted
    
    batch_data = Batch.from_data_list(batch_data_list)
    return batch_data


def drop_edges_biased(batch_data, p_edge_unbiased=0.3, p_edge_biased=0.6):
    batch_data_list = batch_data.to_data_list()
    
    for k in range(len(batch_data_list)):
        addr = batch_data_list[k]["address"]
        idx2impt = np.load(project_dir + "/data/facts/" + addr + "/idx2impt.npy", allow_pickle=True).item()  # 节点重要性

        edge_index = batch_data_list[k].edge_index
        S = [
            (idx2impt[int(edge_index[0, i])] + idx2impt[int(edge_index[1, i])]) / 2
            for i in range(edge_index.size(1))
        ]

        if all(s == S[0] for s in S):
            edge_index_deleted, _ = dropout_adj(edge_index, p=p_edge_unbiased)
            batch_data_list[k].edge_index = edge_index_deleted
        else:
            S_max, S_mean = np.max(S), np.mean(S)
            p_edge_threshold = 0.5 * p_edge_biased
            p = torch.tensor(
                [min(p_edge_threshold, (S_max - S[i]) / (S_max - S_mean) * p_edge_biased) for i in range(len(S))], dtype=torch.float32
            )
            
            keep = torch.bernoulli(1 - p).to(torch.bool)
            keep = keep.to(device)
            edge_index = edge_index[:, keep]
            batch_data_list[k].edge_index = edge_index
            
    batch_data = Batch.from_data_list(batch_data_list)
    return batch_data


def pretrain(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    
    for batch_data in dataloader:
        optimizer.zero_grad()

        if not args.biased:
            batch_data_1 = mask_node_features_unbiased(batch_data)
            batch_data_1 = drop_edges_unbiased(batch_data_1)
            batch_data_2 = mask_node_features_unbiased(batch_data)
            batch_data_2 = drop_edges_unbiased(batch_data_2)

        else:
            if args.ablation == "edge":
                batch_data_1 = mask_node_features_biased(batch_data, p_node=args.pn)
                batch_data_2 = mask_node_features_biased(batch_data, p_node=args.pn)

            elif args.ablation == "feature":
                batch_data_1 = drop_edges_biased(batch_data, p_edge_biased=args.pe)
                batch_data_2 = drop_edges_biased(batch_data, p_edge_biased=args.pe)

            else:
                batch_data_1 = mask_node_features_biased(batch_data, p_node=args.pn)
                batch_data_1 = drop_edges_biased(batch_data_1, p_edge_biased=args.pe)
                batch_data_2 = mask_node_features_biased(batch_data, p_node=args.pn)
                batch_data_2 = drop_edges_biased(batch_data_2, p_edge_biased=args.pe)
        
        z1 = model(batch_data_1.x.to(device), batch_data_1.edge_index.to(device), batch_data_1.batch.to(device))
        z2 = model(batch_data_2.x.to(device), batch_data_2.edge_index.to(device), batch_data_2.batch.to(device))
        loss = model.loss(z1, z2)
        
        loss.backward()
        optimizer.step()
        total_loss += loss
    
    total_loss = total_loss / len(dataloader)
    return total_loss


def pretrain_model(model, dataloader, optimizer, device):
    patience = args.patience
    patience_cur = 0
    best_validation_loss = np.inf
    
    for epoch in range(1, args.epochs + 1):
        loss = pretrain(model, dataloader, optimizer, device)
        if epoch % 1 == 0:
            print(f"Epoch: {epoch:03d}, Loss: {loss:.8f}, patience: {patience_cur}")

        if loss < best_validation_loss:
            patience_cur = 0
            best_validation_loss = loss
            torch.save(model.state_dict(), model_file_dir + model_file_name)
        else:
            patience_cur += 1
            
        if patience_cur > patience:
            print("Early Stop!")
            break 



if __name__ == "__main__":
    args = get_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(device)

    model_file_dir = project_dir + "/algorithm/model_files/"
    model_file_name = "sage_biased_pn{}_pe{}.pt".format(args.pn, args.pe)

    _, _, _, addr_list, label_list, _, _ = get_split_index(args.split)
    dataset = CFGDataset(
        root = project_dir + "/algorithm/dataset_files/dataset_{}dim/".format(args.dim),
        addr_list = addr_list, label_list = label_list
    )
    print(dataset)

    encoder = GraphEncoder(
        in_channels=args.dim, hidden_channels=args.dim * 2, out_channels=args.dim,
        base_model=args.base_model, activation=args.activation, norm=args.norm, pooling=args.pooling,
        use_pool=args.use_pool, use_skip=args.use_skip, num_layers=args.num_layers
    ).to(device)
    model = GCLModel(encoder=encoder, hidden_channels=args.dim, proj_channels=args.proj_dim, tau=args.tau).to(device)
    print(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    dataset = [data for data in dataset if data != None]
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=28)  # 修改 num_workers 加速

    model.load_state_dict(torch.load(model_file_dir + model_file_name))
    pretrain_model(model, dataloader, optimizer, device)