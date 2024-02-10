from graph_encoder import GraphEncoder, GCLModel
from dataset import CFGDataset, get_split_index
from sampler import ImbalancedSampler

import pandas as pd
import numpy as np
import os
import pickle
import random
import argparse

from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Dataset, Batch
from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import classification_report, make_scorer, precision_score, recall_score, f1_score
from sklearn.svm import SVC
from sklearn.linear_model import RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
import xgboost as xgb

import warnings
warnings.filterwarnings("ignore")

# Please modify your project dir
project_dir = "/media/ubuntu/My_Passport/PonziDetector"


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--split", type=str, default="split1")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--downstream_classifier", type=str, default="xgboost")
    
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
    
    args = parser.parse_args()
    return args


@torch.no_grad()
def obtain_embedding(model, dataset, split_idx, device):
    model.eval()
    pretrain_embedding = {"train": None, "val": None, "test": None, "all": None}
    labels = {"train": None, "val": None, "test": None, "all": None}
    
    for key in ["train", "val", "test", "all"]:
        data_list = [dataset[i] for i in split_idx[key]]
        dataloader = DataLoader(data_list, batch_size=64, shuffle=False, num_workers=28)  # 修改 num_workers 加速
        
        for batch_data in dataloader:
            out = model(batch_data.x.to(device), batch_data.edge_index.to(device), batch_data.batch.to(device))
            if pretrain_embedding[key] == None: pretrain_embedding[key] = out
            else: pretrain_embedding[key] = torch.cat((pretrain_embedding[key], out), dim=0)
    
        labels[key] = torch.Tensor([dataset[i].y for i in split_idx[key]])
        print("X_{} {}".format(key, pretrain_embedding[key].shape))
        print("y_{} {}".format(key, labels[key].shape))
        
    return pretrain_embedding, labels


def select_downstream_classifier(y_train):
    
    # XGBoost
    if args.downstream_classifier == "xgboost":
        ratio = float(np.sum(y_train == 0) / np.sum(y_train == 1))
        sample_weight = np.where(y_train == 1, 10.0, 1.0)
        
        clf = xgb.XGBClassifier(
            n_estimators=200, max_depth=4, scale_pos_weight=ratio, min_child_weight=3, random_state=args.random_seed
        )
    return clf


def train_downstream_classifier(clf, X_train, y_train):
    print("\ntraining {}".format(args.downstream_classifier))
    clf.fit(X_train, y_train)


def test_downstream_classifier(clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    print("\ndownstream task result")
    print(classification_report(y_true=y_test, y_pred=y_pred, digits=6))
    return y_pred



if __name__ == "__main__":
    args = get_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(device)

    model_file_dir = project_dir + "/algorithm/model_files/"
    model_file_name = "sage_biased_pn0.6_pe0.6.pt"

    train_idx, test_idx, val_idx, addr_list, label_list, _, _ = get_split_index(args.split)
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
    model.load_state_dict(torch.load(model_file_dir + model_file_name))

    split_idx = {
        "train": train_idx,
        "val": val_idx,
        "test": test_idx,
        "all": train_idx + val_idx + test_idx
    }
    pretrain_embedding, labels = obtain_embedding(model, dataset, split_idx, device)

    clf = select_downstream_classifier(y_train=np.array(labels["train"].cpu()))

    train_downstream_classifier(
        clf,
        X_train = np.array(pretrain_embedding["train"].cpu()),
        y_train = np.array(labels["train"].cpu())
    )
    test_downstream_classifier(
        clf = clf,
        X_test = np.array(pretrain_embedding["test"].cpu()),
        y_test = np.array(labels["test"].cpu())
    )

    clf.save_model(model_file_dir + "xgboost_bisaed_pn0.6_pe0.6_{}.xgb".format(args.split))
    print("save downstream classifier.")