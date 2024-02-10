from torch_geometric.nn import GCNConv, SGConv, SAGEConv, GATConv, GraphConv, GINConv
from torch_geometric.nn import global_mean_pool, global_max_pool, TopKPooling, SAGPooling
from torch_geometric.nn import LayerNorm, BatchNorm
from torch_geometric.data import Data, DataLoader
from sampler import ImbalancedSampler
import torch
import torch.nn as nn
import torch.nn.functional as F


class GCLModel(nn.Module):
    def __init__(
        self, encoder, hidden_channels, proj_channels, tau=0.5
    ):
        super(GCLModel, self).__init__()
        self.encoder = encoder
        self.tau = tau
        self.fc1 = nn.Linear(hidden_channels, proj_channels)
        self.fc2 = nn.Linear(proj_channels, hidden_channels)
    
    def forward(self, x, edge_index, batch):
        return self.encoder(x, edge_index, batch)
    
    def projection(self, z):
        z = self.fc1(z)
        z = F.elu(z)
        z = self.fc2(z)
        return z

    def sim(self, z1, z2):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())
    
    def semi_loss(self, z1, z2):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))
        return -torch.log(between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))
    
    def loss(self, z1, z2, use_mean=True):
        h1 = self.projection(z1)
        h2 = self.projection(z2)
        
        l1 = self.semi_loss(h1, h2)
        l2 = self.semi_loss(h2, h1)
        L = (l1 + l2) * 0.5
        L = L.mean() if use_mean else L.sum()
        return L


class GraphEncoder(nn.Module):
    def __init__(
        self, in_channels, hidden_channels, out_channels,
        base_model="SAGEConv", activation="relu", norm="BatchNorm", pooling="TopKPooling",
        use_pool=False, use_skip=False, num_layers=2
    ):
        super(GraphEncoder, self).__init__()
        
        self.base_model = select_base_model(base_model)
        self.f = select_activation(activation)
        self.norm = select_normalization(norm)
        
        self.num_layers = num_layers
        self.use_pool = use_pool
        self.use_skip = use_skip
        assert self.num_layers >= 2

        self.convs = nn.ModuleList()
        self.convs.append(self.base_model(in_channels, hidden_channels))
        for i in range(1, self.num_layers - 1):
            self.convs.append(self.base_model(hidden_channels, hidden_channels))
        self.convs.append(self.base_model(hidden_channels, out_channels))

        self.norms = nn.ModuleList()
        for i in range(self.num_layers - 1):
            self.norms.append(self.norm(hidden_channels))
        self.norms.append(self.norm(out_channels))

        if self.use_pool:
            self.pooling = select_pooling(pooling)
            self.pools = nn.ModuleList()
            for i in range(self.num_layers - 1):
                self.pools.append(self.pooling(hidden_channels, min_score=0.001))
            self.pools.append(self.pooling(out_channels, min_score=0.001))

        if self.use_skip:
            self.lin = nn.Linear(in_channels, hidden_channels)
    
    def forward(self, x, edge_index, batch):
        if not self.use_skip:
            for i in range(self.num_layers):
                x = self.convs[i](x, edge_index)
                x = self.norms[i](x)
                x = self.f(x)
                if self.use_pool:
                    x, edge_index, _, batch, _, _ = self.pools[i](x=x, edge_index=edge_index, batch=batch)
            out = global_mean_pool(x, batch)
            return out

        else:
            for i in range(self.num_layers):
                h = self.f(self.norms[i](self.convs[i](x, edge_index)))
                
                if i < self.num_layers - 1:
                    h = (self.lin(x) + h) if i == 0 else (x + h)
                if self.use_pool:
                    h, edge_index, _, batch, _, _ = self.pools[i](x=h, edge_index=edge_index, batch=batch)
                x = h
            out = global_mean_pool(x, batch)
            return out
    
    def forward_log_softmax(self, x, edge_index, batch):
        out = self.forward(x, edge_index, batch)
        out = out.log_softmax(dim=-1)
        return out



def select_base_model(name):
    def GAT_wrapper(in_channels, out_channels):
        return GATConv(
            in_channels=in_channels,
            out_channels=out_channels,
            heads=4
        )
    
    def GIN_wrapper(in_channels, out_channels):
        MLP = nn.Sequential(
            nn.Linear(in_channels, 2 * out_channels),
            nn.ELU(),
            nn.Linear(2 * out_channels, in_channels)
        )
        return GINConv(MLP)
    
    base_models = {
        'GCNConv': GCNConv,
        'SGConv': SGConv,
        'SAGEConv': SAGEConv,
        'GATConv': GAT_wrapper,
        'GraphConv': GraphConv,
        'GINConv': GIN_wrapper
    }
    return base_models[name]


def select_activation(name):
    activations = {
        'relu': F.relu,
        'tanh': F.tanh,
        'hardtanh': F.hardtanh,
        'elu': F.elu,
        'leaky_relu': F.leaky_relu,
        'prelu': torch.nn.PReLU(),
        'rrelu': F.rrelu
    }
    return activations[name]


def select_pooling(name):
    poolings = {
        'TopKPooling': TopKPooling,
        'SAGPooling': SAGPooling
    }
    return poolings[name]


def select_normalization(name):
    normalizations = {
        'LayerNorm': LayerNorm,
        'BatchNorm': BatchNorm,
    }
    return normalizations[name]