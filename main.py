import meshio
import numpy as np

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv

from utils.config import config
from utils.loader import DatasetLoader
from torch_geometric.data import DataLoader

dataset =  DatasetLoader.load(config)   
data_size = len(dataset)
print("[INFO] Total datasets", data_size)
train_loader = DataLoader(dataset[:int(data_size * 0.8)], batch_size=64, shuffle=False)
test_loader = DataLoader(dataset[int(data_size * 0.8):], batch_size=64, shuffle=False)

# for batch in train_loader:
#     print(batch)
# exit(0)
# define neural network

print("[INFO] Define the model ...")
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(3, 3)
        self.conv2 = GCNConv(3, 3)
        self.conv3 = GCNConv(3, 3)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index)

        return x


class GNNStack(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNNStack, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(self.build_conv_model(input_dim, hidden_dim))

        self.lns = nn.ModuleList()
        self.lns.append(nn.LayerNorm(hidden_dim))
        self.lns.append(nn.LayerNorm(hidden_dim))

        for l in range(2):
            self.convs.append(self.build_conv_model(hidden_dim, hidden_dim))

        # post-message-passing
        self.post_mp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Dropout(0.25), nn.Linear(hidden_dim, output_dim))    

        self.dropout = 0.25
        self.num_layers = 3

    def build_conv_model(self, input_dim, hidden_dim):
        # refer to pytorch geometric nn module for different implementation of GNNs.
        return GCNConv(input_dim, hidden_dim) # graph convolution operator

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            emb = x
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if not i == self.num_layers - 1:
                x = self.lns[i](x)

        x = self.post_mp(x)

        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = Net().to(device)
# data = data.to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # all the models defined as self... in the init() are gone be optimized

model = GNNStack(3, 32, 3)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def test(loader, model):

    model.eval()
    acc_loss = 0
    for data in loader:

        with torch.no_grad():
            pred = model(data)
            label = data.y

            loss = F.mse_loss(pred, label)
            acc_loss += loss.item() * data.num_graphs

    acc_loss /= len(loader.dataset)
    return acc_loss

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("[INFO] Number of trainable parameters = {}".format(total_params))

print("[INFO] Start Training ...")
for epoch in range(200):

    train_loss = 0.0
    model.train()
    for batch in train_loader:
        
        out = model(batch)
        loss = F.mse_loss(out, batch.y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() #parameters store their own gradient, which is computed in the previsous step!

        train_loss += loss.item() * batch.num_graphs

    train_loss /= len(train_loader.dataset)
    accuracy_loss = test(test_loader, model)
    print('epoch={} train_loss={}, acc_loss={}'.format(epoch,train_loss, accuracy_loss))
    print()