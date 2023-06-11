import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import KarateClub

import networkx as nx
import matplotlib.pyplot as plt

# Define the Graph Convolutional Network model
class GCN(nn.Module):
    def __init__(self, num_features, hidden_dim, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Load the dataset
dataset = Planetoid(root='/tmp/Cora', name='Cora')

# Split the dataset into training, validation, and test sets
data = dataset[0]
x, edge_index, y = data.x, data.edge_index, data.y
train_mask, val_mask, test_mask = data.train_mask, data.val_mask, data.test_mask

# Create the model and define the optimizer
model = GCN(dataset.num_features, hidden_dim=16, num_classes=dataset.num_classes)
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# Convert edge_index to edge list format
edges = edge_index.t().tolist()

# Create a networkx graph object
G = nx.Graph(edges)

# Draw the graph
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=300, font_size=8)

# Show the graph
plt.show()

# Training loop
def train():
    model.train()
    optimizer.zero_grad()
    output = model(x, edge_index)
    loss = F.nll_loss(output[train_mask], y[train_mask])
    loss.backward()
    optimizer.step()

# Evaluation function
def evaluate():
    model.eval()
    output = model(x, edge_index)
    logits, accs = output[test_mask], []
    pred = logits.argmax(dim=1)
    acc = pred.eq(y[test_mask]).sum().item() / test_mask.sum().item()
    return acc

# Train and evaluate the model
for epoch in range(200):
    train()
    acc = evaluate()
    print(f'Epoch: {epoch+1}, Test Accuracy: {acc:.4f}')


# Define the Graph Convolutional Network model
class GCN(nn.Module):
    def __init__(self, num_features, hidden_dim, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Load the KarateClub dataset
dataset = KarateClub()

# Access the data and convert it to a networkx graph object
data = dataset[0]
G = data.to_networkx()

# Draw the graph
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=300, font_size=8)

# Show the graph
plt.show()

# Get the node features and edge indices
x = data.x
edge_index = data.edge_index
num_features = dataset.num_features
num_classes = dataset.num_classes

# Create the model and define the optimizer
model = GCN(num_features, hidden_dim=16, num_classes=num_classes)
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# Training loop
def train():
    model.train()
    optimizer.zero_grad()
    output = model(x, edge_index)
    loss = F.nll_loss(output[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

# Evaluation function
def evaluate():
    model.eval()
    output = model(x, edge_index)
    logits = output.argmax(dim=1)
    acc = logits[data.test_mask].eq(data.y[data.test_mask]).sum().item() / data.test_mask.sum().item()
    return acc

# Train and evaluate the model
for epoch in range(200):
    train()
    acc = evaluate()
    print(f'Epoch: {epoch+1}, Test Accuracy: {acc:.4f}')