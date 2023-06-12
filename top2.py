from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import KarateClub
from torch_geometric.datasets import Flickr

import networkx as nx
import matplotlib.pyplot as plt
"""
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
train_mask, _, _ = data.train_mask, data.val_mask, data.test_mask

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
    pred = output.argmax(dim=1)
    acc = pred.eq(y).sum().item() / y.size(0)
    return acc

# Train and evaluate the model
accuracies = []
for epoch in range(200):
    train()
    acc = evaluate()
    accuracies.append(acc)
    print(f'Epoch: {epoch+1}, Accuracy: {acc:.4f}')

# Calculate average accuracy
avg_accuracy = sum(accuracies) / len(accuracies)
print(f'Average Accuracy: {avg_accuracy:.4f}')

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
dataset = KarateClub()

# Split the dataset into training, validation, and test sets
data = dataset[0]
x, edge_index, y = data.x, data.edge_index, data.y
train_mask = data.train_mask

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
    pred = output.argmax(dim=1)
    acc = pred.eq(y).sum().item() / y.size(0)
    return acc

# Train and evaluate the model
accuracies = []
for epoch in range(200):
    train()
    acc = evaluate()
    accuracies.append(acc)
    print(f'Epoch: {epoch+1}, Accuracy: {acc:.4f}')

# Calculate average accuracy
avg_accuracy = sum(accuracies) / len(accuracies)
print(f'Average Accuracy: {avg_accuracy:.4f}')
"""
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
dataset = Flickr(root="/tmp/Flickr")

# Split the dataset into training set
data = dataset[0]
x, edge_index, y = data.x, data.edge_index, data.y
train_mask = data.train_mask


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
    pred = output.argmax(dim=1)
    acc = pred.eq(y).sum().item() / y.size(0)
    return acc

# Train and evaluate the model
accuracies = []
for epoch in range(200):
    train()
    acc = evaluate()
    accuracies.append(acc)
    print(f'Epoch: {epoch+1}, Accuracy: {acc:.4f}')

# Calculate average accuracy
avg_accuracy = sum(accuracies) / len(accuracies)
print(f'Average Accuracy: {avg_accuracy:.4f}')