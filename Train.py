import torch
import random
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
import networkx as nx
import numpy as np
from GNN import TimingGNN

#Generate a synthetic circuit graph
def generate_data():
    G = nx.DiGraph()
    for i in range(10):
        G.add_node(i, gate_type=random.choice(["AND", "OR", "NAND"]))
        if i > 0:
            G.add_edge(random.randint(0, i-1), i)

    edge_index = torch.tensor(list(G.edges()), dtype=torch.long).t().contiguous()
    node_features = torch.tensor(np.random.rand(len(G.nodes()), 4), dtype=torch.float)
    y = torch.tensor(np.random.rand(len(G.nodes()), 1), dtype=torch.float)
    
    return Data(x=node_features, edge_index=edge_index, y=y)

data = generate_data()

# Initialize Model

input_dim = 4  # Match this with feature extraction in Predict_Timing.py
hidden_dim = 16  # Keep or adjust if needed
output_dim = 1  # Predicting logic depth
model = TimingGNN(input_dim, hidden_dim=16, output_dim=1)

optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# Training Loop
def train(model, data, epochs=100):
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# Train model
train(model, data)

torch.save(model.state_dict(), "trained_model.pth")
print("Model saved as trained_model.pth")
