import torch
import os
import pandas as pd
import networkx as nx
from torch_geometric.data import Data
from GNN import TimingGNN 
from Feature_Extraction import extract_features
from Graph_Generation import generate_circuit_graph 

#Define model dimensions
INPUT_DIM = 4 
HIDDEN_DIM = 16
OUTPUT_DIM = 1

#Load the trained model
model = TimingGNN(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM)
model_path = "trained_model.pth"

if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    print("Model loaded successfully!")
else:
    raise FileNotFoundError(f"Model file {model_path} not found. Run Train.py first.")

for param in model.parameters():
    if torch.isnan(param).any():
        raise ValueError("Model contains NaN weights! Retrain your model.")

def predict_timing(netlist_file):
    """Predict logic depth for a given netlist."""
    
    #Convert netlist to a graph
    G, edge_index = generate_circuit_graph(netlist_file)

    if not G.nodes:
        raise ValueError("Graph generation failed. No nodes found. Check the netlist.")

    #Extract features
    features = extract_features(G)

    if features.empty:
        raise ValueError("Feature extraction failed. Empty DataFrame returned.")

    #Convert features to numeric values, replace NaN with 0
    features = features.apply(pd.to_numeric, errors='coerce').fillna(0)

    #Convert features into tensor format
    X = torch.tensor(features.values, dtype=torch.float32)

    #Validate tensor shape
    print(f"Feature Tensor Shape: {X.shape}")

    #Convert node names (strings) to numerical indices
    node_to_index = {node: idx for idx, node in enumerate(G.nodes)}
    numeric_edges = [(node_to_index[src], node_to_index[dst]) for src, dst in G.edges]

    #Convert edges to tensor format
    edge_index = torch.tensor(numeric_edges, dtype=torch.long).t().contiguous()

    if edge_index.numel() == 0:
        raise ValueError("Edge index is empty. Graph may not be connected properly.")

    #Create PyTorch Geometric Data object
    data = Data(x=X, edge_index=edge_index)

    #Ensure tensor contains no NaN or Inf values
    if torch.isnan(data.x).any() or torch.isinf(data.x).any():
        raise ValueError("Feature tensor contains NaN or Inf values. Check feature extraction.")

    MAX_DEPTH = 10 

    with torch.no_grad():
        prediction = model(data)
        raw_output = prediction.mean().item()
        real_depth = raw_output * MAX_DEPTH 

    print(f"Raw Model Output: {raw_output:.4f}")
    print(f"Corrected Logic Depth: {real_depth:.4f}")

# Run prediction on new netlist
predict_timing("netlists/Simple_Netlist.v")
