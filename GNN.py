import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, BatchNorm

class TimingGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TimingGNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim) 

        self.bn1 = BatchNorm(hidden_dim) 
        self.bn2 = BatchNorm(hidden_dim)  
        self.bn3 = BatchNorm(hidden_dim)  

        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.3) 

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

      
        x = self.conv1(x, edge_index)
        x = self.bn1(x)  
        x = F.relu(x)
        x = self.dropout(x)  

  
        residual = x 
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)


        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout(x)

 
        x = x + residual 

        x = self.fc(x)

        # Output
        x = F.sigmoid(x) * 10 

        return x
