# d:/CAPSTONE/Navigation/neurogait/communication/gat_net.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphAttentionLayer(nn.Module):
    """
    Standard GAT Attention Layer.
    Implements attention-based message propagation between robot agents.
    """
    def __init__(self, in_features, out_features, dropout=0.1, alpha=0.2, concat=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat

        # Projection weights
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        
        # Attention parameter
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        """
        h: node feature representation tensor of shape (num_nodes, in_features)
        adj: adjacency matrix representing communication links (num_nodes, num_nodes)
        """
        # Linear projection
        Wh = torch.matmul(h, self.W) # shape: (num_nodes, out_features)
        num_nodes = Wh.size()[0]

        # Calculate attention coefficients
        a_input = torch.cat([Wh.repeat(1, num_nodes).view(num_nodes * num_nodes, -1), Wh.repeat(num_nodes, 1)], dim=1)
        a_input = a_input.view(num_nodes, num_nodes, 2 * self.out_features)
        
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        # Apply adjacency mask to mask out non-communicating agents
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=-1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime


class GraphAttentionNetwork(nn.Module):
    """
    Phase 3: GAT Network for Robot Team.
    Aggregates exteroceptive observation graphs of R1, R2, etc.
    """
    def __init__(self, feature_dim=128, hidden_dim=64, out_dim=16, num_heads=4, device="cuda:0"):
        super().__init__()
        self.device = device
        
        # Multi-head attention layer
        self.attentions = nn.ModuleList([
            GraphAttentionLayer(feature_dim, hidden_dim, concat=True) 
            for _ in range(num_heads)
        ])
        
        # Output layer
        self.out_att = GraphAttentionLayer(hidden_dim * num_heads, out_dim, concat=False)
        self.to(device)

    def forward(self, node_features: torch.Tensor, adjacency_matrix: torch.Tensor) -> torch.Tensor:
        """
        node_features: (num_nodes, feature_dim)
        adjacency_matrix: (num_nodes, num_nodes)
        """
        # 1. Multi-head aggregation
        x = torch.cat([att(node_features, adjacency_matrix) for att in self.attentions], dim=-1)
        
        # 2. Final output projection
        x = self.out_att(x, adjacency_matrix)
        return x
