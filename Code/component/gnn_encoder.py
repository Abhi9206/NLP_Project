import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleGraphConv(nn.Module):
    """Simple graph convolution layer without torch_geometric dependencies"""

    def __init__(self, in_features, out_features, num_heads=1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads

        # Linear transformations for node features
        self.W = nn.Linear(in_features, out_features * num_heads)
        self.a = nn.Linear(2 * out_features * num_heads, num_heads)

        self.dropout = nn.Dropout(0.1)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x, edge_index):
        # x: [num_nodes, in_features]
        # edge_index: [2, num_edges]

        num_nodes = x.size(0)
        h = self.W(x)  # [num_nodes, out_features * num_heads]
        h = h.view(num_nodes, self.num_heads, self.out_features)  # [num_nodes, num_heads, out_features]

        # Create adjacency matrix
        adj = torch.zeros(num_nodes, num_nodes, device=x.device)
        if edge_index.size(1) > 0:
            adj[edge_index[0], edge_index[1]] = 1.0

        # Simple attention mechanism
        output = torch.zeros_like(h)
        for head in range(self.num_heads):
            h_head = h[:, head, :]  # [num_nodes, out_features]

            # Compute attention scores
            attention_input = []
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if adj[i, j] > 0 or i == j:  # Include self-connections
                        attention_input.append(torch.cat([h_head[i], h_head[j]]))

            if attention_input:
                attention_input = torch.stack(attention_input)  # [num_edges, 2*out_features]
                attention_scores = self.leaky_relu(self.a(attention_input))  # [num_edges, 1]

                # Apply attention
                edge_idx = 0
                for i in range(num_nodes):
                    neighbors = []
                    scores = []
                    for j in range(num_nodes):
                        if adj[i, j] > 0 or i == j:
                            neighbors.append(h_head[j])
                            scores.append(attention_scores[edge_idx])
                            edge_idx += 1

                    if neighbors:
                        neighbors = torch.stack(neighbors)  # [num_neighbors, out_features]
                        scores = torch.stack(scores).squeeze(-1)  # [num_neighbors]
                        scores = F.softmax(scores, dim=0)

                        output[i, head, :] = torch.sum(neighbors * scores.unsqueeze(-1), dim=0)
                    else:
                        output[i, head, :] = h_head[i]
            else:
                output[:, head, :] = h_head

        # Concatenate heads
        if self.num_heads > 1:
            output = output.view(num_nodes, -1)  # [num_nodes, num_heads * out_features]
        else:
            output = output.squeeze(1)  # [num_nodes, out_features]

        return output


class GraphEncoder(nn.Module):
    """
    Simple Graph Neural Network encoder for story graphs.
    Processes graph structure to create constraint-aware embeddings.
    """

    def __init__(self,
                 node_feature_dim=390,
                 hidden_dim=256,
                 num_layers=3,
                 num_heads=4,
                 dropout=0.1):
        super().__init__()

        self.num_layers = num_layers

        # Graph convolution layers
        self.conv_layers = nn.ModuleList()

        # First layer
        self.conv_layers.append(
            SimpleGraphConv(node_feature_dim, hidden_dim, num_heads)
        )

        # Middle layers
        for _ in range(num_layers - 2):
            self.conv_layers.append(
                SimpleGraphConv(hidden_dim * num_heads, hidden_dim, num_heads)
            )

        # Final layer (single head)
        self.conv_layers.append(
            SimpleGraphConv(hidden_dim * num_heads, hidden_dim, 1)
        )

        self.dropout = nn.Dropout(dropout)
        self.output_dim = hidden_dim

    def forward(self, x, edge_index, batch=None):
        """
        Args:
            x: Node features [num_nodes, node_feature_dim]
            edge_index: Graph connectivity [2, num_edges]
            batch: Batch vector [num_nodes] (optional for batching)

        Returns:
            graph_embedding: [1, hidden_dim]
        """
        # Apply graph convolution layers
        for i, conv in enumerate(self.conv_layers):
            x = conv(x, edge_index)
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = self.dropout(x)

        # Global mean pooling to get graph-level embedding
        graph_embedding = x.mean(dim=0, keepdim=True)  # [1, hidden_dim]

        return graph_embedding