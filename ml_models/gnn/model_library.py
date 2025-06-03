#!/usr/bin/env python3

### GNN CLASSIFIER MODELS DEFINITIONS ###

import torch
from torch.nn import Dropout
import torch.nn.functional as F
from torcheval.metrics.functional import r2_score
from torch_geometric.nn import GCNConv, TopKPooling, global_mean_pool
from torch_geometric.nn import GATConv, GlobalAttention
import torch.nn.init as init

# Notation
# hidden channels: number of features per node in the GCN layers

global n_genes

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, TopKPooling, global_mean_pool
from torch.nn import Dropout

class GNNConvDropoutPool(torch.nn.Module):
    '''
    Dynamic class for convolutional GNN with dropout at each topKpooling layer and at the 
    end of the convolutional stage of the network. This version performs classification instead of regression.
    '''
    def __init__(self, in_channels, hidden_channels, num_layers, num_classes=5, nonlin='relu', dropout_rate=0.8, sageconv=False):
        super(GNNConvDropoutPool, self).__init__()
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        
        self.convs = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()
        self.hidden_channels = hidden_channels
        
        # Initialize convolutional and pooling layers
        for i in range(num_layers):
            in_dim = in_channels if i == 0 else hidden_channels * (2 ** i)
            out_dim = hidden_channels * (2 ** (i + 1))
            if not sageconv:
                self.convs.append(GCNConv(in_dim, out_dim))
            else:
                self.convs.append(SAGEConv(in_dim, out_dim))
            self.pools.append(TopKPooling(out_dim, ratio=0.8, nonlinearity=nonlin))
        
        self.dropout = Dropout(dropout_rate)
        
        # First inear layer
        self.lin1 = torch.nn.Linear(hidden_channels * (2 ** num_layers), hidden_channels * (2 ** num_layers))
        
        # Final linear layer for classification introducing cag and sex features
        self.lin2 = torch.nn.Linear(hidden_channels * (2 ** num_layers) + 2, num_classes)  # Output num_classes for classification

        # Apply weight initialization
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize weights using Kaiming Uniform initialization for ReLU activations."""
        for conv in self.convs:
            if isinstance(conv, (GCNConv, SAGEConv)):
                conv.reset_parameters()  # PyG layers have their own reset method

        for pool in self.pools:
            if hasattr(pool, 'reset_parameters'):
                pool.reset_parameters()

        # Apply Kaiming Uniform initialization
        init.kaiming_uniform_(self.lin1.weight, nonlinearity='relu')
        init.zeros_(self.lin1.bias)
        init.kaiming_uniform_(self.lin2.weight, nonlinearity='relu')
        init.zeros_(self.lin2.bias)

    def forward(self, data):
        # Extract required fields from the data object
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        batch = data.batch
        sex = data.sex
        cag = data.cag
        
        # Store intermediate outputs
        graph_idxs = []
        node_idxs = []
        scores = []
        
        for i in range(self.num_layers):
            # GCN layer
            x = F.relu(self.convs[i](x, edge_index))
            # TopK Pooling layer
            x, edge_index, edge_attr, graph_idx, node_idx, score = self.pools[i](x, edge_index, edge_attr, batch)
            graph_idxs.append(graph_idx)
            node_idxs.append(node_idx)
            scores.append(score)
            batch = graph_idx  # Update batch for the next pooling layer

        x = self.dropout(x)

        # Global pooling, pass last state of batch vector
        x = global_mean_pool(x, graph_idxs[-1])

        # Linear layers
        x_lin1 = F.relu(self.lin1(x))
        
        # Concatenate additional features (sex, cag) for classification
        sex = sex.unsqueeze(1)
        cag = cag.unsqueeze(1)
        x = torch.cat([x_lin1, sex, cag], dim=1)
        logits = self.lin2(x)
        
        # Softmax activation for multi-class classification
        prediction = F.softmax(logits, dim=1)

        return prediction, [x_lin1, graph_idxs, node_idxs, scores]

class EdgeGAT(torch.nn.Module):
    '''
    Dynamic class for convolutional GNN with attention mechanism instead of pooling (GATConv).
    Learns attention scores for edges. The node features are transformed by a weight matrix and then 
    aggregated based on the attention scores assigned to their neighbors.
    Takes as input: 
        - in_channels
        - hidden_channels: first layer number of output channels, at each layer the output channels doubles.
        - nonlin: which non linearity is applied after the pooling layers between convolutions
        - num_layers: number of Conv layers
        - dropout_rate: dropout rate before linear layers. (At topKpooling layers the dropout rate is 
                        fixed at 0.8)
        - sageconv: if true, SAGEConv is applied instead of GCNConv layer.
    '''

    def __init__(self, in_channels, hidden_channels, num_layers, num_classes=5, dropout_rate=0.8, heads=1):
        super(EdgeGAT, self).__init__()
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.heads = heads
        
        # Initialize GATConv layers
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            in_dim = in_channels if i == 0 else hidden_channels * (heads ** i)
            out_dim = hidden_channels * (heads ** (i + 1)) if i < num_layers - 1 else hidden_channels * (heads ** i)
            # concat: how the output from multiple attention heads is combined (concatenated or averaged)
            concat = True if i < num_layers - 1 else False  # Do not concatenate on the final layer
            self.convs.append(GATConv(in_dim, out_dim, heads=heads, concat=concat, dropout=dropout_rate))

        # Fully connected layers
        self.lin1 = torch.nn.Linear(hidden_channels * (heads ** num_layers), hidden_channels * (heads ** num_layers))
        self.lin2 = torch.nn.Linear(hidden_channels * (heads ** num_layers) + 2, num_classes)  # Include `sex` and `cag`

        # Apply weight initialization
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize weights using Kaiming Uniform initialization for ReLU activations."""
        for conv in self.convs:
            if isinstance(conv, EdgeGAT):
                conv.reset_parameters()  # PyG layers have their own reset method

        # Apply Kaiming Uniform initialization
        init.kaiming_uniform_(self.lin1.weight, nonlinearity='relu') # this non linearity is the previous to the lin layer
        init.zeros_(self.lin1.bias)
        init.kaiming_uniform_(self.lin2.weight, nonlinearity='relu')
        init.zeros_(self.lin2.bias)
        
    def forward(self, data):
        # Extract required fields from the data object
        x = data.x
        edge_index = data.edge_index
        batch = data.batch
        sex = data.sex
        cag = data.cag

        graph_idxs, node_idxs, scores = [], [], []  # keep empty for output model consistency

        # Pass through GATConv layers
        for i, conv in enumerate(self.convs):
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
            x = conv(x, edge_index)
            if i < self.num_layers - 1:  # Apply activation only to intermediate layers
                x = F.elu(x)

        # Global pooling
        x = global_mean_pool(x, batch)

        # Fully connected layers
        x_lin1 = F.relu(self.lin1(x))
        # Reshape sex and cag to [batch_size, 1]
        sex = sex.unsqueeze(1)
        cag = cag.unsqueeze(1)
        x = torch.cat([x_lin1, sex, cag], dim=1)
        logits = self.lin2(x)
        
        # Softmax activation for multi-class classification
        prediction = F.softmax(logits, dim=1)

        # Return final output and intermediate representations
        return prediction, [x_lin1, graph_idxs, node_idxs, scores]

class GNNConvDropoutGlobalAttention(torch.nn.Module):
    '''
    GNN with multiple Global Attention Pooling stages instead of TopKPooling.
    Attention mechanism on nodes. This version performs classification instead of regression.
    '''
    
    def __init__(self, in_channels, hidden_channels, num_layers=2, num_classes=5, dropout_rate=0.8, sageconv=False, heads=1):
        super(GNNConvDropoutGlobalAttention, self).__init__()
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.hidden_channels = hidden_channels
        
        self.convs = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()
        
        # Initialize convolutional and pooling layers
        for i in range(num_layers):
            in_dim = in_channels if i == 0 else hidden_channels * (heads ** i)
            out_dim = hidden_channels * (heads ** (i + 1))

            # Select convolution type
            if not sageconv:
                self.convs.append(GCNConv(in_dim, out_dim))
            else:
                self.convs.append(SAGEConv(in_dim, out_dim))

            # Define attention-based pooling layer with multiple heads
            gate_nn = torch.nn.Linear(out_dim * heads, heads)  # Learn attention for nodes with multiple heads
            self.pools.append(GlobalAttention(gate_nn))
        
        self.dropout = Dropout(dropout_rate)
        
        # Fully connected layers
        lin_input_dim = hidden_channels * (2 ** num_layers)
        self.lin1 = torch.nn.Linear(lin_input_dim, lin_input_dim)
        self.lin2 = torch.nn.Linear(lin_input_dim + 2, num_classes)  # Output num_classes for classification

    def forward(self, data):
        # Extract required fields from the data object
        x, edge_index, batch = data.x, data.edge_index, data.batch
        sex, cag = data.sex.unsqueeze(1), data.cag.unsqueeze(1)  # Ensure shape [batch_size, 1]

        graph_idxs, node_idxs, scores = [], [], []  # Track pooling outputs

        for i in range(self.num_layers):
            # GCN or SAGE convolution
            x = F.relu(self.convs[i](x, edge_index))

            # Global Attention Pooling
            x = self.pools[i](x, batch)
            graph_idxs.append(batch.clone())  # Save batch state for tracking

        x = self.dropout(x)

        # Fully connected layers
        x_lin1 = F.relu(self.lin1(x))
        x = torch.cat([x_lin1, sex, cag], dim=1)  # Concatenate additional features
        logits = self.lin2(x)

        # Softmax activation for multi-class classification
        prediction = F.softmax(logits, dim=1)

        return prediction, [x_lin1, graph_idxs, node_idxs, scores]

### TRAINING FUNCTIONS ###

def train(model, optimizer, loader, device, scheduler=None, gradient_clipping=None):
    """
    Train the model for one epoch.
    
    Args:
        model (torch.nn.Module): The model to train.
        optimizer (torch.optim.Optimizer): Optimizer for parameter updates.
        loader (DataLoader): Training data loader.
        device (torch.device): Device to run the training on (CPU/GPU).
        scheduler (torch.optim.lr_scheduler, optional): Learning rate scheduler.
        gradient_clipping (float, optional): Maximum gradient norm for clipping.
    
    Returns:
        float: Average training loss for the epoch.
        torch.Tensor: Concatenated true values (y).
        torch.Tensor: Concatenated predicted values (y_pred).
    """
    model.train()
    total_loss = torch.tensor(0.0, device=device)
    true_values = []
    predicted_values = []

    for data in loader:
        data = data.to(device)
        # Ensure inputs are in float32
        data.x = data.x.float()
        # data.y = data.y.long()  # For classification, targets should be long integers
        if hasattr(data, 'edge_attr'):
            data.edge_attr = data.edge_attr.float()
        optimizer.zero_grad()  # Clears the accumulated gradients from the previous iteration

        # Forward pass
        out, _ = model(data)

        # Compute loss
        loss = F.cross_entropy(out, data.y)  # Cross Entropy Loss for classification

        # Backward pass and optimization
        loss.backward()
        if gradient_clipping:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
        optimizer.step()

        # Logging
        total_loss += loss  # Accumulate GPU tensors for efficiency
        true_values.append(data.y)  # Store true values
        predicted_values.append(out.detach())  # Store predicted values

    # Step scheduler if provided
    if scheduler:
        scheduler.step()

    # Compute average loss
    avg_loss = total_loss.item() / len(loader)
    
    # Move tensors to CPU in bulk
    true_values = torch.cat(true_values).cpu()
    predicted_values = torch.cat(predicted_values).cpu()

    # Return metrics
    return avg_loss, true_values, predicted_values

@torch.no_grad()
def test(model, loader, device):
    model.eval()
    total_loss = torch.tensor(0.0, device=device)
    true_values = []
    predicted_values = []

    for data in loader:
        data = data.to(device)

        # Forward pass
        out, _ = model(data)

        # Compute loss
        loss = F.cross_entropy(out, data.y)  # Cross Entropy Loss for classification
        total_loss += loss.item()

        # Logging
        true_values.append(data.y)  # Store true values
        predicted_values.append(out.detach())  # Store predicted values
        
    # Compute average loss
    avg_loss = total_loss.item() / len(loader)
    
    # Move tensors to CPU in bulk
    true_values = torch.cat(true_values).cpu()
    predicted_values = torch.cat(predicted_values).cpu()

    # Return metrics
    return avg_loss, true_values, predicted_values

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    true_values = []
    predicted_values = []

    for data in loader:
        data = data.to(device)

        # Forward pass
        out, _ = model(data)

        # Logging
        true_values.append(data.y.cpu())  # Store true values
        predicted_values.append(out.cpu())  # Store predicted values

    # Concatenate true and predicted values across batches
    true_values = torch.cat(true_values)
    predicted_values = torch.cat(predicted_values)

    # Compute accuracy
    _, predicted_classes = torch.max(predicted_values, 1)
    accuracy = (predicted_classes == true_values).float().mean().item()

    return true_values, predicted_classes, accuracy

class EarlyStopper:
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None

    def early_stop(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            return False
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
            return False