import torch.nn as nn
import torch
from torch_geometric.nn import MessagePassing, BatchNorm
from torch_scatter import scatter_mean

"""
Naming convention:
- N_v: number of nodes
- N_e: number of edges
- N_g: number of graphs
- D_v: node feature size
- D_e: edge feature size
- D_g: graph feature size
"""


# helper function
def mlp(feat_dim):
    layer = []
    for i in range(len(feat_dim) - 1):
        layer.append(nn.Linear(feat_dim[i], feat_dim[i + 1]))
        layer.append(nn.ReLU())
    return nn.Sequential(*layer)


def embed_layer(vocab_size, dim, drop=0.5):
    return nn.Sequential(nn.Embedding(vocab_size, dim),
                         nn.ReLU(),
                         nn.Dropout(drop))


def cosine_distance_torch(x1, x2=None, eps=1e-8):
    """Computes pairwise similarity between two vectors x and y"""
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = x2.norm(p=2, dim=1, keepdim=True)
    return torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)


def cross_attention(x, y, sim=cosine_distance_torch):
    """Computes attention between x an y, and y and x"""
    a = sim(x, y)
    a_x = torch.softmax(a, dim=1)  # i->j
    a_y = torch.softmax(a, dim=0)  # j->i
    attention_x = torch.mm(a_x, y)
    attention_y = torch.mm(torch.transpose(a_y, 1, 0), x)
    return attention_x, attention_y


def batch_pair_cross_attention(feats, batch, **kwargs):
    """Computes the cross graph attention between pairs of graph for a whole batch."""

    # find number of blocks = number of individual graphs in batch
    n_blocks = torch.unique(batch).size()[0]

    # create partitions
    block_feats = []
    for block in range(n_blocks):
        block_feats.append(feats[batch == block, :])

    # loop over all block pairs
    outs = []
    for i in range(0, n_blocks, 2):
        x = block_feats[i]
        y = block_feats[i + 1]
        attention_x, attention_y = cross_attention(x, y, **kwargs)
        outs.append(attention_x)
        outs.append(attention_y)
    results = torch.cat(outs, dim=0)

    return results


def init_weights(m, gain=1.0, bias=0.01):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=gain)
        m.bias.data.fill_(bias)


# graph embedding class
class GraphEncoder(nn.Module):
    """Encoder module that projects node and edge features to learnable embeddings."""

    def __init__(self,
                 node_dim=32,
                 edge_dim=32,
                 node_has_features=False,
                 edge_has_features=False,
                 edge_dim0=8):  # 1 for binary edge attributes (*eg* presence of door or not)
        super(GraphEncoder, self).__init__()

        # Node and edge dimensions
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.node_has_features = node_has_features
        self.edge_has_features = edge_has_features

        # Node and edge encoders are MLPs
        if not node_has_features:  # ged task
            pass
        else:  # floor plan task
            self.cats_one_hot = embed_layer(13, node_dim)
            self.geom_encoder = mlp([5, node_dim])
            self.node_encoder = mlp([2 * node_dim, node_dim])
            self.edge_encoder = mlp([edge_dim0, edge_dim])

            # Initialize weights
            init_weights(self.geom_encoder)
            init_weights(self.node_encoder)
            init_weights(self.edge_encoder)

    # Forward method:
    # Room graphs: x1 = geometry, x2 = category
    # GED: x1 = ones
    def forward(self, x1, x2, edge_feat):

        # node encoding
        if not self.node_has_features:  # ged task
            node_encod = x1
        else:  # floor plan task
            x1 = self.cats_one_hot(x1)  # one-hot categorical encoding // 5  -> D_v
            x2 = self.geom_encoder(x2)  # geometry encoding // 12 -> D_v
            node_encod = torch.cat((x2, x1.squeeze(1)), -1)  # stack // D_v x D_v -> 2*Dv
            node_encod = self.node_encoder(node_encod)  # full node embedding // 2*D_v -> D_v

        # edge embedding
        if not self.edge_has_features:  # ged task
            edge_encod = edge_feat
        else:  # floor plan task
            edge_encod = self.edge_encoder(edge_feat)  # edge encoding // 8 -> D_e

        return node_encod, edge_encod  # D_v, D_e


# Graph Matching Convolution
class GMNConv(MessagePassing):
    def __init__(self, in_node, in_edge, out_node,
                 aggr="add", node_update_type="gru",
                 message_gain=0.1):
        super(GMNConv, self).__init__(aggr=aggr)

        # Hyper settings
        self.node_update_type = node_update_type

        # Message passing (MLP)
        self.f_message = torch.nn.Linear(in_node * 2 + in_edge, out_node)
        init_weights(self.f_message, gain=message_gain)

        # Node update (MLP or GRU)
        if node_update_type == "mlp":
            self.f_node = torch.nn.Linear(out_node + in_node, out_node)
            init_weights(self.f_node)
        elif node_update_type == "gru":
            self.f_node = torch.nn.GRU(out_node + in_node, out_node)
        else:
            raise NotImplementedError("Currently, only MLP and GRU are implemented.")

        # Batch normalization
        self.batch_norm = BatchNorm(out_node)

    def forward(self, edge_index, x, edge_attr, batch):
        # x_transformed = self.lin_node(x)
        return self.propagate(edge_index, x=x, edge_attr=edge_attr, original_x=x, batch=batch)

    def message(self, x_i, x_j, edge_attr):
        x = torch.cat([x_i, x_j, edge_attr], dim=1)  # [own features, neighbouring features, edge features]
        x = self.f_message(x)
        return x

    def update(self, aggr_out, original_x, batch):
        cross_graph_attention = batch_pair_cross_attention(original_x, batch)
        attention_input = original_x - cross_graph_attention
        if self.node_update_type == "gru":
            aggr_out,_ = self.f_node(torch.cat([aggr_out, attention_input], dim=1))
        else:
            aggr_out = self.f_node(torch.cat([aggr_out, attention_input], dim=1))
        aggr_out = self.batch_norm(aggr_out)
        return aggr_out


class GConv(MessagePassing):
    def __init__(self, in_node, in_edge, out_node,
                 aggr="add", node_update_type="gru",
                 message_gain=0.1):
        super(GConv, self).__init__(aggr=aggr)

        # Hyper settings
        self.node_update_type = node_update_type

        # Message passing (MLP)
        self.f_message = torch.nn.Linear(in_node * 2 + in_edge, out_node)
        init_weights(self.f_message, gain=message_gain)

        # Node update (MLP or GRU)
        if node_update_type == "mlp":
            self.f_node = torch.nn.Linear(out_node, out_node)
            init_weights(self.f_node)
        elif node_update_type == "gru":
            self.f_node = torch.nn.GRU(out_node, out_node)
        else:
            raise NotImplementedError("Currently, only MLP and GRU are implemented.")

        # Batch normalization
        self.batch_norm = BatchNorm(out_node)

    def forward(self, edge_index, x, edge_attr, batch):
        # x_transformed = self.lin_node(x)
        return self.propagate(edge_index, x=x, edge_attr=edge_attr, original_x=x, batch=batch)

    def message(self, x_i, x_j, edge_attr):
        x = torch.cat([x_i, x_j, edge_attr], dim=1)  # [own features, neighbouring features, edge features]
        x = self.f_message(x)
        return x

    def update(self, aggr_out, original_x, batch):
        # cross_graph_attention = batch_pair_cross_attention(original_x, batch)
        # attention_input = original_x - cross_graph_attention
        if self.node_update_type == "gru":
            aggr_out,_ = self.f_node(aggr_out)
        else:
            aggr_out = self.f_node(aggr_out)
        aggr_out = self.batch_norm(aggr_out)
        return aggr_out


# Graph Aggregation Module
class GraphAggregator(torch.nn.Module):
    """Computes the graph-level embedding from the final node-level embeddings."""

    def __init__(self, in_node, out_graph):
        super(GraphAggregator, self).__init__()
        self.lin = torch.nn.Linear(in_node, out_graph)
        self.lin_gate = torch.nn.Linear(in_node, out_graph)
        self.lin_final = torch.nn.Linear(out_graph, out_graph)

        # Initialize weights
        init_weights(self.lin)
        init_weights(self.lin_gate)
        init_weights(self.lin_final)

    def forward(self, x, batch):
        x_states = self.lin(x)  # node states // [V x D_v] -> [V x D_F]
        x_gates = torch.nn.functional.softmax(self.lin_gate(x), dim=1)  # node gates // [N_v x D_v] -> [N_v x D_F]
        x_states = x_states * x_gates  # update states based on gate "gated states" // [N_v x D_g]
        x_states = scatter_mean(x_states, batch, dim=0)  # graph-level feature vectors // [N_v x D_g] -> [N_g x D_g]
        x_states = self.lin_final(x_states)  # final graph-level embedding // [N_g x D_g] -> [N_g x D_g]
        return x_states


# Graph matching network
class GraphConvolutionNetwork(torch.nn.Module):
    def __init__(self, cfg):
        super(GraphConvolutionNetwork, self).__init__()

        # hyper settings
        self.config = cfg
        self.node_dim = self.config.model.node_dim
        self.edge_dim = self.config.model.edge_dim
        try:
            self.inter_geom_dim = self.config.model.inter_geom_dim
        except:
            self.inter_geom_dim = None
        self.graph_dim = self.config.model.graph_dim
        self.num_layers = self.config.model.num_layers
        self.node_update_type = self.config.model.node_update_type
        self.message_gain = self.config.model.message_gain
        self.return_node_feats = return_node_feats

        # graph encoder
        self.encoder = GraphEncoder(self.node_dim,
                                    self.edge_dim,
                                    edge_dim0=self.inter_geom_dim,
                                    node_has_features=self.config.model.node_has_features,
                                    edge_has_features=self.config.model.edge_has_features)

        # graph convolutional layers
        self.conv_layers = torch.nn.ModuleList()
        for _ in range(self.num_layers):
            self.conv_layers.append(
                GConv(self.node_dim, self.edge_dim, self.node_dim,
                      node_update_type=self.node_update_type, message_gain=self.message_gain)
            )

        # graph aggregator
        self.aggregator = GraphAggregator(self.node_dim, self.graph_dim)

    def forward(self, edge_index, x1, x2, edge_feats, batch):

        # encode nodes and edges (individual encoders)
        node_feats, edge_feats = self.encoder(x1, x2, edge_feats)

        # compute node embeddings (graph matching convolutional layers)
        for i in range(self.num_layers):
            node_feats = self.conv_layers[i](edge_index, node_feats, edge_feats, batch)


        # compute graph-level embeddings (graph aggregation mean)
        graph_feats = self.aggregator(node_feats, batch)

        # return node- and graph-level embeddings
        return node_feats, graph_feats

    # To return node features only
    def get_node_features(self, edge_index, x1, x2, edge_feats, batch, layers):

        if int(torch.max(layers)) > self.num_layers:
            raise ValueError("The layer index should not exceed the number of layers in the network.")

        # encode nodes and edges (individual encoders)
        node_feats, edge_feats = self.encoder(x1, x2, edge_feats)

        # initialize dictionary for aggregating node features
        node_feats_dict = {}

        # compute node embeddings (graph matching convolutional layers)
        for i in range(int(torch.max(layers))):  # only compute until layer of interest
            node_feats = self.conv_layers[i](edge_index, node_feats, edge_feats, batch)
            if i in layers:
                # Add node features to dictionary:
                # - key = i-th layer;
                # - value = node features at layer i.
                node_feats_dict[i] = node_feats

        return node_feats_dict


class GraphMatchingNetwork(GraphConvolutionNetwork):
    def __init__(self, cfg):
        super(GraphConvolutionNetwork, self).__init__()

        # hyper settings
        self.config = cfg
        self.node_dim = self.config.model.node_dim
        self.edge_dim = self.config.model.edge_dim
        try:
            self.inter_geom_dim = self.config.model.inter_geom_dim
        except:
            self.inter_geom_dim = None
        self.graph_dim = self.config.model.graph_dim
        self.num_layers = self.config.model.num_layers
        self.node_update_type = self.config.model.node_update_type
        self.message_gain = self.config.model.message_gain

        # graph encoder
        self.encoder = GraphEncoder(self.node_dim,
                                    self.edge_dim,
                                    edge_dim0=self.inter_geom_dim,
                                    node_has_features=self.config.model.node_has_features,
                                    edge_has_features=self.config.model.edge_has_features)

        # graph convolutional layers
        self.conv_layers = torch.nn.ModuleList()
        for _ in range(self.num_layers):
            self.conv_layers.append(
                GMNConv(self.node_dim, self.edge_dim, self.node_dim,
                        node_update_type=self.node_update_type, message_gain=self.message_gain)
            )

        # graph aggregator
        self.aggregator = GraphAggregator(self.node_dim, self.graph_dim)
