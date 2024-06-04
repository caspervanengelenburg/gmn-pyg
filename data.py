import numpy as np
import torch
from torch_geometric.data import Dataset
from torch_geometric.utils import from_networkx
import networkx as nx

from utils import generate_binomial_graph, \
    substitute_random_edges, \
    permute_graph_nodes


# Triplet GED dataset of graphs
class TripletDatasetGED(Dataset):
    """Graph edit distance dataset. Creates triplets.
    Samples binomial graphs and substitutes kp edges by kp other edges to form G2;
    and kn edges in the case of G3. G2: positive, G3: negative. kp < kn."""

    def __init__(self, size=10e5, num_nodes=20,
                 kp=1, kn=2, pe=0.2,
                 node_dim=None, edge_dim=None,
                 is_connected=True, permute=True):

        super(TripletDatasetGED).__init__()

        self.size = int(size)
        self.num_nodes = num_nodes
        self.kp = kp
        self.kn = kn
        self.pe = pe
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.is_connected = is_connected
        self.permute = permute

        if self.kp >= self.kn:
            raise ValueError("Number of substitutions to get the positive should be "
                             "strictly lower than the number of substitutions to get the negative.")

    def __len__(self):
        return self.size

    def __getitem__(self, item):

        # Sample G1 (anchor) by generating a random binomial graph
        G1, num_nodes = generate_binomial_graph(num_nodes=self.num_nodes, pe=self.pe,
                                                is_connected=self.is_connected)

        # Modify G1 to get G2 (positive) and G3 (negative) by substituting kp and kn edges, resp.
        G2 = substitute_random_edges(G1, self.kp)
        G3 = substitute_random_edges(G1, self.kn)

        # Make sure that G2 and G3 are also connected
        if self.is_connected:
            while not nx.is_connected(G2):
                G2 = substitute_random_edges(G1, self.kp)
            while not nx.is_connected(G3):
                G3 = substitute_random_edges(G1, self.kn)

        # Permute
        if self.permute:
            G1 = permute_graph_nodes(G1)
            G2 = permute_graph_nodes(G2)
            G3 = permute_graph_nodes(G3)

        # Combine all graphs. G1 is added twice to make training script easier.
        G = nx.disjoint_union_all([G1, G2, G1, G3])

        # Add 2 node attributes
        # 1) "node_feat": all ones (Note (!): can also be added later: set node_dim to None)
        # 2) "order" order index (to distinguish between G1, G2, and G3)
        if self.node_dim is not None:
            G.add_nodes_from([[n, {'node_feat': torch.ones(self.node_dim)}] for n in G.nodes()])  # add node features
        G.add_nodes_from([[n,{'order': int(np.floor(n / num_nodes))}] for n in G.nodes()])  # add order indices

        # Add edge attributes: "edge_feat": all ones
        if self.edge_dim is not None:
            G.add_edges_from([(u,v, {'edge_feat': torch.ones(self.edge_dim)}) for u, v in G.edges()])

        # Convert to PyG graph
        G = from_networkx(G)

        return G