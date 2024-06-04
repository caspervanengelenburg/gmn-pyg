import networkx as nx
import numpy as np
import torch
import random
import copy
import os
import pickle


def generate_binomial_graph(num_nodes=20, pe=0.2, is_connected=True, seed=None):
    n = sample_number_of_nodes(num_nodes)
    p = sample_edge_probability(pe)

    G = nx.erdos_renyi_graph(n, p)

    if is_connected:
        while not nx.is_connected(G):
            G = nx.erdos_renyi_graph(n, p, seed=seed)

    return G, n


def sample_number_of_nodes(num_nodes):
    if isinstance(num_nodes, list):
        n = np.random.randint(num_nodes[0], num_nodes[1])
    elif isinstance(num_nodes, int):
        n = num_nodes
    else:
        raise ValueError("Number of nodes should be a list (of two integers) or integer")

    return n


def sample_edge_probability(pe):
    if isinstance(pe, list):
        p = np.random.uniform(pe[0], pe[1])
    elif isinstance(pe, float):
        p = pe
    else:
        raise ValueError("Edge probability should be a list (of two floats) or float")

    return p


def permute_graph_nodes(G):
    """
    Permute node ordering of a graph, returns a new graph.

    Copied from:
    https://github.com/Lin-Yijie/Graph-Matching-Networks/blob/main/GMN/dataset.py.
    """
    n = G.number_of_nodes()
    G_new = nx.Graph()
    G_new.add_nodes_from(range(n))
    perm = np.random.permutation(n)
    edges = G.edges()
    new_edges = []
    for x, y in edges:
        new_edges.append((perm[x], perm[y]))
    G_new.add_edges_from(new_edges)

    return G_new


def substitute_random_edges(G, k):
    """Substitutes k randomly picked edges in graph G by k other random (and not yet existing) edges."""

    G = copy.deepcopy(G)
    if k > G.number_of_edges():
        raise ValueError("k cannot be larger than the number of edges in the graph")

    # Pick k random edges to remove
    edges_to_remove = random.sample(list(G.edges()), k)

    # Generate k new random edges that do not already exist in the graph
    new_edges = set()
    nodes = list(G.nodes())

    while len(new_edges) < k:
        u, v = random.sample(nodes, 2)
        if (not G.has_edge(u, v)) \
                and ((u, v) not in new_edges) \
                and ((v, u) not in new_edges):
            new_edges.add((u, v))

    # Substitute the edges in the graph
    G.remove_edges_from(edges_to_remove)
    G.add_edges_from(new_edges)

    return G


# helper functions for GMN
def reshape_and_split_tensors(graph_feats, n_splits):
    feats_dim = graph_feats.shape[-1]
    graph_feats = torch.reshape(graph_feats, [-1, feats_dim * n_splits])
    graph_feats_splits = []
    for i in range(n_splits):
        graph_feats_splits.append(graph_feats[:, feats_dim * i: feats_dim * (i + 1)])
    return graph_feats_splits


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1, alpha=0.1):
        self.moving_avg = alpha * val + (1-alpha) * self.val
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, filename):
    torch.save(state, f'{filename}.pth.tar')


def load_checkpoint(model, filename):
    filename = f'{filename}.pth.tar'
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename,
            map_location=lambda storage, loc: storage.cuda() if torch.cuda.is_available() else storage.cpu())

        state_dict = checkpoint['state_dict']

        # new_state_dict = OrderedDict()
        # for k, v in state_dict.items():
        #     name = k[7:] # remove `module.`
        #     new_state_dict[name] = v

        model.load_state_dict(state_dict)
        print("=> loaded checkpoint '{}'"
                .format(filename))
    else:
        print("=> no checkpoint found at '{}'".format(filename))


def save_pickle(object, filename):
    """
    Saves a pickled file.
    """
    with open(filename, 'wb') as f:
        pickle.dump(object, f)
    f.close()


def load_pickle(filename):
    """
    Loads a pickled file.
    """
    with open(filename, 'rb') as f:
        object = pickle.load(f)
        f.close()
    return object
