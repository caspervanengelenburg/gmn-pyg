import torch
import numpy as np
import networkx as nx

def euclidean_distance(x, y):
    """Computes the Euclidean distance between x and y."""
    return torch.sum((x - y) ** 2, dim=-1)


def euclidean_similarity(x,y):
    """Computes a similarity directly based on the
    Euclidean distance between x and y."""
    return - euclidean_distance(x, y)


def approximate_hamming_similarity(x, y):
    """Approximate Hamming similarity."""
    return torch.mean(torch.tanh(x) * torch.tanh(y), dim=1)


def exact_hamming_similarity(x, y):
    """Compute the binary Hamming similarity."""
    match = ((x > 0) * (y > 0)).float()
    return torch.mean(match, dim=1)


def accuracy_threshold(d, label): #d = distance - tau! (for comparing with 0!)
    if isinstance(label.numpy(), int): #for triplets
        if label == 1:
            return torch.sum(d <= 0).item()
        else:
            return torch.sum(d >  0).item()
    else:
        similar_inds = np.where(label.numpy() == 1)[0]
        dissimilar_inds = np.where(label.numpy() == 0)[0]
        corrects = torch.sum(d[similar_inds] <= 0).item() + torch.sum(d[dissimilar_inds] > 0).item()
        return corrects