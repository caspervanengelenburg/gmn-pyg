import torch
from sklearn import metrics

from metrics import euclidean_distance, approximate_hamming_similarity


def pairwise_loss(x, y, labels, config):
    """
    Compute margin loss. Two implementations exist:
    - margin-based loss with Euclidean distance (distance-based).
    - hamming loss with approximate hamming similarity (similarity-based)
    """

    # determine loss type
    loss_type = config.optimize.distance

    labels = labels.float()

    if loss_type == 'margin':
        return torch.relu(config.optimze.margin - labels * (1 - euclidean_distance(x, y)))
    elif loss_type == 'hamming':
        return 0.25 * (labels - approximate_hamming_similarity(x, y)) ** 2
    else:
        raise ValueError('Unknown loss_type %s' % loss_type)


def triplet_loss(x_1p, x_2p, x_1n, x_3n, cfg):
    """
    Compute triplet loss. Two implementations exist:
    - margin-based loss with Euclidean distance (distance-based).
    - hamming loss with approximate hamming similarity (similarity-based)
    """

    # determine loss type
    loss_type = cfg.optimize.distance

    # compute loss
    if loss_type == 'margin':
        # relu(x) is same as max(0, x)
        return torch.relu(cfg.optimize.margin +
                          euclidean_distance(x_1p, x_2p) -
                          euclidean_distance(x_1n, x_3n))
    elif loss_type == 'hamming':
        # hamming loss is encouraged when representation vectors are binary;
        # which is useful for searching through large databases of graphs with low-latency.
        return 0.125 * ((approximate_hamming_similarity(x_1p, x_2p) - 1) ** 2 +
                        (approximate_hamming_similarity(x_1n, x_3n) + 1) ** 2)
    else:
        raise ValueError('Unknown loss_type %s' % loss_type)


def auc(scores, labels, **auc_args):
    """
    Computes the AUC for pair classification.
    """

    scores_max = torch.max(scores)
    scores_min = torch.min(scores)

    # normalize scores to [0, 1] and add a small epislon for safety
    scores = (scores - scores_min) / (scores_max - scores_min + 1e-8)

    labels = (labels + 1) / 2

    fpr, tpr, thresholds = metrics.roc_curve(labels.cpu().detach().numpy(), scores.cpu().detach().numpy())
    return metrics.auc(fpr, tpr)