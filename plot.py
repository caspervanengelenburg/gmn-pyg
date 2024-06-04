import matplotlib.pyplot as plt


def set_figure(nr,
               nc,
               fs=10,
               fs_title=7.5,
               fs_legend=10,
               fs_xtick=3,
               fs_ytick=3,
               fs_axes=4,
               ratio=1,
               fc='black',
               aspect='equal',
               axis='off'):
    """
    Custom figure setup function that generates a nicely looking figure outline.
    It includes "making-sense"-font sizes across all text locations (e.g. title, axes).
    You can always change things later yourself through the outputs or plt.rc(...).
    """

    fig, axs = plt.subplots(nr, nc, figsize=(fs * nc * ratio, fs * nr))
    fig.set_facecolor(fc)

    try:
        axs = axs.flatten()
        for ax in axs:
            ax.set_facecolor(fc)
            ax.set_aspect(aspect)
            ax.axis(axis)
    except:
        axs.set_facecolor(fc)
        axs.set_aspect(aspect)
        axs.axis(axis)

    plt.rc("figure", titlesize=fs * fs_title)
    plt.rc("legend", fontsize=fs * fs_legend)
    plt.rc("xtick", labelsize=fs * fs_xtick)
    plt.rc("ytick", labelsize=fs * fs_ytick)
    plt.rc("axes", labelsize=fs * fs_axes, titlesize=fs * fs_title)

    return fig, axs


# SIMPLE GRAPHS
import networkx as nx
import numpy as np
import copy


# Function to normalize edges
def normalize_edge(u, v):
    return tuple([min(u, v), max(u, v)])


# Function to get particular graph from a batch
def get_graph_from_batch(batch, order_id, batch_id):
    G = batch.subgraph([n for n, d in batch.nodes(data=True)
                        if (d['order'] == order_id) * (d['batch'] == batch_id)])
    return G


# Function to plot a batch of graph triplets.
def plot_graph_batch(batch, bs,
                     fs=4,
                     fs_title=5,
                     background="white",
                     node_color="black",
                     edge_color="black",
                     node_size=None,
                     edge_size=None,
                     pos_type="circular",
                     set_title=True):
    """Plots a batch of triplet graphs."""

    global pos
    if node_size is None:
        node_size = 50 * fs,
    if edge_size is None:
        edge_size = fs / 4

    numbers = [1, 2, 1, 3]

    # set up plot
    _, axs = set_figure(bs, 4, fs=fs, fc=background, fs_title=fs_title)

    for i, ax in enumerate(axs):

        # Set the batch and order indices
        batch_id = int(np.floor(i / 4))
        order_id = i % 4

        # Extract G1
        G1 = get_graph_from_batch(batch, 0, batch_id)

        # Extract G2 (and G3
        if order_id in (0, 2):  # G1
            G2 = get_graph_from_batch(batch, order_id + 1, batch_id)
        elif order_id in (1, 3):  # G2 or G3
            G2 = get_graph_from_batch(batch, order_id, batch_id)
        else:
            raise ValueError("Order should be 0, 1, 2, or 3")

        # Shift G2 (or G3) back
        # - Because graphs are packed as big graph, the nodes have a different index.
        # - Here we shift them back: simply as "node - number of nodes"
        shift = G2.number_of_nodes()
        if order_id in (2, 3):
            shift *= 3
        G2_new = nx.Graph()  # initialize new G2
        G2_new.add_nodes_from([n - shift for n in G2.nodes()])  # shift nodes
        G2_new.add_edges_from([(u - shift, v - shift) for u, v in G2.edges()])  # shift edges
        G2 = copy.deepcopy(G2_new)  # overwrite G2

        # Check whether node sets are equivalent
        if set(G1.nodes()) != set(G2.nodes()):
            raise ValueError("The node sets of the two graphs are not the same")

        if order_id == 0:
            # get fixed node positions if not already given
            if pos_type == "circular":
                pos = nx.circular_layout(G1)
            elif pos_type == "spring":
                pos = nx.spring_layout(G1)
            elif pos_type == "spectral":
                pos = nx.spectral_layout(G1)
            elif pos_type == "random":
                pos = nx.random_layout(G1)
            elif pos_type == "kamada":
                pos = nx.kamada_kawai_layout(G1)
            else:
                raise ValueError("Position can only by 'circular' or 'spring'")

        # Extract edge sets
        edges_1 = set(normalize_edge(u, v) for u, v in G1.edges)
        edges_2 = set(normalize_edge(u, v) for u, v in G2.edges)

        # Compute overlap, removed, and added edge sets
        edges_removed = edges_1.difference(edges_2)  # diff between edge set 1 and edge set 2
        edges_added = edges_2.difference(edges_1)  # diff between edge set 2 and edge set 1
        edges_overlap = edges_1.intersection(edges_2)

        # Plot the graph
        graph_number = numbers[order_id]
        color = "red" if graph_number == 1 else "green"
        if set_title:
            ax.set_title(f"G{graph_number}\n{color}: {'removed' if graph_number == 1 else 'added'} edges")

        if graph_number == 1:
            G = G1
            edges_sub = edges_removed
        else:
            G = G2
            edges_sub = edges_added

        plot_graph_beautiful(G, ax, pos=pos,
                             edge_list=list(edges_overlap),
                             c_node=node_color, c_edge=edge_color,
                             node_size=node_size, edge_size=edge_size)

        plot_graph_beautiful(G, ax, pos=pos,
                             edge_list=list(edges_sub),
                             c_node=node_color, c_edge=color,
                             node_size=node_size, edge_size=edge_size * 4)


def plot_graph_beautiful(
        G,  # graph
        ax,  # axis
        c_node="black",  # color of the nodes (single or list of length N)
        c_edge="black",  # color of the edges (single or list of length E)
        pos="spring",  # node positioning ("single", "circle", or dictionary of node-coordinate values)
        node_size=10,  # marker size of the node
        edge_size=10,  # thickness of the edge
        connectionstyle="arc3",  # style of connection (default: arc3)
        edge_list=None,  # edge list
):
    """
    Function that plots a graph.
    """

    # get edge list
    if edge_list is None:
        edge_list = list(G.edges())

    # node placement
    if pos == "spring":
        pos = nx.spring_layout(G, seed=7)
    elif pos == "circular":
        pos = nx.circular_layout(G)

    # draw nodes and edges
    nx.draw_networkx_nodes(G, pos, ax=ax,
                           node_size=node_size, node_color=c_node)
    nx.draw_networkx_edges(G, pos, ax=ax, edgelist=edge_list,
                           edge_color=c_edge, width=edge_size,
                           connectionstyle=connectionstyle)
