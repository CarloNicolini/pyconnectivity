#    This file is part of pyConnectivity
#
#    pyConnectivity is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 2 of the License, or
#    (at your option) any later version.
#
#    pyConnectivity is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with pyConnectivity. If not, see <http://www.gnu.org/licenses/>.
#
#    Copyright 2014 Carlo Nicolini <carlo.nicolini@iit.it>

__author__ = "Carlo Nicolini <carlo.nicolini@iit.it>"

__all__ = [
    "compute_global_measure", "centrality_distribution", "small_worldness_ER",
    "dist_inv_wei", "efficiency", "local_efficiency", "approximate_cpl",
    "trivial_power_law_estimator", "make_hist", "degrees_to_hist"]

# Back import from base __init__ file in __all__
# http://mikegrouchy.com/blog/2012/05/be-pythonic-__init__py.html

import networkx as nx
import numpy as np
import logging, statistic
import copy
import utils
from utils import is_weighted as is_weighted


def compute_global_measure(inputMatrix, weightedGraph=False, minThreshold=-1.0,
                           maxThreshold=1.0, totalThresholds=100, measuresList=[
        'threshold', 'density',
        'number_of_nodes_giant_component', 'number_of_edges', 'number_connected_components']):
    """Plot a global graph measure depending on threshold value"""
    n = inputMatrix.shape[0]
    # http://www.ncbi.nlm.nih.gov/pmc/articles/PMC2893007/
    # per informazioni migliori su come calcolare average_shortest_path_length
    # Here we adopt a better and maybe faster method to study the threshold
    # dependence, by simply generating all the thresholds linearly
    measures = {}
    for m in measuresList:
        measures[m] = list()

    thresholds = np.linspace(minThreshold, maxThreshold, totalThresholds)

    for n, threshold in enumerate(thresholds):
        graph = graphFromMatrix(
            thresholdMatrixAbsolute(inputMatrix, threshold), weighted=weightedGraph)
        # and nx.is_connected(graph):
        if graph.number_of_nodes() > 1 and graph.number_of_edges() > 2:
            for m in measuresList:
                if m is 'threshold':
                    val = threshold
                elif m is 'density':
                    val = density
                elif m is 'number_of_nodes_giant_component':
                    val = nx.connected_component_subgraphs(
                        graph)[0].number_of_nodes()
                elif m is 'number_of_edges':
                    val = graph.number_of_edges
                elif m is 'number_connected_components':
                    val = nx.number_connected_components(graph)
                elif m is 'degree_assortativity_coefficient':
                    val = nx.degree_assortativity_coefficient(graph)
                elif m is 'average_clustering':
                    val = nx.average_clustering(graph)
                elif m is 'small_worldness':
                    val = small_worldness_ER(graph)
                elif m is 'efficiency':
                    val = efficiency(graph)
                elif m is 'CPL':
                    val = approximate_cpl(graph)
                else:
                    raise Exception('Value not known in the list')
                measures[m].append(val)
    return measures


def centrality_distribution(graph):
    """Returns a centrality distribution of a networkx graph
    Each normalized centrality is divided by the sum of the normalized
    centralities. Note, this assumes the graph is simple.
    """
    centrality = nx.degree_centrality(graph).values()
    centrality = np.np.asarray(centrality)
    centrality /= centrality.sum()
    return centrality


def small_worldness_ER(graph, runs=100):
    """Small World Index (Q) computation
    In this example we do use the Erdos-Renyi random model to compute Q. This
    model is not a good null model for most empirical networks (see [1]_).

    References
    [1] Brian Uzzi, Luis AN Amaral, Felix Reed-Tsochas (2007).
    Small-world networks and management science research: A review. European Management Review. 2(4). pp 77-91
    http://www.kellogg.northwestern.edu/faculty/uzzi/ftp/Uzzi_EuropeanManReview_2007.pdf
    """
    n = graph.number_of_nodes()
    m = graph.number_of_edges()

    CC = nx.average_clustering(graph)   # Actual CC
    L = approximate_cpl(graph)   # Actual L
    Ls = []
    CCs = []
    for i in range(runs):
        r_graph = nx.gnm_random_graph(n, m)   # Create a ER random network
        Ger = nx.connected_component_subgraphs(r_graph)[0]
        CCs.append(nx.average_clustering(Ger))
        Ls.append(approximate_cpl(Ger))
    return (CC / np.mean(CCs)) / (L / np.mean(Ls))


def dist_inv_wei(graph):
    """Distance matrix computed using Dijkstra algorithm
        mat = dist_inv_wei(graph)
     
        The distance matrix contains lengths of shortest paths between all
        pairs of nodes. An entry (u,v) represents the length of shortest path
        from node u to node v. The average shortest path length is the
        characteristic path length of the network.
        Input:      graph,      Directed/undirected graph.
        Output:     mat,      distance (shortest weighted path) matrix
        Modification history:
        2009-08-04: min() function vectorized
        Python Conversion Jean-Christophe KETZINGER, INSERM UMRS678, 2013
        Algorithm: Dijkstra's algorithm.
        Reference:Mika Rubinov, UNSW/U Cambridge, 2007-2012; Rick Betzel and 
        Andrea Avena, IU, 2012
    """
    path_length = nx.all_pairs_dijkstra_path_length(graph, weight='weight')
    mat = []
    sortedcol = keysort(path_length)
    zerospos = []
    for idx, col in enumerate(sortedcol):
        keys = keysort(col)
        line = []
        for (k, v) in enumerate(keys):
            if (v != 0):
                line.append(1.0 / v)
            else:
                line.append(v)
        if (len(line) == 1):
            val = np.argmax(sorted(graph.nodes()) == col.keys()[0])
            zerospos.append(val)
        else:
            mat.append(line)
    mat = np.array(mat)
    for i, val in enumerate(zerospos):
        mat = np.insert(mat, val, 0, axis=0)
        mat = np.insert(mat, val, 0, axis=1)
    return mat


def efficiency(G, wei_loc=None):
    """Efficiency of a network
    https://groups.google.com/forum/#!topic/networkx-discuss/ycxtVuEqePQ"""
    avg = 0.0
    graph = G.copy()
    n = len(graph)

    if is_weighted(graph):   # efficiency_wei
        for (u, v, d) in graph.edges(data=True):
            # Compute the connection-length matrix
            d['weight'] = 1 / (d['weight'] * 1.0)
        if (wei_loc is None):
            for node in graph:
                path_length = nx.single_source_dijkstra_path_length(
                    graph, node)
                avg += sum(1.0 / v for v in path_length.values() if v != 0)
            avg *= 1.0 / (n * (n - 1))
        else:
            mat = dist_inv_wei(graph)
            e = np.multiply(mat, wei_loc) ** (1 / 3.0)
            e_all = np.matrix(e).ravel().tolist()
            avg = sum(e_all[0])
            avg *= 1.0 / (n * (n - 1))  # local efficiency
    else:  # efficiency_bin
        for node in graph:
            path_length = nx.single_source_shortest_path_length(graph, node)
            avg += sum(1.0 / v for v in path_length.values() if v != 0)
        avg *= 1.0 / (n * (n - 1))
    return avg


def local_efficiency(graph):
    """Local efficiency vector
    efficiency_vector = local_efficiency(G)

    This function compute for each node the efficiency of its
    immediate neighborhood and is related to the clustering coefficient.

    Inputs: G,      The graph on which we want to compute the local efficiency

    Output: efficiency_vector,  return as many local efficiency value as
    there are nodes in our graph

    Algorithm: algebraic path count

    Reference: Latora and Marchiori (2001) Phys Rev Lett 87:198701.
    Mika Rubinov, UNSW, 2008-2010
    Jean-Christophe KETZINGER, INSERM UMRS678 PARIS, 2013

    Modification history:
    2010: Original version from BCT (Matlab)
    Python Conversion Jean-Christophe KETZINGER, INSERM UMRS678, 2013
    """
    assert isinstance(graph, nx.Graph)
    efficiency_vector = []
    for node in graph:
        # Get the neighbors of our interest node
        neighbors = graph.neighbors(node)
        neighbors = np.sort(neighbors, axis=None)  # sort the neighbors list
        # Create the subgraph composed exclusively with neighbors
        SG = nx.subgraph(graph, neighbors)
        # assert that the subragh is not only one edge
        if (len(neighbors) > 2):
            if is_weighted(SG):
                GuV = []
                GVu = []
                GWDegree = nx.to_numpy_matrix(graph)
                for neighbor in neighbors:
                    GuV.append(GWDegree[node, neighbor])
                    GVu.append(GWDegree[neighbor, node])
                GVuGuV = (np.outer(np.array(GVu), np.array(GuV).T))
                node_efficiency = efficiency(SG, GVuGuV)
                # compute the global efficiency of this subgraph
                efficiency_vector.append(node_efficiency)
            else:
                efficiency_vector.append(efficiency(SG))
        else:
            efficiency_vector.append(0.0)  # or set it's efficiency value to 0
    return efficiency_vector


def _estimate_s(q, delta, eps):
    delta2 = delta * delta
    delta3 = (1 - delta) * (1 - delta)
    return (2. / (q * q)) * np.log(2. / eps) * delta3 / delta2


def approximate_cpl(graph, q=0.5, delta=0.15, eps=0.05):
    """Computes the approximate CPL for the specified graph
    :param graph: the graph
    :param q: the q-median to use (default 1/2-median, i.e., median)
    :param delta: used to compute the size of the sample
    :param eps: used to compute the size of the sample
    :return: the median
    :rtype: float
    https://ep2013.europython.eu/media/conference/slides/social-network-analysis-in-python.pdf
    """
    assert isinstance(graph, nx.Graph)
    s = _estimate_s(q, delta, eps)
    s = int(np.ceil(s))
    if graph.number_of_nodes() <= s:
        sample = graph.nodes_iter()
    else:
        sample = random.sample(graph.adj.keys(), s)

    averages = []
    for node in sample:
        path_lengths = nx.single_source_shortest_path_length(graph, node)
        average = np.sum(path_lengths.values()) / float(len(path_lengths))
        averages.append(average)

    return np.median(np.array(averages))


def trivial_power_law_estimator(data, x0=None):
    if x0 is None:
        x0 = min(data)
        xs = data
    else:
        xs = [x for x in data if x >= x0]

    s = np.sum(np.log(x / x0) for x in xs)
    return 1. + len(data) / s


def make_hist(xs):
    values = np.zeros(xs[-1] + 1, dtype=int)
    for val in xs:
        values[val] += 1
    return values


def degrees_to_hist(dct):
    xs = dct.values()
    xs.sort()
    values = make_hist(xs)
    return values


def node_properties_color_map(graph, coloring_method):

    node_color = {}
    if coloring_method is 'betwenness_centrality':
        node_color = nx.betweenness_centrality(graph)
    elif coloring_method is 'degree_centrality':
        node_color = nx.degree_centrality(graph)
    elif coloring_method is 'closeness_centrality':
        node_color = nx.closeness_centrality(graph)
    elif coloring_method is 'eigenvector_centrality':
        node_color = nx.eigenvector_centrality(graph)
    elif coloring_method is 'connected_components':
        componentsDictionary = nx.connected_components(graph)
        componentLabel = 0
        for componentNodes in componentsDictionary:
            for node in componentNodes:
                node_color[node] = componentLabel
            componentLabel = componentLabel + 1
    elif coloring_method is 'maximum_modularity_partition':
        node_color = community.best_modularity_partition(graph)
    elif coloring_method is 'clustering_coefficient':
        nx.clustering(graph)
        node_color = nx.square_clustering(graph)
    elif coloring_method is 'emisphere':
        node_color = community.best_partition(graph)
        for n in node_color.keys():
            if n[-1] == 'L':
                node_color[n] = 0.0
            else:
                node_color[n] = 1.0
    elif coloring_method is 'maximum_surprise_partition':
        node_color = community.best_surprise_partition_louvain(graph)
    else:
        raise Exception('Non supported coloring_method')
    return community.__renumber(node_color)

"""
def main():
    G = nx.karate_club_graph()
    print G
    print local_efficiency(G), efficiency(G), approximate_cpl(G),
    dist_inv_wei(G)

main()
"""
