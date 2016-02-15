#    This file is part of pyConnectivity
#
#    pyConnectivity is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 2 of the License, or
#    (at your option) any later version.
#
#    pyConnectivity is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with pyConnectivity. If not, see <http://www.gnu.org/licenses/>.
#
#    Copyright 2014 Carlo Nicolini <carlo.nicolini@iit.it>

#    This module has been downloaded from
#    https://bitbucket.org/taynaud/python-louvain/
#    so check it sometimes to see if there are improvements
#
#    Further informations and API are available at
#    http://perso.crans.org/aynaud/communities/api.html
#
#    This module implements some community detection algorithms


# Back import from base __init__ file in __all__
# http://mikegrouchy.com/blog/2012/05/be-pythonic-__init__py.html
from copy import *
from collections import defaultdict
from itertools import count
import numpy as np
import random

from scipy.special import erf
import mpmath

import numpy as np
from pyconnectivity import nx
from pyconnectivity import copy
from pyconnectivity import os
from pyconnectivity import logging
from pyconnectivity import utils

__all__ = [
    "partition_at_level",
    "modularity",
    "best_modularity_partition",
    "best_infomap_partition",
    "best_surprise_partition_louvain",
    "surprise",
    "surprise_weighted",
    "best_surprise_partition",
    "best_surprise_partition_weighted",
    "modularity_louvain_und_sign",
    "modularity_finetune_und_sign",
    "surprise_gurobi",
    "modularity_gurobi"]

__author__ = "Carlo Nicolini <carlo.nicolini@iit.it>"

__PASS_MAX = -1
__MIN = 0.0000001


def partition_at_level(dendrogram, level):
    """
    Return the partition of the nodes at the given level
    A dendrogram is a tree and each level is a partition of the graph nodes.
    Level 0 is the first partition, which contains the smallest communities,
    and the best is len(dendrogram) - 1.
    The higher the level is, the bigger are the communities
    Parameters:
    dendrogram: list of dict
       a list of partitions, ie dictionnaries where keys of the i+1 are the
       values of the i.
    level: int
       the level which belongs to [0..len(dendrogram)-1]
    Returns:
    partition: dictionnary
       A dictionary where keys are the nodes and the values are the set it
       belongs to.
    See Also:
    best_modularity_partition which directly combines partition_at_level and
    generate_dendrogram to obtain the partition of highest modularity

    Examples
    --------
        G=nx.erdos_renyi_graph(100, 0.01)
        dendo = generate_dendrogram(G)

        for level in range(len(dendo) - 1):
            print "partition at level", level, "is", partition_at_level(dendo, level)
    """
    partition = dendrogram[0].copy()
    for index in range(1, level + 1):
        for node, community in partition.iteritems():
            partition[node] = dendrogram[index][community]

    return partition


def modularity(graph, partition):
    """
    Compute the modularity of a partition of a graph
    Parameters:
    partition: dict
       the partition of the nodes, i.e a dictionary where keys are their nodes
       and values the communities
    graph: networkx.Graph
       the networkx graph which is decomposed
    Returns:
    modularity: float
       The modularity
    Raises:
    KeyError
       If the partition is not a partition of all graph nodes
    ValueError
        If the graph has no link
    TypeError
        If graph is not a networkx.Graph

    References:
    .. 1. Newman, M.E.J. & Girvan, M. Finding and evaluating community
    structure in networks. Physical Review E 69, 26113(2004).
    Examples
    --------
        G=nx.erdos_renyi_graph(100, 0.01)
        part = best_modularity_partition(G)
        modularity(G,part)
    """
    if not isinstance(graph, nx.Graph):
        raise TypeError("Bad graph type, use only non directed graph")

    inc = dict([])
    deg = dict([])
    links = graph.size(weight='weight')
    if links == 0:
        raise ValueError("A graph without link has an undefined modularity")

    for node in graph:
        com = partition[node]
        deg[com] = deg.get(com, 0.) + graph.degree(node, weight='weight')
        for neighbor, datas in graph[node].iteritems():
            weight = datas.get("weight", 1)
            if partition[neighbor] == com:
                if neighbor == node:
                    inc[com] = inc.get(com, 0.) + float(weight)
                else:
                    inc[com] = inc.get(com, 0.) + float(weight) / 2.0
    res = 0.
    for com in set(partition.values()):
        res += (inc.get(com, 0.) / links) - \
            (deg.get(com, 0.) / (2. * links)) ** 2
    return res


def best_modularity_partition(graph, partition=None):
    """Compute the partition of the graph nodes which maximises the modularity
    (or try..) using the Louvain heuristices
    This is the partition of highest modularity, i.e. the highest partition of
    the dendrogram generated by the Louvain algorithm.

    Parameters
    ----------
    graph: networkx.Graph
       the networkx graph which is decomposed
    partition: dict, optionnal
       the algorithm will start using this partition of the nodes.
       It's a dictionary where keys are their nodes and values the communities

    Returns
    -------
    partition: dictionnary
       The partition, with communities numbered from 0 to number of communities

    Notes
    -----
    Uses Louvain algorithm
    1. Blondel, V.D. et al. Fast unfolding of communities in large networks.
    J. Stat. Mech 10008, 1-12(2008).
    """
    dendo = generate_dendrogram(graph, partition)
    return partition_at_level(dendo, len(dendo) - 1)


def generate_dendrogram(graph, part_init=None, method='modularity'):
    """Find communities in the graph and return the associated dendrogram
    A dendrogram is a tree and each level is a partition of the graph nodes.
    Level 0 is the first partition, which contains the smallest communities,
    and the best is len(dendrogram) - 1.
    The higher the level is, the bigger are the communities

    Parameters
    ----------
    graph: networkx.Graph
        the networkx graph which will be decomposed
    part_init: dict, optionnal
        the algorithm will start using this partition of the nodes.
        It's a dictionary where keys are their nodes and values the communities

    Returns
    -------
    dendrogram: list of dictionaries
        a list of partitions, ie dictionnaries where keys of the i + 1 are
        the values of the i. and where keys of the first are the nodes of graph

    """
    if not isinstance(graph, nx.Graph):
        raise TypeError("Bad graph type, use only non directed graph")
     # special case, when there is no link
     # the best partition is everyone in its community
    if graph.number_of_edges() == 0:
        part = dict([])
        for node in graph.nodes():
            part[node] = node
        return part

    current_graph = graph.copy()
    status = Status()
    status.init(current_graph, part_init)
    new_mod, new_surpr = None, None
    mod = __modularity(status, method)
    new_mod = mod
    status_list = list()
    __one_level(current_graph, status, method)
    new_mod = __modularity(status, method)
    partition = __renumber(status.node2com)
    status_list.append(partition)
    mod = new_mod
    current_graph = induced_graph(partition, current_graph)
    status.init(current_graph)

    while True:
        __one_level(current_graph, status, method)
        new_mod = __modularity(status, method)
        if new_mod - mod < __MIN:
            break
        partition = __renumber(status.node2com)
        status_list.append(partition)
        mod = new_mod
        current_graph = induced_graph(partition, current_graph)
        status.init(current_graph)
    return status_list[:]


def induced_graph(partition, graph):
    """Produce the graph where nodes are the communities

    there is a link of weight w between communities if the sum of the weights
    of the links between their elements is w

    Parameters
    ----------
    partition: dict
       a dictionary where keys are graph nodes and  values the part the node belongs to
    graph: networkx.Graph
        the initial graph

    Returns
    -------
    g: networkx.Graph
       a networkx graph where nodes are the parts

    Examples
    --------
        n = 5
        g = nx.complete_graph(2*n)
        part = dict([])
        for node in g.nodes():
            part[node] = node % 2
        ind = induced_graph(part, g)
        goal = nx.Graph()
        goal.add_weighted_edges_from([(0,1,n*n),(0,0,n*(n-1)/2), (1, 1, n*(n-1)/2)])
        nx.is_isomorphic(int, goal)
    True
    """
    ret = nx.Graph()
    ret.add_nodes_from(partition.values())

    for node1, node2, datas in graph.edges_iter(data=True):
        weight = datas.get("weight", 1)
        com1 = partition[node1]
        com2 = partition[node2]
        w_prec = ret.get_edge_data(com1, com2, {"weight": 0}).get("weight", 1)
        ret.add_edge(com1, com2, weight=w_prec + weight)
    return ret


def surprise(graph, partition):
    from surprise import Surprise as Surprise
    S = Surprise(graph, partition, 
                 normalization_method='binary',
                 computation_method='fast')
    return S.surprise()


def best_surprise_partition(graph):
    """Compute the partition of the graph nodes which maximises
    the surprise calling an external combination of programs
    (it uses the SurpriseMe programs)
    This is the partition of highest surprise found.

    Parameters
    ----------
    graph: networkx.Graph
       the networkx graph which is decomposed

    Returns
    -------
    partition: dictionnary
       The partition, with communities numbered from 0 to number of communities

    Raises
    ------
    NetworkXError
       If the graph is not Eulerian.

    References
    ----------
    1. Rodrigo Aldecoa, SurpriseMe.

    """
    has_singleton_nodes = False
    singleton_nodes = []
    for g in nx.connected_component_subgraphs(graph):
        if g.number_of_nodes() == 1:
            singleton_nodes.append(g.nodes()[0])
            has_singleton_nodes = True

    if has_singleton_nodes:
        text_error = "Can't launch SurpriseMe on a graph with singleton nodes: "
        singleton_nodes_in_error = str(singleton_nodes)
        raise Exception(text_error + singleton_nodes_in_error)

    # Since the SurpriseMe program wants nodes as integers we first have to
    # convert the nodes dictionary to integer and then reverse the map)
    node2int, i = {}, 0
    for n in graph.nodes():
        node2int[n] = str(i)
        i += 1
    from copy import copy
    invnodemap = {nodeindex: node for node, nodeindex in node2int.items()}
    graph2 = nx.relabel_nodes(graph, mapping=node2int, copy=True)

    output_graph_file = '/tmp/graph'
    f = open(output_graph_file + '.txt', 'w')
    for l in graph2.edges():
        edgeFrom = copy(l[0])
        edgeTo = copy(l[1])
        if isinstance(edgeFrom, str):
            if edgeFrom.find(' ') != -1:
                edgeFrom = edgeFrom.replace(' ', '_')
            if isinstance(edgeTo, str):
                if edgeTo.find(' ') != -1:
                    edgeTo = edgeTo.replace(' ', '_')
            f.write(str(edgeFrom) + ' ' + str(edgeTo) + '\n')
    f.close()

    logging.info("SurpriseMe running on graph \"" +
                 output_graph_file + ".txt\"")
    commandstring = "cd ~/workspace/PHD/brainets/src/SurpriseMe/; perl SurpriseMe.pl " + \
        output_graph_file + ".txt"
    import subprocess
    subprocess.call(commandstring, shell=True)
    #os.popen(commandstring)  # Pick the partition that maximizes the surprise

    # Now read the graph.S file and select the algorithm with the highest
    # surprise.
    useApproximatedSurprise = True
    if useApproximatedSurprise:
        surprisePartitions = open(output_graph_file + ".S")
        algorithmSurpriseValues = []
        for row in surprisePartitions:
            r = row.rstrip().split()
            if len(r) == 2:
                algorithmSurpriseValues.append((r[0], r[1]))
            else:
                logging.warning("No available surprise for algorithm " + r[0])

        algorithmSurpriseValues = algorithmSurpriseValues[2:]

        logging.info("Selected " + algorithmSurpriseValues[0][0] +
                     " algorithm with surprise S=" + algorithmSurpriseValues[0][1])
        # Load the partition file
        partitionsFile = open(
            "/tmp/graph_" + algorithmSurpriseValues[0][0] + ".part")
        partitionsDictionary = {}
        for rowNumber, rowContent in enumerate(partitionsFile):
            if (rowNumber == 0):  # skip the header
                continue
            r = rowContent.rstrip().split()
            partitionsDictionary[r[0]] = int(r[1])
        # Remap the partitions dictionary using the original labels
        return __renumber({invnodemap[node]: community for node, community in partitionsDictionary.items()})
    else:
        # load all the partitions and compute the surprise for every partition
        process = os.popen('ls /tmp/*.part')
        partitionsFiles = process.read().split('\n')
        partitionsFiles.remove('')
        partitions = {}
        for partFile in partitionsFiles:
            f = open(partFile, 'r')
            partitionsDictionary = {}
            for rowNumber, rowContent in enumerate(f):
                if (rowNumber == 0):  # skip the header
                    continue
                r = rowContent.rstrip().split()
                partitionsDictionary[r[0]] = int(r[1])
            f.close()
            method = partFile.split('_')[1].split('.')[0]
            if method != 'SINGLES' and method != 'ONE':
                partition = __renumber(
                    {invnodemap[node]: community for node, community in partitionsDictionary.items()})
                s_base = surprise(graph, partition)
                s_weighted = surprise(graph, partition)
                partitions[method] = (s_base, s_weighted, partition)
        best_partition = None
        best_surprise_unweighted, best_surprise_weighted = 0.0, 0.0
        best_method = None
        for method, partition in partitions.iteritems():
            if partition[0] > best_surprise_unweighted:
                best_method = method
                best_surprise_unweighted = partition[0]
                best_surprise_weighted = partition[1]
                best_partition = partition[2]
        logging.info('Selected partition ' + best_method + ': (S_uw, S_w)=' +
                     str(best_surprise_unweighted) + ',' + str(best_surprise_weighted))

        return best_partition


def best_infomap_partition(graph, ntrials=10, seed='random'):
    # First save the graph to pajek format and then run the infomap program on
    # it
    G = graph.copy()

    def remove_spaces_from_names(node):
        if isinstance(node, str):
            node = node.replace(' ', '_')
        return node
    #G = nx.relabel_nodes(G, remove_spaces_from_names)

    os.popen("rm /tmp/graph.*")
    nx.write_pajek(G, "/tmp/graph.net")
    partition = {}
    if seed is 'random':
        seed = np.random.randint(10 ** 10)
    infomap_executable = "/home/carlo/workspace/PHD/brainets/src/infomap/Infomap"
    infomap_args = " -s " + \
        str(seed) + " -N " + str(ntrials) + " --clu /tmp/graph.net /tmp"
    commandstring = infomap_executable + infomap_args
    # logging.info(commandstring)
    process = os.popen(commandstring)
    process.read()
    partition_filename = "/tmp/graph.clu"
    partition_file = open(partition_filename, 'r')
    for n, row in enumerate(partition_file):
        row = row.split()
        if row[0] == "*Vertices":
            continue
        partition[n - 1] = int(row[0]) - 1
    """
    for row in partition_file:
        if row[0] == "#":
            continue
        if row.split()[0] == "*Nodes":
            validInput = True
            continue
        if row.split()[0] == "*Links":
            validInput = False
        if validInput:
            r = row.split()
            r[1] += " " + r[2]
            r.remove(r[2])
            partition[r[1].replace('\"', '')] = int(r[0].split(':')[0])
    partition_file.close()
    """
    return partition


def best_cpm_partition(graph, res, ntrials=10):
    import igraph as ig
    import pylouvain as louvain
    iG = utils.to_igraph(graph)
    opt = louvain.Optimiser()
    # Find communities using CPM
    best_part, best_cpm = None, -np.infty
    for n in range(ntrials):
        part = opt.find_partition(
            iG, partition_class=louvain.CPMVertexPartition, resolution=res)
        if part.CPM() > best_cpm:
            best_part = part

    membership = dict(zip(graph.nodes(),best_part.membership))
    #logging.info("CPM value = " + str(part.CPM()) + " Surp=" + str(surprise(graph,membership)))
    return membership


def best_significance_partition(graph, ntrials=10):
    import igraph as ig
    import pylouvain as louvain
    iG = utils.to_igraph(graph)
    opt = louvain.Optimiser()
    # Find communities using significance
    best_part, best_significance = None, 0
    for n in range(ntrials):
        part = opt.find_partition(
            iG, partition_class=louvain.SignificanceVertexPartition)
        if part.significance() > best_significance:
            best_part = part

    logging.info("Significance value = " + str(part.significance()))
    logging.info("Surprise value = " + str(part.surprise()))
    partitions = {}
    for u, v in enumerate(part.membership):
        partitions[u] = v
    return partitions


def best_surprise_partition_louvain(graph, ntrials=1):
    import igraph as ig
    import pylouvain as louvain
    iG = utils.to_igraph(graph)
    opt = louvain.Optimiser()
    # Find communities using surprise
    best_part, best_surprise = None, 0
    for n in range(ntrials):
        part = opt.find_partition(
            iG, partition_class=louvain.SurpriseVertexPartition)
        if part.surprise() > best_surprise:
            best_part = part

    #logging.info("Surprise value = " + str(part.surprise()))
    return dict(zip(graph.nodes(),part.membership))


def best_surprise_asymptotics_partition(graph, ntrials=1):
    import igraph as ig
    import louvain #latest louvain
    iG = utils.to_igraph_fast(graph)
    best_part, best_surprise = None, 0
    for t in range(0, ntrials):
        part = louvain.find_partition(iG, method='Surprise')
        membership = dict(zip(graph.nodes(),part.membership))
        Scorr = surprise(graph, membership)
        Sest = louvain.quality(iG, part, method='Surprise')
        if Scorr > best_surprise:
            best_part = copy.copy(part)
            best_surprise = Scorr

    #logging.info("Surprise value = %g" % (louvain.quality(iG, part, method='Surprise')))
    #logging.info("Surprise value = %g" % surprise(graph, membership))
    return membership


def __renumber(dictionary):
    """
    Renumber the values of the dictionary from 0 to n
    """
    count = 0
    ret = dictionary.copy()
    new_values = dict([])

    for key in dictionary.keys():
        value = dictionary[key]
        new_value = new_values.get(value, -1)
        if new_value == -1:
            new_values[value] = count
            new_value = count
            count += 1
        ret[key] = new_value

    return ret


def reindex_membership(membership):
    """
    This function has the membership as input and output the membership
    where the communities number are ordered by the number of nodes in that community
    """
    ds = {}
    for u, v in membership.iteritems():
        if v not in ds.keys():
            ds[v] = []
        ds[v].append(u)

    S = dict(
        zip(range(0, len(ds)), sorted(ds.values(), key=len, reverse=True)))

    M = {}

    for u, vl in S.iteritems():
        for v in vl:
            M[v] = u
    return M


def reindex_clustering(clustering):
    """
    This function takes a dictionary of community where the key is the community index
    and the values are the nodes in that community and sort output indices from 0 to |C|-1
    with the nodes in it. It's different from __renumber_sort because
    IT DOES NOT TAKE THE NODES MEMBERSHIPS
    """
    return dict(zip(range(0, len(clustering)), sorted(clustering.values(), key=len, reverse=True)))


def __load_binary(data):
    """Load binary graph as used by the cpp implementation of this algorithm"""
    if isinstance(data, types.StringType):
        data = open(data, "rb")

    reader = array.array("I")
    reader.fromfile(data, 1)
    num_nodes = reader.pop()
    reader = array.array("I")
    reader.fromfile(data, num_nodes)
    cum_deg = reader.tolist()
    num_links = reader.pop()
    reader = array.array("I")
    reader.fromfile(data, num_links)
    links = reader.tolist()
    graph = nx.Graph()
    graph.add_nodes_from(range(num_nodes))
    prec_deg = 0

    for index in range(num_nodes):
        last_deg = cum_deg[index]
        neighbors = links[prec_deg:last_deg]
        graph.add_edges_from([(index, int(neigh)) for neigh in neighbors])
        prec_deg = last_deg

    return graph


def __one_level(graph, status, method):
    """Compute one level of communities
    """
    modif = True
    nb_pass_done = 0
    cur_mod = __modularity(status, method)
    new_mod = cur_mod

    while modif and nb_pass_done != __PASS_MAX:
        cur_mod = new_mod
        modif = False
        nb_pass_done += 1

        # xxx random shuffle nodes
        for node in np.random.permutation(graph.nodes()):
            # for node in graph.nodes():
            com_node = status.node2com[node]
            degc_totw = status.gdegrees.get(
                node, 0.) / (status.total_weight * 2.)
            neigh_communities = __neighcom(node, graph, status)
            __remove(node, com_node,
                     neigh_communities.get(com_node, 0.), status)
            best_com = com_node
            best_increase = 0
            for com, dnc in neigh_communities.iteritems():
                incr = dnc - status.degrees.get(com, 0.) * degc_totw
                if incr > best_increase:
                    best_increase = incr
                    best_com = com
            __insert(node, best_com,
                     neigh_communities.get(best_com, 0.), status)
            if best_com != com_node:
                modif = True
        new_mod = __modularity(status, method)
        if new_mod - cur_mod < __MIN:
            break


class Status:

    """
    To handle several data in one struct.
    Could be replaced by named tuple, but don't want to depend on python 2.6
    """
    node2com = {}
    total_weight = 0
    internals = {}
    degrees = {}
    gdegrees = {}
    _graph = nx.Graph()

    def __init__(self):
        self.node2com = dict([])
        self.total_weight = 0
        self.degrees = dict([])
        self.gdegrees = dict([])
        self.internals = dict([])
        self.loops = dict([])
        self._graph = nx.Graph()

    def __str__(self):
        return ("node2com: " + str(self.node2com) + " degrees: "
                + str(self.degrees) + " internals: " + str(self.internals)
                + " total_weight: " + str(self.total_weight))

    def copy(self):
        """Perform a deep copy of status"""
        new_status = Status()
        new_status.node2com = self.node2com.copy()
        new_status.internals = self.internals.copy()
        new_status.degrees = self.degrees.copy()
        new_status.gdegrees = self.gdegrees.copy()
        new_status.total_weight = self.total_weight
        new_status._graph = self._graph.copy()

    def init(self, graph, part=None):
        """Initialize the status of a graph with every node in one community"""
        count = 0
        self.node2com = dict([])
        self.total_weight = 0
        self.degrees = dict([])
        self.gdegrees = dict([])
        self.internals = dict([])
        self._graph = graph.copy()
        self.total_weight = graph.size(weight='weight')
        if part is None:
            for node in graph.nodes():
                self.node2com[node] = count
                deg = float(graph.degree(node, weight='weight'))
                if deg < 0:
                    raise ValueError("Bad graph type, use positive weights")
                self.degrees[count] = deg
                self.gdegrees[node] = deg
                self.loops[node] = float(graph.get_edge_data(node, node,
                                                             {"weight": 0}).get("weight", 1))
                self.internals[count] = self.loops[node]
                count = count + 1
        else:
            for node in graph.nodes():
                com = part[node]
                self.node2com[node] = com
                deg = float(graph.degree(node, weight='weight'))
                self.degrees[com] = self.degrees.get(com, 0) + deg
                self.gdegrees[node] = deg
                inc = 0.
                for neighbor, datas in graph[node].iteritems():
                    weight = datas.get("weight", 1)
                    if weight <= 0:
                        raise ValueError("Bad graph type,use positive weights")
                    if part[neighbor] == com:
                        if neighbor == node:
                            inc += float(weight)
                        else:
                            inc += float(weight) / 2.
                self.internals[com] = self.internals.get(com, 0) + inc


def __neighcom(node, graph, status):
    """Compute the communities in the neighborood of node in the graph given
    with the decomposition node2com
    """
    weights = {}
    for neighbor, datas in graph[node].iteritems():
        if neighbor != node:
            weight = datas.get("weight", 1)
            neighborcom = status.node2com[neighbor]
            weights[neighborcom] = weights.get(neighborcom, 0) + weight

    return weights


def __remove(node, com, weight, status):
    """ Remove node from community com and modify status"""
    status.degrees[com] = (status.degrees.get(com, 0.)
                           - status.gdegrees.get(node, 0.))
    status.internals[com] = float(status.internals.get(com, 0.) -
                                  weight - status.loops.get(node, 0.))
    status.node2com[node] = -1


def __insert(node, com, weight, status):
    """ Insert node into community and modify status"""
    status.node2com[node] = com
    status.degrees[com] = (status.degrees.get(com, 0.) +
                           status.gdegrees.get(node, 0.))
    status.internals[com] = float(status.internals.get(com, 0.) +
                                  weight + status.loops.get(node, 0.))


def __modularity(status, method='modularity'):
    """Compute the modularity of the graph partition using precomputed status
    """
    if method == 'modularity':
        links = float(status.total_weight)
        result = 0.
        for comm in set(status.node2com.values()):
            in_degree = status.internals.get(comm, 0.)
            degree = status.degrees.get(comm, 0.)
            if links > 0:
                result = result + in_degree / links - \
                    ((degree / (2. * links)) ** 2)
        return result
    elif method == 'surprise':
        return surprise(status._graph, status.node2com)


def linear_remap_list(partition):
    """Remap a partition containing communities in non linear-indices list i.e.
        linear_remap_list ( [10,10,10,6,6,3,1,1] )
        [0,0,0,1,1,2,3,3]
    """
    return list(map(lambda i, c=defaultdict(lambda c=count(): next(c)): c[i], partition))


def induced_module(graph, partition):
    graphs = {}  # contains the subgraphs induced by the partition
    comms = {}

    for n, c in partition.iteritems():
        comms[c] = []

    for n, c in partition.iteritems():
        comms[c].append(n)
    # print comms

    for community, nodes in comms.iteritems():
        graphs[community] = nx.subgraph(graph, nodes)

    return graphs


def analyze_modularity_partitions(graph, level):
    c = best_modularity_partition(graph)
    graphs = induced_module(graph, c)

    l = 0
    modularities = [[]]
    while l < level:
        for g in graphs.values():
            cl = best_modularity_partition(g)
            modularities[l].append(modularity(g, cl))
        l += 1
        modularities.append([])

    return modularities


#
# DEPRECATED ##################
#
def refine_best_partition_genetic_optimization(
    graph, method, partition, n_generations=100, multiprocessor=False,
    options={'mutation_prob': 0.25,
             'mating_prob': 0.5}):
    """
    Refine the best surprise partition using it as initial guess for an evolutive algorithm
    This function needs to have DEAP installed
    sudo pip install deap

    Parameters
    ----------
    graph: networkx.Graph
       the networkx graph which is decomposed

    method: can be 'surprise' or 'modularity' it depends on the cost function
            to maximize

    partition: a dictionary { node:integer } representing the initial guess for
    the evolutionary algorithm

    n_generations: number of generations to let the algorithm evolve

    multiprocessor: True or False if has to use multiple threads

    Returns
    -------
    partition: dictionnary
       The partition, with communities numbered from 0 to number of communities

    Raises
    ------
    NetworkXError
       If the graph is not Eulerian.
    """

    def eval_surprise(individual):
        d = {}
        for i, n in enumerate(graph.nodes()):
            d[n] = individual[i]
        return surprise(graph, d),

    def eval_modularity(individual):
        d = {}
        for i, n in enumerate(graph.nodes()):
            d[n] = individual[i]
        return modularity(graph, d),

    if multiprocessor:
        import multiprocessing  # for multicore spawn

    from deap import base
    from deap import creator
    from deap import tools

    from deap import algorithms

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    # weights is the list of weights of our multiobjective optimization function
    # that is just the surprise, so for our case we set the weight to 1
    creator.create("Individual", list, typecode='i',
                   fitness=creator.FitnessMax)
    toolbox = base.Toolbox()

    # Parallel using multiprocessing
    if multiprocessor:
        pool = multiprocessing.Pool()
        toolbox.register("map", pool.map)
        #from scoop import futures
        #toolbox.register("map", futures.map)

    # Attribute generator is a random integer in [0,|V|] for the graph
    toolbox.register("node_community", random.randint,
                     0, graph.number_of_nodes())

    # Structure initializers
    # The function assigned to this alias is tools.initRepeat which takes three arguments: the first one is a data structure
    # constructor, in this case it is our individual constructor creator.Individual, the second one is the function used
    # to generate the content filling for that data structure, and the last one is the number of elements to generate
    # registered individual function is then able to generate individuals composed of 100 random bits.
    # In a similar fashion, we register an operator named population capable
    # of generating a list of 300 individuals
    toolbox.register("individual", tools.initRepeat, creator.Individual,
                     toolbox.node_community, n=graph.number_of_nodes())
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Implement the callbacks
    # Evaluation callback is the evalSurprise function I've just defined
    if method is 'surprise':
        toolbox.register("evaluate", eval_surprise)
    elif method is 'modularity':
        toolbox.register("evaluate", eval_modularity)
    else:
        raise Exception("method can be \'modularity\' or \'surprise\' only")
    # The mating mechanism is in this case the tool.cxTwoPoint
    toolbox.register("mate", tools.cxOnePoint)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutUniformInt, low=0,
                     up=graph.number_of_nodes(), indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=4)

    # random.seed(64)
    pop = toolbox.population(n=16)
    # Insert the initial guess as the best individual
    guess_ind = creator.Individual(partition.values())
    pop.pop()
    pop.insert(0, guess_ind)

    hof = tools.HallOfFame(10)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.05, mutpb=0.05,
                                   ngen=n_generations, stats=stats,
                                   halloffame=hof, verbose=True)

    return pop, log, hof


def modularity_finetune_und_sign(graph, partition=None, qtype='sta'):
    """
    Optimal community structure and modularity
    Ci, Q = modularity_finetune_und_sign(graph)
    """
    if (isinstance(graph, nx.Graph)):
        W = nx.to_numpy_matrix(graph)
    if (isinstance(graph, np.ndarray)):
        W = graph
    n = np.shape(W)[0]
    if partition is not None and isinstance(partition, dict):
        M = partition.values()
    else:
        M = np.asarray(range(0, n))

    W0 = np.multiply(W, (W > 0))
    W1 = -np.multiply(W, (W < 0))
    s0 = np.sum(W0[:])
    s1 = np.sum(W1[:])
    Knm0 = np.asmatrix(np.zeros([n, n]))
    Knm1 = np.asmatrix(np.zeros([n, n]))

    for m in range(0, np.max(M) + 1):
        Knm0[:, m] = np.sum(W0[:, M == m], 1)
        Knm1[:, m] = np.sum(W1[:, M == m], 1)

    Kn0 = np.asarray(Knm0.sum(1))
    Kn1 = np.asarray(Knm1.sum(1))
    # per qualche oscuro motivo di indici di np...
    Km0 = np.asarray(Knm0.sum(0))[0]
    # per qualche oscuro motivo di indici di np...
    Km1 = np.asarray(Knm1.sum(0))[0]

    d0, d1 = None, None
    if qtype is 'smp':
        d0, d1 = 1 / s0, 1 / s1
    elif qtype is 'gja':
        d0, d1 = 1 / (s0 + s1), 1 / (s0 + s1)
    elif qtype is 'sta':
        d0, d1 = 1 / s0, 1 / (s0 + s1)
    elif qtype is 'pos':
        d0, d1 = 1 / s0, 0
    elif qtype is 'neg':
        d0, d1 = 0, 1 / s1
    else:
        raise Exception('qtype unknown')

    if not s0:  # adjust for absent positive weights
        s0, d0 = 1, 0
    if not s1:  # adjust for absent negative weights
        s1, d1 = 1, 0

    f = 1
    while f:
        f = 0
        for u in np.random.permutation(n):
            ma = M[u]
            dQ0 = (Knm0[u, :] + W0[u, u] - Knm0[u, ma]) - \
                np.multiply(Kn0[u], (Km0 + Kn0[u] - Km0[ma])) / s0
            dQ1 = (Knm1[u, :] + W1[u, u] - Knm1[u, ma]) - \
                np.multiply(Kn1[u], (Km1 + Kn1[u] - Km1[ma])) / s1
            dQ = (d0 * dQ0 - d1 * dQ1).flat
            dQ[ma] = 0
            max_dQ, mb = np.max(dQ), np.argmax(dQ)
            if max_dQ > 1E-10:
                f = 1
                M[u] = mb
                Knm0[:, mb] += W0[:, u]
                Knm1[:, mb] += W1[:, u]
                Knm0[:, ma] -= W0[:, u]
                Knm1[:, ma] -= W1[:, u]
                Km0[mb] += Kn0[u]
                Km1[mb] += Kn1[u]
                Km0[ma] -= Kn0[u]
                Km1[ma] -= Kn1[u]

    communities = np.asarray(linear_remap_list(M))
    communities_dict = {}

    for i, n in enumerate(graph.nodes()):
        communities_dict[n] = communities[i]

    return communities_dict


def modularity_louvain_und_sign(graph, qtype='sta'):
    """
    Optimal community structure and modularity

    Ci, Q = modularity_finetune_und_sign(graph)
    """
    if (isinstance(graph, nx.Graph)):
        W = nx.to_numpy_matrix(graph)
    if (isinstance(graph, np.ndarray)):
        W = graph
    N = np.shape(W)[0]

    W0 = np.multiply(W, (W > 0))
    W1 = -np.multiply(W, (W < 0))

    s0 = np.sum(W0[:])
    s1 = np.sum(W1[:])

    d0, d1 = None, None
    if qtype is 'smp':
        d0, d1 = 1 / s0, 1 / s1
    elif qtype is 'gja':
        d0, d1 = 1 / (s0 + s1), 1 / (s0 + s1)
    elif qtype is 'sta':
        d0, d1 = 1 / s0, 1 / (s0 + s1)
    elif qtype is 'pos':
        d0, d1 = 1 / s0, 0
    elif qtype is 'neg':
        d0, d1 = 0, 1 / s1
    else:
        raise Exception('qtype unknown')

    if not s0:  # adjust for absent positive weights
        s0, d0 = 1, 0
    if not s1:  # adjust for absent negative weights
        s1, d1 = 1, 0

    h = 1
    n = N  # number of nodes in hierarchy
    # hierarchical module assignments
    Ci = [np.asarray([]), np.asarray(range(0, n))]
    Q = [-1, 0]  # hierarchical modularity values

    while (Q[h] - Q[h - 1]) > (1E-10):
        Kn0 = np.sum(W0, 1)  # positive node degree
        Kn1 = np.sum(W1, 1)  # negative node degree
        Km0 = Kn0.copy()  # positive module degree !!! IMPORTANT TO DEEP COPY
        Km1 = Kn1.copy()  # negative module degree !!! IMPORTANT TO DEEP COPY
        # positive node-to-module degree !!! IMPORTANT TO DEEP COPY
        Knm0 = W0.copy()
        # negative node-to-module degree !!! IMPORTANT TO DEEP COPY
        Knm1 = W1.copy()
        M = range(0, n)  # initial module assignments
        f = 1  # flag for within-hierarchy search
        while f:
            f = 0
            # loop over all nodes in random order
            for u in np.random.permutation(n):
                ma = M[u]  # current module of u
                dQ0 = Knm0[u, :] + W0[u, u] - Knm0[u, ma] - \
                    (np.multiply(Kn0[u], Km0 + Kn0[u] - Km0[ma])
                     / s0).T  # positive dQ
                dQ1 = Knm1[u, :] + W1[u, u] - Knm1[u, ma] - \
                    (np.multiply(Kn1[u], Km1 + Kn1[u] - Km1[ma])
                     / s1).T  # negative dQ
                dQ = d0 * dQ0 - d1 * dQ1  # rescaled changes in modularity
                dQ.flat[ma] = 0  # no changes for same module
                # maximal increase in modularity and corresponding module
                max_dQ, mb = np.max(dQ.flat), np.argmax(dQ.flat)
                if max_dQ > (1E-10):
                    # if maximal increase is positive (equiv. dQ(mb)>dQ(ma))
                    f = 1
                    M[u] = mb  # reassign module
                    # change positive node-to-module degrees
                    Knm0[:, mb] += W0[:, u]
                    Knm0[:, ma] -= W0[:, u]
                    # change negative node-to-module degrees
                    Knm1[:, mb] += W1[:, u]
                    Knm1[:, ma] -= W1[:, u]
                    Km0[mb] += Kn0[u]  # change positive module degrees
                    Km0[ma] -= Kn0[u]
                    Km1[mb] += Kn1[u]  # change negative module degrees
                    Km1[ma] -= Kn1[u]

        h += 1
        Ci.append(np.zeros(N, dtype='int'))
        M = linear_remap_list(M)
        # loop through initial module assignment
        for u in range(0, n):
            Ci[h][Ci[h - 1] == u] = M[u]

        n = len(np.unique(M))  # number of new nodes
        w0 = np.zeros([n, n])
        w1 = np.zeros([n, n])

        for u in range(0, n):
            for v in range(u, n):
                MS = np.asarray(M)
                # pool positive weights of nodes in same module
                w0[u, v] = np.sum(np.sum((W0[MS == u, :])[:, MS == v]))
                # pool negative weights of nodes in same module
                w1[u, v] = np.sum(np.sum((W1[MS == u, :])[:, MS == v]))
                w0[v, u] = w0[u, v]
                w1[v, u] = w1[u, v]

        W0 = w0.copy()
        W1 = w1.copy()

        # compute modularity
        W02 = np.dot(W0, W0)
        W12 = np.dot(W1, W1)
        # contribution of positive weights
        Q0 = np.sum(np.diag(W0)) - np.sum(np.sum(W02)) / s0
        # contribution of negative weights
        Q1 = np.sum(np.diag(W1)) - np.sum(np.sum(W12)) / s1
        Q.append(d0 * Q0 - d1 * Q1)

    Ci = Ci[-1]
    Q = Q[-1]

    communities = np.asarray(linear_remap_list(Ci))
    communities_dict = {}
    for i, n in enumerate(graph.nodes()):
        communities_dict[n] = communities[i]

    return communities_dict, Q


def pairwise_community_to_community_dict(pairwise_comm):
    np.fill_diagonal(pairwise_comm, 0)
    comms = {}
    communities = 0
    for g in nx.connected_components(nx.from_numpy_matrix(pairwise_comm)):
        for n in g:
            comms[n] = communities
        communities += 1
    return comms


def modularity_exact_maximization_din_thai(graph):
    # Refers to the problem
    # Towards Optimal Community Detection : From Trees to General Weighted Networks
    # T.Dinh, M.Thai

    import glpk
    import cvxopt
    from cvxopt import glpk

    Aij = nx.to_numpy_matrix(graph)
    # n is the number of nodes in the graph
    n = graph.number_of_nodes()
    # m the number of edges if the graph is unweighted
    m = graph.size(weight='weight')
    np.fill_diagonal(Aij, 0)
    k = Aij.sum(axis=0).flatten()
    dij = np.outer(k, k) / (2.0 * m)

    np.fill_diagonal(dij, 0)
    # Bij is the modularity matrix, page 4 of T.Dinh, M.Thai
    Bij = -(Aij - dij) / m

    indices = []
    for i in range(0, n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                indices.append([i, j, k])

    nconstraints = 3 * n * (n - 1) * (n - 2) / 6
    nvariables = n * n

    # Create the constraints matrix.
    A = np.zeros([nconstraints, nvariables])
    for count, ind in enumerate(indices):
        i, j, k = ind[0], ind[1], ind[2]
        ij = i * n + j
        ik = i * n + k
        jk = j * n + k

        A[count * 3, ij] = 1
        A[count * 3, jk] = 1
        A[count * 3, ik] = -1

        A[count * 3 + 1, ij] = 1
        A[count * 3 + 1, jk] = -1
        A[count * 3 + 1, ik] = 1

        A[count * 3 + 2, ij] = -1
        A[count * 3 + 2, jk] = 1
        A[count * 3 + 2, ik] = 1

    Bij = Bij.reshape(nvariables, 1)
    b = np.ones([nconstraints, 1])

    (status, sol) = cvxopt.glpk.ilp(c=cvxopt.matrix(Bij),   # c parameter
                                    G=cvxopt.matrix(A),     # G parameter
                                    h=cvxopt.matrix(b),     # h parameter
                                    I=set(range(0, nvariables)),
                                    B=set(range(0, nvariables))
                                    )
    # print np.dot(Bij.T,sol)
    # np.savetxt('c.txt',Bij,fmt='%1.2f')
    # np.savetxt('G.txt',A,fmt='%d')
    # np.savetxt('h.txt',b,fmt='%d')

    sol = np.array(sol)
    # IT'S OF FUNDAMENTAL IMPORTANCE JUST TO TAKE THE
    # UPPER TRIANGULAR PART OF THE SOLUTION BECAUSE THE
    # CONSTRAINTS ARE ENFORCED JUST FOR i<j<k
    sol = np.triu(sol.reshape(n, n))
    return pairwise_community_to_community_dict(sol)

def get_intraedges_intranodes(graph, membership):
    # returns the constant potts model hamiltonian with given gamma constant
    groups = membership_to_groups(graph,membership)
    intraedges = 0
    intranodes = 0
    h_cpm = 0
    comm_intra = []
    for comm, nbunch in groups.iteritems():
        g = nx.subgraph(graph, nbunch)
        nc = g.number_of_nodes()
        ec = g.number_of_edges()
        h_cpm += ec - nc*nc*gamma
        comm_intra.append([ec,nc*nc])
    return comm_intra


def surprise_gurobi(graph, sparse_problem=False, use_progressbar=False):
    from gurobipy import LinExpr, Model, GRB, quicksum
    from surprise import Surprise
    n = graph.number_of_nodes()
    m = graph.number_of_edges()

    def ind2lin(i, j, n):
        return (n * (n - 1) / 2) - (n - i) * ((n - i) - 1) / 2 + j - i - 1
    model = Model()
    model.setAttr(GRB.Attr.ModelName, "Surprise Optimization")
    model.setAttr(GRB.Attr.ModelSense, GRB.MINIMIZE)
    model.params.OutputFlag = False
    nvar = (n * (n - 1) / 2)
    # Per convertire indici:
    # http://stackoverflow.com/questions/27086195/linear-index-upper-triangular-matrix
    for i in range(0, nvar):
        model.addVar(0, 1, 1, GRB.BINARY)
    model.update()

    sumedges = quicksum([LinExpr(model.getVars()[ind2lin(e[0], e[1], n)])
                         for e in graph.edges()])
    obj = {}
    obj['exact'] = quicksum(model.getVars())
    obj['gap'] = obj['exact'] - sumedges
    model.setObjective(obj['gap'])

    if sparse_problem:
        # Create the auxiliary data structures to hold minimum-st-node-cuts
        from networkx.algorithms.connectivity import build_auxiliary_node_connectivity
        from networkx.algorithms.connectivity import minimum_st_node_cut
        auxgraph = build_auxiliary_node_connectivity(graph)
        from networkx.algorithms.flow import build_residual_network
        resgraph = build_residual_network(auxgraph, 'capacity')

    def separator(graph, auxgraph, resgraph, u, v):
        from networkx.algorithms.connectivity import minimum_node_cut
        return minimum_node_cut(graph, u, v)

    # Create transitivity constraints
    modelvars = model.getVars()
    for i in range(0, n):
        for j in range(i + 1, n):
            gg = graph.copy()
            if gg.has_edge(i, j):
                gg.remove_edge(i, j)
            if sparse_problem:
                krange = separator(graph, auxgraph, resgraph, i, j)
            else:
                krange = range(j + 1, n)
            for k in krange:
                ij = ind2lin(i, j, n)
                ik = ind2lin(i, k, n)
                jk = ind2lin(j, k, n)
                xij = modelvars[ij]
                xik = modelvars[ik]
                xjk = modelvars[jk]
                model.addConstr(LinExpr(xij + xjk - xik), GRB.LESS_EQUAL, 1)
                model.addConstr(LinExpr(xij - xjk + xik), GRB.LESS_EQUAL, 1)
                model.addConstr(LinExpr(-xij + xjk + xik), GRB.LESS_EQUAL, 1)
    model.update()
    model.tune()
    # Fleck noted GAP variant with no relax (i.e. sum edges == k) is fastest
    # but in our case we want to solve the exact problem

    parts = {}
    # Add a constraint that can be changed after
    constr = model.addConstr(sumedges, GRB.GREATER_EQUAL, 1)
    model.update()
    if use_progressbar:
        from progressbar import ProgressBar
        pbar = ProgressBar(maxval=m + 1)
        pbar.start()
    for K in range(0, m + 1):
        # Constraint on k
        # Tutto questo giro serve per poter variare ultimo constraint su
        # intracluster edges
        model.remove(constr)
        constr = model.addConstr(sumedges, GRB.GREATER_EQUAL, K)
        model.update()
        model.optimize()
        if model.status is GRB.OPTIMAL:
            membership = model2membership(model, graph)
            surprise = Surprise(graph, membership).surprise()
            print surprise, get_intraedges_intranodes(graph, membership)
            parts[K] = membership
        if use_progressbar:
            pbar.update(K)
    if use_progressbar:
        pbar.finish()

    return parts


def model2membership(model, graph):
    n = graph.number_of_nodes()
    # Collect the solution
    sol = np.zeros([n, n])

    def lin2ind(k, n):
        i = n - 2 - int(np.sqrt(-8 * k + 4 * n * (n - 1) - 7) / 2.0 - 0.5)
        j = k + i + 1 - n * (n - 1) / 2 + (n - i) * ((n - i) - 1) / 2
        return i, j

    for k, v in enumerate(model.getVars()):
        i, j = lin2ind(k, n)
        sol[i][j] = v.x
        sol[j][i] = v.x

    membership = pairwise_community_to_community_dict(sol)
    return membership


def CPM_hamiltonian(graph, membership, gamma):
    # returns the constant potts model hamiltonian with given gamma constant
    groups = membership_to_groups(graph,membership)
    intraedges = 0
    intranodes = 0
    h_cpm = 0
    for comm, nbunch in groups.iteritems():
        g = nx.subgraph(graph, nbunch)
        intraedges += g.number_of_edges()
        intranodes += g.number_of_nodes()
        h_cpm += intraedges - intranodes*intranodes*gamma
    return h_cpm


def modularity_gurobi(graph, sparse_problem=False, use_progressbar=False):
    from gurobipy import LinExpr, Model, GRB, quicksum
    n = graph.number_of_nodes()
    m = graph.number_of_edges()

    def ind2lin(i, j, n):
        return (n * (n - 1) / 2) - (n - i) * ((n - i) - 1) / 2 + j - i - 1

    def lin2ind(k, n):
        i = n - 2 - int(np.sqrt(-8 * k + 4 * n * (n - 1) - 7) / 2.0 - 0.5)
        j = k + i + 1 - n * (n - 1) / 2 + (n - i) * ((n - i) - 1) / 2
        return i, j

    model = Model()
    model.setAttr(GRB.Attr.ModelName, "Modularity Optimization")
    model.setAttr(GRB.Attr.ModelSense, GRB.MAXIMIZE)
    model.params.OutputFlag = False
    nvar = n * (n - 1) / 2
    # Per convertire indici:
    # http://stackoverflow.com/questions/27086195/linear-index-upper-triangular-matrix
    for i in range(0, nvar):
        model.addVar(0, 1, 1, GRB.BINARY)
    model.update()

    # Create the auxiliary data structures to hold minimum-st-node-cuts
    """
    from networkx.algorithms.connectivity import build_auxiliary_node_connectivity
    auxgraph = build_auxiliary_node_connectivity(graph)
    from networkx.algorithms.flow import build_residual_network
    resgraph = build_residual_network(auxgraph, 'capacity')
    """
    def separator(graph, u, v):
        from networkx.algorithms.connectivity import minimum_node_cut
        return minimum_node_cut(graph, u, v)

    # Create the modularity matrix B
    A = nx.to_numpy_matrix(graph)
    np.fill_diagonal(A, 0)
    k = A.sum(axis=0).flatten()
    D = np.outer(k, k)
    np.fill_diagonal(D, 0)
    B = A - (D / (2.0 * m))

    # ILP complete
    obj_func = LinExpr()
    for xijIndex, xij in enumerate(model.getVars()):
        i, j = lin2ind(xijIndex, n)
        obj_func += LinExpr(B[i, j] * xij)
    obj_func = 1.0 / (2.0 * m) * obj_func

    # Set the objective function of modularity as in Dinh-Thai
    model.setObjective(obj_func)
    model.update()
    # Create the internal edges constraint
    sumedges = quicksum([LinExpr(model.getVars()[ind2lin(e[0], e[1], n)]) for e in graph.edges()])
    # Create transitivity constraints
    # print "Adding constraints"
    modelvars = model.getVars()
    from copy import deepcopy
    # MUST OBTAIN 1441 for karate club, 5743 for Dolphins

    for i in range(0, n):
        for j in range(i + 1, n):
            gg = graph.copy()
            if gg.has_edge(i, j):
                gg.remove_edge(i, j)
            if sparse_problem:
                krange = separator(graph, i, j)
            else:
                krange = range(j + 1, n)
            for k in krange:
                ij = ind2lin(i, j, n)
                ik = ind2lin(i, k, n)
                jk = ind2lin(j, k, n)
                xij = modelvars[ij]
                xik = modelvars[ik]
                xjk = modelvars[jk]
                model.addConstr(LinExpr(xij + xjk - xik), GRB.LESS_EQUAL, 1)
                model.addConstr(LinExpr(xij - xjk + xik), GRB.LESS_EQUAL, 1)
                model.addConstr(LinExpr(-xij + xjk + xik), GRB.LESS_EQUAL, 1)
    
    """
    from itertools import combinations
    for u, v in combinations(range(n), r=2):
        xuv = modelvars[ind2lin(u, v, n)]
        gg = deepcopy(graph)
        if gg.has_edge(u, v):
            gg.remove_edge(u, v)
        for w in separator(gg, u, v):
            xuw = modelvars[ind2lin(u, w, n)]
            xwv = modelvars[ind2lin(w, v, n)]
            model.addConstr(LinExpr(xuw + xwv - xuv), GRB.LESS_EQUAL, 1)
    model.update()
    """
    #print len(model.getConstrs()), " constraints added"

    model.tune()
    model.optimize()
    if model.status is GRB.OPTIMAL:
        membership = model2membership(model, graph)
    #print modularity(graph, membership), len(np.unique(membership.values()))
    return membership
    """
    parts = {}
    # Add a constraint that can be changed after
    constr = model.addConstr(sumedges, GRB.GREATER_EQUAL, 1)
    model.update()
    if use_progressbar:
        from progressbar import ProgressBar
        pbar = ProgressBar(maxval=m + 1)
        pbar.start()
    for K in range(0, m + 1):
        # Constraint on k
        # Tutto questo giro serve per poter variare ultimo constraint su
        # intracluster edges
        model.remove(constr)
        constr = model.addConstr(sumedges, GRB.GREATER_EQUAL, K)
        model.update()
        model.tune()
        model.optimize()
        if model.status is GRB.OPTIMAL:
            membership = model2membership(model, graph)
            parts[K] = membership
        if use_progressbar:
            pbar.update(K)
    if use_progressbar:
        pbar.finish()
    return parts
    """


def edge_local_reweighting(graph):
    W = nx.to_numpy_matrix(graph)
    n = W.shape[0]
    Ls = graph.number_of_edges()
    Ws = graph.size(weight='weight')
    Wbar = Ls * W / Ws

    delta = (W != 0).astype(int)
    sumW = np.sum(W, axis=0).reshape(n, 1)
    C = np.zeros(W.shape)

    for i in range(0, n):
        for j in range(0, n):
            dd = np.multiply(delta[i, :], delta[j, :])
            C[i, j] += 2.0 * (W[i, j] + np.multiply(dd,
                                                    W[i, :] + W[j, :]).sum() * 0.5) / (sumW[i] + sumW[j])

    Wtilde = np.multiply(Wbar, C)

    sumWtilde = np.sum(Wtilde, axis=0).reshape(n, 1)
    G = Wtilde.copy()

    for i in range(0, n):
        G[i, :] /= sumWtilde[i]
    del C, sumW, Wtilde, Ws, Wbar

    return nx.from_numpy_matrix(G)


def community_subgraphs(graph, membership):
    ds = {}
    for u, v in membership.iteritems():
        if v not in ds.keys():
            ds[v] = []
        ds[v].append(u)
    for nbunch in ds.values():
        yield nx.subgraph(graph, nbunch)


def membership_to_groups(graph, membership):
    group = {}
    for x, g in enumerate(community_subgraphs(graph, membership)):
        group[x] = set(g.nodes())
    return group


def best_fagso_partition(graph, nruns=1):
    import fagso
    return fagso.fagso(graph, show_percentage=False)


def best_fagso_partition_cpp(graph, nruns=1, method=0, kfold=1):
    # First save the graph as a temporary adjacency matrix
    # to avoid collision in multiple runs
    import glob
    import subprocess
    import os
    graph_id = np.random.randint(2 ** 32)
    graphname = '/tmp/fagso_graph_' + str(graph_id) + '.adj'
    # Save the adjacency matrix to feed to fagso (must be a square 0-1 matrix)
    np.savetxt(graphname, nx.to_numpy_matrix(graph) != 0, fmt='%d', delimiter=' ')

    fagso_dir = '/home/carlo/workspace/PHD/brainets/src/Surprise++/build/'
    fagso_executable = fagso_dir + 'fagso '
    fagso_options = ' -r %d -s %d -v %d -o /tmp/'.rstrip().lstrip() % (nruns,method, kfold)
    status = subprocess.call(fagso_executable + ' ' +fagso_options + ' ' + graphname, shell=True)
    status = subprocess.call('echo ' + fagso_executable + ' ' +fagso_options + ' ' + graphname, shell=True)
    # Get the results back
    membership = {}
    membership_file = glob.glob("/tmp/fagso_graph_" + str(graph_id) + "*.memb")[0]
    # Membership files are separated by ':'
    m = np.loadtxt(membership_file, delimiter=':').astype(int)
    membership = dict(zip(m[:, 0], m[:, 1]))
    membership = reindex_membership(membership)
    groups = reindex_clustering(membership_to_groups(graph, membership))
    # Remove the temporary files
    os.popen('rm /tmp/fagso_graph_*')
    return membership


def label_propagation(graph, nruns=1):
    # Implements label propagation from http://arxiv.org/pdf/0910.5516.pdf
    n = graph.number_of_nodes()
    C = []
    t = 0
    C.append(range(0, n))
    t = 1
    X = np.random.permutation(range(0, n))

    def highest_freq_label(graph, x, C):

        def most_common(lst):
            return max(set(lst), key=lst.count)

        return most_common(array(C)[graph.neighbors(x)].tolist())

    for i, n in enumerate(graph.nodes()):
        C.append()

# Author: Maxwell Bertolero, bertolero@berkeley.edu, mbertolero@gmail.com
# Author: Maxwell Bertolero, bertolero@berkeley.edu, mbertolero@gmail.com


def within_module_degree(graph, partition, weighted=False):
    '''
    Computes the within-module degree for each node (Guimera et al. 2005)

    ------
    Inputs
    ------
    graph = Networkx Graph, unweighted, undirected.
    partition = dictionary from modularity partition of graph using Louvain method

    ------
    Output
    ------
    Dictionary of the within-module degree of each node.

    '''
    new_part = {}
    for m, n in zip(partition.values(), partition.keys()):
        try:
            new_part[m].append(n)
        except KeyError:
            new_part[m] = [n]
    partition = new_part
    wd_dict = {}

    # loop through each module, look at nodes in modules
    for m in partition.keys():
        mod_list = partition[m]
        mod_wd_dict = {}
        # get within module degree of each node
        for source in mod_list:
            count = 0
            for target in mod_list:
                if (source, target) in graph.edges() or (target, source) in graph.edges():
                    if weighted:
                        count += graph.get_edge_data(source, target)['weight']
                        # i assume this will only get one weighted edge.
                        count += graph.get_edge_data(target, source)['weight']
                    else:
                        count += 1
            mod_wd_dict[source] = count
        # z-score
        all_mod_wd = mod_wd_dict.values()
        avg_mod_wd = float(sum(all_mod_wd) / len(all_mod_wd))
        std = np.std(all_mod_wd)
        # add to dictionary
        for source in mod_list:
            wd_dict[source] = (mod_wd_dict[source] - avg_mod_wd) / std
    return wd_dict


def participation_coefficient(graph, partition, weighted=False):
    '''
    Computes the participation coefficient for each node (Guimera et al. 2005).

    ------
    Inputs
    ------
    graph = Networkx graph
    partition = modularity partition of graph

    ------
    Output
    ------
    Dictionary of the participation coefficient for each node.

    '''
    # this is because the dictionary output of Louvain is "backwards"
    new_part = {}
    for m, n in zip(partition.values(), partition.keys()):
        try:
            new_part[m].append(n)
        except KeyError:
            new_part[m] = [n]
    partition = new_part
    pc_dict = {}
    all_nodes = set(graph.nodes())
    # loop through modules
    if not weighted:
        for m in partition.keys():
            # set of nodes in modules
            mod_list = set(partition[m])
            # set of nodes outside that module
            between_mod_list = list(set.difference(all_nodes, mod_list))
            for source in mod_list:
                # degree of node
                degree = float(nx.degree(G=graph, nbunch=source))
                count = 0
                # between module degree
                for target in between_mod_list:
                    if (source, target) in graph.edges() or(source, target) in graph.edges():
                        count += 1
                bm_degree = float(count)
                if bm_degree == 0.0:
                    pc = 0.0
                else:
                    pc = 1 - ((float(bm_degree) / float(degree)) ** 2)
                pc_dict[source] = pc
        return pc_dict
        # this is because the dictionary output of Louvain is "backwards"
    if weighted:
        for m in partition.keys():
            # set of nodes in modules
            mod_list = set(partition[m])
            # set of nodes outside that module
            between_mod_list = list(set.difference(all_nodes, mod_list))
            for source in mod_list:
                # degree of node
                degree = 0
                edges = G.edges([source], data=True)
                for edge in edges:
                    degree += graph.get_edge_data(edge[0], edge[1])['weight']
                count = 0
                # between module degree
                for target in between_mod_list:
                    if (source, target) in graph.edges() or(source, target) in graph.edges():
                        count += graph.get_edge_data(source, target)['weight']
                        # i assume this will only get one weighted edge.
                        count += graph.get_edge_data(target, source)['weight']
                bm_degree = float(count)
                if bm_degree == 0.0:
                    pc = 0.0
                else:
                    pc = 1 - ((float(bm_degree) / float(degree)) ** 2)
                pc_dict[source] = pc
        return pc_dict
