import random
import networkx as nx
import numpy as np


def compare_with_ties(a, b):
    diff = cmp(a, b)
    return diff if diff else random.choice([-1, 1])


def sort_edges_by_jaccard_index(graph, edges):
    #from random import shuffle
    # shuffle(edges)
    edges_sorted = sorted(list(nx.jaccard_coefficient(graph, edges)), key=lambda l: l[
                          2], reverse=True, cmp=compare_with_ties)

    unique_j_edges = len(np.unique(np.array(edges_sorted)[:, 2]))
    total_j_edges = len(edges)
    print "Degeneracy= ", 1.0 - float(unique_j_edges) / float(total_j_edges)
    return [(row[0], row[1]) for row in edges_sorted], [row[2] for row in edges_sorted]


def sort_edges_by_resource_allocation(graph, edges):
    edges_sorted = sorted(list(nx.resource_allocation_index(
        graph, edges)), key=lambda l: l[2], reverse=True, cmp=compare_with_ties)
    return [(row[0], row[1]) for row in edges_sorted], [row[2] for row in edges_sorted]


def sort_edges_by_adamic_adar_index(graph, edges):
    edges_sorted = sorted(list(nx.adamic_adar_index(graph, edges)), key=lambda l: l[
                          2], reverse=True, cmp=compare_with_ties)
    return [(row[0], row[1]) for row in edges_sorted], [row[2] for row in edges_sorted]


def sort_edges_by_preferential_attachment(graph, edges):
    edges_sorted = sorted(list(nx.preferential_attachment(
        graph, edges)), key=lambda l: l[2], reverse=True, cmp=compare_with_ties)
    return [(row[0], row[1]) for row in edges_sorted], [row[2] for row in edges_sorted]

from collections import defaultdict
import copy

# http://stackoverflow.com/questions/9767773/calculating-simrank-using-networkx
# G. Jeh and J. Widom. SimRank: a measure of structural-context
# similarity. In KDD'02 pages 538-543. ACM Press, 2002.


def simrank(G, r=0.9, max_iter=100):
    # init. vars
    sim_old = defaultdict(list)
    sim = defaultdict(list)
    for n in G.nodes():
        sim[n] = defaultdict(int)
        sim[n][n] = 1
        sim_old[n] = defaultdict(int)
        sim_old[n][n] = 0

    # recursively calculate simrank
    for iter_ctr in range(max_iter):
        if _is_converge(sim, sim_old):
            break
        sim_old = copy.deepcopy(sim)
        for u in G.nodes():
            for v in G.nodes():
                if u == v:
                    continue
                s_uv = 0.0
                for n_u in G.neighbors(u):
                    for n_v in G.neighbors(v):
                        s_uv += sim_old[n_u][n_v]
                sim[u][v] = (
                    r * s_uv / (len(G.neighbors(u)) * len(G.neighbors(v))))
    return sim


def _is_converge(s1, s2, eps=1e-4):
    for i in s1.keys():
        for j in s1[i].keys():
            if abs(s1[i][j] - s2[i][j]) >= eps:
                return False
    return True


def sort_edges_by_simrank(graph, edges):
    S = simrank(graph)
    edges_simrank = [(e[0], e[1], S[e[0]][e[1]]) for e in edges]
    edges_sorted = sorted(
        edges_simrank, key=lambda l: l[2], reverse=True, cmp=compare_with_ties)
    return [(row[0], row[1]) for row in edges_sorted], [S[e[0]][e[1]] for e in edges]
