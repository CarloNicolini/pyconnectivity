# -*- coding: utf-8 -*-
"""
Created on Fri Oct 31 10:28:06 2014

@author: carlo
"""
#!/usr/bin/python
import sys
sys.path.append('../../')
from surprise import compute_surprise
from DisjointSet import DisjointSet as DS
import networkx as nx


def getSurprise(graph, paredges, num_nodes, num_edges):
    n = num_nodes
    p = n * (n - 1) / 2
    m = num_edges
    mi, pi = 0, 0

    dse = DS()
    for e in paredges:
        dse.add(e[0], e[1])

    for x in dse.group.values():
        ni = len(x)
        pi += ni * (ni - 1) / 2
        mi += graph.subgraph(x).number_of_edges()
    return compute_surprise(p, pi, m, mi), p, pi, m, mi
    
    
def bestSurpriseEdge(edgesList, graph, par):
    tmpPar = par.copy()
    bestEdge = None
    baseS = getSurprise(graph, par)[0]
    bestDeltaS = sys.float_info.min
    for e in edgesList:
        tmpPar.add_edge(*e)
        tmpS, p, m, pi, mi = getSurprise(graph, tmpPar)
        if tmpS - baseS > bestDeltaS:
            bestDeltaS = tmpS - baseS
            bestEdge = e
        tmpPar = par.copy()
    # print "Choosen", bestEdge,bestDeltaS+baseS
    return bestEdge


def buildLocalCandEdges(graph, edge):
    locCandEdges = set(nx.edges(graph, edge))
    locCandEdges.remove(edge)
    return list(locCandEdges)


def buildLocalCandEdges2(graph, edge, addedEdges):
    locCandEdges = set(nx.edges(graph, edge)) - set(addedEdges) - set(edge)
    # locCandEdges.remove(edge)
    return list(locCandEdges)


def agso(graph):
    par = nx.Graph()
    par.add_nodes_from(graph.nodes())
    curS = getSurprise(graph, par)[0]
    canedges = graph.edges()
    # shuffle(canedges)
    #canedges[0], canedges[11] = canedges[11], canedges[0]
    #canedges = sort_edges_by_jaccard_index(graph,canedges)
    addedEdges = []
    while canedges:
        # print "Total=", float(graph.number_of_edges() - len(canedges)), " %"
        bestEdge = bestSurpriseEdge(canedges, graph, par)
        if bestEdge is not None:
            par.add_edge(bestEdge[0], bestEdge[1])
            addedEdges.append(bestEdge)
            curS = getSurprise(graph, par)[0]
            canedges.remove(bestEdge)
            # print "Removed", bestEdge
            locCandEdges = buildLocalCandEdges2(graph, bestEdge, addedEdges)
            #locCandEdges = buildLocalCandEdges(graph, bestEdge)
            while locCandEdges:
                bestLocEdge = bestSurpriseEdge(locCandEdges, graph, par)
                # print "bestLocEdge", bestLocEdge
                if bestLocEdge is not None:
                    par.add_edge(bestLocEdge[0], bestLocEdge[1])
                    addedEdges.append(bestLocEdge)
                    curS = getSurprise(graph, par)[0]
                    # print "curS=", curS
                else:
                    break
        else:
            break
    return curS, par