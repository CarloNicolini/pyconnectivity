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

__author__ = "Carlo Nicolini <carlo.nicolini@iit.it>"

import numpy as np
import networkx as nx
import mpmath
from mpmath import mpf
from copy import copy
from hyperhelper import log_hyper_probability, sum_log_probabilities, hyper1, compute_surprise

__ONE_OVER_LOG10 = 0.43429448190325176


# Ref: Intensity and coherence of motifs in weighted complex networks
# Jukka-Pekka


def subgraph_intensity(graph):
    if graph.number_of_edges() == 0:
        return 0
    if graph.number_of_nodes() == 1:
        return 0
    k = graph.number_of_edges()
    prod = mpf(1.0)
    for e in graph.edges(data=True):
        wij = e[2].get('weight', 1.0)
        prod *= wij
    if prod == 1:
        return prod
    return float(mpmath.power(prod, mpf(1.0 / k)))


def subgraph_intensity2(graph):
    # use the transformation in log and exp
    # http://en.wikipedia.org/wiki/Geometric_mean
    s = 0.0
    for e in graph.edges(data=True):
        wij = e[2].get('weight', 1.0)
        s += np.log(wij)
    return np.exp(s / graph.number_of_edges())


# Ref: Intensity and coherence of motifs in weighted complex networks
# Jukka-Pekka
def subgraph_coherence(graph):
    if graph.number_of_edges() == 0:
        return 1
    if graph.number_of_nodes() == 1:
        return 1
    k = graph.number_of_edges() + mpf(10 ** -18)
    prod = mpf(1.0)
    sumweights = 0.0
    for e in graph.edges(data=True):
        wij = e[2].get('weight', 1)
        prod *= wij
        sumweights += wij
    return float(mpmath.power(prod, mpf(1.0 / k)) / sumweights * k)


class Surprise:

    def __init__(self, graph,
                 partition,
                 normalization_method='max',
                 computation_method='accurate',
                 use_quadrature=False):

        self.graph = graph.copy()
        self.partition = copy(partition)
        self.computation_method = computation_method

        if not isinstance(partition, dict):
            raise Exception(
                'partition is a dictionary node:comm where node is the graph\
                node, comm is linear index in [0,...,|C|]')
        if len(graph) != len(partition):
            raise Exception(
                'Not a valid partition for this graph, check partition length!')
        # Enforce symmetry
        self.adjacency = nx.to_numpy_matrix(
            graph).astype(np.float64)  # to save memory
        # Check if the input matrix is symmetric
        if not (self.adjacency.transpose() == self.adjacency).all():
            raise Exception(
                'Graph is directed, adjacency matrix is not symmetric')

        self.adjacency = (self.adjacency + self.adjacency.T) * 0.5
        # Enforce no-self-loops
        np.fill_diagonal(self.adjacency, 0)

        # Normalize edge weights for the maximum weight
        self.adjacency = self.__normalize(normalization_method)
        # Reconstruct the current class graph
        self.graph = nx.from_numpy_matrix(self.adjacency)
        # Build the clustering where the key is the community and for
        # every community there is a list of nodes contained
        self.clustering = {}
        for n, c in partition.iteritems():
            self.clustering[c] = []
        for n, c in partition.iteritems():
            self.clustering[c].append(n)

        # Compute the surprise parameters
        # mi is the number of intracluster edges
        # pi is the number of intracluster pairs
        # m is the number of edges
        # p is the number of pairs
        self.mi, self.pi, self.m, self.p, self.ni = self.__compute_surprise_parameters()
        #self.mi = np.floor(self.mi)
        #self.pi = np.ceil(self.pi)
        #self.m = np.floor(self.m)
        if use_quadrature:
            self.surprise_value = self.__compute_surprise_weighted_by_quadrature(
                self.mi, self.pi, self.m, self.p)
        else:
            if self.computation_method is 'accurate':
                self.surprise_value = self.__compute_surprise_weighted(
                    self.mi, self.pi, self.m, self.p)
            elif self.computation_method is 'fast':
                self.surprise_value = compute_surprise(self.mi, self.pi, self.m, self.p)


        # These are the condition that a full, mutually exclusive clustering
        # imposes, if these are not met,
        # then the clustering is impossible, or does not cover all nodes
        # or it imposes overlapping communities
        condition_1 = (self.m - self.mi) <= (self.p - self.pi)
        condition_2 = (0 <= self.mi <= self.pi <= self.p)
        condition_3 = (self.mi <= self.m)
        error_message = 'Non valid combination of parameters because '
        if not condition_1:
            error_message += '(m-mi)>(p-pi), '
            error_message += ('mi=%g,pi=%g,m=%g,p=%g') % (self.mi,
                                                          self.m, self.pi, self.p)
            raise Exception(error_message)

        if not condition_2:
            error_message += 'not 0 <= mi <= pi <= p'
            error_message += ('mi=%g,pi=%g,m=%g,p=%g') % (self.mi,
                                                          self.m, self.pi, self.p)
            raise Exception(error_message)

        if not condition_3:
            error_message += 'mi > m'
            error_message += ('mi=%g,pi=%g,m=%g,p=%g') % (self.mi,
                                                          self.m, self.pi, self.p)
            raise Exception(error_message)

        return None

    def __get__(self):
        return self.surprise_value

    def __normalize(self, method):
        if method is None:
            return self.adjacency
        if method == 'binary':
            self.adjacency = (self.adjacency != 0).astype(float)
            self.computation_method = 'fast'
            return self.adjacency
        else:
            raise Exception(method + ' method not implemented')

    def get_parameters(self):
        return {'mi': self.mi, 'pi': self.pi, 'm': self.m, 'p': self.p}

    def surprise(self):
        return self.surprise_value

    def __compute_surprise_parameters(self):
        mi, pi = 0, 0
        for community, nodes in self.clustering.iteritems():
            subgraph_adjacency_matrix = self.adjacency[nodes, :][:, nodes]
            mi += subgraph_adjacency_matrix.sum() * 0.5 - \
                subgraph_adjacency_matrix.diagonal().sum()
            ni = subgraph_adjacency_matrix.shape[0]
            pi += ni * (ni - 1) / 2 * subgraph_adjacency_matrix.max()

        m = self.adjacency.sum() * 0.5
        n = self.adjacency.shape[0]
        p = float(n * (n - 1) / 2) * self.adjacency.max()
        return mi, pi, m, p, ni


    # Uses the hypergeometric 3F2 function as described in
    #  Bogolubsky and Skorokhodov,
    # "Fast evaluation of the hypergeometric function pFp-1(a;b;z)
    # at the singular point z=1 by means of the Hurwitz zeta function x(a,s)",
    # Programming and Computer Software, 2006, Volume 32, Issue 3, pp. 145-153
    # also described in
    # http://fredrikj.net/blog/2014/06/easy-hypergeometric-series-at-unity/
    def __compute_surprise_weighted(self, mi, pi, m, p):
        b1 = mpmath.binomial(pi, mi)
        b2 = mpmath.binomial(p - pi, m - mi)
        b3 = mpmath.binomial(p, m)
        h3f2 = mpmath.hyp3f2(
            1, mi - m, mi - pi, mi + 1, -m + mi + p - pi + 1, 1)
        #h3f2 = hyper1([mpf(1),mi-m, mi-pi], [mpf(1),mpf(1)+mi, mi+p-pi-m+mpf(1)], 10, 10)
        log10cdf = mpmath.log10(
            b1) + mpmath.log10(b2) + mpmath.log10(h3f2) - mpmath.log10(b3)
        return -float(log10cdf)

    def __compute_surprise_weighted_by_quadrature(self, mi, pi, m, p):
        f = lambda i: mpmath.binomial(
            self.pi, i) * mpmath.binomial(self.p - self.pi, self.m - i) / \
            mpmath.binomial(self.p, self.m)
        # +- 0.5 is because of the continuity correction
        return float(-mpmath.log10(mpmath.quad(f,
                                               [self.mi - 0.5, self.m + 0.5])))


"""
def get_surprise_parameters(graph, partition):
    # if len(partition) != len(graph):
    #    raise Exception('Partition size is different from number of nodes')
    # Compute the Surprise parameter of a partition for a given graph,
    p = graph.number_of_nodes() * (graph.number_of_nodes() - 1) / 2
    m = graph.number_of_edges()

    memberships, pi = None, None
    # M e quello che nell articolo teorico e chiamato i_p(\Xi)
    # i_p(\Xi) = i_p(C_1)+\ldots i_p(C_k) se ci sono k clusters

    if isinstance(partition, dict):
        memberships = Counter(partition.values())
        # Calculate parameter M the maximum possible number of intracommunity
        # links actually observed in that partition
        mvals = np.asarray(memberships.values())
        pi = (mvals * (mvals - 1) / 2).sum()
    if isinstance(partition, list):
        memberships = Counter(partition).values()
        pi = (np.asarray(memberships) *
              (np.asarray(memberships) - 1) / 2).sum()

    # Calculate parameter p
    # If both a node and its neighbor are in the same partition count +1
    mi = 0
    for e in graph.edges():
        if partition[e[0]] == partition[e[1]]:
            mi += 1
    return mi, pi, m, p
"""
"""
def surprise_gaussian_limit(mi, pi, m, p):
    mui = m * pi / p
    sigmahi = np.sqrt((m * (-m + p) * (p - pi) * pi) / ((p - 1) * p ** 2))

    def phi(x):
        return 0.5 + 0.5 * erf(x / np.sqrt(2))

    alpha2 = phi((m - mui) / sigmahi + 1.0 / (2.0 * sigmahi))
    alpha1 = phi((mi - mui) / sigmahi - 1.0 / (2.0 * sigmahi))
    return np.log10(alpha2 - alpha1) + log_binomial(p, m) * __ONE_OVER_LOG10


def surprise_taylor_expansion(mi, pi, m, p):
    arg = (1 + m - mi) / \
        (np.sqrt(-((m * (m - p) * (p - pi) * pi) / ((-1 + p) * p ** 2)))
            * np.sqrt(2 * np.pi))
    return np.log10(arg) + log_binomial(p, m) * __ONE_OVER_LOG10
"""
