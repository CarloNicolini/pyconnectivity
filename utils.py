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
#
#    Further informations and API are available at
#    http://perso.crans.org/aynaud/communities/api.html

__author__ = "Carlo Nicolini <carlo.nicolini@iit.it>"
__all__ = ["reverse", "issymmetric", "rescaleMatrix",
           "sortrows", "load_matlab_matrix", "load_average_subject_matrix",
           "graph_from_matrix", "is_weighted", "LFR_weighted_benchmark",
           "to_igraph", "ring_cliques_increasing_benchmark",
           "ring_cliques_benchmark","threshold_matrix_absolute",
           "threshold_giant_component","threshold_matrix_absolute_module",
           "threshold_matrix_garlaschelli"]

# http://mikegrouchy.com/blog/2012/05/be-pythonic-__init__py.html
import networkx as nx
import numpy as np
import scipy.io
import copy


def progress(header, current, maximum):
    import sys
    percent = float(current) / float(maximum) * 100
    percentstring = '... %.1f' % percent + ' %'
    sys.stderr.write('\r' + header + percentstring)


def reverse(data, value):
    """returns the x value corresponding to a given y value (the one that minimizes the difference (it works just for monotonic data)"""
    sortedData = data[np.argsort(data[:, 0])]
    return sortedData[np.argmin(abs(sortedData[:, 1] - value))][0]


def issymmetric(A):
    return (A[:, :].T == A[:, :]).all()


def rescaleMatrix(matrix):
    """Rescale the matrix in [0,1] interval"""
    #tmp = copy.copy(matrix)
    # return ((tmp-tmp.min())/(tmp.max()-tmp.min()))
    return matrix


def sortrows(array, column=0):
    """sorts the array in rows with values in column.  default value of column is 0"""
    sorted = array[np.argsort(array[:, 0], column)]
    return sorted


def load_VOI_names(filename, maxNamesLength=None):
    """
    Returns a list of strings representing the VOI names loaded from a .txt file
    """
    VOI = list()
    f = open(filename, 'r')
    for row in f:
        VOIname = row.strip('\n')
        if maxNamesLength is not None:
            VOIname = VOIname[0:maxNamesLength]
        # remove carriage returns if any...
        VOI.append(VOIname.replace('\r', ''))
    return VOI


def threshold_matrix_garlaschelli(a_ij, tau):
    # Follow the method of thresholding of 
    # Community Detection for Correlation Matrices
    # Mel MacMahon and Diego Garlaschelli
    # Phys. Rev. X 5, 021006
    # on page 14,16
    n = a_ij.shape[0]
    z_tau = tau * 1.0 / np.sqrt(n - 3)
    c_tau = np.tanh(z_tau)

    return threshold_matrix_absolute_module(a_ij, c_tau)


def threshold_matrix_absolute(a_ij, threshold):
    """
    This function thresholds the connectivity matrix by absolute weight
    magnitude. All weights below the given threshold, and all weights
    on the main diagonal (self-self connections) are set to 0.
    """
    a_ij_np = copy.copy(np.asarray(a_ij))
    a_ij_np[a_ij < threshold] = 0  # All low values set to 0
    return a_ij_np


def threshold_matrix_absolute_module(a_ij, threshold):
    """
    Threshold a matrix keeping the strongest connection
    """
    a_ij_np = copy.copy(a_ij)
    # Where values positively or negatively correlated
    low_values_indices = np.abs(a_ij) < threshold
    a_ij_np[low_values_indices] = 0  # All low values set to 0
    return a_ij_np


def threshold_giant_component(a_ij):
    """
    Compute the threshold under which the graph splits in two components
    """
    tol = 1E-6
    a = a_ij.max()
    b = a_ij.min()
    i = 1
    imax = len(np.nonzero(a_ij)[0])/2
    if ncc(a_ij) > 1: # has already more than one connected component
        return b

    from networkx import number_connected_components as ncc
    from networkx import from_numpy_matrix as as_mat
    cut_threshold = b
    while i<imax and abs(b-a)/2.0 > tol:
        c = (a + b) / 2
        gc = as_mat(threshold_matrix_absolute(a_ij,c))
        ga = as_mat(threshold_matrix_absolute(a_ij,a))
        if ncc(gc) == 1 or ((b - a) / 2 < tol):
            cut_threshold = c
        if np.sign(ncc(gc)-1) == np.sign(ncc(ga)-1):
            a = c
        else:
            b = c
        i += 1
    return cut_threshold


def correlation_to_fisher_z_score(r):
    """
    Transform a correlation matrix in z-score
    """
    return np.arctanh(r)


def fisher_z_score_to_correlation(z):
    """
    Transform a Z score matrix to correlation coefficient
    http://support.sas.com/documentation/cdl/en/procstat/63104/HTML/default/viewer.htm#procstat_corr_sect018.htm
    """
    return (np.exp(2 * z) - 1) / (np.exp(2 * z) + 1)


def load_matlab_matrix(filename):
    """Load a Matlab matrix to memory and as np.array"""
    return scipy.io.loadmat(filename).values()


def load_average_subject_matrix(folder, matrixRescale=False, forceSymmetry=False, fisherCorrection=True):
    """
    Load a tuple all the matrices and make an average matrix and returns a list of all matrices
    """
    import glob
    subjectMatricesFileNames = glob.glob(folder + '*.mat')
    # Create the average matrix
    # matrixDimension = load_matlab_matrix(subjectMatricesFileNames[0]).shape
    allMatrices = []
    for subject in subjectMatricesFileNames:
        A = load_matlab_matrix(subject)[1]
        if forceSymmetry:
            A = (A + A.T) * 0.5
        if fisherCorrection:
            allMatrices.append(correlation_to_fisher_z_score(A))
        else:
            allMatrices.append(A)

    if fisherCorrection:
        if matrixRescale:
            return rescaleMatrix(fisher_z_score_to_correlation(np.mean(allMatrices, 0)))
        else:
            return fisher_z_score_to_correlation(np.mean(allMatrices, 0))
    else:
        if matrixRescale:
            return rescaleMatrix(np.mean(allMatrices, 0))
        else:
            return np.mean(allMatrices, 0)


def graph_from_matrix(matrix, graphName=None, weighted=True, labels=None, originalMatrixForWeights=None):
    """Generate a graph from a numpy adjaency matrix. If labels are provided, then nodes are labeled"""
    G = nx.Graph()
    G.__setattr__('is_weighted', weighted)

    # this contains the inverted dictionary node name to row number
    G.__setattr__('row_node_dictionary', dict())
    if graphName is not None:
        G.__setattr__('name', graphName)

    if (labels is not None):
        for i in range(0, matrix.shape[0]):
            G.add_node(labels[i])
            G.row_node_dictionary[labels[i]] = i

        for (sourceNode, destNode), value in np.ndenumerate(matrix):
            if (value != 0):
                if G.is_weighted:
                    G.add_edge(labels[sourceNode],
                               labels[destNode], weight=abs(value))
                else:
                    G.add_edge(labels[sourceNode], labels[destNode])
    else:
        G.add_nodes_from(range(0, matrix.shape[0]))

        for (sourceNode, destNode), value in np.ndenumerate(matrix):
            if (value != 0 and sourceNode != destNode):
                if G.is_weighted:
                    G.add_edge(sourceNode, destNode, weight=abs(value))
                else:
                    G.add_edge(sourceNode, destNode)
    return G


def write_edges_pairs_file(graph, filename):
    """
    Save the graph as edges pairs to file
    """
    # Write the edges pairs file and the partition
    f = open(filename, 'w')
    for l in graph.edges():
        f.write(l[0].replace(' ', '_') + ' ' + l[1].replace(' ', '_') + '\n')
    f.close()


def graph_from_sparse_matrix(sparseMatrix, graphName=None, weighted=True, labels=None):
    """Generate a graph from a scipy sparse adjaency matrix.
    If labels are provided, then nodes are labeled
    """
    G = nx.from_scipy_sparse_matrix(matrix)
    G.__setattr__('is_weighted', weighted)
    # this contains the inverted dictionary node name to row number
    G.__setattr__('row_node_dictionary', dict())
    if graphName is not None:
        G.__setattr__('name', graphName)
    return G


def write_sparse_matrix_to_pajek(sparseCooMatrix, netfilename):
    """Save a sparse matrix to pajek .net format"""
    outputfile = open(netfilename, 'w')
    outputfile.write("*Vertices " + str(sparseCooMatrix.shape[0]) + "\n")
    outputfile.write("*Arcs\n")

    for i, j, v in zip(sparseCooMatrix.row, sparseCooMatrix.col, sparseCooMatrix.data):
        if i != j:
            outputfile.write(str(i) + " " + str(j) + " " + str(v) + "\n")
    outputfile.close()


def is_weighted(graph):
    """Return True if a graph is weighted, False if it's not weighted or has no the 'is_weighted' attribute"""
    if hasattr(graph, 'is_weighted'):
        return graph.is_weighted
    else:
        return False


def writeNodeAttributes(G, attributes):
    """Utility function to print the node + various attribute in csv format"""
    if type(attributes) is not list:
        attributes = [attributes]
    for node in G.nodes():
        vals = [str(dict[node])
                for dict in [nx.get_node_attributes(G, x) for x in attributes]]
        logging.info(str(node) + str(vals))


def ralign(X, Y):
    """Rigid alignment of two sets of points in k-dimensional Euclidean space.
    Given two sets of points in
    correspondence, this function computes the scaling
    rotation, and translation that define the transform TR
    that minimizes the sum of squared errors between TR(X)
    and its corresponding points in Y.  This routine takes
    O(n k^3)-time.

    Inputs:
        X - a k x n matrix whose columns are points
        Y - a k x n matrix whose columns are points that correspond to
           the points in X
    Outputs:
        c, R, t - the scaling, rotation matrix, and translation vector defining
        the linear map TR as TR(x) = c * R * x + t
        such that the average norm of TR(X(:, i) - Y(:, i)) is minimized.
    """
    m, n = X.shape

    mx = X.mean(1)
    my = Y.mean(1)
    Xc = X - np.tile(mx, (n, 1)).T
    Yc = Y - np.tile(my, (n, 1)).T

    sx = np.mean(np.sum(Xc * Xc, 0))
    Sxy = np.dot(Yc, Xc.T) / n

    U, D, V = np.linalg.svd(Sxy, full_matrices=True, compute_uv=True)
    V = V.T.copy()
    # print U,"\n\n",D,"\n\n",V
    r = np.rank(Sxy)
    S = np.eye(m)
    if r > (m - 1):
        if (np.det(Sxy) < 0):
            S[m, m] = -1
        elif (r == m - 1):
            if (np.det(U) * np.det(V) < 0):
                S[m, m] = -1
        else:
            R = np.eye(2)
            c = 1
            t = np.zeros(2)
            return R, c, t

    R = np.dot(np.dot(U, S), V.T)

    c = np.trace(np.dot(np.diag(D), S)) / sx
    t = my - c * np.dot(R, mx)

    return R, c, t


def barthelemy_benchmark(n_er, prob, n_cliques, k_cliques):
    graph = nx.connected_component_subgraphs(
        nx.erdos_renyi_graph(n_er, prob))[0]
    graph_k = nx.disjoint_union_all(
        [nx.complete_graph(k_cliques) for i in range(0, n_cliques)])
    graph = nx.disjoint_union(graph, graph_k).copy()
    for i in range(0, n_cliques):
        graph.add_edge(np.random.randint(n_er), n_er + i * n_cliques)
    return graph


def ring_cliques_benchmark(n, r):
    """
    Returns a networkx graph of ring of cliques with r cliques of n nodes
    n number of nodes in every clique
    r number of connected cliques
    """
    graph = nx.disjoint_union_all([nx.complete_graph(n) for i in range(0, r)])
    graph.add_edges_from([(u, u + n + 1) for u in range(0, r * (n), n)])
    graph.remove_node(n * r + 1)
    graph.add_edge(n - 1, n * r - 1)
    return graph


def ring_cliques_increasing_benchmark(n_min, n_max):
    """
    Returns a graph ring of cliques where the cliques are in 
    increasing dimension from n_min to n_max
    n_min minimum size of first clique
    n_max maximum size of second clique
    """
    graph = nx.Graph()

    for i in range(n_min, n_max):
        g = nx.complete_graph(i)
        graph = nx.disjoint_union(graph, g)
        k = sum(range(3, i))
        if i > n_min:
            # add edges between last node in the previous clique
            # and first node in this clique
            graph.add_edge(k, k - 1)
    # Add last edge to close the ring
    graph.add_edge(graph.nodes()[0], graph.nodes()[-1])
    return graph


def anonymize(graph):
    return nx.convert_node_labels_to_integers(graph)


def to_igraph(graph):
    # Initialise the optimiser, using default settings
    import networkx as nx
    import igraph as ig
    g = ig.Graph()
    if anonymize:
        nx.write_graphml(anonymize(graph), "/tmp/test_graph.graphml")
    else:
        nx.write_graphml(graph, "/tmp/test_graph.graphml")
    return ig.read("/tmp/test_graph.graphml", format='graphml')


def to_igraph_fast(graph):
    # Initialise the optimiser, using default settings
    import networkx as nx
    import igraph as ig
    g = ig.Graph()
    graph_anonym = nx.convert_node_labels_to_integers(graph)
    g.add_vertices(graph_anonym.nodes())
    g.add_edges(graph.edges())
    return g


def makeargs(dict):
    args = []
    for k, v in dict.items():
        args.extend(['-' + k, str(v)])
    return args


def leval(s):
    try:
        return eval(s)
    except (ValueError, SyntaxError):
        return s


def read_file(fname):
    for line in open(fname):
        line = line.strip().split()
        yield tuple(leval(x) for x in line)


def LFR_weighted_benchmark(**kwargs):
    """Undirected weighted networks with overlapping nodes.
    This program is an implementation of the algorithm described in
    the paper\"Directed, weighted and overlapping benchmark graphs for
    community detection algorithms\", written by Andrea Lancichinetti
    and Santo Fortunato. In particular, this program is to produce
    undirected weighted networks with overlapping nodes.  Each
    feedback is very welcome. If you have found a bug or have
    problems, or want to give advises, please contact us:
    -N              [number of nodes]
    -k              [average degree]
    -maxk           [maximum degree]
    -mut            [mixing parameter for the topology]
    -muw            [mixing parameter for the weights]
    -beta           [exponent for the weight distribution]
    -t1             [minus exponent for the degree sequence]
    -t2             [minus exponent for the community size distribution]
    -minc           [minimum for the community sizes]
    -maxc           [maximum for the community sizes]
    -on             [number of overlapping nodes]
    -om             [number of memberships of the overlapping nodes]
    -C              [average clustering coefficient]
    -o              suffix for outputfiles (string)
    -N, -k, -maxk, -muw have to be specified. For the others, the
    program can use default values:

    t1=2, t2=1, on=0, om=0, beta=1.5, mut=muw, minc and maxc will be
    chosen close to the degree sequence extremes.  If you set a
    parameter twice, the latter one will be taken.

    To have a random network use:
    -rand
    Using this option will set muw=0, mut=0, and minc=maxc=N, i.e.
    there will be one only community.
    Use option -sup (-inf) if you want to produce a benchmark whose
    distribution of the ratio of external degree/total degree is
    superiorly (inferiorly) bounded by the mixing parameter.

    The flag -C is not mandatory. If you use it, the program will
    perform a number of rewiring steps to increase the average cluster
    coefficient up to the wished value.  Since other constraints must
    be fulfilled, if the wished value will not be reached after a
    certain time, the program will stop (displaying a warning).

    Example1:
    ./benchmark -N 1000 -k 15 -maxk 50 -muw 0.1 -minc 20 -maxc 50
    Example2:
    ./benchmark -f flags.dat -t1 3
    """
    import os

    args = makeargs(kwargs)
    suffix = kwargs['o']

    # First clear the previous files
    if os.path.isfile('network_' + suffix + '.dat'):
        os.remove('network_' + suffix + '.dat')

    if os.path.isfile('community_' + suffix + '.dat'):
        os.remove('community_' + suffix + '.dat')

    if os.path.isfile('time_seed.dat'):
        os.remove('time_seed.dat')

    if os.path.isfile('statistics_' + suffix + '.dat'):
        os.remove('statistics_' + suffix + '.dat')

    progdir = '~/workspace/PHD/brainets/src/LFRBenchmark/weighted_networks/benchmark_mod '
    commandstring = progdir + ' '.join(args)
    process = os.popen(commandstring)
    log = process.read()
    G = nx.Graph()
    # Use zero based node numbering

    for row in open('network_' + suffix + '.dat'):
        r = row.split()
        G.add_edge(int(r[0]) - 1, int(r[1]) - 1, weight=float(r[2]))

    M = {}
    for row in open('community_' + suffix + '.dat'):

        r = row.split()
        M[int(r[0]) - 1] = int(r[1])

    os.remove('statistics_' + suffix + '.dat')
    os.remove('time_seed.dat')
    os.remove('community_' + suffix + '.dat')
    os.remove('network_' + suffix + '.dat')
    return G, M, log
