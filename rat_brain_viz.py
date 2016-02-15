
# coding: utf-8

# In[43]:

from __future__ import division
import sys
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = 20,20
import numpy as np
import networkx as nx
import visualization
import utils

nodespos = visualization.load_node_positions_2D('anatomical_coordinates_symmetrized_LR_N50.txt')
nodeslist = [n.rstrip() for n in open('nodes.txt')]

func_b6_mat = np.loadtxt('funct_b6.csv',delimiter=',')
func_btbr_mat = np.loadtxt('funct_btbr.csv',delimiter=',')
struct_b6_mat = np.loadtxt('struct_b6.csv',delimiter=',')
struct_btbr_mat = np.loadtxt('struct_btbr.csv',delimiter=',')

# Make thresholding operations
func_b6_graph = nx.from_numpy_matrix(utils.threshold_matrix_absolute(func_b6_mat,0.28))
func_btbr_graph = nx.from_numpy_matrix(utils.threshold_matrix_absolute(func_btbr_mat,0.28))
struct_b6_graph = nx.from_numpy_matrix(utils.threshold_matrix_absolute(struct_b6_mat,5))
struct_btbr_graph = nx.from_numpy_matrix(utils.threshold_matrix_absolute(struct_btbr_mat,5))

mapping = dict(zip(range(0,50),nodeslist))
func_b6_graph = nx.relabel_nodes(func_b6_graph, mapping)
func_btbr_graph = nx.relabel_nodes(func_btbr_graph, mapping)
struct_b6_graph = nx.relabel_nodes(struct_b6_graph, mapping)
struct_btbr_graph = nx.relabel_nodes(struct_btbr_graph, mapping)

func_b6_graph.name = 'Functional B6'
func_btbr_graph.name = 'Functional BTBR'
struct_b6_graph.name = 'Structural B6'
struct_btbr_graph.name = 'Structural BTBR'

visualization.draw_network(func_b6_graph,dict(zip(func_b6_graph.nodes(),50*[0])),
                                 node_pos='anatomical_coordinates_symmetrized_LR_N50.txt',
                                 draw_labels=True,
                                 draw_edges=True,
                                 nodes_size=5000,
                                 node_alpha=1,
                                 color_map=plt.cm.binary,
                                 font_size=25,
                             edges_width=0.5,
                             output_file_name='func_b6_028.pdf')

visualization.draw_network(func_btbr_graph,dict(zip(func_btbr_graph.nodes(),50*[0])),
                                 node_pos='anatomical_coordinates_symmetrized_LR_N50.txt',
                                 draw_labels=True,
                                 draw_edges=True,
                                 nodes_size=5000,
                                 node_alpha=1,
                                 color_map=plt.cm.binary,
                                 font_size=25,
                                 edges_width=0.5,
                                 output_file_name='func_btbr_028.pdf')

visualization.draw_network(struct_b6_graph,dict(zip(struct_b6_graph.nodes(),50*[0])),
                                 node_pos='anatomical_coordinates_symmetrized_LR_N50.txt',
                                 draw_labels=True,
                                 draw_edges=True,
                                 nodes_size=5000,
                                 node_alpha=1,
                                 color_map=plt.cm.binary,
                                 font_size=25,
                                 edges_width=0.5,
                                 output_file_name='struct_b6_5.pdf')


visualization.draw_network(struct_btbr_graph,dict(zip(struct_btbr_graph.nodes(),50*[0])),
                                 node_pos='anatomical_coordinates_symmetrized_LR_N50.txt',
                                 draw_labels=True,
                                 draw_edges=True,
                                 nodes_size=5000,
                                 node_alpha=1,
                                 color_map=plt.cm.binary,
                                 font_size=25,
                                 edges_width=0.5,
                                 output_file_name='struct_btbr_5.pdf')

