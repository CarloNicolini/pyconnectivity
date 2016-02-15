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

__author__ = 'Carlo Nicolini <carlo.nicolini@iit.it>'
__all__ = ['plot_measures', 'draw_network',
           'draw_network_3D', 'load_node_positions_2D','visualize_communities']

import networkx as nx
import numpy as np
import logging, statistic
import copy
import matplotlib.pyplot as plt

def plot_measures(averageThresholdMeasures, dataLabel, colors=['r', 'b']):
    '''Plot all the graph measures vs threshold and density'''
    totalplots = 100 * len(averageThresholdMeasures) + 10 + 0
    for i in range(0, len(averageThresholdMeasures)):
        currentKey = averageThresholdMeasures.keys()[i]
        if currentKey is not 'threshold':
            # Now add the legend with some customizations.
            plt.subplot(totalplots + i + 1)
            plt.plot(averageThresholdMeasures['threshold'], averageThresholdMeasures[currentKey], '-' +
                     colors[0], label=dataLabel + ' threshold')
            #plt.plot( averageThresholdMeasures['density'],  averageThresholdMeasures[currentKey],'-o'+colors[1], label=dataLabel+' density')
            plt.xlim([np.min(averageThresholdMeasures['threshold']),
                      np.max(averageThresholdMeasures['threshold'])])
            plt.xlabel('Threshold')
            plt.ylabel(currentKey[0:12])
            plt.title(currentKey)
            plt.xticks(np.arange(np.min(averageThresholdMeasures['threshold']), np.max(
                averageThresholdMeasures['threshold']), 0.01), rotation=45)
            plt.grid(True)
            pylab.legend(loc=3)

    plt.tight_layout()


def load_node_positions_2D(positionsFileName):
    '''Load a dictionary where every row contains the node and its position as 2D coordinates'''
    pos = {}
    positionFile = open(positionsFileName, 'r')
    for row in positionFile:
        r = row.rstrip().split()
        if r[0] == '#' or row == '':  # skip comments
            continue
        # r[0] == region name
        # r[1] == region side
        # r[2] == region position x
        # r[3] == region position y
        pos[r[0] + ' ' + r[1]] = (float(r[2]), float(r[3]))
    return pos


def community_subgraphs(graph, membership):
    ds = {}
    for u, v in membership.iteritems():
        if v not in ds.keys():
            ds[v] = []
        ds[v].append(u)
    for nbunch in ds.values():
        yield nx.subgraph(graph, nbunch)


def draw_network(graph, nodes_membership, **kwargs):
    ''' Draw the graph with its labels '''
    plot_prop = {}
    plot_prop['nodes_size'] = kwargs.get('nodes_size', 10)
    plot_prop['edges_width'] = kwargs.get('edges_width', 1)
    plot_prop['color_map'] = kwargs.get('color_map', plt.cm.Spectral)
    plot_prop['font_size'] = kwargs.get('font_size', 15)
    plot_prop['layout_method'] = kwargs.get('layout_method', 'neato')
    plot_prop['node_alpha'] = kwargs.get('node_alpha', 0.8)
    plot_prop['draw_edges'] = kwargs.get('draw_edges', False)
    plot_prop['draw_edges_weights'] = kwargs.get('draw_edges_weights', False)
    plot_prop['axisX'] = kwargs.get('axisX', 0)
    plot_prop['axisY'] = kwargs.get('axisY', 2)
    plot_prop['mst_sparsification'] = kwargs.get('mst_sparsification', False)
    plot_prop['highlight_comm'] = kwargs.get('highlight_comm', None)

    original_weights_matrix = kwargs.get('original_weights_matrix', None)
    split_hemispheres = kwargs.get('split_hemispheres', None)
    output_file_name = kwargs.get('output_file_name', None)
    node_pos = kwargs.get('node_pos', None)

    if node_pos is None:
        node_pos = nx.graphviz_layout(graph, prog=plot_prop['layout_method'])
    elif isinstance(node_pos, str):
        #logging.info('Ignoring layout_method, using loaded node coordinates')
        node_pos = load_node_positions_2D(node_pos)
    else:
        if np.shape(node_pos)[1] == 3:
            #logging.info('Ignoring layout_method, using provided 3D node coordinates')
            node_pos_dict = {}
            for i, r in enumerate(node_pos):
                node_pos_dict[
                    i] = -node_pos[i, plot_prop['axisX']], node_pos[i, plot_prop['axisY']]
            node_pos = node_pos_dict.copy()

    # Nodewise size
    nodesize = [plot_prop['nodes_size'] / 25.0 *
                (graph.degree(n) + 1) for n in graph.nodes()]
    nodesize = [plot_prop['nodes_size'] * 2.5 * (float(nodes_membership.get(
        n, False) == plot_prop['highlight_comm']) + 0.25) for n in graph.nodes()]

    nodes_membership_highlight = {}
    for n, c in nodes_membership.iteritems():
        if c == plot_prop['highlight_comm']:
            nodes_membership_highlight[n] = 1
        else:
            nodes_membership_highlight[n] = 0

    edges_width = None  # [1 for (n,m,w) in G.edges(data=True)]
    edges_color = None  # [1 for (n,m,w) in G.edges(data=True)]
    if not hasattr(graph, 'is_weighted'):
        graph.is_weighted = False

    if graph.is_weighted:
        edges_width = [plot_prop['edges_width'] *
                       np.log(2 * w['weight'] * 2 + 1)
                       for (n, m, w) in graph.edges(data=True)]

        edges_color = [plot_prop['edges_width'] *
                       np.log(2 * w['weight'] + 1)
                       for (n, m, w) in graph.edges(data=True)]
    else:
        edges_width = [plot_prop['edges_width'] * (graph.degree(n)
                                                   + graph.degree(m)) / 2 for (n, m, w) in graph.edges(data=True)]

        edges_color = [(graph.degree(n) + graph.degree(m)) / 2 + 5
                       for (n, m, w) in graph.edges(data=True)]
        # Inverted dictionary of nodes is used to lookup dataF
        if original_weights_matrix is not None:
            edges_color = [original_weights_matrix[graph.row_node_dictionary[n]]
                           [graph.row_node_dictionary[m]]
                           for (n, m, w) in graph.edges(data=True)]

    #edges_width = plot_prop['edges_width']

    if split_hemispheres:
        posShifted = {}
        for regionName, regionPosition in node_pos.iteritems():
            if regionName[-1] == 'L':
                posShifted[regionName] = np.array(
                    regionPosition) - np.array([400, 0])
            if regionName[-1] == 'R':
                posShifted[regionName] = np.array(
                    regionPosition) + np.array([400, 0])

        node_pos = copy(posShifted)

    if plot_prop['draw_edges'] is True:
        if plot_prop['mst_sparsification'] is False:
            nx.draw_networkx_edges(graph, node_pos, alpha=0.3, width=edges_width,
                                   edge_cmap=plt.cm.gist_yarg, edge_vmin=np.min(
                                       edges_color) - 0.1,
                                   edge_vmax=np.max(edges_color), with_labels=False)
        else:
            el = list(community_subgraphs(graph, nodes_membership))[
                plot_prop['highlight_comm']]
            nx.draw_networkx_edges(nx.minimum_spanning_tree(el), node_pos, alpha=0.3, width=edges_width,
                                   edge_cmap=plt.cm.Spectral, edge_vmin=np.min(
                                       edges_color) - 0.1,
                                   edge_vmax=np.max(edges_color), with_labels=False)

    if plot_prop['draw_edges_weights'] is True:
        weight_labels = dict(((u, v), '%.3f' % d.get('weight', 1))
                             for u, v, d in graph.edges(data=True))
        nx.draw_networkx_edge_labels(
            graph, node_pos, edge_labels=weight_labels)

    nx.draw_networkx_nodes(graph, node_pos, linewidths=0.2, node_size=nodesize,
                           node_color=nodes_membership.values(),
                           alpha=plot_prop['node_alpha'],
                           cmap=plot_prop['color_map'])

    if kwargs.get('draw_labels', False):
        node_pos_shift = {}
        for key, val in node_pos.iteritems():
            node_pos_shift[key] = (val[0], val[1] + plot_prop['font_size'] / 2)
        nx.draw_networkx_labels(graph, node_pos_shift, font_size=plot_prop['font_size'])

    plt.title(graph.name + ' |E|=' + str(graph.number_of_edges()) + ' |Comm|= ' +
              str(len(np.unique(nodes_membership.values()))))  # + ' Method= ' + measureNodeColor )
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())

    if output_file_name is not None:
        plt.savefig(output_file_name)


def visualize_communities(graph, membership, line_color='red', line_width=1):
    ci = np.array(membership.values())
    from bct import grid_communities
    import pylab

    def grayify_cmap(cmap):
        """Return a grayscale version of the colormap"""
        cmap = plt.cm.get_cmap(cmap)
        colors = cmap(np.arange(cmap.N))

        # convert RGBA to perceived greyscale luminance
        # cf. http://alienryderflex.com/hsp.html
        rgb_weight = [0.299, 0.587, 0.114]
        luminance = np.sqrt(np.dot(colors[:, :3] ** 2, rgb_weight))
        colors[:, :3] = luminance[:, np.newaxis]
        return cmap.from_list(cmap.name + "_grayscale", colors, cmap.N)

    bounds, ixes = grid_communities(ci)
    adj = nx.to_numpy_matrix(graph)
    pylab.imshow(adj[np.ix_(ixes, ixes)], interpolation='none', cmap=grayify_cmap('gray_r'))
    pylab.axis('off')
    bounds = bounds.tolist()
    bounds = [0] + bounds
    # Draw the bounding rectangles 1
    for n, b in enumerate(bounds):
        l = membership.values().count(n)
        plt.hlines(y=b, xmin=b, xmax=b + l, color=line_color,
                   linewidth=line_width, zorder=1)
        plt.vlines(x=b, ymin=b, ymax=b + l, color=line_color,
                   linewidth=line_width, zorder=1)
    # Draw the bounding rectangles 2
    for n, b in enumerate(bounds):
        if n == 0:
            continue
        l = membership.values().count(n)
        plt.hlines(y=b, xmin=bounds[n - 1], xmax=bounds[n], color=line_color, linewidth=line_width, zorder=1)
        plt.vlines(x=b, ymin=bounds[n - 1], ymax=bounds[n], color=line_color, linewidth=line_width, zorder=1)
    #plt.colorbar()


def desaturate(rgb_in, desat_scale, alpha):
    """
    Desaturate an RGB value by some amount desat_scale
    """
    import colorsys
    hsv = list(colorsys.rgb_to_hsv(rgb_in[0], rgb_in[1], rgb_in[2]))
    rgb_out = list(colorsys.hsv_to_rgb(hsv[0], hsv[1] * desat_scale, hsv[2]))
    return np.array(rgb_out + [rgb_in[3] * alpha])


def draw_network_3D(graph, nodes_membership, **kwargs):
    ''' Draw the graph with its labels '''
    from mayavi import mlab
    plot_prop = {}
    plot_prop['nodes_size'] = kwargs.get('nodes_size', 10)
    plot_prop['edges_width'] = kwargs.get('edges_width', 1)
    plot_prop['color_map'] = kwargs.get('color_map', plt.cm.Spectral)
    plot_prop['font_size'] = kwargs.get('font_size', 15)
    plot_prop['layout_method'] = kwargs.get('layout_method', 'neato')
    plot_prop['node_alpha'] = kwargs.get('node_alpha', 0.8)
    plot_prop['draw_edges'] = kwargs.get('draw_edges', False)
    plot_prop['draw_edges_weights'] = kwargs.get('draw_edges_weights', False)
    plot_prop['axisX'] = kwargs.get('axisX', 0)
    plot_prop['axisY'] = kwargs.get('axisY', 2)
    plot_prop['mst_sparsification'] = kwargs.get('mst_sparsification', False)
    plot_prop['highlight_comm'] = kwargs.get('highlight_comm', None)

    plot_prop['v_min'] = kwargs.get('v_min',np.min(nodes_membership.values()) )
    plot_prop['v_max'] = kwargs.get('v_max',np.max(nodes_membership.values()) )
    # http://colorbrewer2.org/index.html?type=qualitative&scheme=Paired&n=12
    import brewer2mpl
    # Generates perceptually diverging maps to better see differences of
    # sequential data
    lut = np.array(brewer2mpl.get_map('Paired', 'Qualitative', 12).mpl_colors)
    # Add the alpha channel
    lut = np.hstack((lut, np.ones((lut.shape[0], 1)))) * 255
    # Reduce the alpha of non highlighted community
    if plot_prop['highlight_comm'] is not None:
        for i, colorRGB in enumerate(lut):
            if i != plot_prop['highlight_comm']:
                lut[i, :] = desaturate(colorRGB, kwargs.get('desat_scale',0.8), alpha=kwargs.get('alpha',1))

    xyz = kwargs.get('node_pos', None)
    
    #output_file_name = kwargs.get('output_file_name', None)
    #xyz=np.array([node_pos[v] for v in sorted(graph.nodes())])
    scalars = nodes_membership.values()
    #mlab.options.backend = 'envisage'
    mlab.figure(1, bgcolor=(0, 0, 0))
    # mlab.clf()
    # http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
    pts = mlab.points3d(xyz[:, 0], xyz[:, 1], xyz[:, 2],
                        scalars,
                        scale_factor=plot_prop['nodes_size'],
                        scale_mode='none',
                        colormap='Set2',
                        resolution=12,
                        reset_zoom=True,
                        vmax=plot_prop['v_max'],
                        vmin=plot_prop['v_min'])

    # Set the new colormap look-up-table
    pts.module_manager.scalar_lut_manager.lut.table = lut

    if plot_prop['draw_edges'] is not None:
        if plot_prop['highlight_comm'] is not None:
            highlight_subgraph = list(community_subgraphs(graph, nodes_membership))[plot_prop['highlight_comm']]
            if plot_prop['mst_sparsification']:
                el = nx.minimum_spanning_tree(highlight_subgraph).edges()
            else:
                el = highlight_subgraph.edges()
        else:
            if plot_prop['mst_sparsification']:
                el = nx.minimum_spanning_tree(graph).edges()
            else:
                el = graph.edges()
                
        pts.mlab_source.dataset.lines = np.array(el)
        tube = mlab.pipeline.tube(pts, tube_radius=plot_prop['edges_width'])
        tube.filter.radius_factor = 1.0
        tube.filter.vary_radius = 'vary_radius_by_scalar'
        
        if plot_prop['highlight_comm'] is not None:
            tubecolor = tuple(lut[plot_prop['highlight_comm'], 0:3] / 255.0)
            mlab.pipeline.surface(tube, color = tubecolor)
    
    mlab.show()

    ''' RISPOSTA DI AESTRIVEX CHE USA ANCHE LUI QUESTI DATI
    http://stackoverflow.com/questions/22253298/mayavi-points3d-with-different-size-and-colors/
    nodes = points3d(x,y,z)
    nodes.glyph.scale_mode = 'scale_by_vector'

    #this sets the vectors to be a 3x5000 vector showing some random scalars
    nodes.mlab_source.dataset.point_data.vectors = np.tile( np.random.random((5000,)), (3,1))

    nodes.mlab_source.dataset.point_data.scalars = np.random.random((5000,))
    '''
    '''
    # Visualize the data
    from mayavi import mlab

    mlab.figure(1, bgcolor=(0, 0, 0))
    mlab.clf()
    colors = statistic.node_properties_color_map(graph,'maximum_modularity_partition').values()
    
    pts = mlab.points3d(pos[:, 0], pos[:, 1], pos[:, 2],  colors, colormap='spectral', scale_factor=1.00, resolution=3)
    pts.glyph.color_mode = 'color_by_scalar'
    pts.mlab_source.dataset.lines = np.array(graph.edges())

    # Use a tube fiter to plot tubes on the link, varying the radius with the
    # scalar value
    tube = mlab.pipeline.tube(pts, tube_radius=0.1)
    tube.filter.radius_factor = 1.
    tube.filter.vary_radius = 'vary_radius_by_scalar'
    #mlab.pipeline.surface(tube, color=(0.8, 0.8, 0))

    # Visualize the local atomic density
    mlab.pipeline.volume(mlab.pipeline.gaussian_splatter(pts))

    gs = mlab.pipeline.gaussian_splatter(pts)
    gs.filter.radius = 0.05
    #iso = mlab.pipeline.iso_surface(gs, colormap = 'RdBu', opacity=0.03)
    #gsvol = mlab.pipeline.volume(gs)

    mlab.view(49, 31.5, 52.8, (4.2, 37.3, 20.6))

    mlab.show()
    '''


def draw_degree_distribution(graph):
    plt.hist(graph.degree().values())
    plt.title('Gaussian Histogram')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.show()


def draw_communities_histogram(graph, comms):
    plt.cla()
    plt.xlabel('community id')
    plt.ylabel('nodes in community')
    plt.hist(comms.values(), np.unique(comms.values()))
    plt.show()
