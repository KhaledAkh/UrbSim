# This file contains the function that will be used in Urban Simulation Assignment Part 1 code file "Part1_networks.py"

import networkx as nx
import numpy as np
import pandas as pd
from shapely import MultiPoint
from shapely.ops import nearest_points


# returns a copy graph after removing a node + connects disconnected components "if any" with a high weighted edge
def remove_n(G, node_r):
    SG = G.copy()
    SG.remove_node(node_r)
    node_group = []
    for i, WG in enumerate(nx.strongly_connected_components(SG)):
        group = {}
        for node in nx.neighbors(G, node_r):
            if node in WG:
                # print(i,nx.get_node_attributes(G,'coords')[node])
                group[node] = nx.get_node_attributes(G, 'coords')[node]
        node_group.append(group)
        # node_group.append(MultiPoint(group.values()))
    closest_p = []
    for i in range(len(node_group) - 1):
        closest_p.append([[x.x, x.y] for x in list(
            nearest_points(MultiPoint(list(node_group[i].values())), MultiPoint(list(node_group[i + 1].values()))))])
    avg_weight = 10 + 10 * (
            np.max([w for (o, d), w in nx.get_edge_attributes(G, 'weight').items() if d == node_r]) // 10)
    print(avg_weight, node_r)
    for i in range(len(node_group) - 1):
        node1 = [x for x, y in node_group[i].items() if y in closest_p[i]][0]
        node2 = [x for x, y in node_group[i + 1].items() if y in closest_p[i]][0]
        # print(node1,node2)
        SG.add_edge(node1, node2, weight=avg_weight, inv_weight=1 / avg_weight)
        SG.add_edge(node2, node1, weight=avg_weight, inv_weight=1 / avg_weight)
    return SG


# apply node removal on topological network based on the passed a centrality function and global measurement and if
# sequential removal is required or not
# returns dataframe of the removed nodes, measured global measure, % change compared to original
def node_removal(G, centrality_func, gloabal_measu, seq=False, n=10):
    # getting list of nodes to remove
    node_list = []
    if seq:
        SG = G.copy()
        for i in range(n):
            node = pd.DataFrame.from_dict(centrality_func(SG), orient='index', columns=['centrality']).sort_values(
                'centrality', ascending=False).index.values[0]
            node_list.append(node)
            SG = remove_n(SG, node)
    else:
        node_list = pd.DataFrame.from_dict(centrality_func(G), orient='index', columns=['centrality']).sort_values(
            'centrality', ascending=False).index.values[:n]

    # removing nodes and calucalting the measurement
    full_G_measure = gloabal_measu(G)
    SG = G.copy()
    nrdf = pd.DataFrame(index=range(1, n + 1), columns=['Removed_node', 'Measurement', 'Difference_from_full'])
    for i in nrdf.index.values:
        nrdf['Removed_node'].loc[i] = node_list[i - 1]
        print(nrdf['Removed_node'].loc[i])
        SG = remove_n(SG, nrdf['Removed_node'].loc[i])
        # print(nx.number_strongly_connected_components(SG))
        nrdf['Measurement'].loc[i] = gloabal_measu(SG)
        nrdf['Difference_from_full'].loc[i] = (nrdf['Measurement'].loc[i] - full_G_measure) / full_G_measure

    return nrdf


# returns a dictionary with {node:normalised utilisation} of each node in the network using the flows in the edges
def utilisation_centrality(G):
    node_flows = {}
    for node in G:
        node_flows[node] = (sum([att['flows_Di'] for u, v, att in G.edges(data=True) if v == node]))
    return {node: x / max(node_flows.values()) for node, x in node_flows.items()}


# returns a list of top n nodes using the passed centrality function and sequential node removing
def seq_cent(G, centrality_func, n=3):
    node_list = []
    SG = G.copy()
    for i in range(n):
        node = pd.DataFrame.from_dict(centrality_func(SG), orient='index', columns=['centrality']).sort_values(
            'centrality', ascending=False).index.values[0]
        node_list.append(node)
        SG = remove_n(SG, node)
    return node_list


# returns a dictionary of {edge: flow} after calculating the edge flow of the passed network and OD flows table
def calculate_flows(G, OD_flows):
    flows = {(u, v): 0 for u, v in G.edges}
    for i, row in OD_flows.iterrows():
        source = row.station_origin
        target = row.station_destination
        # get the shortest path
        path = nx.dijkstra_path(G, source, target)
        # our path is a list of nodes, we need to turn this to a list of edges
        path_edges = list(zip(path, path[1:]))
        for u, v in path_edges:
            flows[(u, v)] += row.flows
    return flows


# Returns a list of the top n nodes using the passed centrality function and sequential node removing
# and recalculating edge flow after each iteration
def seq_cent_flwo(G, centrality_func, OD_flows, n=3):
    node_list = []
    adjusted_OD = OD_flows.copy()
    SG = G.copy()
    for i in range(n):
        node = pd.DataFrame.from_dict(centrality_func(SG), orient='index', columns=['centrality']).sort_values(
            'centrality', ascending=False).index.values[0]
        node_list.append(node)
        adjusted_OD = adjusted_OD.replace(node, next(nx.neighbors(SG, node)))
        SG = remove_n(SG, node)
        flows = calculate_flows(SG, adjusted_OD)
        nx.set_edge_attributes(SG, flows, 'flows_Di')
    return node_list


# apply node removal on flow weighted network based on the passed list of nodes and flow table (OD flows)
# returns dataframe of the removed nodes, measured avg_trip_length_change and avg_high_util_change
def node_removal_flows(G, node_list, OD_flows):
    adjusted_OD = OD_flows.copy()
    full_G_avg_trip_length = sum(nx.get_edge_attributes(G, 'flows_Di').values()) / adjusted_OD['flows'].sum()
    full_G_avg_high_uti = max(nx.get_edge_attributes(G, 'flows_Di').values()) / adjusted_OD['flows'].sum()
    SG = G.copy()
    nrdf = pd.DataFrame(index=range(1, 3 + 1),
                        columns=['Removed_node', 'avg_trip_length', 'avg_trip_length_change', 'avg_high_util',
                                 'avg_high_util_change'])
    for i in nrdf.index.values:
        nrdf['Removed_node'].loc[i] = node_list[i - 1]
        print(nrdf['Removed_node'].loc[i])
        adjusted_OD = adjusted_OD.replace(nrdf['Removed_node'].loc[i],
                                          next(nx.neighbors(SG, nrdf['Removed_node'].loc[i])))
        SG = remove_n(SG, nrdf['Removed_node'].loc[i])
        print(nx.number_strongly_connected_components(SG))
        flows = calculate_flows(SG, adjusted_OD)
        nx.set_edge_attributes(SG, flows, 'flows_Di')
        nrdf['avg_trip_length'].loc[i] = sum(nx.get_edge_attributes(SG, 'flows_Di').values()) / adjusted_OD[
            'flows'].sum()
        nrdf['avg_trip_length_change'].loc[i] = (nrdf['avg_trip_length'].loc[
                                                     i] - full_G_avg_trip_length) / full_G_avg_trip_length
        nrdf['avg_high_util'].loc[i] = max(nx.get_edge_attributes(SG, 'flows_Di').values()) / adjusted_OD['flows'].sum()
        nrdf['avg_high_util_change'].loc[i] = (nrdf['avg_high_util'].loc[i] - full_G_avg_high_uti) / full_G_avg_high_uti
    return nrdf
