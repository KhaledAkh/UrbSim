# This file contains the code for the analysis used for the assignment of Urban Simulation module
# It is intended to be run line by line (or block by block) in the console similar to a Jupyter notebook ot R studio

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from functools import partial

# function that will be used in this code
from functions_for_part1 import *

# Loading the  graph that was exported from Practical 10 exercise
G = nx.read_gml('LU_GDi.gml')
print(type(G))

# creating "weight" attribute of 1 for all edges. This will be used to compare with created edges that connect
# disconnected graph components after node removal
weights = {(u, v): 1 for u, v in G.edges()}
nx.set_edge_attributes(G, weights, 'weight')
nx.set_edge_attributes(G, weights, 'inv_weight')

# Initial plotting
pos = nx.get_node_attributes(G, 'coords')
fig, ax = plt.subplots(dpi=500, layout='compressed', frameon=False)
edg = nx.draw_networkx_edges(G, pos, width=0.2, arrows=False)
nx.draw_networkx_nodes(G, pos=pos, node_color='dodgerblue', node_size=10, edgecolors='w', linewidths=0.2)
plt.axis("off")
plt.show()

# Histogram for the Degrees
degree_df = pd.DataFrame.from_dict(dict(G.degree()), orient='index', columns=['Degree'])
degree_df.groupby(by=['Degree']).count()
degree_df['ones'] = 1
degree_df.groupby('Degree').count().plot.bar()
plt.show()

### Unweighted network
## I.1. Centrality measures
# Create a dataframe of nodes with centrality measures
cdf = pd.DataFrame.from_dict(nx.betweenness_centrality(G), orient='index', columns=['betweeness'])
cdf['eigen'] = nx.eigenvector_centrality(G, max_iter=10000)
cdf['closeness'] = nx.closeness_centrality(G)

# get the top 10 in each centrality measure
cdf['betweeness'].sort_values(ascending=False).head(10)
cdf['eigen'].sort_values(ascending=False).head(10)
cdf['closeness'].sort_values(ascending=False).head(10)

# Plot nodes with centrality as size and color
centrality_to_plot = cdf['eigen']

fig, ax = plt.subplots(dpi=500, layout='compressed', frameon=False)
edg = nx.draw_networkx_edges(G, pos, width=0.2, arrows=False)
nx.draw_networkx_nodes(G, pos=pos, node_color=centrality_to_plot, node_size=centrality_to_plot * 1000, edgecolors='w',
                       linewidths=0.2, cmap='copper')
for i, node in enumerate(centrality_to_plot.sort_values(ascending=False).head(10).index.values):
    ax.annotate(i + 1, xy=nx.get_node_attributes(G, 'coords')[node], size=10, c='w', ha='center', va='center')
plt.xlim([519600, 543000])
plt.ylim([176000, 189000])
plt.axis("off")
plt.show()

# check how they are correlated with each other
cdf.corr()

## I.2. Impact measures:
# Average shortest path
full_avg_sp = nx.average_shortest_path_length(G)
print(full_avg_sp)

# Global Average Clustering coeffecient
full_avg_ce = nx.average_clustering(G)
print(full_avg_ce)

# I.3. Node removal:

# Adjust the centrality to take weight into account weight in order to pass as argument to the node removal function
betweeness_weighted = partial(nx.betweenness_centrality, weight='weight')
closeness_weighted = partial(nx.closeness_centrality, distance='weight')
# Adjust the centrality to raise default finding eigenvector iterations
high_iter_eigenvector = partial(nx.eigenvector_centrality, max_iter=10000)
# Adjust the average_shortest_path_length to take weight into account
av_sp = partial(nx.average_shortest_path_length, weight='weight')

# All is done through the "node_removal" function that takes arguments:
# 1- centrality
# 2- global measure
# 3- if sequential removing to be used
# results will be saved in this dataframe in
nrdfs = []
for seq in [False, True]:
    for measure in [av_sp, nx.average_clustering]:
        nrdf = pd.DataFrame(index=range(1, 11))
        for centrality in [betweeness_weighted, closeness_weighted, high_iter_eigenvector]:
            nrdf = nrdf.merge(node_removal(G, centrality, measure, seq=seq), left_index=True, right_index=True,
                              suffixes=('', f'_{centrality=}'[:4]))
        nrdfs.append(nrdf)

len(nrdfs)
# Average Shortest Path Non-sequential removal:
print(nrdfs[0])
nrdfs[0].to_csv('tables\\Average_sp_nonseq.csv')
# Average Clustering coefficient Non-sequential removal:
print(nrdfs[1])
nrdfs[1].to_csv('tables\\Average_cl_nonseq.csv')
# Average Shortest Path sequential removal:
print(nrdfs[2])
nrdfs[2].to_csv('tables\\Average_sp_seq.csv')
# Average Clustering coefficient sequential removal:
print(nrdfs[3])
nrdfs[3].to_csv('tables\\Average_cl_seq.csv')

### Flow weighted network

# Getting the flows between stations dataframe for recalcuating the edge flows when needed
london_OD_AMpeak = pd.read_csv('london_flows.csv')

# add inverted flows as weight to be used as distance in some of the centrality as needed
inverted_flows = {k: 1 / v if v != 0 else 1 for k, v in nx.get_edge_attributes(G, 'flows_Di').items()}
nx.set_edge_attributes(G, inverted_flows, 'inv_flows')

## II.1. Centrality measures
cdfw = pd.DataFrame.from_dict(nx.betweenness_centrality(G, weight='inv_flows'), orient='index', columns=['betweeness'])
cdfw['eigen'] = nx.eigenvector_centrality(G, max_iter=10000, weight='flows_Di')
# calculating node "utilisation"
cdfw['utilisation'] = utilisation_centrality(G)

# get the top 10 in each centrality measure
cdfw['betweeness'].sort_values(ascending=False).head(10)
cdfw['eigen'].sort_values(ascending=False).head(10)
cdfw['utilisation'].sort_values(ascending=False).head(10)

## Node removal on the flow weighted network

# adjust the centralites to include flows
high_iter_eigenvector_flow = partial(nx.eigenvector_centrality, max_iter=10000, weight='flows_Di')
betweeness_flow = partial(nx.betweenness_centrality, weight='inv_flows')

# calculate the sequential removal for the new flow wighted centralities
# seq_cent_flwo calculate the top nodes and recalculate edge flow after each iteration
node_list_bet_flow = seq_cent_flwo(G, betweeness_flow, london_OD_AMpeak)
node_list_eig_flow = seq_cent_flwo(G, high_iter_eigenvector_flow, london_OD_AMpeak)
node_list_utilisation_flow = seq_cent_flwo(G, utilisation_centrality, london_OD_AMpeak)

# applying the node removal with the flows using the "node_removal_flows" function
# the function takes the graph and list of nodes to remove and dataframe of the flows between stations "OD table"
nr_bet_geo = node_removal_flows(G, seq_cent(G, betweeness_weighted), london_OD_AMpeak)
nr_bet_geo.to_csv('tables\\flows\\betweeness.csv')

nr_eig_geo = node_removal_flows(G, seq_cent(G, high_iter_eigenvector), london_OD_AMpeak)
nr_eig_geo.to_csv('tables\\flows\\eigen.csv')

nr_clo_geo = node_removal_flows(G, seq_cent(G, closeness_weighted), london_OD_AMpeak)
nr_clo_geo.to_csv('tables\\flows\\closeness.csv')

nr_bet_flow = node_removal_flows(G, node_list_bet_flow, london_OD_AMpeak)
nr_bet_flow.to_csv('tables\\flows\\betweeness_flow.csv')

nr_eig_flow = node_removal_flows(G, node_list_eig_flow, london_OD_AMpeak)
nr_eig_flow.to_csv('tables\\flows\\eigen_flow.csv')

nr_flow_flow = node_removal_flows(G, node_list_utilisation_flow, london_OD_AMpeak)
nr_flow_flow.to_csv('tables\\flows\\utilisation_flow.csv')
