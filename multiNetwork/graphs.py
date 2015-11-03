import networkx as nx
import operator
import pandas as pd
import numpy as np


class create_graph:

    def __init__(self):
        data = 'csv/Forex.csv'
        headers = ['start', 'end', 'weights']
        self.Network = pd.read_csv(data, skiprows=1, names=headers)
        self.graph = np.asarray(self.Network[['start', 'end']])
        self.nodes = np.unique(self.graph)
        self.weights = list(map(float, self.Network['weights']))

    def networkList(self):
        G = nx.DiGraph()
        for node in self.nodes:
            G.add_node(node)

        self.graph_l = []
        for edge in self.graph:
            G.add_edge(edge[0], edge[1])
            self.graph_l.append((edge[0], edge[1]))

        labels = dict(list(zip(self.graph_l, self.weights)))
        return G, labels


class graphStats:

    def calculate_degree_centrality(self, graph):
        dgc_key = []
        dgc_value = []
        g = graph
        dc = nx.degree_centrality(g)
        nx.set_node_attributes(g, 'degree_cent', dc)
        degcent_sorted = sorted(dc.items(), key=operator.itemgetter(1), reverse=True)
        for key, value in degcent_sorted:
            dgc_key.append(str(key))
            dgc_value.append(value)
        return dgc_key, dgc_value

    def calculate_betweenness_centrality(self, graph):
        btc_key = []
        btc_value = []
        g = graph
        bc = nx.betweenness_centrality(g)
        betcent_sorted = sorted(bc.items(), key=operator.itemgetter(1), reverse=True)
        for key, value in betcent_sorted:
            btc_key.append(str(key))
            btc_value.append(value)
        return btc_key, btc_value
