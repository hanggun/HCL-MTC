import networkx as nx
import codecs
import numpy as np

G = nx.DiGraph()
with codecs.open('./data/rcv1.taxonomy', 'r', encoding='utf-8') as f:
    for idx, line in enumerate(f):
        if idx > 0:
            line = line.split()
            root = line.pop(0)
            for node in line:
                G.add_edge(root, node)
print(G.edges)
array = np.array(nx.to_numpy_matrix(G))
np.fill_diagonal(array, 1)
print(1)
