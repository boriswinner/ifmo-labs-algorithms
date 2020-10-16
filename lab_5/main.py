import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from constants import EDGES, VERTEXES
from algo import dfs_init, dfs, bfs

def create_adjacency_matrix():
    matrix = np.zeros((VERTEXES, VERTEXES))
    current_edges_number = 0
    while current_edges_number < EDGES:
        i, j = np.random.randint(0, VERTEXES, 2)
        if (i != j) and (matrix[i][j] == 0):
            matrix[i][j] = 1
            matrix[j][i] = 1
            current_edges_number += 1
    return matrix

def adjacency_matrix_to_list(m):
    array = []
    for i in range(VERTEXES):
        array.append([])
        for j in range(VERTEXES):
            if m[i][j] == 1:
                array[-1].append(j)
    return array

def print_matrix(m):
    print("__", end=" ")
    for i in range(VERTEXES):
        print(str(i).zfill(2), end=" ")
    print()
    for i in range(VERTEXES):
        print(str(i).zfill(2), end=" ")
        for j in range(VERTEXES):
            print(str(int(m[i][j])).zfill(2), end=" ")
        print()

def print_list(l):
    for i in range(VERTEXES):
        print(str(i).zfill(2), end="->")
        for j in l[i]:
            print(str(j).zfill(2), end=" ")
        print()

def print_graph(g):
    gr = nx.Graph()
    for i in range(VERTEXES):
        gr.add_node(i)
    for i in range(VERTEXES):
        for j in g[i]:
            gr.add_edge(i, j)
    nx.draw(gr, with_labels=True)
    plt.show()
    return gr

m = create_adjacency_matrix()
print_matrix(m)
l = adjacency_matrix_to_list(m)
print_list(l)
gr = print_graph(l)
dfs_init(l)
bfs(l, 5)