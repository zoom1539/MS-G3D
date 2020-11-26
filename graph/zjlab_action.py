import sys
sys.path.insert(0, '')
sys.path.extend(['../'])

import numpy as np

from graph import tools

# Joint index:
# {0,  "Nose"}
# {1,  "LEye"},
# {2,  "REye"},
# {3,  "LEar"},
# {4,  "REar"},
# {5,  "LShoulder"},
# {6,  "RShoulder"},
# {7,  "LElbow"},
# {8,  "RElbow"},
# {9,  "LWrist"},
# {10, "RWrist"},
# {11, "LHip"},
# {12, "RHip"},
# {13, "LKnee"},
# {14, "Rknee"},
# {15, "LAnkle"},
# {16, "RAnkle"},
# {17, "Neck"},

num_node = 18
self_link = [(i, i) for i in range(num_node)]
inward = [(10, 8), (8, 6), (9, 7), (7, 5), (15, 13), (13, 11), (16, 14), (14, 12),
          (11, 5), (12, 6), (5, 17), (6, 17), (0, 17), (1, 0), (2, 0), (3, 1),
          (4, 2)]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward


class AdjMatrixGraph:
    def __init__(self, *args, **kwargs):
        self.num_nodes = num_node
        self.edges = neighbor
        self.self_loops = [(i, i) for i in range(self.num_nodes)]
        self.A_binary = tools.get_adjacency_matrix(self.edges, self.num_nodes)
        self.A_binary_with_I = tools.get_adjacency_matrix(self.edges + self.self_loops, self.num_nodes)


if __name__ == '__main__':
    graph = AdjMatrixGraph()
    A_binary = graph.A_binary
    import matplotlib.pyplot as plt
    print(A_binary)
    plt.matshow(A_binary)
    plt.show()
