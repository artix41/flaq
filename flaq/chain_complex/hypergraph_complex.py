import functools as ft
from itertools import combinations
from typing import List

from ldpc.codes import ring_code
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.sparse import csr_array


class HypergraphComplex:
    def __init__(self, parity_check_matrices: List[np.ndarray]):
        for H in parity_check_matrices:
            if not isinstance(H, np.ndarray):
                raise ValueError("The matrices should be given as numpy arrays")
        self.parity_check_matrices = parity_check_matrices
        self.dimension = len(parity_check_matrices)
        self.space_dim = [[H.shape[1], H.shape[0]] for H in parity_check_matrices]

        self.boundary_operators = self.construct_boundary_operators()

        print("d_0", np.unique(np.sum(self.boundary_operators[0], axis=0)))
        print("d_{-1}", np.unique(np.sum(self.boundary_operators[-1], axis=1)))

        if not self.is_valid_complex():
            raise ValueError("Not a valid complex")

    def is_valid_complex(self):
        for i in range(len(self.boundary_operators)-1):
            if not np.all((self.boundary_operators[i] @ self.boundary_operators[i+1]) % 2 == 0):
                return False

        return True

    def construct_boundary_operators(self):
        boundary_operators = []
        space_labels = []
        for k in range(self.dimension+1):
            list_indices = combinations(range(self.dimension), k)
            space_labels.append([
                np.array([1 if i in indices else 0 for i in range(self.dimension)])
                for indices in list_indices
            ])

        for k in range(self.dimension):
            blocks = np.zeros((len(space_labels[k+1]), len(space_labels[k]))).tolist()
            for i in range(len(space_labels[k])):
                for j in range(len(space_labels[k+1])):
                    diff_indices = ((space_labels[k][i] + space_labels[k+1][j]) % 2).nonzero()[0]
                    if len(diff_indices) > 1:
                        n_rows = np.prod([
                            self.space_dim[m][space_labels[k+1][j][m]]
                            for m in range(self.dimension)
                        ])
                        n_cols = np.prod([
                            self.space_dim[m][space_labels[k][i][m]]
                            for m in range(self.dimension)
                        ])
                        block = np.zeros((n_rows, n_cols))

                    else:
                        non_triv_index = diff_indices[0]
                        ops = [
                            np.eye(self.space_dim[m][space_labels[k][i][m]])
                            for m in range(self.dimension)
                        ]
                        ops[non_triv_index] = self.parity_check_matrices[non_triv_index]
                        block = ft.reduce(np.kron, ops)

                    blocks[j][i] = block

            boundary_op = np.array(np.block(blocks), dtype='uint8')
            boundary_operators.append(boundary_op.T)

        return boundary_operators

    def draw_tanner_graph(self, index_boundary_operator=0):
        graph = nx.algorithms.bipartite.from_biadjacency_matrix(
            csr_array(self.boundary_operators[index_boundary_operator])
        )
        coordinates = nx.spring_layout(graph)
        print(graph._node)

        nx.draw(
            graph, coordinates,
            node_color='black', node_size=200, width=2,
            with_labels=True, font_color='white', font_size=7
        )

        plt.show()


if __name__ == "__main__":
    H = ring_code(2)
    complex = HypergraphComplex([H, H])
    complex.draw_tanner_graph(1)
