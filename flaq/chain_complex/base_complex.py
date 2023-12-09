from abc import ABC, abstractmethod

from typing import List

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.sparse import csr_array


class BaseComplex(ABC):
    def __init__(self, sanity_check=True):
        self._boundary_operators = None

        if sanity_check and not self.is_valid_complex():
            raise ValueError("Not a valid complex")

    def is_valid_complex(self):
        for i in range(len(self.boundary_operators)-1):
            if not np.all((self.boundary_operators[i] @ self.boundary_operators[i+1]) % 2 == 0):
                return False

        return True

    @abstractmethod
    def construct_boundary_operators(self) -> List[np.ndarray]:
        """Construct boundary operators of the chain complex

        Returns
        -------
        List[np.ndarray]
            Boundary matrix for each level
        """
        pass

    @property
    def boundary_operators(self) -> List[np.ndarray]:
        if self._boundary_operators is None:
            self._boundary_operators = self.construct_boundary_operators()

        return self._boundary_operators

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
