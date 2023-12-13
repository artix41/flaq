from abc import ABC, abstractmethod

from typing import List

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from pyvis.network import Network
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

    def draw_tanner_graph(self, index_boundary_operator=0, notebook=False):
        graph = nx.algorithms.bipartite.from_biadjacency_matrix(
            csr_array(self.boundary_operators[index_boundary_operator])
        )
        node_type = [graph.nodes[i]['bipartite'] for i in range(len(graph.nodes))]

        graph = nx.convert_node_labels_to_integers(graph)

        nt = Network(notebook=notebook, cdn_resources='remote')
        nt.from_nx(graph)

        for edge in nt.get_edges():
            edge['width'] = 7

        for node_id in nt.get_nodes():
            node = nt.get_node(node_id)
            node['label'] = str(node['id'])
            node['shape'] = ['box', 'circle'][node_type[node_id]]
            if not node_type[node_id]:
                node['color'] = 'orange'

            node['font'] = {'size': 45}

        nt.show('nx.html')

        return nt
