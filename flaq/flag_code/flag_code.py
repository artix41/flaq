from itertools import combinations
from typing import List, Set, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt

import networkx as nx
from panqec.codes import Toric2DCode, Planar2DCode, Color666PlanarCode


class FlagCode:
    def __init__(
        self,
        boundary_operators: List[np.ndarray],
        cell_positions: List[List[Tuple]] = None,
        x: int = 1,
        z: int = 1
    ):
        """Flag code constructor

        Parameters
        ----------
        boundary_operators : List[np.ndarray]
            List of boundary operators for the input D-dimensional chain complex
        cell_positions : _type_, optional
            , by default None
        x : int, optional
            Size of Pauli X pinned set types (i.e. x in x-pinned set), by default 1
        z : int, optional
            Size of Pauli Z pinned set types (i.e. z in z-pinned set), by default 2
        """
        self.boundary_operators = boundary_operators
        self.cell_positions = cell_positions
        self.dimension = len(self.boundary_operators)
        self.n_levels = self.dimension + 1
        self.n_colors = self.n_levels
        self.all_colors = list(range(1, self.n_colors+1))
        self.x = x
        self.z = z

        for i in range(self.dimension-1):
            if self.boundary_operators[i].shape[1] != self.boundary_operators[i+1].shape[0]:
                raise ValueError(
                    f"Incorrect dimensions for input adjacency matrices {i} and {i+1}"
                )

        self.n_cells = [boundary_operators[0].shape[0]]
        for i in range(self.dimension):
            self.n_cells.append(boundary_operators[i].shape[1])

        self.flags = []
        self.flag_to_index = dict()
        self.flag_coordinates = []

        self._Hx: np.ndarray = None
        self._Hz: np.ndarray = None

        self.construct_graph(show=True)

    def construct_graph(self, show=False):
        """Find all flags from the chain complex and construct the graph"""

        # Find all the flags recursively

        def get_completed_flags(flag_beginning: Tuple[int]):
            if len(flag_beginning) == self.n_levels:
                return [flag_beginning]

            completedFlags = []
            for cell in boundary_operators[len(flag_beginning)-1][flag_beginning[-1]].nonzero()[1]:
                for flag in get_completed_flags((*flag_beginning, cell)):
                    completedFlags.append(flag)

            return completedFlags

        for v0 in range(self.n_cells[0]):
            flags = get_completed_flags((v0,))
            for flag in flags:
                self.flag_to_index[flag] = len(self.flags)
                self.flags.append(flag)
                cell_coords = [self.cell_positions[i][flag[i]] for i in range(self.n_levels)]
                flag_coords = np.mean(cell_coords, axis=0)
                self.flag_coordinates.append(flag_coords)

        # Build the colored graph of flags by exploring it recursively

        def get_adjacent_flags(flag: Tuple[int], level: int):
            left_adjacent_flags = None
            right_adjacent_flags = None

            if level > 0:
                left_adjacent_flags = set(
                    self.boundary_operators[level-1][flag[level-1]].nonzero()[1]
                )

            if level < self.dimension:
                right_adjacent_flags = set(
                    self.boundary_operators[level].T[flag[level+1]].nonzero()[1]
                )

            if left_adjacent_flags is None:
                new_cells = right_adjacent_flags
            elif right_adjacent_flags is None:
                new_cells = left_adjacent_flags
            else:
                new_cells = left_adjacent_flags.intersection(right_adjacent_flags)

            adjacent_flags = [(*flag[:level], cell, *flag[level+1:]) for cell in new_cells]

            return adjacent_flags

        self.n_flags = len(self.flags)
        self.flag_adjacency = np.zeros((self.n_flags, self.n_flags), dtype=np.uint8)
        explored_flags = set()

        def explore_graph_from_flag(flag: Tuple[int]):
            if flag in explored_flags:
                return

            explored_flags.add(flag)

            for level in range(self.n_levels):
                for next_flag in get_adjacent_flags(flag, level):
                    if next_flag != flag:
                        idx1, idx2 = self.flag_to_index[flag], self.flag_to_index[next_flag]
                        self.flag_adjacency[idx1, idx2] = level + 1
                        self.flag_adjacency[idx2, idx1] = level + 1

                        explore_graph_from_flag(next_flag)

        explore_graph_from_flag(self.flags[0])

        self.graph = nx.from_numpy_array(self.flag_adjacency)

        edges, weights = zip(*nx.get_edge_attributes(self.graph, 'weight').items())

        if show:
            nx.draw(
                self.graph, self.flag_coordinates,
                node_color='black', node_size=200,
                edge_color=np.array(weights)-1, edge_cmap=plt.cm.tab20, width=2,
                with_labels=True, font_color='white', font_size=7
            )

            plt.show()

    def get_all_maximal_subgraphs(
        self,
        colors: Set[int],
        return_arrays: bool = False
    ) -> Union[List[Set[int]], List[np.ndarray]]:
        visited_nodes = set()
        maximal_subgraphs = []

        def get_max_subgraph(node):
            if node in visited_nodes:
                return {}

            visited_nodes.add(node)
            subgraph = {node}

            adj_nodes = self.flag_adjacency[node].nonzero()[0]
            for adj_node in adj_nodes:
                if self.flag_adjacency[node, adj_node] in colors:
                    subgraph.update(get_max_subgraph(adj_node))

            return subgraph

        for node in range(len(self.flag_adjacency)):
            if node not in visited_nodes:
                subgraph_set = get_max_subgraph(node)

                if return_arrays:
                    subgraph_array = np.zeros(self.n_flags, dtype='uint8')
                    subgraph_array[list(subgraph_set)] = 1
                    maximal_subgraphs.append(subgraph_array)
                else:
                    maximal_subgraphs.append(subgraph_set)

        print("Maximal subgraphs\n", maximal_subgraphs)
        return maximal_subgraphs

    def is_pin_code_relation(self) -> bool:
        for color in self.all_colors:
            max_subgraphs = self.get_all_maximal_subgraphs({color})
            for subgraph in max_subgraphs:
                if len(subgraph) % 2 == 1:
                    return False

        return True

    def get_stabilizers(self, n_pinned: int) -> np.ndarray:
        max_subgraphs = []
        for pinned_colors in combinations(self.all_colors, n_pinned):
            free_colors = set(self.all_colors) - set(pinned_colors)

            max_subgraphs.extend(
                self.get_all_maximal_subgraphs(free_colors, return_arrays=True)
            )

        return np.array(max_subgraphs)

    @property
    def Hx(self):
        if self._Hx is None:
            self._Hx = self.get_stabilizers(self.x)

        return self._Hx

    @property
    def Hz(self):
        if self._Hz is None:
            self._Hz = self.get_stabilizers(self.z)

        return self._Hz

    def is_valid_css(self):
        return np.all((self.Hx @ self.Hz.T) % 2 == 0)


if __name__ == "__main__":
    code = Toric2DCode(2)
    boundary_operators = [
        code.Hz.todense(),
        code.Hx.todense().T
    ]
    positions = [
        np.array(code.stabilizer_coordinates)[code.z_indices],
        code.qubit_coordinates,
        np.array(code.stabilizer_coordinates)[code.x_indices]
    ]

    flag_code = FlagCode(boundary_operators, positions)
    # flag_code.get_all_maximal_subgraphs([0, 2])
    print(f"Is it a valid pin code? {flag_code.is_pin_code_relation()}")
    print(f"Is it a valid CSS code? {flag_code.is_valid_css()}")
