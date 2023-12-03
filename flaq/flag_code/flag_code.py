from itertools import combinations
import sys
from typing import List, Set, Tuple, Union

from ldpc.mod2 import rank
from ldpc.codes import ring_code
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from panqec.codes import Toric2DCode, Toric3DCode, Planar2DCode, Color666PlanarCode

from flaq.utils import get_all_logicals
from flaq.chain_complex import HypergraphComplex

sys.setrecursionlimit(10000)


class FlagCode:
    def __init__(
        self,
        boundary_operators: List[np.ndarray],
        cell_positions: List[List[Tuple]] = None,
        x: int = 1,
        z: int = 1,
        add_boundary_pins=True
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
        for H in boundary_operators:
            if not isinstance(H, np.ndarray):
                raise ValueError("The matrices should be given as numpy arrays")

            if isinstance(H, np.matrix):
                raise ValueError("The matrices should be arrays, not np.matrix objects")

        self.boundary_operators = boundary_operators
        self.cell_positions = cell_positions
        self.dimension = len(self.boundary_operators)
        self.n_levels = self.dimension + 1
        self.n_colors = self.n_levels
        self.all_colors = list(range(1, self.n_colors+1))
        self.x = x
        self.z = z

        if x + z > self.dimension:
            raise ValueError(
                f"x+z must be below the dimension. "
                f"Currently {x}+{z} > {self.dimension}"
            )

        for i in range(self.dimension-1):
            if self.boundary_operators[i].shape[1] != self.boundary_operators[i+1].shape[0]:
                raise ValueError(
                    f"Incorrect dimensions for input adjacency matrices {i} and {i+1}"
                )

        self.has_boundary_vertex = False
        self.has_boundary_cell = False

        if add_boundary_pins:
            self.add_boundary_pins()

        self.n_cells = [boundary_operators[0].shape[0]]
        for i in range(self.dimension):
            self.n_cells.append(boundary_operators[i].shape[1])

        self.flags = []
        self.flag_to_index = dict()
        self.flag_coordinates = []

        self._Hx: np.ndarray = None
        self._Hz: np.ndarray = None
        self._k: int = None
        self._d_x: int = None
        self._d_z: int = None
        self._d: int = None
        self._x_logicals: np.ndarray = None
        self._z_logicals: np.ndarray = None

        self.construct_graph(show=False)

    def add_boundary_pins(self):
        # print(self.boundary_operators[0].shape)
        new_column = np.array([np.sum(self.boundary_operators[-1], axis=1) % 2]).T

        if not np.all(new_column == 0):
            self.has_boundary_vertex = True
            self.boundary_operators[-1] = np.hstack([
                self.boundary_operators[-1],
                new_column
            ])

        new_row = np.sum(self.boundary_operators[0], axis=0) % 2

        if not np.all(new_row == 0):
            self.has_boundary_cell = True
            self.boundary_operators[0] = np.vstack([
                self.boundary_operators[0],
                new_row
            ])

        if self.cell_positions is not None:
            eps1, eps2 = np.random.random(), np.random.random()
            if self.has_boundary_vertex:
                self.cell_positions[0] = np.vstack([self.cell_positions[0], [eps1, eps1]])
            if self.has_boundary_cell:
                self.cell_positions[-1] = np.vstack([self.cell_positions[-1], [eps2, eps2]])

    def construct_graph(self, show=False):
        """Find all flags from the chain complex and construct the graph"""

        # Find all the flags recursively

        def get_completed_flags(flag_beginning: Tuple[int]):
            if len(flag_beginning) == self.n_levels:
                return [flag_beginning]

            completedFlags = []
            cells = self.boundary_operators[len(flag_beginning)-1][flag_beginning[-1]]
            for cell in cells.nonzero()[0]:
                for flag in get_completed_flags((*flag_beginning, cell)):
                    completedFlags.append(flag)

            return completedFlags

        for v0 in range(self.n_cells[0]):
            flags = get_completed_flags((v0,))
            for flag in flags:
                self.flag_to_index[flag] = len(self.flags)
                self.flags.append(flag)

                if self.cell_positions is not None:
                    cell_coords = [self.cell_positions[i][flag[i]] for i in range(self.n_levels)]
                    flag_coords = np.mean(cell_coords, axis=0)
                    self.flag_coordinates.append(flag_coords)

        # Build the colored graph of flags by exploring it recursively

        def get_adjacent_flags(flag: Tuple[int], level: int):
            left_adjacent_flags = None
            right_adjacent_flags = None

            if level > 0:
                left_adjacent_flags = set(
                    self.boundary_operators[level-1][flag[level-1]].nonzero()[0]
                )

            if level < self.dimension:
                right_adjacent_flags = set(
                    self.boundary_operators[level].T[flag[level+1]].nonzero()[0]
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

        if self.cell_positions is None:
            self.flag_coordinates = nx.spring_layout(self.graph)

        if show:
            nx.draw(
                self.graph, self.flag_coordinates,
                node_color='black', node_size=200,
                edge_color=np.array(weights)-1, edge_cmap=plt.cm.tab20, width=2,
                with_labels=True, font_color='white', font_size=7
            )

            plt.show()

    def get_all_rainbow_subgraphs(k: int):
        """Get all subgraphs where each node is connected to k edges
        of k different colors
        """
        visited_nodes = set()
        rainbow_subgraphs = []

        def get_rainbow_subgraph(node: int):
            if node in visited_nodes:
                return {}

            visited_nodes.add(node)
            subgraph = {node}

            adj_nodes = self.flag_adjacency[node].nonzero()[0]
            for adj_node in adj_nodes:
                if self.flag_adjacency[node, adj_node] in colors:
                    subgraph.update(get_max_subgraph(adj_node))

            return subgraph


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

        # print("Maximal subgraphs\n", maximal_subgraphs)
        return maximal_subgraphs

    def is_pin_code_relation(self) -> bool:
        for color in self.all_colors:
            max_subgraphs = self.get_all_maximal_subgraphs({color})
            for subgraph in max_subgraphs:
                if len(subgraph) % 2 == 1:
                    print("Bad subgraph", subgraph)
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

    def get_logicals(self) -> np.ndarray:
        logicals = get_all_logicals(self.Hx, self.Hz, self.k)

        self._x_logicals = logicals['X']
        self._z_logicals = logicals['Z']

        return logicals

    @property
    def x_logicals(self) -> np.ndarray:
        if self._x_logicals is None:
            self.get_logicals()

        return self._x_logicals

    @property
    def z_logicals(self) -> np.ndarray:
        if self._z_logicals is None:
            self.get_logicals()

        return self._z_logicals

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

    @property
    def n(self):
        return self.n_flags

    @property
    def k(self):
        if self._k is None:
            self._k = self.n - rank(self.Hx) - rank(self.Hz)

        return self._k

    @property
    def d_x(self):
        if self._d_x is None:
            self._d_x = np.min(np.sum(self.x_logicals, axis=1))

        return self._d_x

    @property
    def d_z(self):
        if self._d_z is None:
            self._d_z = np.min(np.sum(self.z_logicals, axis=1))

        return self._d_z

    @property
    def d(self):
        if self._d is None:
            self._d = min(self.d_x, self.d_z)

        return self._d

    def x_stabilizer_weights(self):
        weights, count = np.unique(np.sum(self.Hx, axis=1), return_counts=True)

        return {w: c for w, c in zip(weights, count)}

    def z_stabilizer_weights(self):
        weights, count = np.unique(np.sum(self.Hz, axis=1), return_counts=True)

        return {w: c for w, c in zip(weights, count)}

    def is_valid_css(self):
        return np.all((self.Hx @ self.Hz.T) % 2 == 0)

    def is_triorthogonal(self, pauli='X'):
        H = self.Hx
        L = self.x_logicals

        # Check stabilizer triorthogonality
        for i1 in range(H.shape[0]):
            for i2 in range(H.shape[0]):
                for i3 in range(H.shape[0]):
                    if np.sum(H[i1] * H[i2] * H[i3]) % 2 != 0:
                        print("Not triorthogonal at", i1, i2, i3)
                        print(self.Hx[i1].nonzero()[0])
                        print(self.Hx[i2].nonzero()[0])
                        print(self.Hx[i3].nonzero()[0])
                        return False

        print("It is stabilizer triorthogonal")

        # Check one-logical triorthogonality
        for i1 in range(H.shape[0]):
            for i2 in range(H.shape[0]):
                for i3 in range(L.shape[0]):
                    if np.sum(H[i1] * H[i2] * L[i3]) % 2 != 0:
                        return False

        print("It is one-logical triorthogonal")

        # Check one-logical triorthogonality
        for i1 in range(H.shape[0]):
            for i2 in range(L.shape[0]):
                for i3 in range(L.shape[0]):
                    if np.sum(H[i1] * L[i2] * L[i3]) % 2 != 0:
                        return False

        print("It is two-logical triorthogonal")

        return True


def generate_random_parity_check_matrix(n_rows=3, n_cols=4):
    matrix = []

    while len(matrix) < n_cols:
        col = np.random.choice([0, 1], n_rows)
        if np.sum(col) % 2 == 0:
            matrix.append(col)

    return np.transpose(matrix)


if __name__ == "__main__":
    # code = Toric2DCode(4)
    # boundary_operators = [
    #     np.array(code.Hz.todense()),
    #     np.array(code.Hx.todense().T)
    # ]
    # print(boundary_operators)
    positions = None

    # positions = [
    #     np.array(code.stabilizer_coordinates)[code.z_indices],
    #     code.qubit_coordinates,
    #     np.array(code.stabilizer_coordinates)[code.x_indices]
    # ]

    H = ring_code(3)

    H = np.zeros((1, 1))
    while rank(H) == 0:
        H = generate_random_parity_check_matrix(6, 8)

    new_col = np.array([np.sum(H, axis=1) % 2]).T
    if not np.all(new_col == 0):
        H = np.hstack([H, new_col])

    print(H)

    complex = HypergraphComplex([H, H])
    boundary_operators = complex.boundary_operators
    positions = None

    flag_code = FlagCode(
        boundary_operators,
        positions,
        x=1,
        z=1,
        add_boundary_pins=False
    )

    # flag_code.get_all_maximal_subgraphs([0, 2])
    print(f"Is it a valid pin code? {flag_code.is_pin_code_relation()}")
    print(f"Is it a valid CSS code? {flag_code.is_valid_css()}")
    print(f"n: {flag_code.n}")
    print(f"k: {flag_code.k}")
    print("X stabilizer weights", flag_code.x_stabilizer_weights())
    print("Z stabilizer weights", flag_code.z_stabilizer_weights())
    print(f"X Logical: {flag_code.x_logicals}")
    print(f"Z Logical: {flag_code.z_logicals}")
    print(f"Distance: {flag_code.d}")
    print(f"Is triorthogonal: {flag_code.is_triorthogonal()}")
