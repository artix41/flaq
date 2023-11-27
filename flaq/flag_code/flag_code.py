from typing import List, Set, Tuple

import numpy as np
import matplotlib.pyplot as plt

import networkx as nx
from panqec.codes import Toric2DCode, Planar2DCode, Color666PlanarCode


class FlagCode:
    def __init__(self, boundary_operators: List[np.ndarray], cell_positions=None):
        self.boundary_operators = boundary_operators
        self.cell_positions = cell_positions
        self.dimension = len(self.boundary_operators)

        for i in range(self.dimension-1):
            if self.boundary_operators[i].shape[1] != self.boundary_operators[i+1].shape[0]:
                raise ValueError(
                    f"Incorrect dimensions for input adjacency matrices {i} and {i+1}"
                )

        self.n_cells = [boundary_operators[0].shape[0]]
        for i in range(self.dimension):
            self.n_cells.append(boundary_operators[i].shape[1])

        # Find all the flags recursively

        self.flags = []
        self.flag_to_index = dict()
        self.flag_coordinates = []

        def get_completed_flags(flag_beginning: Tuple[int]):
            if len(flag_beginning) == self.dimension+1:
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
                cell_coords = [cell_positions[i][flag[i]] for i in range(self.dimension+1)]
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
                newCells = right_adjacent_flags
            elif right_adjacent_flags is None:
                newCells = left_adjacent_flags
            else:
                newCells = left_adjacent_flags.intersection(right_adjacent_flags)

            adjacent_flags = [(*flag[:level], cell, *flag[level+1:]) for cell in newCells]

            return adjacent_flags

        self.flag_adjacency = np.zeros((len(self.flags), len(self.flags)), dtype=np.uint8)
        explored_flags = set()

        def explore_graph_from_flag(flag: Tuple[int]):
            if flag in explored_flags:
                return

            explored_flags.add(flag)

            for level in range(self.dimension+1):
                for next_flag in get_adjacent_flags(flag, level):
                    if next_flag != flag:
                        # print("Adj flags", nextFlag)
                        idx1, idx2 = self.flag_to_index[flag], self.flag_to_index[next_flag]
                        self.flag_adjacency[idx1, idx2] = level + 1
                        self.flag_adjacency[idx2, idx1] = level + 1

                        explore_graph_from_flag(next_flag)

        explore_graph_from_flag(self.flags[0])

        self.graph = nx.from_numpy_array(self.flag_adjacency)

        edges, weights = zip(*nx.get_edge_attributes(self.graph, 'weight').items())

        nx.draw(
            self.graph, self.flag_coordinates, node_color='black', edge_color=np.array(weights)-1,
            edge_cmap=plt.cm.tab20, node_size=200, width=2, with_labels=True, font_color='white',
            font_size=7
        )

        plt.show()

    def get_all_maximal_subgraphs(self, colors: Set[int]) -> List[Set[int]]:
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
                maximal_subgraphs.append(get_max_subgraph(node))

        print("Maximal subgraphs\n", maximal_subgraphs)
        return maximal_subgraphs

    def is_pin_code_relation(self) -> bool:
        for excluded_color in range(self.dimension+1):
            colors = set(range(self.dimension+1)) - {excluded_color}

            max_subgraphs = self.get_all_maximal_subgraphs(colors)
            for subgraph in max_subgraphs:
                if len(subgraph) % 2 == 1:
                    return False

        return True


if __name__ == "__main__":
    code = Toric2DCode(3)
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
    flag_code.get_all_maximal_subgraphs([1, 2])
    print(f"Is it a pin code? {flag_code.is_pin_code_relation()}")
    # print("Number of flags", len(flagCode.flags))
