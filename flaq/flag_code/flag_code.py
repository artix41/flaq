import copyreg
from itertools import combinations, product
from functools import partial
import sys
from typing import List, Set, Tuple, Union, Dict, Optional

from ldpc.mod2 import rank, nullspace
import networkx as nx
import numpy as np
from pyvis.network import Network
from p_tqdm import p_map

from flaq.utils import get_all_logicals
from flaq.algorithms import Graph, get_rainbow_subgraphs

sys.setrecursionlimit(10000)


class FlagCode:
    def __init__(
        self,
        boundary_operators: List[np.ndarray],
        cell_positions: Optional[List[List[Tuple]]] = None,
        x: int = 2,
        z: int = 2,
        add_boundary_pins: bool = True,
        stabilizer_types: Optional[Dict[str, Dict[Tuple, str]]] = None,
        reduce_logicals_iter: int = 0,
        osd_order=6,
        verbose: bool = False
    ):
        """Flag code constructor

        Parameters
        ----------
        boundary_operators : List[np.ndarray]
            List of boundary operators for the input D-dimensional chain complex
        cell_positions : List[List[Tuple]], optional
            For each level, list of positions as 2-tuples (x, y)
            for all the cells at that level. By default None
        x : int, optional
            Number of colors of X stabilizers subgraphs, by default 2
        z : int, optional
            Number of colors of Z stabilizers subgraphs, by default 2
        add_boundary_pins: bool, optional
            If True, add some boundary pins to make the 0-cells and D-cells even-weight
            (warning: it tends to increase the number of qubits by a large amount).
            By default False.
        reduce_logicals_iter: int, optional
            Number of iterations to reduce to size of the logicals
            By default: 0.
        osd_order: int, optional
            Order of the OSD algorithms to use when finding the logicals
            Default: 6
        verbose: bool, optional
            If True, print some information when running the code
        """
        for H in boundary_operators:
            if not isinstance(H, np.ndarray):
                raise ValueError("The matrices should be given as numpy arrays")

            if isinstance(H, np.matrix):
                raise ValueError("The matrices should be arrays, not np.matrix objects")

        self.boundary_operators = boundary_operators
        self.cell_positions = cell_positions
        self.add_boundary_pins = add_boundary_pins
        self.osd_order = osd_order
        self.dimension = len(self.boundary_operators)
        self.n_levels = self.dimension + 1
        self.n_colors = self.n_levels
        self.all_colors = list(range(1, self.n_colors+1))
        self.x = x
        self.z = z
        self.stabilizer_types = stabilizer_types
        self.reduce_logicals_iter = reduce_logicals_iter
        self.verbose = verbose

        if x < 2 or z < 2:
            raise ValueError("x and z must be greater than 2")
        if x > self.dimension or z > self.dimension:
            raise ValueError(
                f"x and z can't be strictly greater than the dimension of the system"
                f"(i.e. {self.dimension})"
            )
        if x + z < self.dimension+2:
            raise ValueError(
                f"x+z must be above D+2. Currently {x}+{z} < {self.dimension}+2"
            )

        for i in range(self.dimension-1):
            if self.boundary_operators[i].shape[1] != self.boundary_operators[i+1].shape[0]:
                raise ValueError(
                    f"Incorrect dimensions for input adjacency matrices {i} and {i+1}"
                )

        if self.stabilizer_types is None:
            self.stabilizer_types = {
                'X': {
                    'maximal': list(map(tuple, combinations(self.all_colors, self.x)))
                },
                'Z': {
                    'maximal': list(map(tuple, combinations(self.all_colors, self.z)))
                }
            }

        if set(self.stabilizer_types.keys()) != {'X', 'Z'}:
            raise ValueError("'stabilizer_types' should have keys 'X' and 'Z'")

        for pauli in [('X', self.x), ('Z', self.z)]:
            all_colors = (
                self.stabilizer_types[pauli[0]]['maximal']
                + self.stabilizer_types[pauli[0]]['rainbow']
            )
            if set(all_colors) != set(combinations(self.all_colors, pauli[1])):
                raise ValueError(f"'stabilizers_types['{pauli[0]}'] does not contain all colors")

        self.has_boundary_vertex = False
        self.has_boundary_cell = False

        if add_boundary_pins:
            self.construct_boundary_pins()

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

        self.construct_graph()

    def args(self):
        return (self.boundary_operators,)

    def kwargs(self):
        return {
            'cell_positions': self.cell_positions,
            'x': self.x,
            'z': self.z,
            'add_boundary_pins': self.add_boundary_pins,
            'stabilizer_types': self.stabilizer_types,
            'reduce_logical_iter': self.reduce_logicals_iter,
            'verbose': self.verbose
        }

    def log(self, *args, **kwargs):
        """Print function, conditioned on the self.verbose attribute.
        Takes all the arguments of the standard print function
        """
        if self.verbose:
            print(*args, **kwargs)

    def construct_boundary_pins(self):
        """Changes the first and last boundary operators to make all the weights even"""

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

    def construct_graph(self):
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
        self.flag_adjacency_matrix = np.zeros((self.n_flags, self.n_flags), dtype=np.uint8)
        self.flag_graph = Graph(self.n_flags)
        explored_flags = set()

        def explore_graph_from_flag(flag: Tuple[int]):
            if flag in explored_flags:
                return

            explored_flags.add(flag)

            for level in range(self.n_levels):
                for next_flag in get_adjacent_flags(flag, level):
                    if next_flag != flag:
                        idx1, idx2 = self.flag_to_index[flag], self.flag_to_index[next_flag]
                        self.flag_adjacency_matrix[idx1, idx2] = level + 1
                        self.flag_adjacency_matrix[idx2, idx1] = level + 1

                        self.flag_graph.add_edge(idx1, idx2, level + 1)

                        explore_graph_from_flag(next_flag)

        explore_graph_from_flag(self.flags[0])

        self.nx_graph = nx.from_numpy_array(self.flag_adjacency_matrix)

    def get_rainbow_subgraphs(
        self,
        colors: Union[Set[int], List[int]],
        graph: Graph = None,
        return_format: str = "set"
    ) -> Union[List[Set[int]], List[np.ndarray], List[Graph]]:
        """Get all subgraphs where each node is connected to k edges
        of k different colors.

        Parameters
        ----------
        colors : Set[int]
            Set of colors that the subgraph should have at each node
        graph: Graph, optional
            Graph on which to do the search. By default, it uses the whole flag graph.
        return_format: str
            Can be 'set' or 'array'.
            If 'set', returns each graph as the Set of its nodes.
            If 'array, returns each graph as a binary numpy array of size `n_flags`,
            with a one for all the node indices present in the subgraph.
        Returns
        -------
        Union[List[Set[int]], List[np.ndarray], List[Graph]]
            List of rainbow subgraphs, where the subgraph format is specified
            by the `return_format` argument
        """

        if graph is None:
            graph = self.flag_graph

        return get_rainbow_subgraphs(graph, colors, return_format=return_format)

    def get_all_maximal_subgraphs(
        self,
        colors: Set[int],
        graph: Graph = None,
        return_format: str = "set"
    ) -> Union[List[Set[int]], List[np.ndarray]]:
        """Get all the subgraphs where no more edge of a color present in `colors`
        can be added.

        Parameters
        ----------
        colors : Set[int]
            Set of colors that the subgraph should have at each node
        graph: Graph, optional
            Graph on which to do the search. By default, it uses the whole flag graph.
        return_format: str
            Can be 'set', 'graph' or 'array'.
            If 'set', returns each graph as the Set of its nodes.
            If 'array, returns each graph as a binary numpy array of size `n_flags`,
            with a one for all the node indices present in the subgraph.
        Returns
        -------
        Union[List[Set[int]], List[np.ndarray], List[Graph]]
            List of maximal subgraphs, where the subgraph format is specified
            by the `return_format` argument
        """

        if return_format not in ['set', 'graph', 'array']:
            raise ValueError(
                f"'return_format' must take value in ['set', 'graph', 'array'],"
                f"not {return_format}"
            )

        if graph is None:
            graph = self.flag_graph

        visited_nodes = set()
        maximal_subgraphs = []

        def get_max_subgraph(node):
            if node in visited_nodes:
                return Graph(self.n_flags)

            visited_nodes.add(node)
            subgraph = Graph(self.n_flags)

            for adj_node, color in graph[node]:
                if color in colors:
                    adj_max_subgraph = get_max_subgraph(adj_node)
                    subgraph.merge_with(adj_max_subgraph)
                    subgraph.add_edge(node, adj_node, color)

            return subgraph

        for node in graph.nodes:
            if node not in visited_nodes:
                subgraph = get_max_subgraph(node)
                if return_format == 'array':
                    subgraph_array = np.zeros(self.n_flags, dtype='uint8')
                    subgraph_array[list(subgraph.nodes)] = 1
                    maximal_subgraphs.append(subgraph_array)
                elif return_format == 'set':
                    maximal_subgraphs.append(subgraph.nodes)
                else:
                    maximal_subgraphs.append(subgraph)

        return maximal_subgraphs

    def is_pin_code_relation(self) -> bool:
        """Check whether we have our flag graph defines a valid pin code relation,
        i.e. where each D-pinned set (i.e. 1-maximal subgraph) has even weight

        Returns
        -------
        bool
            True if the flag graph defines a pin code relation
        """
        for color in self.all_colors:
            max_subgraphs = self.get_all_maximal_subgraphs({color})
            for subgraph in max_subgraphs:
                if len(subgraph) % 2 == 1:
                    return False

        return True

    def get_stabilizers(
        self,
        stabilizer_types: Dict[str, List[Tuple]] = None
    ) -> np.ndarray:
        """Get the parity-check matrix corresponding to stabilizers defined
        by k-subgraphs. The subgraphs are either rainbow or maximal,
        as specified in `stabilizer_types`.

        Parameters
        ----------
        stabilizer_types : Dict[str, List[Tuple]], optional
            Dictionary that specifies the list of tuples of free colors that
            we want to be either 'rainbow' or 'maximal' subgraphs.
            By default None ('maximal' for every subgraph)

        Returns
        -------
        np.ndarray
            Parity-check matrix of size m x n, where m is the number of stabilizers
            and n the number of qubits.
        """

        maximal_subgraphs = {}
        final_subgraphs = []

        for colors in stabilizer_types['maximal'] + stabilizer_types['rainbow']:
            maximal_subgraphs[colors] = self.get_all_maximal_subgraphs(
                colors, return_format='graph'
            )

        for colors in stabilizer_types['maximal']:
            final_subgraphs.extend([
                max_subgraph.as_array()
                for max_subgraph in maximal_subgraphs[colors]
            ])

        list_args = []
        for colors in stabilizer_types['rainbow']:
            list_args.extend([(colors, g) for g in maximal_subgraphs[colors]])

        worker_func = lambda x: get_rainbow_subgraphs(
            graph=x[1],
            colors=x[0],
            return_format='array'
        )

        results = p_map(
            worker_func,
            list_args,
            desc="Getting rainbow subgraphs"
        )

        final_subgraphs.extend([result for sublist in results for result in sublist])

        return np.array(final_subgraphs)

    def get_logicals(self, verbose: bool = False) -> Dict[str, np.ndarray]:
        """Get the low-weight X and Z logical operators of the code

        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary with two keys 'X' and 'Z', corresponding to X and Z
            logical operators. For each of them, numpy array of size k x n,
            where k is the number of logical qubits and n the number of physical
            qubits.
        """
        logicals = get_all_logicals(
            self.Hx, self.Hz, self.k, reduce_iter=self.reduce_logicals_iter,
            osd_order=self.osd_order, verbose=verbose
        )

        self._x_logicals = logicals['X']
        self._z_logicals = logicals['Z']

        self._d, self._d_x, self._d_z = None, None, None

        return logicals

    @property
    def x_logicals(self) -> np.ndarray:
        if self._x_logicals is None:
            self.get_logicals(verbose=self.verbose)

        return self._x_logicals

    @property
    def z_logicals(self) -> np.ndarray:
        if self._z_logicals is None:
            self.get_logicals(verbose=self.verbose)

        return self._z_logicals

    @property
    def Hx(self):
        if self._Hx is None:
            self._Hx = self.get_stabilizers(self.stabilizer_types['X'])

        return self._Hx

    @property
    def Hz(self):
        if self._Hz is None:
            self._Hz = self.get_stabilizers(self.stabilizer_types['Z'])

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
            self._d_x = 0
            if len(self.x_logicals) > 0:
                self._d_x = np.min(np.sum(self.x_logicals, axis=1))

        return self._d_x

    @property
    def d_z(self):
        if self._d_z is None:
            self._d_z = 0
            if len(self.z_logicals) > 0:
                self._d_z = np.min(np.sum(self.z_logicals, axis=1))

        return self._d_z

    @property
    def d(self):
        if self._d is None:
            self._d = min(self.d_x, self.d_z)

        return self._d

    def x_stabilizer_weights(self) -> Dict[int, int]:
        return get_operator_weights(self.Hx)

    def z_stabilizer_weights(self) -> Dict[int, int]:
        return get_operator_weights(self.Hz)

    def x_logical_weights(self) -> Dict[int, int]:
        return get_operator_weights(self.x_logicals)

    def z_logical_weights(self) -> Dict[int, int]:
        return get_operator_weights(self.z_logicals)

    def is_valid_css(self) -> bool:
        """Check whether the X and Z parity-check matrices commute

        Returns
        -------
        bool
            True if it is a valid CSS code with commuting X and Z
            stabilizers
        """
        return np.all((self.Hx @ self.Hz.T) % 2 == 0)

    def is_multiorthogonal(
        self,
        k: int = 3,
        pauli: str = 'X',
        levels: Optional[List[int]] = None
    ):

        if pauli == 'X':
            H = self.Hx
            L = self.x_logicals
        else:
            H = self.Hz
            L = self.z_logicals

        if levels is None:
            levels = list(range(k))

        for n_L in levels:
            indices = list(product(
                *[range(L.shape[0]) for _ in range(n_L)],
                *[range(H.shape[0]) for _ in range(k-n_L)]
            ))
            for i, index in enumerate(indices):
                if i % 100 == 0:  # used to reduce the print rate (useful in notebooks)
                    print(
                        f"{k}-orthogonality with {n_L} logicals: "
                        f"{round((i+1) / len(indices) * 100)}%",
                        end='\r'
                    )

                logical_rows = [L[i] for i in index[:n_L]]
                stab_rows = [H[i] for i in index[n_L:]]

                prod = np.prod([*logical_rows, *stab_rows], axis=0)

                if np.sum(prod) % 2 != 0:
                    print(f"Not {k}-orthogonal when including {n_L} logicals")
                    return False

            self.log(f"Multiorthogonal with {n_L} logicals")

        return True

    def is_manifold(self):
        for node in self.flag_graph.nodes:
            colors_at_node = list(map(lambda x: x[1], self.flag_graph[node]))
            if len(colors_at_node) != len(set(colors_at_node)):
                return False

        return True

    def is_unfoldable(self) -> bool:
        free_colors = set(range(1, self.dimension+1))
        free_colors = {1, 3}

        maximal_subgraphs: Set[Graph] = self.get_all_maximal_subgraphs(
            free_colors, return_format='graph'
        )

        for max_subgraph in maximal_subgraphs:
            # print("Max subgraph", max_subgraph.nodes)
            # Get all maximal 1-graphs (aka unit graphs)
            unit_graphs: List[Graph] = []
            for color in free_colors:
                unit_graphs.extend(
                    list(self.get_all_maximal_subgraphs(
                        {color}, max_subgraph, return_format='graph'
                    ))
                )

            # print("Unit graphs", [g.nodes for g in unit_graphs])
            n = len(unit_graphs)
            # print("Number of unit graphs", n)

            n_ancillas = n - len(max_subgraph.nodes)
            # print("n_ancillas", n_ancillas)

            # Get all X operators of the overlap toric code group,
            # by considering the adjacency matrix of connected unit graphs
            adj_matrix = np.zeros((n, n), dtype='uint8')
            for i in range(n):
                for j in range(n):
                    if i != j and len(unit_graphs[i].nodes & unit_graphs[j].nodes) != 0:
                        adj_matrix[i, j] = 1
                        adj_matrix[j, i] = 1

            # print("Adj matrix", adj_matrix)

            n_x_ops_tc = rank(adj_matrix)
            # print("Number of X ops TC", n_x_ops_tc)

            # Get all X operators of the overlap color code group
            unit_graph_arrays = np.array(
                [unit_graph.as_array() for unit_graph in unit_graphs]
            )

            n_x_ops_cc = n - 2*(n - rank(unit_graph_arrays))
            # print("Number of X ops CC", n_x_ops_cc)

            if n_x_ops_cc + n_ancillas != n_x_ops_tc:
                return False

            # Get the center of the overlap groups
            n_gen_center_tc = rank(nullspace(adj_matrix))

            # print("Nullspace TC", nullspace(adj_matrix))

            n_gen_center_cc = 2*rank(nullspace(unit_graph_arrays[:, list(max_subgraph.nodes)]))

            # print("n_gen_center_tc", n_gen_center_tc)
            # print("n_gen_center_cc", n_gen_center_cc)

            if n_gen_center_cc + n_ancillas != n_gen_center_tc:
                return False

        return True

    def get_syndrome(self, error: Union[List[int], Set[int]], pauli: str):
        error = list(error)
        error_vector = np.zeros(self.n, dtype='uint8')
        error_vector[error] = 1
        H = {'X': self.Hz, 'Z': self.Hx}[pauli]

        syndrome = (H.dot(error_vector) % 2).nonzero()[0]

        return syndrome

    def is_stabilizer(self, stabilizer: Union[List[int], Set[int]], pauli: str):
        stabilizer = list(stabilizer)
        stab_array = np.zeros(self.n, dtype='uint8')
        stab_array[stabilizer] = 1
        H = {'X': self.Hx, 'Z': self.Hz}[pauli]

        return rank(np.vstack([H, stab_array])) == rank(H)

    def __hash__(self):
        return hash(self.flag_graph)

    def draw(
        self,
        restricted_colors: List[int] = None,
        restricted_nodes: List[int] = None,
        restricted_depth: int = 0,
        notebook: bool = True,
        edge_width: int = 7,
        node_size: int = 45,
        colors: Dict[int, str] = None
    ) -> Network:
        """Draw the flag graph of the code using PyVis.

        Parameters
        ----------
        restricted_colors : List[int], optional
            Only edges of color present in this list will be displayed.
            By default None (no restriction).
        restricted_nodes : List[int], optional
            Only nodes of index present in this list (and potentially their neighborhood
            if `restricted_depth` > 0) will be displayed. By default None (no restriction).
        restricted_depth : int, optional
            When `restricted_nodes` is given, `restricted_depth` gives the depth of the
            neighborhood to display around those nodes. By default 0
        notebook : bool, optional
            Whether we want to display the graph in a Jupyter notebook environment,
            by default True.
        edge_width : int, optional
            Width of the edges in the visualization, by default 7
        node_size : int, optional
            Size of the nodes in the visualization, by default 45
        colors : Dict[int, str], optional
            Dictionary that associates a real color (as a string) to each flag graph color.
            By default None.

        Returns
        -------
        nt: Network
            PyVis Network object. Use nt.show('nx.html') to visualize the graph in a
            Jupyter notebook.
        """
        nt = Network(notebook=notebook, cdn_resources='in_line', filter_menu=True)
        nt.from_nx(self.nx_graph)

        if colors is None:
            colors = {1: 'blue', 2: 'green', 3: 'red', 4: 'yellow'}

        highlight_restricted_nodes = (restricted_nodes is not None)
        if restricted_colors is None:
            restricted_colors = self.all_colors

        if restricted_nodes is None:
            restricted_nodes = list(range(self.n_flags))

        displayed_nodes = self.flag_graph.get_neighborhood_nodes(
            restricted_nodes, restricted_depth
        )

        for edge in nt.get_edges():
            color_id = self.flag_adjacency_matrix[edge['from'], edge['to']]
            edge['color'] = colors[color_id]
            edge['width'] = edge_width
            if (color_id not in restricted_colors
                    or edge['from'] not in displayed_nodes
                    or edge['to'] not in displayed_nodes):
                edge['hidden'] = True
                edge['physics'] = False

        for node_id in nt.get_nodes():
            node = nt.get_node(node_id)
            node['label'] = str(node['id'])
            node['shape'] = 'circle'
            node['font'] = {'size': node_size}

            if node_id not in displayed_nodes:
                node['hidden'] = True
                node['physics'] = False

            if highlight_restricted_nodes and node_id in restricted_nodes:
                node['font']['color'] = 'red'

        nt.show_buttons(filter_=['physics'])

        nt.show('nx.html')

        return nt


def random_code(n_rows: int = 3, n_cols: int = 4):
    matrix = []

    while len(matrix) < n_cols:
        col = np.random.choice([0, 1], n_rows)
        if np.sum(col) % 2 == 0:
            matrix.append(col)

    return np.transpose(matrix)


def make_even(H: np.ndarray) -> np.ndarray:
    """Take a (classical) parity-check matrix and make its rows and columns even-weight
    by potentially adding a new row and a new column.

    Parameters
    ----------
    H : np.ndarray
       Parity-check matrix

    Returns
    -------
    np.ndarray
        Even-weight parity-check matrix
    """
    new_col = np.array([np.sum(H, axis=1) % 2]).T
    if not np.all(new_col == 0):
        H = np.hstack([H, new_col])

    new_row = np.sum(H, axis=0) % 2
    if not np.all(new_row == 0):
        H = np.vstack([H, new_row])

    return H


def get_operator_weights(operator: np.ndarray) -> Dict[int, int]:
    """Giving a parity-check or logical matrix, returns the number of
    operators (dict value) that have a given weight (keys)

    Parameters
    ----------
    operator : np.ndarray
       2D array

    Returns
    -------
    Dict[int, int]
        Weight count, i.e. number of  operators (dict values)
        that have a given weight (keys)
    """
    weights, count = np.unique(np.sum(operator, axis=1), return_counts=True)

    return {w: c for w, c in zip(weights, count)}


def reduce_flag_code(obj: FlagCode):
    return (FlagCode, (obj.args, obj.kwargs))  # Adjust the arguments as needed


copyreg.pickle(FlagCode, reduce_flag_code)


if __name__ == "__main__":
    from flaq.chain_complex import DoubleSquareComplex

    complex = DoubleSquareComplex(10, 10, periodic=True, sanity_check=True)

    stabilizer_types = {
        'X': {(1, 2): 'rainbow', (1, 3): 'maximal', (2, 3): 'rainbow'},
        'Z': {(1, 2): 'rainbow', (1, 3): 'maximal', (2, 3): 'rainbow'}
    }

    flag_code = FlagCode(
        complex.boundary_operators,
        x=2,
        z=2,
        add_boundary_pins=False,
        stabilizer_types=stabilizer_types,
        verbose=True
    )

    flag_code.get_rainbow_subgraphs([1, 2])
