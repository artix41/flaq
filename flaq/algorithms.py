import pickle
from typing import Union, Set, List
import sys

import numpy as np


sys.setrecursionlimit(10000)


class Graph:
    def __init__(self, n_nodes: int):
        self.n_nodes = n_nodes

        self.adj_list = [set() for _ in range(n_nodes)]
        self.edges = set()
        self.nodes = set()
        self.color_count = dict()

    def add_edge(self, node1: int, node2: int, color: int):
        self.adj_list[node1].add((node2, color))
        self.adj_list[node2].add((node1, color))
        self.edges.add((node1, node2))
        self.edges.add((node2, node1))
        self.nodes.add(node1)
        self.nodes.add(node2)

        if color in self.color_count:
            self.color_count[color] += 1
        else:
            self.color_count[color] = 1

    def remove_edge(self, node1: int, node2: int, color: int):
        self.adj_list[node1].remove((node2, color))
        self.adj_list[node2].remove((node1, color))
        self.edges.remove((node1, node2))
        self.edges.remove((node2, node1))

        if len(self.adj_list[node1]) == 0:
            self.nodes.remove(node1)

        if len(self.adj_list[node2]) == 0:
            self.nodes.remove(node2)

        self.color_count[color] -= 1
        if self.color_count[color] == 0:
            self.color_count.pop(color)

    def merge_with(self, other_graph: "Graph"):
        for node in range(self.n_nodes):
            self.adj_list[node] = self[node].union(other_graph[node])

        self.edges = self.edges.union(other_graph.edges)
        self.nodes = self.nodes.union(other_graph.nodes)

    def copy(self):
        # new_graph.adj_list = deepcopy(self.adj_list)
        # adj_list = list(map(lambda x: x.copy(), self.adj_list))
        # adj_list = pickle.loads(pickle.dumps(self.adj_list, -1))

        new_graph = Graph(self.n_nodes)
        new_graph.adj_list = pickle.loads(pickle.dumps(self.adj_list, -1))

        return new_graph

    def connected_colors(self, node):
        return [adj_node[1] for adj_node in self.adj_list[node]]

    def is_subgraph(self, bigger_graph):
        answer = True
        for node in range(self.n_nodes):
            answer = answer and self.adj_list[node].issubset(bigger_graph[node])

        return answer

    def as_array(self):
        array = np.zeros(self.n_nodes, dtype='uint8')
        array[list(self.nodes)] = 1

        return array

    def get_neighborhood_nodes(self, nodes: Union[Set, List], depth: int):
        explorable_nodes = set(nodes.copy())
        visited_nodes = set()

        current_depth = 0
        while len(explorable_nodes) > 0 and current_depth <= depth:
            for node in explorable_nodes.copy():
                explorable_nodes.remove(node)
                visited_nodes.add(node)

                for adj_node, _ in self[node]:
                    if adj_node not in visited_nodes:
                        explorable_nodes.add(adj_node)

            current_depth += 1

        return visited_nodes

    def __getitem__(self, index: int):
        return self.adj_list[index]

    def __repr__(self):
        adj_dict = {}

        for node in range(self.n_nodes):
            if len(self.adj_list[node]) != 0:
                adj_dict[node] = set(map(lambda x: x[0], self.adj_list[node]))

        return f"Graph({adj_dict})"

    def __hash__(self):
        hashed_value = hash(tuple(map(
            lambda nodes: tuple(sorted(nodes, key=lambda adj_node: adj_node[0])),
            self.adj_list
        )))

        return hashed_value

    def __eq__(self, g):
        return (g.adj_list == self.adj_list)


def get_rainbow_subgraphs(
    graph: Graph,
    colors: Union[Set[int], List[int]],
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

    if return_format not in ['set', 'array']:
        raise ValueError(
            f"'return_format' must take value in ['set', 'array'],"
            f"not {return_format}"
        )

    colors = set(colors)
    n_nodes = graph.n_nodes

    rainbow_subgraphs = set()

    def find_rainbow_subgraphs(
        partial_subgraph: Graph,
        explorable_nodes: Set[int],
        finished_nodes: Set[int]
    ):
        if len(explorable_nodes) == 0:
            # We found a rainbow subgraph
            edges = tuple(sorted(partial_subgraph.edges))
            nodes = tuple(sorted(partial_subgraph.nodes))
            rainbow_subgraphs.add(nodes)
            return {edges}

        max_subgraphs = set()

        for node in explorable_nodes.copy():
            explored_colors = {color for _, color in partial_subgraph[node]}
            remaining_colors = colors - explored_colors

            for adj_node, color in graph[node]:
                if (color in remaining_colors
                        and not (adj_node in finished_nodes)
                        and color not in partial_subgraph.connected_colors(adj_node)
                        and colors.issubset(graph.connected_colors(adj_node))):
                    prev_explorable_nodes = explorable_nodes.copy()
                    prev_finished_nodes = finished_nodes.copy()

                    partial_subgraph.add_edge(node, adj_node, color)
                    explorable_nodes.add(adj_node)

                    if len(partial_subgraph.connected_colors(adj_node)) == len(colors):
                        explorable_nodes.remove(adj_node)
                        finished_nodes.add(adj_node)

                    if len(partial_subgraph.connected_colors(node)) == len(colors):
                        explorable_nodes.remove(node)
                        finished_nodes.add(node)

                    subgraph_of_max_subgraph = False
                    for max_subgraph in max_subgraphs:
                        if (node, adj_node) in max_subgraph:
                            subgraph_of_max_subgraph = True
                            break

                    if not subgraph_of_max_subgraph:
                        subgraphs = find_rainbow_subgraphs(
                            partial_subgraph,
                            explorable_nodes,
                            finished_nodes
                        )
                        max_subgraphs.update(subgraphs)

                    partial_subgraph.remove_edge(node, adj_node, color)

                    explorable_nodes = prev_explorable_nodes
                    finished_nodes = prev_finished_nodes

        if len(max_subgraphs) == 0:
            max_subgraphs = {tuple(sorted(partial_subgraph.edges))}

        return max_subgraphs

    max_subgraphs = set()
    for node in range(n_nodes):
        # print(f"Explore node {node+1} / {self.n_flags}", end='\r')
        partial_subgraph = Graph(n_nodes)
        explorable_nodes = {node}
        finished_nodes = set()

        has_all_colors = (
            colors.issubset(graph.connected_colors(node))
        )

        node_included_in_max_subgraph = False
        if has_all_colors:
            for max_subgraph in max_subgraphs:
                for edge in max_subgraph:
                    if edge[0] == node or edge[1] == node:
                        node_included_in_max_subgraph = True

        if has_all_colors and not node_included_in_max_subgraph:
            max_subgraphs.update(
                find_rainbow_subgraphs(partial_subgraph, explorable_nodes, finished_nodes)
            )

    if return_format == 'array':
        subgraphs_output = []
        for subgraph_nodes in rainbow_subgraphs:
            subgraph_array = np.zeros(n_nodes, dtype='uint8')
            subgraph_array[list(subgraph_nodes)] = 1
            subgraphs_output.append(subgraph_array)
    elif return_format == 'set':
        subgraphs_output = list(map(set, rainbow_subgraphs))

    return subgraphs_output
