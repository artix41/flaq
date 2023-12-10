
import numpy as np

from . import BaseComplex


class DoubleOpenSquareComplex(BaseComplex):
    """Chain complex built from a square complex with open boundary conditions,
    by duplicating each vertex.
    Each edge (or rather hyperedge) in the bulk of the complex is then connected
    to four vertices
    """

    def __init__(self, Lx: int = 2, Ly: int = 2, sanity_check=True):
        """Constructor of the double square complex

        Parameters
        ----------
        Lx : int, optional
            Size of the complex (number of faces) in the horizontal direction, by default 2
        Ly : int, optional
            Size of the complex (number of faces) in the vertical direction, by default 2
        """
        self.Lx = Lx
        self.Ly = Ly

        super().__init__(sanity_check=sanity_check)

    def construct_boundary_operators(self):
        boundary_operators = []

        vertex_coordinates = []
        for x in range(0, 2*self.Lx+1, 2):
            for y in range(0, 2*self.Ly+1, 2):
                vertex_coordinates.append((x, y, 0))
                vertex_coordinates.append((x, y, 1))

        edge_coordinates = []

        # Vertical edges
        for x in range(0, 2*self.Lx+1, 2):
            for y in range(1, 2*self.Ly, 2):
                edge_coordinates.append((x, y))

        # Horizontal edges
        for x in range(1, 2*self.Lx, 2):
            for y in range(0, 2*self.Ly+1, 2):
                edge_coordinates.append((x, y))

        face_coordinates = []
        for x in range(1, 2*self.Lx, 2):
            for y in range(1, 2*self.Ly, 2):
                face_coordinates.append((x, y))

        vertex_index = {coord: i for i, coord in enumerate(vertex_coordinates)}
        edge_index = {coord: i for i, coord in enumerate(edge_coordinates)}
        n_vertices = len(vertex_coordinates)
        n_edges = len(edge_coordinates)
        n_faces = len(face_coordinates)

        # Face to edges operator
        delta_0 = np.zeros((n_edges, n_faces), dtype='uint8')

        for i_face, face_coord in enumerate(face_coordinates):
            delta = [(0, 1), (1, 0), (0, -1), (-1, 0)]

            for d in delta:
                edge_coord = (face_coord[0] + d[0], face_coord[1] + d[1])

                if edge_coord in edge_coordinates:
                    delta_0[edge_index[edge_coord], i_face] = 1

        # Edge to vertices operator
        delta_1 = np.zeros((n_vertices, n_edges), dtype='uint8')

        for i_edge, edge_coord in enumerate(edge_coordinates):
            if edge_coord[0] % 2 == 0:
                delta = [(0, -1), (0, 1)]
            else:
                delta = [(-1, 0), (1, 0)]

            for d in delta:
                vertex_coord = (edge_coord[0] + d[0], edge_coord[1] + d[1], 0)
                if vertex_coord in vertex_coordinates:
                    delta_1[vertex_index[vertex_coord], i_edge] = 1

                vertex_coord = (edge_coord[0] + d[0], edge_coord[1] + d[1], 1)
                if vertex_coord in vertex_coordinates:
                    delta_1[vertex_index[vertex_coord], i_edge] = 1

        boundary_operators = [delta_1, delta_0]

        return boundary_operators
