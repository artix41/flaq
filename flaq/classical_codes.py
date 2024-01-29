from typing import List, Tuple

import numpy as np


def double_rep_code(size: int, periodic: bool = True) -> np.ndarray:
    n_vertices = 2*size
    n_edges = size if periodic else size - 1

    H = np.zeros((n_edges, n_vertices), dtype=np.uint8)

    for i in range(n_edges):
        for j in range(4):
            if periodic or 2*i+j < n_vertices:
                H[i, (2*i+j) % n_vertices] = 1

    return H


def ising_code(size: Tuple[int, int], periodic: bool = True) -> np.ndarray:
    Lx, Ly = size
    size = np.array([Lx, Ly])

    data_bit_coordinates: List[Tuple[int, int]] = []
    for x in range(0, 2*Lx+1, 2):
        for y in range(0, 2*Ly+1, 2):
            data_bit_coordinates.append([x, y])
    data_bit_coordinates = np.array(data_bit_coordinates, dtype=np.int32)

    check_bit_coordinates: List[Tuple[int, int]] = []
    max_x = 2*Lx+2 if periodic else 2*Lx
    for x in range(1, max_x, 2):
        for y in range(0, 2*Ly+1, 2):
            check_bit_coordinates.append((x, y))

    max_y = 2*Ly+2 if periodic else 2*Ly
    for x in range(0, 2*Lx+1, 2):
        for y in range(1, max_y, 2):
            check_bit_coordinates.append([x, y])
    check_bit_index = {tuple(coord): i for i, coord in enumerate(check_bit_coordinates)}
    check_bit_coordinates = np.array(check_bit_coordinates, dtype=np.int32)

    n, m = len(data_bit_coordinates), len(check_bit_coordinates)

    H = np.zeros((m, n), dtype=np.uint8)

    delta = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]])
    for j in range(n):
        for d in delta:
            check_coord = data_bit_coordinates[j] + d
            if periodic:
                check_coord %= (2*size+2)

            if tuple(check_coord) in check_bit_index:
                H[check_bit_index[tuple(check_coord)], j] = 1

    return H
