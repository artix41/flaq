from abc import ABC, abstractmethod

from typing import List

from ldpc.mod2 import rank
import networkx as nx
import numpy as np
from pyvis.network import Network
from scipy.sparse import csr_array

from flaq.utils import get_all_logicals, draw_bipartite_graph


class BaseComplex(ABC):
    def __init__(self, sanity_check: bool = True):
        self._boundary_operators = None
        self._n = None
        self._k = None
        self._d = None

        if sanity_check and not self.is_valid_complex():
            raise ValueError("Not a valid complex")

    def is_valid_complex(self) -> bool:
        """Check whether the boundary of a boundary is always zero

        Returns
        -------
        bool
            True if it is a valid chain complex (i.e. the boundary of a boundary is always zero)
        """
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

    @property
    def n(self) -> int:
        if self._n is None:
            self._n = self.get_n()

        return self._n

    @property
    def k(self) -> int:
        if self._k is None:
            self._k = self.get_k()

        return self._k

    @property
    def d(self) -> int:
        if self._d is None:
            self._d = self.get_d()

        return self._d

    def get_logicals(
        self,
        left_index: int = 0,
        reduce_iter: int = 0,
        osd_order=6,
        verbose: bool = False
    ) -> int:
        logicals = get_all_logicals(
            self.boundary_operators[left_index],
            self.boundary_operators[left_index+1].T,
            reduce_iter=reduce_iter,
            osd_order=osd_order,
            verbose=verbose
        )

        return logicals

    def get_d(
        self,
        left_index: int = 0,
        reduce_iter: int = 0,
        osd_order=6,
        verbose: bool = False
    ) -> int:
        """Get the distance of the code corresponding to the complex in-between
        left_index and left_index+2
        """

        logicals = self.get_logicals(
            left_index=left_index,
            reduce_iter=reduce_iter,
            osd_order=osd_order,
            verbose=verbose
        )

        d = 0
        if len(logicals['X']) > 0:
            d = min(
                np.min(np.sum(logicals['X'], axis=1)),
                np.min(np.sum(logicals['Z'], axis=1))
            )

        return d

    def get_n(self, left_index: int = 0) -> int:
        """Get the number of physical qubits of the code corresponding to the complex in-between
        left_index and left_index+2
        """

        return self.boundary_operators[left_index].shape[1]

    def get_k(self, left_index: int = 0) -> int:
        """Get the number of logical qubits of the code corresponding to the complex in-between
        left_index and left_index+2
        """

        Hx = self.boundary_operators[left_index]
        Hz = self.boundary_operators[left_index+1].T

        return self.get_n(left_index) - rank(Hx) - rank(Hz)

    def draw_tanner_graph(
        self,
        index_boundary_operator: int = 0,
        notebook: bool = False
    ) -> Network:
        nt = draw_bipartite_graph(
            self.boundary_operators[index_boundary_operator],
            notebook=notebook
        )

        return nt
