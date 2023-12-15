import pytest

from itertools import combinations
from ldpc.codes import ring_code

from flaq.flag_code import FlagCode
from flaq.chain_complex import HypergraphComplex


square_complexes = [
    (2, HypergraphComplex([ring_code(2), ring_code(2)])),
    (3, HypergraphComplex([ring_code(3), ring_code(3)])),
    (4, HypergraphComplex([ring_code(4), ring_code(4)])),
]

manifold_2d_complexes = [
    HypergraphComplex([ring_code(2), ring_code(2)]),
    HypergraphComplex([ring_code(3), ring_code(2)]),
    HypergraphComplex([ring_code(3), ring_code(3)]),
]

manifold_3d_complexes = [
    HypergraphComplex([ring_code(2), ring_code(2), ring_code(2)]),
]


class TestFlagCode:
    @pytest.mark.parametrize("complex", square_complexes)
    def test_square_complex(self, complex):
        L = complex[0]

        flag_code = FlagCode(
            complex[1].boundary_operators,
            None,
            x=1,
            z=1,
            add_boundary_pins=False,
            verbose=False
        )
        assert 8*L**2 == len(flag_code.flags)

    @pytest.mark.parametrize("complex", manifold_2d_complexes)
    def test_manifold_2d_complex(self, complex):
        flag_code = FlagCode(
            complex.boundary_operators,
            None,
            x=1,
            z=1,
            add_boundary_pins=False,
            verbose=False
        )

        # Test that maximal and rainbow subgraphs are equal for manifolds
        for colors in combinations([1, 2, 3], 2):
            max_subgraphs = flag_code.get_all_maximal_subgraphs(colors)
            rainbow_subgraphs = flag_code.get_all_rainbow_subgraphs(colors)

            assert len(max_subgraphs) == len(rainbow_subgraphs)

            for subgraph in max_subgraphs:
                assert subgraph in rainbow_subgraphs

    @pytest.mark.parametrize("complex", manifold_3d_complexes)
    def test_manifold_3d_complex(self, complex):
        flag_code = FlagCode(
            complex.boundary_operators,
            None,
            x=2,
            z=1,
            add_boundary_pins=False,
            verbose=False
        )

        # Test that maximal and rainbow subgraphs are equal for manifolds
        for colors in combinations([1, 2, 3], 2):
            max_subgraphs = flag_code.get_all_maximal_subgraphs(colors)
            rainbow_subgraphs = flag_code.get_all_rainbow_subgraphs(colors)

            assert len(max_subgraphs) == len(rainbow_subgraphs)

            for subgraph in max_subgraphs:
                assert subgraph in rainbow_subgraphs
