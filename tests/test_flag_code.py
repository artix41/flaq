import pytest

from itertools import combinations
from ldpc.codes import ring_code

from flaq.flag_code import FlagCode
from flaq.chain_complex import HypergraphComplex, DoubleSquareComplex


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
            x=2,
            z=2,
            add_boundary_pins=False,
            verbose=False
        )
        assert 8*L**2 == flag_code.n

    @pytest.mark.parametrize("complex", manifold_2d_complexes)
    def test_manifold_2d_complex(self, complex):
        flag_code = FlagCode(
            complex.boundary_operators,
            None,
            x=2,
            z=2,
            add_boundary_pins=False,
            verbose=False
        )

        assert flag_code.is_pin_code_relation()
        assert flag_code.is_valid_css()
        assert flag_code.is_manifold()
        assert flag_code.is_multiorthogonal(2)

        # Test that maximal and rainbow subgraphs are equal for manifolds
        for colors in combinations([1, 2, 3], 2):
            max_subgraphs = flag_code.get_maximal_subgraphs(colors)
            rainbow_subgraphs = flag_code.get_rainbow_subgraphs(colors)

            assert len(max_subgraphs) == len(rainbow_subgraphs)

            for subgraph in max_subgraphs:
                assert subgraph in rainbow_subgraphs

    @pytest.mark.parametrize("complex", manifold_3d_complexes)
    def test_manifold_3d_complex(self, complex):
        flag_code = FlagCode(
            complex.boundary_operators,
            None,
            x=3,
            z=2,
            add_boundary_pins=False,
            verbose=False
        )

        assert flag_code.is_pin_code_relation()
        assert flag_code.is_manifold()
        assert flag_code.is_valid_css()
        assert flag_code.is_multiorthogonal(3)

        # Test that maximal and rainbow subgraphs are equal for manifolds
        for colors in [*combinations([1, 2, 3, 4], 2), *combinations([1, 2, 3, 4], 3)]:
            max_subgraphs = flag_code.get_maximal_subgraphs(colors)
            rainbow_subgraphs = flag_code.get_rainbow_subgraphs(colors)

            assert len(max_subgraphs) == len(rainbow_subgraphs)

            for subgraph in max_subgraphs:
                assert subgraph in rainbow_subgraphs

    def test_3d_color_code(self):
        complex = HypergraphComplex([ring_code(2), ring_code(2), ring_code(2)])
        color_code = FlagCode(
            complex.boundary_operators,
            x=2,
            z=3,
            add_boundary_pins=False,
            verbose=False
        )

        assert color_code.n == 384
        assert color_code.k == 9
        assert color_code.d == 4

    def test_double_square_rainbow_code(self):
        stabilizer_types = {
            'X': {'rainbow': [(1, 2), (2, 3)], 'maximal': [(1, 3)]},
            'Z': {'rainbow': [(1, 2), (2, 3)], 'maximal': [(1, 3)]},
        }

        for sizes in [(2, 2), (3, 3), (4, 4)]:
            complex = DoubleSquareComplex(*sizes, periodic=True, sanity_check=True)

            flag_code = FlagCode(
                complex.boundary_operators,
                x=2,
                z=2,
                stabilizer_types=stabilizer_types)

            assert flag_code.is_pin_code_relation()
            assert flag_code.is_valid_css()
            assert not flag_code.is_manifold()
            assert flag_code.k == 4
            assert flag_code.d == 4*sizes[0]
