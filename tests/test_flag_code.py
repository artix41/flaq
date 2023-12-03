import pytest

import numpy as np
from panqec.codes import Toric2DCode

from flaq.flag_code import FlagCode


class TestFlagCode:
    def test_flags(self):
        for L in range(2, 5):
            code = Toric2DCode(L)
            boundary_operators = [
                np.array(code.Hz.todense()),
                np.array(code.Hx.todense().T)
            ]

            flag_code = FlagCode(boundary_operators)
            assert 8*L**2 == len(flag_code.flags)

    def test_rainbow_subgraph(self):
        for L in range(2, 5):
            code = Toric2DCode(L)
            boundary_operators = [
                np.array(code.Hz.todense()),
                np.array(code.Hx.todense().T)
            ]

            flag_code = FlagCode(boundary_operators)

            # Test that maximal and rainbow subgraphs are equal for manifolds
            assert flag_code.get_all_maximal_subgraphs() == flag_code.get_all_rainbow_subgraphs()
