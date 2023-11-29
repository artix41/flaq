import pytest

import numpy as np
from panqec.codes import Toric2DCode

from flaq.flag_code import FlagCode


class TestFlagCode:
    def test_flags(self):
        for L in range(2, 5):
            code = Toric2DCode(L)
            boundaryOperators = [
                np.array(code.Hz.todense()),
                np.array(code.Hx.todense().T)
            ]

            flagCode = FlagCode(boundaryOperators)
            assert 8*L**2 == len(flagCode.flags)
