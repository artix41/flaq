import pytest

from panqec.codes import Toric2DCode

from flaq.flag_code import FlagCode


class TestFlagCode:
    def test_flags(self):
        for L in range(2, 5):
            code = Toric2DCode(L)
            boundaryOperators = [
                code.Hz.todense(),
                code.Hx.todense().T
            ]

            flagCode = FlagCode(boundaryOperators)
            assert 8*L**2 == len(flagCode.flags)
