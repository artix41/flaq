import numpy as np
import networkx as nx
from panqec.codes import Toric2DCode


class FlagCode:
    def __init__(self, boundaryOperators):
        self.boundaryOperators = boundaryOperators
        self.dimension = len(self.boundaryOperators)

        print("Boundary operators\n", boundaryOperators)

        for i in range(self.dimension-1):
            if self.boundaryOperators[i].shape[1] != self.boundaryOperators[i+1].shape[0]:
                raise ValueError(
                    f"Incorrect dimensions for input adjacency matrices {i} and {i+1}"
                )

        self.nCells = [boundaryOperators[0].shape[0]]
        for i in range(self.dimension):
            self.nCells.append(boundaryOperators[i].shape[1])

        self.flags = []
        self.flagToIndex = dict()

        def getCompletedFlags(flagBeginning):
            if len(flagBeginning) == self.dimension+1:
                return [flagBeginning]

            completedFlags = []
            for cell in boundaryOperators[len(flagBeginning)-1][flagBeginning[-1]].nonzero()[1]:
                for flag in getCompletedFlags((*flagBeginning, cell)):
                    completedFlags.append(flag)

            return completedFlags

        for v0 in range(self.nCells[0]):
            flags = getCompletedFlags((v0,))
            for flag in flags:
                self.flagToIndex[flag] = len(self.flags)
                self.flags.append(flag)


if __name__ == "__main__":
    code = Toric2DCode(3)
    boundaryOperators = [
        code.Hz.todense(),
        code.Hx.todense().T
    ]

    flagCode = FlagCode(boundaryOperators)
    print(flagCode.flags)
    print("Number of flags", len(flagCode.flags))
