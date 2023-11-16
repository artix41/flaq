import numpy as np
import matplotlib.pyplot as plt

import networkx as nx
from panqec.codes import Toric2DCode, Planar2DCode


class FlagCode:
    def __init__(self, boundaryOperators, cellPositions=None):
        self.boundaryOperators = boundaryOperators
        self.cellPositions = cellPositions
        self.dimension = len(self.boundaryOperators)

        for i in range(self.dimension-1):
            if self.boundaryOperators[i].shape[1] != self.boundaryOperators[i+1].shape[0]:
                raise ValueError(
                    f"Incorrect dimensions for input adjacency matrices {i} and {i+1}"
                )

        self.nCells = [boundaryOperators[0].shape[0]]
        for i in range(self.dimension):
            self.nCells.append(boundaryOperators[i].shape[1])

        # Find all the flags recursively

        self.flags = []
        self.flagToIndex = dict()
        self.flagCoordinates = []

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
                cellCoords = [cellPositions[i][flag[i]] for i in range(self.dimension+1)]
                flagCoords = np.mean(cellCoords, axis=0)
                self.flagCoordinates.append(flagCoords)

        # Build the colored graph of flags by exploring it recursively

        def getAdjacentFlags(flag, level):
            print(level)
            leftAdjacentFlags = None
            rightAdjacentFlags = None

            if level > 0:
                leftAdjacentFlags = set(
                    self.boundaryOperators[level-1][flag[level-1]].nonzero()[1]
                )

            if level < self.dimension:
                rightAdjacentFlags = set(
                    self.boundaryOperators[level].T[flag[level+1]].nonzero()[1]
                )

            print("LR", leftAdjacentFlags, rightAdjacentFlags)

            if leftAdjacentFlags is None:
                newCells = rightAdjacentFlags
            elif rightAdjacentFlags is None:
                newCells = leftAdjacentFlags
            else:
                newCells = leftAdjacentFlags.intersection(rightAdjacentFlags)

            adjacentFlags = [(*flag[:level], cell, *flag[level+1:]) for cell in newCells]

            return adjacentFlags

        self.flagAdjacency = np.zeros((len(self.flags), len(self.flags)), dtype=np.uint8)
        exploredFlags = set()

        def exploreGraphFromFlag(flag):
            print("Current flag", flag)
            if flag in exploredFlags:
                return

            exploredFlags.add(flag)

            for level in range(self.dimension+1):
                for nextFlag in getAdjacentFlags(flag, level):
                    if nextFlag != flag:
                        print("Adj flags", nextFlag)
                        idx1, idx2 = self.flagToIndex[flag], self.flagToIndex[nextFlag]
                        self.flagAdjacency[idx1, idx2] = level + 1
                        self.flagAdjacency[idx2, idx1] = level + 1

                        exploreGraphFromFlag(nextFlag)

        exploreGraphFromFlag(self.flags[0])

        graph = nx.from_numpy_array(self.flagAdjacency)

        edges, weights = zip(*nx.get_edge_attributes(graph, 'weight').items())

        nx.draw(
            graph, self.flagCoordinates, node_color='black', edge_color=np.array(weights)-1,
            edge_cmap=plt.cm.tab20b, node_size=40, width=2
        )

        plt.show()


if __name__ == "__main__":
    code = Planar2DCode(4)
    boundaryOperators = [
        code.Hz.todense(),
        code.Hx.todense().T
    ]
    positions = [
        np.array(code.stabilizer_coordinates)[code.z_indices],
        code.qubit_coordinates,
        np.array(code.stabilizer_coordinates)[code.x_indices]
    ]

    flagCode = FlagCode(boundaryOperators, positions)
    # print(flagCode.flags)
    # print("Number of flags", len(flagCode.flags))
