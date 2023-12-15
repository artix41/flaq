from ldpc import bposd_decoder
from ldpc.mod2 import rank, nullspace
import numpy as np


def get_all_logicals(Hx, Hz, k=None):
    min_x_logicals = []
    min_z_logicals = []

    arbitrary_x_logicals = get_arbitrary_logicals(Hx, Hz, 'X', k)

    for anticommuting_x_logical in arbitrary_x_logicals:
        min_z_logicals.append(get_minimum_logical(Hx, anticommuting_x_logical))

    for anticommuting_z_logical in min_z_logicals:
        min_x_logicals.append(get_minimum_logical(Hz, anticommuting_z_logical))

    min_x_logicals = np.array(min_x_logicals, dtype='uint8')
    min_z_logicals = np.array(min_z_logicals, dtype='uint8')

    return {'X': min_x_logicals, 'Z': min_z_logicals}


def get_minimum_logical(H, anticommuting_logical):
    H_with_logical = np.vstack([H, anticommuting_logical])

    decoder = bposd_decoder(
        H_with_logical,
        error_rate=0.1,
        max_iter=200,
        bp_method='msl',
        osd_method="osd0",
        osd_order=0
    )

    syndrome = np.array([0] * H.shape[0] + [1])
    predicted_logical_binary = decoder.decode(syndrome)

    return predicted_logical_binary


def get_arbitrary_logicals(Hx, Hz, pauli='X', k=None):
    print(f"Get logicals {pauli}")
    if pauli == 'X':
        H = [Hx, Hz]
    else:
        H = [Hz, Hx]

    rank_H0 = rank(H[0])
    augmented_H = nullspace(H[1])

    if k is None:
        k = Hx.shape[1] - rank_H0 - rank(H[1])

    logicals = []
    for i, s in enumerate(augmented_H):
        print(f"{i} / {len(augmented_H)}", end="\r")

        new_H0 = np.vstack([H[0], s])
        if rank(new_H0) > rank_H0:
            H[0] = new_H0
            rank_H0 += 1

            logicals.append(s)

        if len(logicals) == k:
            break

    print()
    print("Done")

    return logicals
