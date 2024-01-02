from ldpc import bposd_decoder
from ldpc.mod2 import rank, nullspace
import numpy as np


def log(*args, verbose=False, **kwargs):
    if verbose:
        print(*args, **kwargs)


def get_all_logicals(Hx, Hz, k=None, reduce_iter=0, osd_order=6, verbose=False):
    arbitrary_x_logicals = get_arbitrary_logicals(Hx, Hz, 'X', k, verbose=verbose)

    min_z_logicals = get_minimum_logicals(
        Hx, arbitrary_x_logicals, osd_order=osd_order, verbose=verbose
    )
    min_x_logicals = get_minimum_logicals(
        Hz, min_z_logicals, osd_order=osd_order, verbose=verbose
    )

    min_x_logicals = np.array(min_x_logicals, dtype='uint8')
    min_z_logicals = np.array(min_z_logicals, dtype='uint8')

    for i in range(min_x_logicals.shape[0]):
        min_x_logicals[i] = reduce_logical(Hx, min_x_logicals[i], reduce_iter)

    for i in range(min_z_logicals.shape[0]):
        min_z_logicals[i] = reduce_logical(Hz, min_z_logicals[i], reduce_iter)

    return {'X': min_x_logicals, 'Z': min_z_logicals}


def reduce_logical(H, logical, n_iter, random=False):
    current_weight = len(logical.nonzero()[0])

    for _ in range(n_iter):
        for i in range(H.shape[0]):
            if random:
                stab_i = np.random.randint(0, H.shape[0])
            else:
                stab_i = i

            stab = H[stab_i]
            new_logical = np.logical_xor(logical, stab)
            new_weight = len(new_logical.nonzero()[0])

            if new_weight < current_weight:
                logical = new_logical
                current_weight = new_weight

            elif current_weight == new_weight:
                if np.random.rand() < 0.5:
                    logical = new_logical

    return logical


def get_minimum_logicals(H, anti_logical_basis, osd_order=6, verbose=False):
    H_with_logical = np.vstack([H, anti_logical_basis])

    decoder = bposd_decoder(
        H_with_logical,
        error_rate=0.1,
        max_iter=200,
        bp_method='msl',
        osd_method="osd_cs",
        osd_order=osd_order
    )

    logicals = []
    for i in range(len(anti_logical_basis)):
        log(f"Finding logicals {i+1}/{len(anti_logical_basis)}  ", end='\r', verbose=verbose)
        logical_syndrome = [0 for _ in range(len(anti_logical_basis))]
        logical_syndrome[i] = 1

        syndrome = np.array([0] * H.shape[0] + logical_syndrome)

        predicted_logical_binary = decoder.decode(syndrome)

        logicals.append(predicted_logical_binary)

    return logicals


def get_arbitrary_logicals(Hx, Hz, pauli='X', k=None, verbose=False):
    if verbose:
        log(f"Get logicals {pauli}", verbose=verbose)

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
        log(f"{i} / {len(augmented_H)}", end="\r", verbose=verbose)

        new_H0 = np.vstack([H[0], s])
        if rank(new_H0) > rank_H0:
            H[0] = new_H0
            rank_H0 += 1

            logicals.append(s)

        if len(logicals) == k:
            break

    log(verbose=verbose)
    log("Done", verbose=verbose)

    return logicals
