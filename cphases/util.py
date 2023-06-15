'''
author: Rowan
date: 2023-05-26
'''

import numpy as np
import itertools
from scipy.stats import chi2
from matplotlib import pyplot as plt


def get_adjacent_triplets(num):
    '''
        Return an array of indexes corresponding only to triplets immediately adjacent to each other
    '''
    triplets = []
    for i in range(num - 2):
        triplets.append([i, i+1, i + 2])

    return np.sort(np.array(triplets))


def get_triplets_increasing(num, force=None, all=False, max=None, omit_phis=[]):
    '''
        Get an array of indicies corresponding to all possible triplets with increasing indices. This will include dependent/redundant closures phases

        force: force the first index to be a specific value, setting force=0 returns the standard basis of closure phases and ensures each triplet is independent

        all: return all permutations of triplets

        max: restrict temporal baseline to be less than or equal to this value

        omit_phis: omit triplets that contain any of the specified phases, this is a list of strings of the form 'ij' where i and j are the indexes of the phase to omit
    '''
    numbers = np.arange(0, num, 1)
    permutations = np.array(list(itertools.permutations(numbers, 3)))
    if all:
        return permutations
    combinations = np.sort(permutations, axis=1)
    combinations = np.unique(combinations, axis=0)
    if force is not None:
        combinations = np.array(
            [triplet for triplet in combinations if triplet[0] == force])

    if max is not None:
        combinations = np.array(
            [triplet for triplet in combinations if triplet.max() <= max])

    if len(omit_phis) > 0:
        for omit_phi in omit_phis:
            combinations = np.array(
                [triplet for triplet in combinations if int(omit_phi[0]) and int(omit_phi[1]) not in triplet])

    return combinations


def build_A(triplets, coherence, zero_indexes=[], omit_phis=[]):
    '''
      Given an array of triplet indicies and a coherence matrix, construct a design matrix that maps phases to closure phases.
    '''

    phi_indexes = collapse_indexes(coherence)

    A = np.zeros((triplets.shape[0], len(phi_indexes)))
    for i in range(triplets.shape[0]):
        a = A[i]
        triplet = triplets[i]

        i1string = f'{triplet[0]}{triplet[1]}'
        i2string = f'{triplet[0]}{triplet[2]}'
        i3string = f'{triplet[1]}{triplet[2]}'

        i1 = np.where(phi_indexes == i1string)
        i2 = np.where(phi_indexes == i2string)
        i3 = np.where(phi_indexes == i3string)

        if i1 not in omit_phis and i2 not in omit_phis and i3 not in omit_phis:
            if i1string in zero_indexes:
                a[i1] = 0
            else:
                a[i1] = 1

            if i2string in zero_indexes:
                a[i2] = 0
            else:
                a[i2] = -1

            if i3string in zero_indexes:
                a[i3] = 0
            else:
                a[i3] = 1
    return A


def collapse_indexes(cov):
    '''
        Get the canonical indexes of the phases of a given covariance matrix for input into other functions
    '''
    index_matrix = []
    for i in range(cov.shape[0]):
        row = []
        for j in range(cov.shape[1]):
            row.append(f'{i}{j}')
        index_matrix.append(row)

    cov_indexes = np.array(index_matrix)
    cov_indexes = cov_indexes[np.triu_indices(cov_indexes.shape[0], 1)]
    return cov_indexes


def coherence_to_phivec(coherence: np.complex64) -> np.complex64:
    '''
      Collapse a coherence matrix into a vector of phases
    '''
    utri = np.triu_indices_from(coherence, 1)
    phi_vec = coherence[utri]

    return phi_vec


def phivec_to_coherence(phi_vec: np.complex64, n: np.int8) -> np.complex64:
    '''
        Convert a vector of complex phases back into a coherence matrix
    '''

    coherence = np.ones((n, n), dtype=np.cdouble)
    utri = np.triu_indices_from(coherence, 1)
    coherence[utri] = phi_vec
    coherence = coherence * coherence.T.conj()

    return coherence


def phi_to_closure(A, phic) -> np.complex64:
    '''
        Use matrix mult to generate vector of phase closures such that 
        the angle xi = phi_12 + phi_23 - phi_13
    '''
    return np.exp(1j * A @ np.angle(phic))


def cumulative_mask(triplets):
    '''
        Return a mask of triplets such that they are only consecutive closure phases
    '''
    mask = np.full((triplets.shape[0]), False, dtype=bool)
    for i in range(mask.shape[0]):
        if (triplets[i, 2] - triplets[i, 1] == 1) and (triplets[i, 1] - triplets[i, 0] == 1):
            mask[i] = True

    return mask
