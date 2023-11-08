'''
    The goal of this script is to explore the impact of weighting on phase linking solutions.
    1. Does a standard inner product really perform better than the coherence weighted inner product?
    2. Can weighting by r/dB improve results further?

    '''


from closig.model import SeasonalVegLayer
import numpy as np
from matplotlib import pyplot as plt
import cphases.util as util
from triplets.itriplet import IntensityTriplet
from closig.expansion import TwoHopBasis
from model.models import LSClosureModel
from closig.linking import EMI, EMI_py, CutOffRegularizer, NearestNeighbor, EVD
import seaborn as sns


def main():

    shrubs = SeasonalVegLayer(n_mean=1.2 - 0.01j, n_std=0.01, n_amp=0.05,
                              n_t=0, P_year=30, density=0.1, dcoh=0.8, coh0=0.5, h=0.2, name='Shrubs')

    P = 30

    twoHop = TwoHopBasis(P)
    As = twoHop.basis_matrix_std_basis()
    twoHop.evaluate_covariance(shrubs.covariance(P))

    covariance = shrubs.covariance(P)

    # Establish possible intensity evolution and compute difference
    intensity = 10 * np.log10(4e2 * np.abs(np.diag(covariance)))
    istack = np.stack([intensity for i in range(len(intensity))])
    idifference = istack.T - istack

    # Setup triplets
    indices = twoHop.basis_indices()
    indices_max = indices
    timescales = twoHop.ptau
    times = twoHop.pt

    itriplet = IntensityTriplet(
        intensity, function='arctan', kappa=0.2)

    itriplets = itriplet.get_triplets(indices)

    # Compute inferred phase error from triplet and standard inner product
    cmodel = LSClosureModel([itriplet], shrubs.covariance(P))
    cmodel.train(indices_max)
    cmatrix = cmodel.get_correction_matrix(indices)
    cmatrix_weighted = cmodel.get_correction_matrix(
        indices, weights=idifference)

    # Compute phase history residual from EVD
    C = shrubs.covariance(P, coherence=False)
    pl = EVD().link(shrubs.covariance(P), G=np.abs(C))[:, np.newaxis]

    shrubs.plot_matrices(P, coherence=False)

    C_res = C * (pl @ pl.T.conj()).conj()

    plt.scatter(idifference.flatten(), np.angle(
        C.flatten()), label='True Error', alpha=0.8, s=15, marker='x')
    plt.scatter(idifference.flatten(), np.angle(
        cmatrix.flatten()), label='Standard Inner Product', alpha=0.3)
    plt.scatter(idifference.flatten(), np.angle(
        C_res.flatten()), label='Coherence Weighted', alpha=0.3)
    plt.scatter(idifference.flatten(), np.angle(
        cmatrix_weighted.flatten()), label='R weighted', alpha=0.8, s=15)
    plt.legend(loc='best')

    plt.show()


main()
