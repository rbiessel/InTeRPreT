import numpy as np
from model.models import LSClosureModel
from triplets.itriplet import IntensityTriplet
from closig.expansion import TwoHopBasis
from greg import simulation


def bootstrap_correlation(C, l, triplets, nsample=100, fitLine=False, zeroPhi=True):

    R2 = np.zeros((nsample))

    if fitLine:
        coeffs = np.zeros((2, nsample))

    if zeroPhi:
        C = C * C.conj() / np.abs(C)

    for i in range(nsample):

        sim_data = simulation.circular_normal(
            (l, l, C.shape[0]), Sigma=C)

        sim_data = np.swapaxes(sim_data, 0, 2)
        sim_data = np.swapaxes(sim_data, 1, 2)

        # evaluate covariance
        C_sim = C.copy()
        for i in range(C.shape[0]):
            for j in range(i, C.shape[1]):
                print(i, j)
                # C_sim[i, j] = np.mean(sim_data[i, j, :])
                # C_sim[j, i] = np.mean(sim_data[i, j, :])

        tripletModel = IntensityTriplet(
            20 * np.log10(np.abs(np.diag(C_sim))), triplets)

        indices = TwoHopBasis(C.shape[0]).basis_indices()
        predModel = LSClosureModel([tripletModel], C_sim)

        R2[i] = predModel.get_R(indices)

        if fitLine:
            coeffs[:, i] = predModel.train(indices).params

    if fitLine:
        return R2, coeffs
    else:
        return R2


def main():
    raise NotImplementedError(' Bootstrapping is still TODO')
    # Get a magnitude matrix from closig
    # Bootstrap possible R^2 for high and low coherence scenarios


main()
