import numpy as np
from model.models import LSClosureModel
from closig.model import PrecipScatterSoilLayer
from triplets.itriplet import IntensityTriplet
from closig.expansion import TwoHopBasis
from greg import simulation
from matplotlib import pyplot as plt
import seaborn as sns


def get_random_C(C, l, coherence=False):
    sim_data = simulation.circular_normal(
        (l, l, C.shape[0]), Sigma=C)
    # evaluate covariance
    C_sim = C.copy()
    for i in range(C.shape[0]):
        for j in range(i, C.shape[1]):
            C_sim[i, j] = np.mean(sim_data[:, :, i] *
                                  sim_data[:, :, j].conj())
            C_sim[j, i] = C_sim[i, j].conj()

    # Normalize
    if coherence:
        for i in range(C.shape[0]):
            for j in range(i, C.shape[1]):
                C_sim[i, j] = C_sim[i, j] / np.sqrt(C_sim[i, i] * C_sim[j, j])

    return C_sim


def bootstrap_correlation(C, l, nsample=100, fitLine=False, zeroPhi=True):

    R2 = np.zeros((nsample))

    if fitLine:
        coeffs = np.zeros((2, nsample))

    if zeroPhi:
        C = C * C.conj() / np.abs(C)

    indices = TwoHopBasis(C.shape[0]).basis_indices()
    for i in range(nsample):

        C_sim = get_random_C(C, l)

        tripletModel = IntensityTriplet(
            10 * np.log10(7e3 * np.abs(np.diag(C_sim))))

        predModel = LSClosureModel([tripletModel], C_sim)

        R2[i] = predModel.compute_R(indices)

        if fitLine:
            coeffs[:, i] = predModel.train(indices).params

    if fitLine:
        return R2, coeffs
    else:
        return R2


def main():
    model = PrecipScatterSoilLayer(
        f=20, tau=1, dcoh=0.99, coh0=0.0, offset=0.1, scale=0.1)

    C = model.covariance(10, coherence=False)
    C_samp = get_random_C(C, 50, coherence=False)

    indices = TwoHopBasis(C.shape[0]).basis_indices()
    tripletModel = IntensityTriplet(
        10 * np.log10(7e3 * np.abs(np.diag(C_samp))), kappa=1)

    predModel = LSClosureModel([tripletModel], C_samp)
    predModel.plot_scatter(indices=indices)
    # return
    # Plot the coherence and angle of the root and sample covariance matrices
    fig, ax = plt.subplots(nrows=2, ncols=2)

    ax[0, 0].imshow(np.abs(C))
    ax[1, 0].imshow(np.abs(C_samp))
    ax[0, 1].imshow(np.angle(C), cmap=plt.cm.seismic)
    ax[1, 1].imshow(np.angle(C_samp), cmap=plt.cm.seismic)
    ax[1, 0].set_xlabel('Coherence')
    ax[1, 1].set_xlabel('Phase')
    ax[1, 0].set_ylabel('Sample')
    ax[0, 0].set_ylabel('True')
    plt.show()

    R = bootstrap_correlation(C_samp, l=100, nsample=100,
                              fitLine=False, zeroPhi=True)

    sns.kdeplot(R, bw_adjust=0.2)
    plt.show()


if __name__ == '__main__':
    main()