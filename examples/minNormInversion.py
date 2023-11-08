from closig.model import SeasonalVegLayer
import numpy as np
from matplotlib import pyplot as plt
import cphases.util as util
from triplets.itriplet import IntensityTriplet
from closig.expansion import TwoHopBasis
from model.models import LSClosureModel

import seaborn as sns


def main():

    shrubs = SeasonalVegLayer(n_mean=1.2 - 0.01j, n_std=0.1, n_amp=0.1,
                              n_t=0, P_year=30, density=0.1, dcoh=1, h=0.05, name='Shrubs')

    shrubs =  SeasonalVegLayer(n_mean=1.2 - 0.1j, n_std=0, n_amp=0,
                            n_t=.01, P_year=30, density=1, dcoh=1, h=0.5, name='Shrubs')
    P = 30

    twoHop = TwoHopBasis(P)
    As = twoHop.basis_matrix_std_basis()
    twoHop.evaluate_covariance(shrubs.covariance(P))

    covariance = shrubs.covariance(P)
    intensity = 10 * np.log10(4e2 * np.abs(np.diag(covariance)))
    dIntensity = np.tile(intensity, [len(intensity), 1])
    print(dIntensity.shape)
    dIntensity = dIntensity.T - dIntensity

    indices = twoHop.basis_indices()
    indices_max = indices

    timescales = twoHop.ptau
    times = twoHop.pt

    kappa = 9
    itriplet = IntensityTriplet(
        intensity, function='tanh', kappa=kappa)

    itriplets = itriplet.get_triplets(indices)

    cmodel = LSClosureModel([itriplet], shrubs.covariance(P))
    cmodel.train(indices_max)
    print('Parameters: ', cmodel.params)
    print(f'Kappa: {kappa}')
    cmodel.plot_scatter(indices=indices)
    cmatrix = cmodel.get_correction_matrix(indices)



    # fig, ax = plt.subplots(nrows=1, ncols=2)

    # ax[0].set_title('Predicted Error')
    # ax[0].imshow(np.angle(cmatrix), cmap=plt.cm.seismic)

    # ax[1].set_title('True Error')
    # ax[1].imshow(np.angle(covariance), cmap=plt.cm.seismic)
    # plt.show()

    plt.scatter(dIntensity.flatten(), np.angle(cmatrix).flatten(), label='Predicted Error', s=10)
    plt.scatter(dIntensity.flatten(), np.angle(covariance).flatten(), label='Full Error', s=10)


    slope_null = np.polyfit(dIntensity.flatten(), np.angle(covariance * cmatrix.conj()).flatten(), 1)
    plt.scatter(dIntensity.flatten(), np.angle(covariance * cmatrix.conj()).flatten(), label='Difference', s=10)
    plt.plot(dIntensity.flatten(), dIntensity.flatten() * slope_null[0], label='Difference Fit')

    print(slope_null)
    plt.legend(loc='best')
    plt.show()

    return

    colors = plt.cm.viridis(timescales / np.max(timescales))

    plt.scatter(itriplets,
                np.angle(cmodel.compute_bicoh(indices)), c=colors, alpha=0.5)
    cbar = plt.colorbar(orientation='vertical')
    R = cmodel.compute_R(indices)
    cbar.ax.set_title(r'$\tau$')
    plt.title(f"R: {np.round(R, 3)}")
    plt.xlabel('Intensity Triplet [-]')
    plt.ylabel('Closure Phase [$\mathrm{rad}$]')
    plt.show()

    cphases_pred = cmodel.predict(indices)
    cphases_true = np.angle(cmodel.compute_bicoh(indices))

    cmodel.get_correction_matrix(indices, apply_in_place=True)

    plt.scatter(timescales, np.angle(cmodel.compute_bicoh(indices)))
    plt.xlabel('timescale')
    plt.ylabel('Closure Phase [$\mathrm{rad}$]')
    plt.show()

    plt.scatter(cphases_pred, cphases_true, s=5, alpha=0.2, color='black')
    plt.plot(cphases_true, cphases_true,
             color='tomato', linewidth=2, label='1:1')
    plt.legend(loc='best')
    plt.show()


main()
