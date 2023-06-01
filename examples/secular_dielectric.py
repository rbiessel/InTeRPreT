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

    P = 30

    twoHop = TwoHopBasis(P)
    As = twoHop.basis_matrix_std_basis()
    twoHop.evaluate_covariance(shrubs.covariance(P))

    covariance = shrubs.covariance(P)
    intensity = 10 * np.log10(4e2 * np.abs(np.diag(covariance)))
    indices = twoHop.basis_indices()
    indices_max = indices

    timescales = twoHop.ptau
    times = twoHop.pt

    itriplet = IntensityTriplet(
        intensity, function='arctan', kappa=0.2)

    itriplets = itriplet.get_triplets(indices)

    cmodel = LSClosureModel([itriplet], shrubs.covariance(P))
    cmodel.train(indices_max)
    cmatrix = cmodel.get_correction_matrix(indices)

    plt.imshow(np.angle(cmatrix), cmap=plt.cm.seismic)
    plt.show()

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
