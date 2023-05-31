from closig.model import SeasonalVegLayer
import numpy as np
from matplotlib import pyplot as plt
import cphases.matrix as cpmatrix
from triplets.itriplet import IntensityTriplet
from model.models import LSClosureModel

import seaborn as sns


def main():

    shrubs = SeasonalVegLayer(n_mean=1.2 - 0.01j, n_std=0, n_amp=0.01,
                              n_t=.1, P_year=30, density=0.5, dcoh=1, h=0.5, name='Shrubs')

    P = 30
    covariance = shrubs.covariance(P)
    intensity = 10 * np.log10(4e2 * np.abs(np.diag(covariance)))
    indices = cpmatrix.get_triplets_increasing(P, force=0)
    indices_max = cpmatrix.get_triplets_increasing(P, max=30, force=0)

    timescales = np.zeros(indices.shape[0])
    for i in range(indices.shape[0]):
        timescales[i] = np.max(indices[i]) - np.min(indices[i])

    itriplet = IntensityTriplet(
        intensity, function='tanh', kappa=10)

    itriplets = itriplet.get_triplets(indices)

    cmodel = LSClosureModel([itriplet], shrubs.covariance(P))
    cmodel.train(indices_max)
    colors = plt.cm.viridis(timescales / np.max(timescales))

    plt.scatter(itriplets,
                np.angle(cmodel.compute_bicoh(indices)), c=colors, alpha=0.5)
    cbar = plt.colorbar(orientation='vertical')
    cbar.ax.set_title(r'$\tau$')
    plt.xlabel('Intensity Triplet [-]')
    plt.ylabel('Closure Phase [$\mathrm{rad}$]')
    plt.show()

    cphases_pred = cmodel.predict(indices)
    cphases_true = np.angle(cmodel.compute_bicoh(indices))

    plt.scatter(cphases_pred, cphases_true, s=5, alpha=0.2, color='black')
    plt.plot(cphases_true, cphases_true,
             color='tomato', linewidth=2, label='1:1')
    plt.legend(loc='best')
    plt.show()


main()
