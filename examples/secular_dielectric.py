from closig.model import SeasonalVegLayer
import numpy as np
from matplotlib import pyplot as plt
import cphases.matrix as cpmatrix
from triplets.itriplet import IntensityTriplet
import seaborn as sns


def main():

    shrubs = SeasonalVegLayer(n_mean=1.2 - 0.01j, n_std=0, n_amp=0.01,
                              n_t=.1, P_year=30, density=0.5, dcoh=1, h=0.5, name='Shrubs')

    P = 30
    covariance = shrubs.covariance(P)
    intensity = 10 * np.log10(4e2 * np.abs(np.diag(covariance)))
    # Plot the covariance matrix
    # shrubs.plot_matrices(P)
    plt.plot(intensity)
    plt.show()
    print(covariance.shape)
    indices = cpmatrix.get_triplets_increasing(P)
    print(indices)

    itriplets = IntensityTriplet(
        intensity, function='tanh').get_triplets(indices)
    sns.kdeplot(itriplets, bw_adjust=0.5)
    plt.show()


main()
