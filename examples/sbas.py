'''
    Proof of concept: 

    The intensity triplet can reconstruct dielectric driven closure phases at longer baselines, reaching performance similar to full-network estimators. The assumption here is that the intensity change over time is linear with respect to the dielectric change over time. This is based on current soil moisture estimation theory. For C-band and vegetation, its unlikely this will hold and its also unlikely that the coherence will be sufficient.

    Drawbacks:

    - Temporally consistent errors still cannot be constrained
    - A large number of looks is needed to resolve the systematic portion of the phase error (~400)
    - SBAS networks drastically reduce the available triplets for training and thus increases the sensitivity to decorrelation and outliers. For short networks, it may be advisable to persue a more robust estimator such as Theil-Sen or a brute force grid search
    - estimates of the slope between intensity and phase triplet may be slightly biased due to an additional timescale dependence. It's hard to resolve this without near-perfect coherence so in reality, the uncertainty driven by decorrelation is likely to be much larger than that due to these timescale dependencies. 
    - Assumes intensity triplet and closure phase are correlated and that other contributions to both these quantities are negligible. Both may be subject to other variability that aren't correlated with each other
'''

from closig.model import SeasonalVegLayer, PrecipScatterSoilLayer
from closig.expansion import SmallStepBasis, TwoHopBasis
from scripts.plotting import triangle_plot
from matplotlib import pyplot as plt
from closig.linking import EMI, EMI_py, CutOffRegularizer, NearestNeighbor, EVD
import cphases.util as util
from triplets.itriplet import IntensityTriplet
from closig.expansion import TwoHopBasis
from model.models import LSClosureModel
from cphases.bootstrapping import get_random_C
import matplotlib as mpl

import numpy as np


# define a veg layer with a secular trend
model = veg =  SeasonalVegLayer(n_mean=1.2 - 0.1j, n_std=0, n_amp=0,
                            n_t=0.1, P_year=30, density=2, dcoh=1, h=5, name='Shrubs')


# Analysis & Plotting
P = 90

fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))
model.plot_matrices(
    P, coherence=False, displacement_phase=False, ax=[ax[0, 0], ax[0, 1]])

# Compute closure phases and plot
smallStep = SmallStepBasis(P)
twoHop = TwoHopBasis(P)
c_phases_ss = smallStep.evaluate_covariance(
    model.covariance(P), compl=True, normalize=False)

c_phases_th = twoHop.evaluate_covariance(
    model.covariance(P), compl=True, normalize=False)
print(f'Mean Closure Phase: {np.angle(c_phases_ss).mean()}')

triangle_plot(smallStep, c_phases_ss,
              ax=ax[1, 1], cmap=plt.cm.seismic, vabs=180)
triangle_plot(twoHop, c_phases_th,
              ax=ax[1, 0], cmap=plt.cm.seismic, vabs=45)
ax[1, 1].set_title(
    f'Small Step Basis\n mean: {np.round(np.angle(c_phases_ss).mean(), 3)} rad')
ax[1, 0].set_title(
    f'Two Hop Basis\n mean: {np.round(np.angle(c_phases_th).mean(), 3)} rad')
plt.tight_layout()
plt.show()


## Introduce speckle or not into covariance matrix
cov_full = model.covariance(P, coherence=False)
# cov_full = get_random_C(cov_full, 400, coherence=False)

pl_evd_full = EVD().link(cov_full, G=np.abs(cov_full))

# Setup intensity triple
intensity = 10 * np.log10(4e2 * np.abs(np.diag(cov_full)))
indices_all = twoHop.basis_indices()
itriplet = IntensityTriplet(intensity, function='arctan', kappa=11.5)
# itriplet = IntensityTriplet(intensity, function='arctan', kappa=11)


full_model = LSClosureModel([itriplet], cov_full)
full_model.plot_scatter(indices_all)

cphases_pred = full_model.train(indices_all).predict(indices_all)
cphases_true = np.angle(full_model.compute_bicoh(indices_all)) 
timescales = twoHop.ptau
colors = plt.cm.viridis(timescales / np.max(timescales))
scat = plt.scatter(cphases_pred, cphases_true, s=5, alpha=1, color=colors)
cbar = plt.colorbar(scat)
cbar.ax.set_title(r'Normed p$\tau$')
plt.xlabel('True Closure Phases')
plt.ylabel('Predicted Closure Phases')
plt.plot(cphases_true, cphases_true,
            color='tomato', linewidth=2, label='1:1')
plt.legend(loc='best')
plt.show()

## Temporal baseline experiment
# taus = np.arange(5, P + 4, 10)
taus = np.array([1, 2, 4, 60, 90])
l2error = np.zeros((len(taus)))
colors = plt.cm.viridis(np.linspace(0, 1, len(taus)))
ierrors = np.zeros((len(taus)))
for tau, color, i in zip(taus, colors, range(len(taus))):
    G = np.abs(model.covariance(P, coherence=True))
    G = CutOffRegularizer().regularize(G, tau_max=tau)
    cov = cov_full * G
    pl_evd = EVD().link(cov, G=G)

    if tau >= 4:
        subset_indices = twoHop.basis_indices(cutoff=tau)
        closure_model = LSClosureModel([itriplet], cov)
        correction = closure_model.train(subset_indices).get_correction_matrix(indices_all)
        pl_evd_corr = EVD().link(cov * correction.conj(), G=G)
        plt.plot(np.angle(pl_evd_corr), '--', label=f'bw-{tau}-trip', color = color)

    plt.plot(np.angle(pl_evd), label=f'bw-{tau}', color = color)


plt.legend(loc='best')
plt.xlabel('t')
plt.ylabel('Phase [rad]')
plt.grid(alpha=0.5)
plt.tight_layout()
plt.show()

'''

'''
