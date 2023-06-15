'''
    Simulating a semi-arid environment with sparse shrubbery with seasonal moisture content
'''
from closig.model import SeasonalVegLayer
from closig.expansion import SmallStepBasis, TwoHopBasis
from scripts.plotting import triangle_plot
from matplotlib import pyplot as plt
from closig.linking import EMI, EMI_py, CutOffRegularizer, NearestNeighbor, EVD
import cphases.util as util
from triplets.itriplet import IntensityTriplet
from closig.expansion import TwoHopBasis
from model.models import LSClosureModel
from cphases.bootstrapping import get_random_C

import numpy as np


# define a veg layer with a secular trend
model = SeasonalVegLayer(n_mean=1.2 - 0.01j, n_std=0, n_amp=0.05,
                            n_t=0.2, P_year=30, density=0.05, dcoh=0.8, coh0=0.1, h=2, name='Shrubs')


# Analysis & Plotting

# Analysis & Plotting
P = 60

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


taus = np.arange(5, P, 10)
l2error = np.zeros((len(taus)))
colors = plt.cm.viridis(np.linspace(0, 1, len(taus)))
ierrors = np.zeros((len(taus)))

# Compute full-network solution
cov_full = model.covariance(P, coherence=False)
# cov_full = get_random_C(cov_full, 1000, coherence=False)

# Simulate speckle
pl_evd_full = EVD().link(cov_full, G=np.abs(cov_full))

# Setup intensity triple
intensity = 10 * np.log10(4e2 * np.abs(np.diag(cov_full)))
indices_all = twoHop.basis_indices()
itriplet = IntensityTriplet(intensity, function='tanh', kappa=11.5)
# itriplet = IntensityTriplet(intensity, function='tanh', kappa=1)

LSClosureModel([itriplet], cov_full).plot_scatter(indices_all)


for tau, color, i in zip(taus, colors, range(len(taus))):
    G = np.abs(model.covariance(P, coherence=True))
    G = CutOffRegularizer().regularize(G, tau_max=tau)
    cov = cov_full * G

    subset_indices = twoHop.basis_indices(cutoff=tau)
    closure_model = LSClosureModel([itriplet], cov)
    correction = closure_model.train(subset_indices).get_correction_matrix(indices_all)


    pl_evd = EVD().link(cov, G=G)
    pl_evd_corr = EVD().link(cov * correction.conj(), G=G)

    plt.plot(np.angle(pl_evd), label=f'bw-{tau}', color = color)
    plt.plot(np.angle(pl_evd_corr), '--', label=f'bw-{tau}-trip', color = color)


plt.legend(loc='best')
plt.xlabel('t')
plt.ylabel('Phase [rad]')
plt.grid(alpha=0.5)
plt.tight_layout()
plt.show()

'''

'''
