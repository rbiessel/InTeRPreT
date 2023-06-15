import numpy as np
from abc import abstractmethod
from triplets import Triplet
from typing import List
from cphases import util
import scipy.stats as stats


class ClosurePredictor():
    '''
        Objective: ingest arbitrary set of triplets
            train the model on a specified subset of triplets
            predict the remaining triplets

        For use with full-network phase linking, train on all triplets. The functionality to train with only a subset is designed for SBAS purposes.
    '''

    def __init__(self, triplets: List[Triplet], covariance):
        assert len(triplets) > 0, 'Must provide at least one triplet'
        self.triplets = triplets
        self.covariance = covariance
        # Quality metrics and parameters, to be defined during training

        self.params = None
        self.r_squared = None
        self.rmse = None

    def compute_bicoh(self, indices) -> np.complex64:
        '''
            Compute the closures for a given set of indices
            returns the complex bicoherence for a set of indices
        '''
        bicoherence = np.zeros(indices.shape[0], dtype=np.complex64)
        for i in range(indices.shape[0]):
            trip = indices[i]
            bicoherence[i] = self.covariance[trip[0], trip[1]] * \
                self.covariance[trip[1], trip[2]] * \
                self.covariance[trip[2], trip[0]]

        return bicoherence

    def plot_scatter(self, indices, ax=None, model=False):
        '''
            Plot the scatter plot of the model
        '''
        from matplotlib import pyplot as plt
        returnax = False
        if ax is None:
            fig, ax = plt.subplots(nrows=1, ncols=len(self.triplets))
        else:
            assert len(ax) == len(self.triplets)
            returnax = True

        if model:
            assert self.params is not None, 'Model must be trained before plotting model'

        if len(self.triplets) == 1:
            ax = [ax]

        closures = np.angle(self.compute_bicoh(indices))

        for i in range(len(ax)):
            eval_triplets = self.triplets[i].get_triplets(
                indices)

            # Plot fit
            if model:
                ostd = np.std(eval_triplets)
                x = np.linspace(np.min(eval_triplets) - ostd, np.max(
                    eval_triplets) + ostd, len(eval_triplets) * 4)
                y = x * self.params[i]
                ax[i].plot(x, y, color='tomato', linewidth=2)
                ax[i].plot(x, y + self.params[-1], '--',
                           color='gray', linewidth=2, alpha=0.4)

            # Plot scatter
            ax[i].scatter(eval_triplets, closures, s=5,
                          alpha=0.2, color='black')
            ax[i].set_xlabel(f'{self.triplets[i].name} [-]')
            ax[i].set_ylabel('Closures [$\mathrm{rad}$]')
        if returnax:
            return ax
        else:
            plt.tight_layout()
            plt.show()

    def predict_phase(self, indices, weights=None) -> np.complex64:
        '''
            Return a correction matrix containing a minimum norm solution for the predicted phases
        '''
        assert self.params is not None, 'Model must be trained before predicting phase'

        cphases = self.predict(indices)
        A = util.build_A(indices, self.covariance)

        if weights is None:
            phases = np.exp(1j * np.linalg.lstsq(A, cphases)
                            [0], dtype=np.complex64)
            return phases
        else:

            weights = util.coherence_to_phivec(
                weights)
            weights = np.diag(np.abs(weights**2))
            # assert len(
            #     weights.shape) == 2 and weights.shape[0] == weights.shape[1], 'Weights must be a square matrix'
            Qinv = np.linalg.pinv(weights)
            phases = Qinv @ A.T @ np.linalg.pinv(A @ Qinv @ A.T) @ cphases
            return np.exp(1j * phases, dtype=np.complex64)

    def get_correction_matrix(self, indices, apply_in_place=False, weights=None) -> np.complex64:
        '''
            Return the already correction matrix for a given set of indices. 
        '''
        phivec = self.predict_phase(indices, weights=weights)

        if apply_in_place:
            self.covariance *= util.phivec_to_coherence(
                phivec, self.covariance.shape[0]).conj()
        else:
            return util.phivec_to_coherence(phivec, self.covariance.shape[0])

    def compute_RMSE(self):
        assert self.params is not None, 'Model must be trained before computing RMSE'
        pass

    def compute_R(self, indices):
        if len(self.triplets) > 1:
            raise NotImplementedError(
                'R^2 not implemented for multiple triplets')
            assert self.params is not None, 'Model must be trained before computing R^2'

        return stats.pearsonr(self.triplets[0].get_triplets(indices), np.angle(self.compute_bicoh(indices)))[0]

    @ abstractmethod
    def train(self, indices):
        pass

    def predict(self, indices) -> np.float64:
        assert self.params is not None, 'Model must be trained before predicting'
        pass
