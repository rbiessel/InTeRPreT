
import numpy as np
from triplets import Triplet


class IntensityTriplet(Triplet):
    '''
        A triplet of intensity measurements. It's phase analog has a sigmoidal form. The triplet is then its linear mapping and is consistent with the closure phase mapping.
    '''

    def __init__(self, intensity, function='tanh', L=1, kappa=1, cubic=False, norm=False):
        '''
            Available functions are 'tanh', 'arctan', and 'logistic'
            kappa is the steepness of the function and is related to the dynamic range of the intensity measurements. Should be consistent accross all triplets in the same pixel.
        '''
        self.L = L
        self.k = kappa
        self.cubic = cubic
        self.norm = norm
        self.intensity = intensity
        self.name = 'Intensity Triplet'

        if function == 'tanh':
            self.form = self._tanh
        elif function == 'arctan':
            self.form = self._arctan
        else:
            self.form = self._logistic

    def _logistic(self, x):
        return ((1/(1 + np.exp(-x * self.k))) - (1/2))

    def _arctan(self, x):
        return np.arctan(self.k * x)

    def _tanh(self, x):
        return np.tanh(self.k * x)

    def get_triplet(self, i1, i2, i3):
        i1 = self.intensity[i1]
        i2 = self.intensity[i2]
        i3 = self.intensity[i3]

        triplet = (self.form((i2 - i1)) +
                   self.form((i3 - i2)) - self.form((i3 - i1)))
        triplet *= self.L

        if self.norm:
            triplet = triplet / np.sqrt(i1**2 + i2**2 + i3**2)

        return triplet
