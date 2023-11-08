import numpy as np
from interpret.model import ClosurePredictor
from typing import Type


class LSClosureModel(ClosurePredictor):

    '''
        Multiple unweighted least squares for each triplet.

        'ax + by + ... + epsilon = c'
    '''

    def train(self, indices) -> ClosurePredictor:
        closures = np.angle(self.compute_bicoh(indices))

        if len(self.triplets) == 1:
            coeff = np.polyfit(
                self.triplets[0].get_triplets(indices), closures, 1)
            self.params = coeff
        else:
            raise NotImplementedError(
                'Multiple regression not implemented yet')

        self.compute_R(indices)

        return self

    def predict(self, indices) -> np.float64:
        super().predict(indices)
        if len(self.triplets) == 1:
            return self.triplets[0].get_triplets(indices) * self.params[0]
        else:
            raise NotImplementedError(
                'Multiple regression not implemented yet')


class TheilSenClosureModel(ClosurePredictor):

    def __init__(self, triplets, closures):
        raise NotImplementedError()
        assert len(
            triplets) == 1, 'Theil-Sen does not support multiple regression at this time'
        super().__init__(triplets, closures)

    def train(self, indices):
        super().train(indices)

        pass
