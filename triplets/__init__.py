import numpy as np
from abc import abstractmethod


class Triplet():

    @abstractmethod
    def get_triplet(self, i1, i2, i3):
        pass

    def get_triplets(self, indices):
        '''
            Given an array of triplet indices, return an array of triplets.
        '''
        triplets = np.zeros(indices.shape[0])
        for i in range(indices.shape[0]):
            triplets[i] = self.get_triplet(
                indices[i][0], indices[i][1], indices[i][2])
        return triplets
