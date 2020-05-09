# -*- coding: utf-8 -*-
"""
Created on Sat May  2 11:40:50 2020

@author: Carlos
"""
import numpy as np


class Normalizer:

    def __init__(self):

        self.normalization = None

    def normalize_features(self, X, reuse=False):
        """
        Normalize features matrix with mean normalization
        and feature (max-min) scaling
        Saves the Normalization for use in reuse_

        Parameters
        ----------
        X : np.ndarray
            n*m+1 features matrix

        Returns
        -------
        X : np.ndarray
            New array of n*m+1 features matrix normalized

        """
        X_norm = np.copy(X)

        if not reuse:
            self.normalization = np.empty(X.shape[1], dtype=tuple)

        # Si hay que reutilizar, pero el objeto no está inicializado
        elif self.normalization is None:
            raise Exception("Not initialized")

        elif X.shape[1] != self.normalization.size:
            raise Exception("Different size")

        # For every feature
        for i in range(X.shape[1]):

            if reuse:
                mean = self.normalization[i][0]
                rango = self.normalization[i][1]
            else:
                mean = np.mean(X[:, i])
                rango = X[:, i].max() - X[:, i].min()
                self.normalization[i] = (mean, rango)

            # Normalize
            if rango != 0:
                X_norm[:, i] = (X[:, i] - mean) / rango

        return X_norm


if __name__ == "__main__":
    a = np.array([[1., 2., 3.], [7., 1., 40.]])
    norm = Normalizer()
    b = norm.normalize_features(a)
    c = np.array([[10., 11.]])
    d = norm.normalize_features(c, reuse=True)
    pass
