"""Preprocessing module. All preprocessing must be fit on training fold only."""
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA


class FoldSafePreprocessor:
    """Preprocessing that fits on train fold only, transforms both."""

    def __init__(self, do_impute=True, do_scale=False, pca_dim=None):
        self.do_impute = do_impute
        self.do_scale = do_scale
        self.pca_dim = pca_dim
        self.imputer = None
        self.scaler = None
        self.pca = None

    def fit_transform(self, X_trn):
        """Fit on training data, return transformed training data."""
        X = X_trn.copy()

        if self.do_impute:
            self.imputer = SimpleImputer(strategy='median')
            X = self.imputer.fit_transform(X)

        if self.do_scale:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)

        if self.pca_dim is not None and self.pca_dim < X.shape[1]:
            self.pca = PCA(n_components=self.pca_dim)
            X = self.pca.fit_transform(X)
            var_explained = self.pca.explained_variance_ratio_.sum()
        else:
            var_explained = 1.0

        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return X, var_explained

    def transform(self, X_val):
        """Transform validation data using fitted preprocessors."""
        X = X_val.copy()

        if self.imputer is not None:
            X = self.imputer.transform(X)

        if self.scaler is not None:
            X = self.scaler.transform(X)

        if self.pca is not None:
            X = self.pca.transform(X)

        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return X


class TreePreprocessor:
    """Minimal preprocessing for tree models (impute only)."""

    def __init__(self):
        self.imputer = None

    def fit_transform(self, X_trn):
        self.imputer = SimpleImputer(strategy='median')
        X = self.imputer.fit_transform(X_trn)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return X, 1.0

    def transform(self, X_val):
        X = self.imputer.transform(X_val)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return X
