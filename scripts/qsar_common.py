"""Ortak parcalar: CV icinde calisan on-isleme (eksik doldurma, sabit sutun, korelasyon filtresi)."""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class CorrFilter(BaseEstimator, TransformerMixin):
    """Egitim verisinde |r| > threshold olan ciftlerden daha sonraki sutunu atar. Yalniz fit verisine bakar."""
    def __init__(self, threshold=0.9): self.threshold = threshold
    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        sd = X.std(0); keep = sd > 1e-8
        idx = np.where(keep)[0]
        C = np.abs(np.corrcoef(X[:, idx], rowvar=False)) if len(idx) > 1 else np.ones((1, 1))
        C = np.nan_to_num(C)
        dropped = np.zeros(len(idx), dtype=bool)
        for i in range(len(idx)):
            if dropped[i]: continue
            hit = C[i] > self.threshold; hit[:i + 1] = False
            dropped |= hit
        self.keep_ = idx[~dropped]
        return self
    def transform(self, X): return np.asarray(X, dtype=float)[:, self.keep_]

class Clean(BaseEstimator, TransformerMixin):
    """inf/NaN -> egitim medyani; asiri degerleri egitim 0.5-99.5 yuzdeligine kirpar (RDKit Ipc vb.)."""
    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float).copy(); X[~np.isfinite(X)] = np.nan
        self.med_ = np.nan_to_num(np.nanmedian(X, 0))
        self.lo_, self.hi_ = np.nanpercentile(X, 0.5, 0), np.nanpercentile(X, 99.5, 0)
        self.lo_ = np.nan_to_num(self.lo_); self.hi_ = np.nan_to_num(self.hi_)
        return self
    def transform(self, X):
        X = np.asarray(X, dtype=float).copy(); X[~np.isfinite(X)] = np.nan
        r, c = np.where(np.isnan(X)); X[r, c] = self.med_[c]
        return np.clip(X, self.lo_, self.hi_)
