"""
Pure numpy/scipy stand-in for sklearn's TfidfVectorizer, for EVALUATION ONLY.

Used when scikit-learn cannot be imported (e.g. a Windows Application Control policy blocks one of its DLLs).
`install()` registers fake `sklearn.feature_extraction.text` modules so `linkage.py` imports unchanged.

Reproduces the settings linkage.py uses: analyzer='char', ngram_range=(2, 3), min_df=1, sublinear_tf=True,
and sklearn's defaults for the rest (lowercase, whitespace collapsed, smooth idf = ln((1+n)/(1+df)) + 1,
L2 row normalisation, terms unseen at fit time ignored). Not a general replacement.
"""
import re
import sys
import types

import numpy as np
import scipy.sparse as sp

_WS = re.compile(r'\s\s+')


class TfidfVectorizer:
    def __init__(self, analyzer='char', ngram_range=(1, 1), min_df=1, sublinear_tf=False, dtype=np.float64, **_):
        if analyzer != 'char':
            raise NotImplementedError("shim supports analyzer='char' only")
        self.ngram_range, self.min_df, self.sublinear_tf, self.dtype = ngram_range, min_df, sublinear_tf, dtype
        self.vocabulary_ = {}
        self.idf_ = None

    def _grams(self, text):
        t = _WS.sub(' ', str(text).lower())
        lo, hi = self.ngram_range
        n = len(t)
        return [t[i:i + k] for k in range(lo, hi + 1) for i in range(n - k + 1)]

    def _counts(self, docs, grow):
        indptr, indices, data = [0], [], []
        for doc in docs:
            row = {}
            for g in self._grams(doc):
                j = self.vocabulary_.get(g)
                if j is None:
                    if not grow:
                        continue
                    j = self.vocabulary_.setdefault(g, len(self.vocabulary_))
                row[j] = row.get(j, 0) + 1
            indices.extend(row.keys())
            data.extend(row.values())
            indptr.append(len(indices))
        return indptr, indices, data

    def fit_transform(self, docs):
        docs = list(docs)
        self.vocabulary_ = {}
        indptr, indices, data = self._counts(docs, grow=True)
        X = sp.csr_matrix((np.array(data, dtype=np.float64), indices, indptr), shape=(len(docs), len(self.vocabulary_)))
        df = np.bincount(X.indices, minlength=X.shape[1])
        keep = np.flatnonzero(df >= self.min_df)
        if len(keep) != X.shape[1]:
            remap = {old: new for new, old in enumerate(keep)}
            self.vocabulary_ = {g: remap[j] for g, j in self.vocabulary_.items() if j in remap}
            X, df = X[:, keep], df[keep]
        self.idf_ = np.log((1.0 + len(docs)) / (1.0 + df)) + 1.0
        return self._finish(X)

    def transform(self, docs):
        docs = list(docs)
        indptr, indices, data = self._counts(docs, grow=False)
        X = sp.csr_matrix((np.array(data, dtype=np.float64), indices, indptr), shape=(len(docs), len(self.vocabulary_)))
        return self._finish(X)

    def _finish(self, X):
        if self.sublinear_tf:
            X.data = 1.0 + np.log(X.data)
        X = X.multiply(self.idf_).tocsr()
        norms = np.sqrt(np.asarray(X.multiply(X).sum(axis=1)).ravel())
        norms[norms == 0] = 1.0
        X = sp.diags(1.0 / norms) @ X
        return X.tocsr().astype(self.dtype)


def install():
    """Register the shim as sklearn.feature_extraction.text unless the real one imports fine."""
    import os
    try:
        if os.environ.get('FORCE_TFIDF_SHIM'):
            raise ImportError('shim forced')
        import sklearn.feature_extraction.text  # noqa: F401
        return False
    except Exception:
        for name in [m for m in sys.modules if m == 'sklearn' or m.startswith('sklearn.')]:
            del sys.modules[name]
    pkg, fe, text = types.ModuleType('sklearn'), types.ModuleType('sklearn.feature_extraction'), \
        types.ModuleType('sklearn.feature_extraction.text')
    text.TfidfVectorizer = TfidfVectorizer
    pkg.feature_extraction, fe.text = fe, text
    sys.modules.update({'sklearn': pkg, 'sklearn.feature_extraction': fe, 'sklearn.feature_extraction.text': text})
    return True
