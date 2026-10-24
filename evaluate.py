"""
Evaluation of the linkage method on synthetic benchmarks: baselines, ablations, per-noise-type recall.

  python evaluate.py                       # 3 seeds, default size
  python evaluate.py --seeds 5 --n-true 600 --out results

Systems
  Proposed (auto)          pairs the matcher labels perfect/high after one-to-one assignment
  Proposed (auto+review)   ... plus the 'review' pairs a human would look at
  Exact                    normalized-whitespace name equality AND district+ward found in an address
  Fuzzy                    rapidfuzz name + address similarity, threshold, greedy one-to-one
  Fellegi-Sunter           unsupervised EM on comparison vectors, posterior > 0.5, greedy one-to-one
Ablations (each removes ONE component from the proposed method; reported as auto+review and auto)
  no_fold, no_roman, no_guards, per_address_only, name_only, hungarian
All systems see the same data. Reviewer decisions are not used: every run starts from an empty in-memory store.
"""
import argparse
import os
import re
import sys
import tempfile
import unicodedata
from collections import defaultdict
from contextlib import contextmanager

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, distance
from scipy.optimize import linear_sum_assignment

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tfidf_shim            # noqa: E402
SHIM_USED = tfidf_shim.install()   # numpy TF-IDF when scikit-learn cannot be imported (evaluation only)
from sklearn.feature_extraction.text import TfidfVectorizer   # noqa: E402
import linkage as M          # noqa: E402

# explicit column mapping: the released files use neutral column names
MAPPING1 = {'name': 'full_name'}
MAPPING2 = {'name': 'full_name'}
from synth import generate, NAME_OPS, ADDR_OPS   # noqa: E402

DIGITS = str.maketrans('०१२३४५६७८९', '0123456789')


# ------------------------------------------------------------------ metrics
def prf(pred: set, gold: set):
    tp = len(pred & gold)
    p = tp / len(pred) if pred else 1.0
    r = tp / len(gold) if gold else 1.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def score_system(pred: set, gold_df: pd.DataFrame) -> dict:
    gold = {(a, b) for a, b in zip(gold_df.A_id, gold_df.B_id) if b}
    p, r, f = prf(pred, gold)
    no_match = set(gold_df[gold_df.B_id == ''].A_id)
    fl = sum(1 for a, _ in pred if a in no_match) / max(1, len(no_match))   # false-link rate
    by_op = {}
    t = gold_df[gold_df.B_id != '']
    for op in NAME_OPS + ADDR_OPS + ['clean', 'namesake_near']:
        sub = t[t.ops.str.split('+').apply(lambda x: op in x)]
        if len(sub):
            by_op[op] = sum(1 for a, b in zip(sub.A_id, sub.B_id) if (a, b) in pred) / len(sub)
    return {'precision': p, 'recall': r, 'f1': f, 'false_link_rate': fl, **{f'rec_{k}': v for k, v in by_op.items()}}


# ------------------------------------------------------------------ proposed method (+ ablations)
def run_matcher(A: pd.DataFrame, B: pd.DataFrame, hungarian: bool = False):
    store = M.FeedbackStore()
    rm = M.RecordMatcher(feedback_store=store)
    s = rm.match_datasets(A, B, MAPPING1, MAPPING2)
    assign = list(s.assign)
    if hungarian:
        assign = hungarian_assign(s)
    auto, anyp = set(), set()
    for i, j in enumerate(assign):
        if j is None:
            continue
        inf = s._info(i, j)
        pair = (s.d1[i]['id'], s.d2[j]['id'])
        if inf['match_category'] in ('perfect', 'high', 'verified'):
            auto.add(pair)
        if inf['match_category'] in ('perfect', 'high', 'verified', 'review'):
            anyp.add(pair)
    return auto, anyp


def hungarian_assign(s):
    """Optimal one-to-one assignment over the same scored candidates (maximises total confidence)."""
    n1 = len(s.d1)
    cols, entries = {}, []
    for i, cands in enumerate(s.candidates):
        for j, inf in s.rank(cands):
            if inf['match_category'] in ('perfect', 'high', 'review'):
                entries.append((i, cols.setdefault(j, len(cols)), j, inf['confidence_score']))
    # cost -score on candidate edges, 0 elsewhere: a 0-cost cell means "left unassigned", so the solver is free
    # to skip a weak edge instead of being forced into a maximum-cardinality matching.
    C = np.zeros((n1, len(cols)))
    for i, c, j, sc in entries:
        C[i, c] = min(C[i, c], -sc)
    rows, cs = linear_sum_assignment(C)
    inv = {c: j for j, c in cols.items()}
    out = [None] * n1
    for i, c in zip(rows, cs):
        if C[i, c] < 0:
            out[i] = inv[c]
    return out


def _rebuild_tables():
    """Recompute everything in linkage.py that was derived from normalize_nepali_text at import time."""
    forms = {}
    for code, names in M.NEPAL_DISTRICTS.items():
        for n in names:
            norm = M.normalize_nepali_text(n)
            if norm:
                forms.setdefault(norm, code)
    M._DISTRICT_FORMS = forms
    M._DISTRICT_FORMS_SORTED = sorted(forms.items(), key=lambda kv: -len(kv[0]))
    M.LOCAL_LEVEL_SUFFIX_TOKENS = {M.normalize_nepali_text(s) for s in [
        'गाउँपालिका', 'गाउपालिका', 'गाउपालीका', 'गाउँपालीका', 'गापा', 'गा.पा.', 'नगरपालिका', 'नगरपालीका', 'नपा',
        'महानगरपालिका', 'उपमहानगरपालिका', 'उपमहानपा', 'महानपा', 'गाविस', 'पालिका']}


@contextmanager
def patched(ablation: str):
    names = ['CHAR_FOLDS', 'PHRASE_REPLACEMENTS', 'to_roman_phonetic', 'evaluate_names', 'score_address',
             '_DISTRICT_FORMS', '_DISTRICT_FORMS_SORTED', 'LOCAL_LEVEL_SUFFIX_TOKENS']
    saved = {n: getattr(M, n) for n in names}
    try:
        if ablation == 'no_fold':
            M.CHAR_FOLDS, M.PHRASE_REPLACEMENTS = [], []
            _rebuild_tables()
        elif ablation == 'no_roman':
            M.to_roman_phonetic = lambda t: ''
        elif ablation == 'no_guards':
            orig = saved['evaluate_names']

            def no_guards(raw1, norm1, rom1, raw2, norm2, rom2, fs=None):
                out = orig(raw1, norm1, rom1, raw2, norm2, rom2, fs)
                if out['status'] == 'Review' and out['score'] == 75.0 and 'फरक' in out['reason'] and '(' in out['reason']:
                    sc = max(fuzz.token_sort_ratio(norm1, norm2), fuzz.token_sort_ratio(rom1, rom2),
                             distance.JaroWinkler.similarity(norm1, norm2) * 100.0,
                             distance.JaroWinkler.similarity(rom1, rom2) * 100.0)
                    if len(norm1.split()) >= 2 and len(norm2.split()) >= 2:
                        sc = max(sc, fuzz.token_set_ratio(norm1, norm2), fuzz.token_set_ratio(rom1, rom2))
                    sc = round(sc, 1)
                    return {'score': sc, 'reason': 'guard removed', 'status': 'High' if sc >= 85 else 'Review'}
                return out
            M.evaluate_names = no_guards
        elif ablation == 'name_only':
            orig_addr = saved['score_address']

            def flat(d1, d2):
                out = orig_addr(d1, d2)
                out.update(score=100.0, matched=True)
                return out
            M.score_address = flat
        yield
    finally:
        for n, v in saved.items():
            setattr(M, n, v)


# ------------------------------------------------------------------ shared blocking for the baselines
def _raw_text(row_vals):
    return ' '.join(str(v) for v in row_vals if str(v) not in ('', 'nan'))


def baseline_candidates(A, B, k=30):
    ta = [_raw_text(r) for r in A[['full_name', 'district', 'locallevel']].values]
    tb = [_raw_text(r) for r in B[['full_name', 'PerAddress', 'TemAddress', 'OtherAddress']].values]
    v = TfidfVectorizer(analyzer='char', ngram_range=(2, 3), sublinear_tf=True, dtype=np.float32)
    mb = v.fit_transform(tb)
    ma = v.transform(ta)
    sims = (ma @ mb.T).toarray()
    return [list(np.argsort(-row)[:k]) for row in sims]


def _addr_texts(b):
    return [str(b[c]) for c in ('PerAddress', 'TemAddress', 'OtherAddress') if str(b[c]) not in ('', 'nan')]


def _ward_in(ward, text):
    return ward in [str(int(x)) for x in re.findall(r'\d+', text.translate(DIGITS))]


def greedy_one_to_one(triples, threshold, A, B):
    used_a, used_b, out = set(), set(), set()
    for sc, i, j in sorted(triples, key=lambda t: -t[0]):
        if sc < threshold or i in used_a or j in used_b:
            continue
        used_a.add(i), used_b.add(j)
        out.add((A.id.iloc[i], B.id.iloc[j]))
    return out


def run_exact(A, B, cands):
    pairs = []
    for i, js in enumerate(cands):
        a = A.iloc[i]
        na = ' '.join(a.full_name.lower().split())
        for j in js:
            b = B.iloc[j]
            if ' '.join(str(b.full_name).lower().split()) != na:
                continue
            ok = any(a.district in t and _ward_in(str(a.wardno).translate(DIGITS), t) for t in _addr_texts(b))
            if ok:
                pairs.append((1.0, i, j))
    return greedy_one_to_one(pairs, 0.5, A, B)


def fuzzy_triples(A, B, cands):
    pairs = []
    for i, js in enumerate(cands):
        a = A.iloc[i]
        for j in js:
            b = B.iloc[j]
            name = fuzz.token_sort_ratio(a.full_name, str(b.full_name))
            ads = _addr_texts(b)
            addr = max([(fuzz.partial_ratio(str(a.district), t) + fuzz.partial_ratio(str(a.locallevel), t)
                         + (100 if _ward_in(str(a.wardno).translate(DIGITS), t) else 0)) / 3 for t in ads] or [0])
            pairs.append(((name + addr) / 2, i, j))
    return pairs


def fs_vectors(A, B, cands):
    rows = []
    for i, js in enumerate(cands):
        a = A.iloc[i]
        w = str(a.wardno).translate(DIGITS)
        for j in js:
            b = B.iloc[j]
            jw = distance.JaroWinkler.similarity(a.full_name, str(b.full_name))
            name = 0 if a.full_name == str(b.full_name) else 1 if jw >= .92 else 2 if jw >= .8 else 3
            ads = _addr_texts(b)
            rows.append((i, j, name, int(any(str(a.district) in t for t in ads)),
                         int(any(str(a.locallevel) in t for t in ads)), int(any(_ward_in(w, t) for t in ads))))
    return np.array(rows)


def fs_triples(A, B, cands, iters=40):
    X = fs_vectors(A, B, cands)
    levels = [4, 2, 2, 2]
    F = X[:, 2:].astype(int)
    p = 0.03
    # u (agreement among NON-matches) is estimated from uniformly random A-B pairs, as in standard practice: pairs
    # that survived blocking are far more similar than non-matches in general, which would corrupt u.
    rng = np.random.default_rng(0)
    R = fs_vectors(A, B, [list(rng.choice(len(B), 20, replace=False)) for _ in range(len(A))])[:, 2:].astype(int)
    u = [np.bincount(R[:, k], minlength=levels[k]) / len(R) + 1e-4 for k in range(4)]
    m = [np.array([.6, .2, .1, .1]), np.array([.05, .95]), np.array([.1, .9]), np.array([.2, .8])]
    for _ in range(iters):            # EM updates m and the match prior only
        lm = np.log(p) + sum(np.log(m[k][F[:, k]]) for k in range(4))
        lu = np.log(1 - p) + sum(np.log(u[k][F[:, k]]) for k in range(4))
        g = 1 / (1 + np.exp(lu - lm))
        p = g.mean()
        for k in range(4):
            for lv in range(levels[k]):
                m[k][lv] = (g[F[:, k] == lv].sum() + 1e-3) / (g.sum() + 1e-3 * levels[k])
    lm = np.log(p) + sum(np.log(m[k][F[:, k]]) for k in range(4))
    lu = np.log(1 - p) + sum(np.log(u[k][F[:, k]]) for k in range(4))
    w = lm - lu          # log-odds match weight (posterior > 0.5  <=>  w > 0); posteriors saturate at 1.0, weights rank
    return [(w[r], int(X[r, 0]), int(X[r, 1])) for r in range(len(X))]


FUZZY_GRID = [50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
FS_GRID = [-10, -5, 0, 3, 5, 7, 8, 9, 10, 12, 15]    # thresholds on the match weight
DEV_SEED = 10_000        # threshold tuning set: never one of the evaluation seeds


def tune_baselines(n_true):
    """Best-F1 threshold for each baseline on a separate dev dataset, so baselines are not handicapped by a guess."""
    A, B, gold = generate(n_true=n_true, seed=DEV_SEED)
    cands = baseline_candidates(A, B)
    out = {}
    for name, fn, grid in (('fuzzy', fuzzy_triples, FUZZY_GRID), ('fs', fs_triples, FS_GRID)):
        tri = fn(A, B, cands)
        out[name] = max(grid, key=lambda t: score_system(greedy_one_to_one(tri, t, A, B), gold)['f1'])
    print(f"baseline thresholds tuned on dev seed {DEV_SEED}: {out}", flush=True)
    return out


# ------------------------------------------------------------------ driver
def one_seed(seed, n_true, thr, hungarian_too=True):
    A, B, gold = generate(n_true=n_true, seed=seed)
    results = {}
    with patched('none'):
        auto, anyp = run_matcher(A, B)
    results['Proposed (auto+review)'], results['Proposed (auto)'] = anyp, auto
    for ab in ('no_fold', 'no_roman', 'no_guards', 'name_only'):
        with patched(ab):
            auto, anyp = run_matcher(A, B)
        results[f'  - {ab} (auto+review)'], results[f'  - {ab} (auto)'] = anyp, auto
    B1 = B.copy()
    B1['TemAddress'] = B1['OtherAddress'] = ''
    auto, anyp = run_matcher(A, B1)
    results['  - per_address_only (auto+review)'], results['  - per_address_only (auto)'] = anyp, auto
    if hungarian_too:
        auto, anyp = run_matcher(A, B, hungarian=True)
        results['  ~ hungarian (auto+review)'], results['  ~ hungarian (auto)'] = anyp, auto
    cands = baseline_candidates(A, B)
    results['Baseline: Exact'] = run_exact(A, B, cands)
    results[f"Baseline: Fuzzy (thr {thr['fuzzy']}, tuned)"] = greedy_one_to_one(fuzzy_triples(A, B, cands), thr['fuzzy'], A, B)
    results[f"Baseline: Fellegi-Sunter (weight>{thr['fs']}, tuned)"] = greedy_one_to_one(fs_triples(A, B, cands), thr['fs'], A, B)
    return {k: score_system(v, gold) for k, v in results.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--n-true', type=int, default=600)
    ap.add_argument('--out', default=os.path.join(HERE, 'results'))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = []
    thr = tune_baselines(a.n_true)
    for seed in range(1, a.seeds + 1):
        print(f'seed {seed} ...', flush=True)
        for sysname, m in one_seed(seed, a.n_true, thr).items():
            rows.append({'seed': seed, 'system': sysname, **m})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, 'per_seed.csv'), index=False, encoding='utf-8-sig')
    order = list(dict.fromkeys(df.system))
    main_cols = ['precision', 'recall', 'f1', 'false_link_rate']
    agg = df.groupby('system')[main_cols].agg(['mean', 'std']).reindex(order)
    summary = pd.DataFrame({c: [f"{agg[(c, 'mean')][s]:.3f} ± {agg[(c, 'std')][s]:.3f}" for s in order] for c in main_cols},
                           index=order)
    summary.to_csv(os.path.join(a.out, 'summary.csv'), encoding='utf-8-sig')
    op_cols = [c for c in df.columns if c.startswith('rec_')]
    per_op = df.groupby('system')[op_cols].mean().reindex(order).round(3)
    per_op.to_csv(os.path.join(a.out, 'recall_by_noise_type.csv'), encoding='utf-8-sig')
    pd.set_option('display.width', 250, 'display.max_columns', 50)
    print(f"\nMean ± sd over {a.seeds} seeds (n_true={a.n_true}); TF-IDF: "
          f"{'numpy shim (scikit-learn unavailable)' if SHIM_USED else 'scikit-learn'}\n")
    print(summary.to_string())
    print("\nRecall by noise type (mean over seeds)\n")
    print(per_op.to_string())


if __name__ == '__main__':
    main()
