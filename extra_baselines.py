"""
Two further baselines on the same synthetic benchmark as evaluate.py:

  Splink              probabilistic (Fellegi-Sunter) linkage with Splink's standard unsupervised training
                      (pip install "splink<5"; run and tested with Splink 4.0.17 / DuckDB 2.0.0; the Splink 5 code path is
                      written but untested because pyarrow could not be loaded on the test machine)
  Supervised (GBM)    gradient-boosted trees on pair features, trained on labeled pairs of a SEPARATE synthetic seed

Both give their scores to the same greedy one-to-one step as the other baselines, with the threshold tuned on
separate synthetic seeds (never the evaluation seeds). Splink uses its own candidate generation (blocking rules);
the supervised model uses the TF-IDF candidates of the other baselines.

  python extra_baselines.py --seeds 5 --n-true 600 --out results
Set SPLINK_PATH to a folder with a private Splink install if it is not installed in the active environment.
"""
import argparse
import os
import re
import sys
import warnings

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
if os.environ.get('SPLINK_PATH'):
    sys.path.insert(0, os.environ['SPLINK_PATH'])
warnings.filterwarnings('ignore')
import evaluate as E                         # noqa: E402
from synth import generate                   # noqa: E402

DIGITS = E.DIGITS
SPLINK_GRID = [-10, -5, 0, 2, 4, 6, 8, 10, 12, 15, 20]     # thresholds on the match weight
GBM_GRID = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]   # thresholds on predicted probability
DEV_SEEDS = (10000, 10001)


# ------------------------------------------------------------------ Splink
def _splink_frames(A, B):
    def nums(row):
        txt = ' '.join(str(row[c]) for c in ('PerAddress', 'TemAddress', 'OtherAddress') if str(row[c]) not in ('', 'nan'))
        found = sorted({str(int(x)) for x in re.findall(r'\d+', txt.translate(DIGITS))})
        return ('|' + '|'.join(found) + '|') if found else None

    a = pd.DataFrame({'unique_id': A['id'], 'name': A['full_name'], 'district': A['district'].replace('', None),
                      'locallevel': A['locallevel'].replace('', None),
                      'ward': A['wardno'].map(lambda w: str(int(re.findall(r'\d+', str(w).translate(DIGITS))[0]))
                                              if re.findall(r'\d+', str(w).translate(DIGITS)) else None),
                      'addr_text': None, 'ward_nums': None})
    addr = B[['PerAddress', 'TemAddress', 'OtherAddress']].astype(str).apply(
        lambda r: ' '.join(v for v in r if v not in ('', 'nan')) or None, axis=1)
    b = pd.DataFrame({'unique_id': B['id'], 'name': B['full_name'], 'district': None, 'locallevel': None, 'ward': None,
                      'addr_text': addr, 'ward_nums': B.apply(nums, axis=1)})
    for col in ('district', 'locallevel', 'ward', 'addr_text', 'ward_nums'):
        a[col] = a[col].astype(object).where(a[col].notna(), None)
        b[col] = b[col].astype(object).where(b[col].notna(), None)
    return a, b


def splink_triples(A, B):
    import importlib.metadata as md
    from splink import Linker, SettingsCreator
    from splink.backends.duckdb import DuckDBAPI
    import splink.comparison_library as cl
    import splink.comparison_level_library as cll

    a, b = _splink_frames(A, B)
    # Register through DuckDB's own pandas support (no pyarrow needed), then hand Splink the table names.
    import duckdb
    con = duckdb.connect(':memory:')
    con.register('tbl_a', a)
    con.register('tbl_b', b)
    db = DuckDBAPI(con)
    major = int(md.version('splink').split('.')[0])
    if major >= 5:
        inputs = [db.register('tbl_a', dataset_display_name='a'), db.register('tbl_b', dataset_display_name='b')]
    else:
        inputs = ['tbl_a', 'tbl_b']

    def contains_comparison(name, cond, null_cond):
        return cl.CustomComparison(
            output_column_name=name, comparison_description=name,
            comparison_levels=[cll.CustomLevel(null_cond, label_for_charts='null').configure(is_null_level=True),
                               cll.CustomLevel(cond, label_for_charts='found in address'),
                               cll.ElseLevel()])

    settings = SettingsCreator(
        link_type='link_only', unique_id_column_name='unique_id',
        comparisons=[
            cl.JaroWinklerAtThresholds('name', [0.92, 0.8]),
            contains_comparison('district_in_addr', "contains(addr_text_r, district_l)", "district_l IS NULL OR addr_text_r IS NULL"),
            contains_comparison('local_in_addr', "contains(addr_text_r, locallevel_l)", "locallevel_l IS NULL OR addr_text_r IS NULL"),
            contains_comparison('ward_in_addr', "contains(ward_nums_r, '|' || ward_l || '|')", "ward_l IS NULL OR ward_nums_r IS NULL"),
        ],
        blocking_rules_to_generate_predictions=["contains(r.addr_text, l.district)",
                                                "substr(l.name, 1, 3) = substr(r.name, 1, 3)"],
        retain_intermediate_calculation_columns=False)
    linker = (Linker(inputs, settings) if major >= 5
              else Linker(inputs, settings, db_api=db, input_table_aliases=['a', 'b']))
    t = linker.training
    t.estimate_probability_two_random_records_match(
        ["l.name = r.name AND contains(r.ward_nums, '|' || l.ward || '|')"], recall=0.6)
    t.estimate_u_using_random_sampling(max_pairs=2_000_000)
    t.estimate_parameters_using_expectation_maximisation("contains(r.addr_text, l.district)")
    t.estimate_parameters_using_expectation_maximisation("contains(r.ward_nums, '|' || l.ward || '|')")
    pred = linker.inference.predict(threshold_match_weight=-10).as_pandas_dataframe()
    pos_a = {v: i for i, v in enumerate(A['id'])}
    pos_b = {v: j for j, v in enumerate(B['id'])}
    # unique_id_l is from file A and _r from file B (datasets are ordered by name 'a' < 'b'); guard against the reverse
    if len(pred) and str(pred['unique_id_l'].iloc[0]) not in pos_a:
        pred = pred.rename(columns={'unique_id_l': 'unique_id_r', 'unique_id_r': 'unique_id_l'})
    return [(float(w), pos_a[str(x)], pos_b[str(y)])
            for w, x, y in zip(pred['match_weight'], pred['unique_id_l'], pred['unique_id_r'])]


# ------------------------------------------------------------------ supervised
def pair_features(A, B, cands):
    from rapidfuzz import fuzz, distance
    rows, ids = [], []
    for i, js in enumerate(cands):
        a = A.iloc[i]
        w = str(a.wardno).translate(DIGITS)
        for j in js:
            b = B.iloc[j]
            na, nb = str(a.full_name), str(b.full_name)
            ads = E._addr_texts(b)
            best_loc = max([fuzz.partial_ratio(str(a.locallevel), t) for t in ads] or [0])
            best_dis = max([fuzz.partial_ratio(str(a.district), t) for t in ads] or [0])
            rows.append([fuzz.token_sort_ratio(na, nb), fuzz.token_set_ratio(na, nb), fuzz.partial_ratio(na, nb),
                         distance.JaroWinkler.similarity(na, nb) * 100, abs(len(na) - len(nb)),
                         len(na.split()), len(nb.split()),
                         int(any(str(a.district) in t for t in ads)), int(any(str(a.locallevel) in t for t in ads)),
                         int(any(E._ward_in(w, t) for t in ads)), best_loc, best_dis, len(ads)])
            ids.append((i, j))
    return np.array(rows, dtype=float), ids


def gbm_fit(n_true):
    from sklearn.ensemble import HistGradientBoostingClassifier
    A, B, gold = generate(n_true=n_true, seed=DEV_SEEDS[0])
    X, ids = pair_features(A, B, E.baseline_candidates(A, B))
    truth = {(a, b) for a, b in zip(gold.A_id, gold.B_id) if b}
    y = np.array([(A.id.iloc[i], B.id.iloc[j]) in truth for i, j in ids])
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=0).fit(X, y)


def gbm_triples(model, A, B):
    X, ids = pair_features(A, B, E.baseline_candidates(A, B))
    p = model.predict_proba(X)[:, 1]
    return [(float(p[r]), i, j) for r, (i, j) in enumerate(ids)]


# ------------------------------------------------------------------ driver
def best_threshold(triples, grid, A, B, gold):
    return max(grid, key=lambda t: E.score_system(E.greedy_one_to_one(triples, t, A, B), gold)['f1'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--n-true', type=int, default=600)
    ap.add_argument('--out', default=os.path.join(HERE, 'results'))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    print('training supervised model and tuning thresholds on separate synthetic seeds ...', flush=True)
    model = gbm_fit(a.n_true)
    thr = {}
    # Splink threshold: tuned on dev seed 10001 (its own unsupervised training on that seed's data)
    dA, dB, dG = generate(n_true=a.n_true, seed=DEV_SEEDS[1])
    thr['splink'] = best_threshold(splink_triples(dA, dB), SPLINK_GRID, dA, dB, dG)
    thr['gbm'] = best_threshold(gbm_triples(model, dA, dB), GBM_GRID, dA, dB, dG)
    print('tuned thresholds:', thr, flush=True)

    rows = []
    for seed in range(1, a.seeds + 1):
        print(f'seed {seed} ...', flush=True)
        A, B, gold = generate(n_true=a.n_true, seed=seed)
        preds = {f"Splink (weight>{thr['splink']}, tuned)": E.greedy_one_to_one(splink_triples(A, B), thr['splink'], A, B),
                 f"Supervised GBM (p>{thr['gbm']}, tuned)": E.greedy_one_to_one(gbm_triples(model, A, B), thr['gbm'], A, B)}
        for name, pred in preds.items():
            rows.append({'seed': seed, 'system': name, **E.score_system(pred, gold)})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, 'extra_per_seed.csv'), index=False, encoding='utf-8-sig')
    cols = ['precision', 'recall', 'f1', 'false_link_rate']
    agg = df.groupby('system')[cols].agg(['mean', 'std'])
    summ = pd.DataFrame({c: [f"{agg[(c, 'mean')][s]:.3f} ± {agg[(c, 'std')][s]:.3f}" for s in agg.index] for c in cols}, index=agg.index)
    summ.to_csv(os.path.join(a.out, 'extra_summary.csv'), encoding='utf-8-sig')
    op = [c for c in df.columns if c.startswith('rec_')]
    df.groupby('system')[op].mean().round(3).to_csv(os.path.join(a.out, 'extra_recall_by_noise_type.csv'), encoding='utf-8-sig')
    pd.set_option('display.width', 200)
    print(f'\nMean ± sd over {a.seeds} seeds (n_true={a.n_true})\n')
    print(summ.to_string())


if __name__ == '__main__':
    main()
