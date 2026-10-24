"""
Evaluate the linkage method on REAL data that stays on this computer, using the human-verified pairs as labels.

PRIVACY: the script reads your files locally, makes no network calls, never reads any stored reviewer decisions
or database, and writes/prints ONLY aggregate numbers: counts, rates, confidence intervals, and breakdown tables in
which any cell smaller than --min-cell is replaced by "<N". It never prints a name, an address or a record ID.
Errors are reported by type only (add --debug to see the full message on your own screen).
Review the files in the output folder before sharing anything, and get the data owner's permission to publish them.

Usage (run on the computer that holds the data, from this folder):
  python real_eval.py --d1 path\\to\\file1.xlsx --d2 path\\to\\file2.xlsx --verified path\\to\\verified_matches.csv
Options: --sheet1/--sheet2 (Excel sheet name), --ablations (adds ablation runs), --min-cell 10, --out real_results

Labels: a CSV with columns d1_id,d2_id, one row per verified pair (IDs as they appear in file 1 and file 2).
Use the SAME two files the verified pairs came from, so that the IDs agree.

What the numbers mean (state these in the paper):
  * Only records that reviewers verified have a known correct partner. Metrics are computed on those records.
  * Reviewers saw pairs the system proposed, so recall is optimistic (a true partner the system never proposed was never
    verified). Precision is the share of predicted partners for labeled records that equal the verified partner.
  * Global precision and the false-link rate cannot be estimated from these labels alone.
"""
import argparse
import json
import math
import os
import platform
import sys
import tempfile
import warnings

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
warnings.filterwarnings('ignore')
import logging  # noqa: E402
logging.disable(logging.CRITICAL)     # the matcher logs; keep any such text off the screen and out of files

import evaluate as E   # noqa: E402  (installs the numpy TF-IDF only if scikit-learn cannot be imported)
M = E.M


# ------------------------------------------------------------------ loading
class SafeError(Exception):
    """Our own error messages: they never contain data values, so they are always shown."""


def read_table(path, sheet=None):
    ext = os.path.splitext(path)[1].lower()
    if ext in ('.xlsx', '.xlsm', '.xls'):
        return pd.read_excel(path, dtype=str, keep_default_na=False, sheet_name=sheet if sheet else 0)
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding='utf-8-sig')


def to_schema(df1, df2):
    """Rename the real columns to the names the benchmark code expects; returns frames and found-field flags."""
    c1 = M.detect_columns(list(df1.columns), 1)
    c2 = M.detect_columns(list(df2.columns), 2)
    for need, c, label in (('name', c1, 'file 1'), ('name', c2, 'file 2')):
        if not c.get(need):
            raise SafeError(f'{label}: name column not found (rename the header or check the sheet)')
    other = next((v for k, v in c2.items() if k.endswith('_address') and k not in ('per_address', 'tem_address')), None)
    if not (c2.get('per_address') or c2.get('tem_address') or other):
        raise SafeError('file 2: no address column found')

    def col(df, c):
        return df[c].astype(str) if c else pd.Series([''] * len(df), index=df.index)

    A = pd.DataFrame({'id': col(df1, c1.get('id')), 'full_name': col(df1, c1['name']),
                      'district': col(df1, c1.get('district')), 'locallevel': col(df1, c1.get('locallevel')),
                      'wardno': col(df1, c1.get('wardno'))}).reset_index(drop=True)
    B = pd.DataFrame({'id': col(df2, c2.get('id')), 'full_name': col(df2, c2['name']),
                      'PerAddress': col(df2, c2.get('per_address')), 'TemAddress': col(df2, c2.get('tem_address')),
                      'OtherAddress': col(df2, other)}).reset_index(drop=True)
    # records without an ID get row-position ids D1_row_N / D2_row_N (the labels file can use the same form)
    A['id'] = [v if str(v).strip() else f'D1_row_{i + 1}' for i, v in enumerate(A['id'])]
    B['id'] = [v if str(v).strip() else f'D2_row_{i + 1}' for i, v in enumerate(B['id'])]
    found = {'file1': {k: bool(c1.get(k)) for k in ('name', 'district', 'locallevel', 'wardno', 'id')},
             'file2': {'name': True, 'per_address': bool(c2.get('per_address')), 'tem_address': bool(c2.get('tem_address')),
                       'other_address': bool(other), 'id': bool(c2.get('id'))}}
    return A, B, found


def load_gold(path, A, B):
    g = pd.read_csv(path, dtype=str, keep_default_na=False, encoding='utf-8-sig')
    g.columns = [c.strip().lower() for c in g.columns]
    if not {'d1_id', 'd2_id'} <= set(g.columns):
        raise SafeError('verified file needs columns d1_id and d2_id')
    ids_a = A['id'].value_counts()
    ids_b = B['id'].value_counts()
    info = {'verified_rows': int(len(g))}
    g = g[(g.d1_id.str.strip() != '') & (g.d2_id.str.strip() != '')].drop_duplicates()
    ok = g.d1_id.map(lambda x: ids_a.get(x, 0) == 1) & g.d2_id.map(lambda x: ids_b.get(x, 0) == 1)
    info['dropped_id_not_found_or_not_unique'] = int((~ok).sum())
    g = g[ok]
    # one-to-one labels only
    g = g[~g.d1_id.duplicated(keep=False) & ~g.d2_id.duplicated(keep=False)]
    info['labeled_pairs_used'] = int(len(g))
    return dict(zip(g.d1_id, g.d2_id)), info


# ------------------------------------------------------------------ statistics
def wilson(k, n, z=1.96):
    if n == 0:
        return (float('nan'), float('nan'))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def system_metrics(pred, gold):
    """Metrics restricted to labeled records. pred: set of (a_id, b_id)."""
    L = set(gold)
    pl = {(a, b) for a, b in pred if a in L}
    correct = sum(1 for a, b in pl if gold[a] == b)
    n = len(L)
    r_lo, r_hi = wilson(correct, n)
    p_lo, p_hi = wilson(correct, len(pl))
    return {'labeled_records': n, 'predicted_for_labeled': len(pl), 'correct': correct,
            'recall': correct / n if n else float('nan'), 'recall_ci95_low': r_lo, 'recall_ci95_high': r_hi,
            'precision_on_labeled': correct / len(pl) if pl else float('nan'), 'precision_ci95_low': p_lo,
            'precision_ci95_high': p_hi,
            'wrong_partner_rate': (len(pl) - correct) / n if n else float('nan'),
            'no_link_rate': (n - len(pl)) / n if n else float('nan')}


# ------------------------------------------------------------------ proposed method with stratification
def run_proposed(A, B, gold):
    store = M.FeedbackStore()
    session = M.RecordMatcher(feedback_store=store).match_datasets(A, B, E.MAPPING1, E.MAPPING2)
    pos_a = {r['id']: i for i, r in enumerate(session.d1)}
    pos_b = {r['id']: j for j, r in enumerate(session.d2)}
    auto, anyp = set(), set()
    for i, j in enumerate(session.assign):
        if j is None:
            continue
        cat = session._info(i, j)['match_category']
        pair = (session.d1[i]['id'], session.d2[j]['id'])
        if cat in ('perfect', 'high'):
            auto.add(pair)
        if cat in ('perfect', 'high', 'review'):
            anyp.add(pair)
    rows, cand_hit, top5_hit = [], 0, 0
    rm = session.matcher
    for a, b in gold.items():
        i, j = pos_a[a], pos_b[b]
        info = rm.compute_similarity(session.d1[i], session.d2[j])
        cand_hit += any(jj == j for jj, _ in session.candidates[i])
        top5_hit += any(jj == j for jj, _ in session.recommendations(i))
        ns = info['name_score']
        rows.append({
            'name_evidence': ('score 100' if ns >= 100 else 'score 98-99 (spelling/romanization)' if ns >= 98
                              else 'score 85-97' if ns >= 85 else 'score 50-84' if ns >= 50 else 'score <50'),
            'address_evidence': info['matched_address_col'] or 'no address match',
            'addr_matched': info['district_score'] >= 70,
            'category_when_scored': info['match_category'],
            'auto_correct': (a, b) in auto, 'any_correct': (a, b) in anyp,
        })
    n = len(gold)
    extra = {'gold_partner_in_candidates': cand_hit / n if n else float('nan'),
             'gold_partner_in_top5_recommendations': top5_hit / n if n else float('nan')}
    return auto, anyp, pd.DataFrame(rows), extra


def strata_table(df, col, min_cell):
    out = []
    for k, g in df.groupby(col):
        n = len(g)
        if n < min_cell:
            out.append({'stratum_type': col, 'stratum': k, 'labeled_pairs': f'<{min_cell}', 'share': '',
                        'recall_auto': '', 'recall_auto_plus_review': ''})
        else:
            out.append({'stratum_type': col, 'stratum': k, 'labeled_pairs': n, 'share': round(n / len(df), 3),
                        'recall_auto': round(g.auto_correct.mean(), 3),
                        'recall_auto_plus_review': round(g.any_correct.mean(), 3)})
    return out


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--d1', required=True)
    ap.add_argument('--d2', required=True)
    ap.add_argument('--verified', required=True)
    ap.add_argument('--sheet1')
    ap.add_argument('--sheet2')
    ap.add_argument('--out', default=os.path.join(HERE, 'real_results'))
    ap.add_argument('--min-cell', type=int, default=10)
    ap.add_argument('--ablations', action='store_true')
    ap.add_argument('--debug', action='store_true', help='show full error messages (may contain data)')
    a = ap.parse_args()
    try:
        df1, df2 = read_table(a.d1, a.sheet1), read_table(a.d2, a.sheet2)
        A, B, found = to_schema(df1, df2)
        gold, ginfo = load_gold(a.verified, A, B)
        if len(gold) < 2 * a.min_cell:
            raise SafeError(f'only {len(gold)} usable labeled pairs (need at least {2 * a.min_cell}); too few to report safely')
        info = {'records_file1': len(A), 'records_file2': len(B), **ginfo, 'columns_found': found,
                'min_cell': a.min_cell, 'python': platform.python_version(),
                'tfidf': 'numpy shim' if E.SHIM_USED else 'scikit-learn'}
        print(f"records: file1={len(A)} file2={len(B)}; labeled pairs used: {ginfo['labeled_pairs_used']}", flush=True)

        results = {}
        auto, anyp, strata, extra = run_proposed(A, B, gold)
        results['Proposed (auto)'], results['Proposed (auto+review)'] = auto, anyp
        if a.ablations:
            for ab in ('no_fold', 'no_roman', 'no_guards', 'name_only'):
                with E.patched(ab):
                    r_auto, r_any = E.run_matcher(A, B)
                results[f'  - {ab} (auto)'], results[f'  - {ab} (auto+review)'] = r_auto, r_any
            B1 = B.copy()
            B1['TemAddress'] = B1['OtherAddress'] = ''
            r_auto, r_any = E.run_matcher(A, B1)
            results['  - per_address_only (auto)'], results['  - per_address_only (auto+review)'] = r_auto, r_any
        thr = E.tune_baselines(600)       # thresholds tuned on SYNTHETIC data, not on the real labels
        cands = E.baseline_candidates(A, B)
        results['Baseline: Exact'] = E.run_exact(A, B, cands)
        results[f"Baseline: Fuzzy (thr {thr['fuzzy']})"] = E.greedy_one_to_one(E.fuzzy_triples(A, B, cands), thr['fuzzy'], A, B)
        results[f"Baseline: Fellegi-Sunter (weight>{thr['fs']})"] = E.greedy_one_to_one(E.fs_triples(A, B, cands), thr['fs'], A, B)

        summary = pd.DataFrame([{'system': k, **system_metrics(v, gold)} for k, v in results.items()])
        for k, v in extra.items():
            summary.loc[summary.system.str.startswith('Proposed'), k] = v
        num = summary.select_dtypes('number').columns
        summary[num] = summary[num].round(4)

        tables = []
        for col in ('name_evidence', 'address_evidence', 'category_when_scored'):
            tables += strata_table(strata, col, a.min_cell)
        strata_df = pd.DataFrame(tables)

        os.makedirs(a.out, exist_ok=True)
        summary.to_csv(os.path.join(a.out, 'real_summary.csv'), index=False, encoding='utf-8-sig')
        strata_df.to_csv(os.path.join(a.out, 'real_strata.csv'), index=False, encoding='utf-8-sig')
        with open(os.path.join(a.out, 'real_run_info.json'), 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=1)
        pd.set_option('display.width', 220, 'display.max_columns', 30)
        print('\nAggregate metrics on labeled records (no record-level output):\n')
        print(summary[['system', 'labeled_records', 'recall', 'recall_ci95_low', 'recall_ci95_high',
                       'precision_on_labeled', 'wrong_partner_rate', 'no_link_rate']].to_string(index=False))
        print('\nProposed method, breakdown by evidence type (cells below the minimum size suppressed):\n')
        print(strata_df.to_string(index=False))
        print(f'\nWritten to: {a.out}  (aggregates only; review before sharing)')
    except SafeError as e:
        print(f'FAILED: {e}')
        sys.exit(1)
    except Exception as e:
        print(f'FAILED: {type(e).__name__}' + (f': {e}' if a.debug else ' (run with --debug to see details on your screen)'))
        sys.exit(1)


if __name__ == '__main__':
    main()
