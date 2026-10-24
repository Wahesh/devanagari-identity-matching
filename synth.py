"""
Seeded synthetic benchmark generator for the record-linkage method.

Every record is made up (random combinations of common Nepali names and real place names); nothing comes from
any real dataset. Noise is injected per true pair, each operation independently, and every pair is labelled
with the operations applied, so recall can be reported per noise type.

Vocabulary (names, places) is in vocab.py.

  python synth.py --out synth_data --seed 1        # writes A.csv, B.csv, gold.csv

or `from synth import generate` and use the returned DataFrames.
"""
import argparse
import os
import random
from typing import Dict, List, Tuple

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

from vocab import (MALE_FIRST, MALE_MIDDLE, FEMALE_FIRST, FEMALE_MIDDLE, SURNAMES, CLAN_SUFFIX, PLACES,  # noqa: E402
                   SUFFIXES, ENGLISH, NP_DIGITS)

NAME_OPS = ['spelling', 'abbrev', 'clan', 'honorific', 'typo', 'english']
ADDR_OPS = ['other_column', 'old_vdc', 'ward_missing', 'no_local', 'addr_missing', 'wrong_ward']
# probability that each operation is applied to a true pair (name ops and address ops independent)
OP_P = {'spelling': .25, 'abbrev': .20, 'clan': .10, 'honorific': .08, 'typo': .12, 'english': .06,
        'other_column': .20, 'old_vdc': .12, 'ward_missing': .10, 'no_local': .06, 'addr_missing': .04, 'wrong_ward': .05}


class Gen:
    def __init__(self, seed: int):
        self.r = random.Random(seed)
        self.used = set()

    def person(self, force_english=False) -> dict:
        r = self.r
        while True:
            if r.random() < 0.72:
                first, middle = r.choice(MALE_FIRST), r.choice(MALE_MIDDLE)
            else:
                first, middle = r.choice(FEMALE_FIRST), r.choice(FEMALE_MIDDLE)
            sur = r.choice(SURNAMES)
            if force_english:
                first = r.choice([x for x in MALE_FIRST if x in ENGLISH])
                middle = r.choice([x for x in MALE_MIDDLE if x in ENGLISH or not x])
                sur = r.choice([x for x in SURNAMES if x in ENGLISH])
            d = r.choice(list(PLACES))
            key = (first, middle, sur, d)
            if key in self.used:
                continue
            self.used.add(key)
            return {'first': first, 'middle': middle, 'sur': sur, 'district': d,
                    'local': r.choice(PLACES[d][0]), 'ward': r.randint(1, 14)}

    def name(self, p, middle=None) -> str:
        m = p['middle'] if middle is None else middle
        return ' '.join(x for x in (p['first'], m, p['sur']) if x)

    def ward_text(self, w) -> str:
        return str(w).translate(NP_DIGITS) if self.r.random() < .6 else str(w)

    def address(self, local, ward, district, with_ward=True) -> str:
        r = self.r
        loc = f"{local} {r.choice(SUFFIXES)}".strip()
        w = self.ward_text(ward)
        if not with_ward:
            return r.choice([f"{loc}, {district}", f"{district}, {loc}"])
        return r.choice([f"{loc}-{w}, {district}", f"{loc} वडा नं. {w} {district}",
                         f"{district}, {loc} - {w}", f"{loc} {w} {district}"])

    def elsewhere(self) -> str:
        d = self.r.choice(list(PLACES))
        return self.address(self.r.choice(PLACES[d][0]), self.r.randint(1, 12), d)

    # ----- name noise -----
    def spelling(self, name: str) -> str:
        swaps = [('श', 'स'), ('ष', 'स'), ('ी', 'ि'), ('ि', 'ी'), ('ब', 'व'), ('ं', ''), ('ू', 'ु'), ('ण', 'न')]
        self.r.shuffle(swaps)
        for a, b in swaps:
            if a in name:
                return name.replace(a, b, 1)
        return name

    def typo(self, name: str) -> str:
        toks = name.split()
        k = self.r.randrange(len(toks))
        t = toks[k]
        if len(t) > 3:
            i = self.r.randrange(1, len(t))
            t = t[:i] + t[i + 1:] if self.r.random() < .5 else t[:i] + t[i] + t[i:]
        toks[k] = t
        return ' '.join(toks)

    def noisy_name(self, p, ops: List[str]) -> Tuple[str, List[str]]:
        applied, name, r = [], self.name(p), self.r
        if 'abbrev' in ops and p['middle']:
            name = self.name(p, r.choice([p['middle'][0] + '.', '']))
            applied.append('abbrev')
        if 'clan' in ops and p['sur'] in CLAN_SUFFIX:
            name += ' ' + CLAN_SUFFIX[p['sur']]
            applied.append('clan')
        if 'english' in ops:
            parts = [ENGLISH.get(x) for x in name.split()]
            if all(parts):
                name = ' '.join(parts)
                applied.append('english')
        if 'spelling' in ops:
            new = self.spelling(name)
            if new != name:
                name = new
                applied.append('spelling')
        if 'typo' in ops and 'english' not in applied:
            name = self.typo(name)
            applied.append('typo')
        if 'honorific' in ops:
            name = r.choice(['स्व.', 'श्री', 'स्वर्गीय']) + ' ' + name
            applied.append('honorific')
        return name, applied

    def addresses(self, p, ops: List[str]) -> Tuple[Dict[str, str], List[str]]:
        r = self.r
        a = {'PerAddress': '', 'TemAddress': '', 'OtherAddress': ''}
        applied = []
        own = self.address(p['local'], p['ward'], p['district'])
        if 'addr_missing' in ops:
            return a, ['addr_missing']
        if 'old_vdc' in ops:
            old = r.choice(PLACES[p['district']][1])
            own = self.address(old, (p['ward'] % 9) + 1, p['district']).replace('नगरपालिका', 'गा.वि.स.')
            applied.append('old_vdc')
        elif 'no_local' in ops:
            own = f"{p['district']}"
            applied.append('no_local')
        elif 'ward_missing' in ops:
            own = self.address(p['local'], p['ward'], p['district'], with_ward=False)
            applied.append('ward_missing')
        elif 'wrong_ward' in ops:
            own = self.address(p['local'], ((p['ward'] + r.randint(1, 5) - 1) % 14) + 1, p['district'])
            applied.append('wrong_ward')
        if 'other_column' in ops:
            col = r.choice(['TemAddress', 'OtherAddress'])
            a['PerAddress'], a[col] = self.elsewhere(), own
            applied.append('other_column')
        else:
            a['PerAddress'] = own
        return a, applied


def generate(n_true: int = 600, n_only_a: int = 100, n_only_b: int = 100, n_relatives: int = 50,
             n_namesake_near: int = 40, n_namesake_far: int = 40, seed: int = 1
             ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns (A, B, gold). gold has a row per A record: B_id ('' if no true match), ops, kind."""
    g = Gen(seed)
    r = g.r
    A, B, gold = [], [], []
    ids_a, ids_b = iter(range(1, 10 ** 6)), iter(range(1, 10 ** 6))

    def add_a(p, kind, b_id='', ops=''):
        aid = f"A{next(ids_a):05d}"
        A.append({'id': aid, 'full_name': g.name(p), 'district': p['district'], 'locallevel': p['local'],
                  'wardno': g.ward_text(p['ward'])})
        gold.append({'A_id': aid, 'B_id': b_id, 'kind': kind, 'ops': ops})
        return aid

    def add_b(name, addr):
        bid = f"B{next(ids_b):05d}"
        B.append({'id': bid, 'full_name': name, **addr})
        return bid

    for _ in range(n_true):
        ops = [o for o in NAME_OPS + ADDR_OPS if r.random() < OP_P[o]]
        p = g.person(force_english='english' in ops)
        name, a1 = g.noisy_name(p, ops)
        addr, a2 = g.addresses(p, ops)
        bid = add_b(name, addr)
        add_a(p, 'true', bid, '+'.join(a1 + a2) or 'clean')

    # relative: same surname/address, different first name; the true person is absent from B
    for _ in range(n_relatives):
        p = g.person()
        rel = dict(p, first=r.choice([f for f in MALE_FIRST + FEMALE_FIRST if f != p['first']]))
        add_b(g.name(rel), g.addresses(p, [])[0])
        add_a(p, 'relative')

    # same name, same district, different local level / ward: both people exist, told apart by address
    for _ in range(n_namesake_near):
        p = g.person()
        q = dict(p, local=r.choice([x for x in PLACES[p['district']][0] if x != p['local']] or [p['local']]),
                 ward=(p['ward'] % 12) + 1)
        add_a(p, 'true', add_b(g.name(p), g.addresses(p, [])[0]), 'namesake_near')
        add_a(q, 'true', add_b(g.name(q), g.addresses(q, [])[0]), 'namesake_near')

    # same name in another district only: no true match
    for _ in range(n_namesake_far):
        p = g.person()
        q = dict(p, district=r.choice([d for d in PLACES if d != p['district']]))
        q['local'] = r.choice(PLACES[q['district']][0])
        add_b(g.name(q), g.addresses(q, [])[0])
        add_a(p, 'namesake_far')

    for _ in range(n_only_a):
        add_a(g.person(), 'only_a')
    for _ in range(n_only_b):
        p = g.person()
        add_b(g.name(p), g.addresses(p, [])[0])

    r.shuffle(A)
    r.shuffle(B)
    return pd.DataFrame(A), pd.DataFrame(B), pd.DataFrame(gold)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(HERE, 'synth_data'))
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--n-true', type=int, default=600)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    A, B, gold = generate(n_true=a.n_true, seed=a.seed)
    for name, df in (('A', A), ('B', B), ('gold', gold)):
        df.to_csv(os.path.join(a.out, f'{name}.csv'), index=False, encoding='utf-8-sig')
    print(f"A={len(A)} B={len(B)} true pairs={(gold.B_id != '').sum()}")
    print(gold.kind.value_counts().to_string())
