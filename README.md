# Rule-based record linkage for Nepali names and addresses

A transparent, rule-based method for linking records of the same person across two files that share **no unique
identifier**, when names and addresses are free text in Devanagari, spelled inconsistently, and addresses follow
Nepal's administrative hierarchy (district, local level, ward). Every score comes with a stated reason. No training
data is needed. This repository contains the method, a seeded **synthetic** benchmark generator, baselines, ablations
and the results reported in the accompanying article.

> **All reported results are on synthetic data.** The records are random combinations of common Nepali names and real
> place names; noise types and rates were chosen by the author. The results show that the method handles those
> variations. They do not establish accuracy on real records. Nothing in this repository comes from real data.

## How it works (short)
1. Orthographic normalization and phonetic romanization of Devanagari names (so Devanagari and Latin spellings compare).
2. A knowledge base of the 77 districts, local-level suffix handling, and cautious ward-number extraction.
3. Candidate generation by character 2-3-gram TF-IDF (overall and within the record's district).
4. Rule-based scores: a name cascade with first-name / surname **conflict guards** (a shared surname and address with a
   different first name usually means a relative), and an address score taken from any one of three address fields.
   Confidence = mean(name score, address score); pairs are labeled perfect / high / review / none.
5. Global one-to-one assignment, re-solved after reviewer decisions.

## Files
| File | Purpose |
|---|---|
| `linkage.py` | the method |
| `synth.py`, `vocab.py` | seeded synthetic benchmark generator (12 noise types, relatives, same-name people) |
| `evaluate.py` | baselines (exact, fuzzy, Fellegi-Sunter), ablations, recall by noise type |
| `extra_baselines.py` | Splink and a supervised gradient-boosted model |
| `real_eval.py` | for data holders: evaluate on your own labeled data privately (aggregates only) |
| `tfidf_shim.py` | numpy TF-IDF fallback if scikit-learn cannot be imported |
| `results/` | per-replication and summary results (5 seeds) |

## Quick start
```bash
git clone https://github.com/Wahesh/devanagari-identity-matching.git
cd devanagari-identity-matching
pip install -r requirements.txt
python synth.py --out synth_data --seed 1          # write one synthetic dataset (A.csv, B.csv, gold.csv)
python evaluate.py --seeds 5 --n-true 600          # baselines + ablations  -> results/
pip install "splink<5"                             # optional, for extra_baselines.py (tested with 4.0.17)
python extra_baselines.py --seeds 5 --n-true 600   # Splink + supervised model -> results/
```
```python
import linkage as L
matcher = L.RecordMatcher(feedback_store=L.FeedbackStore())
session = matcher.match_datasets(df_a, df_b, {'name': 'full_name'}, {'name': 'full_name'})
# df_a columns: id, full_name, district, locallevel, wardno
# df_b columns: id, full_name, PerAddress, TemAddress, OtherAddress
for i, j in enumerate(session.assign):      # j is None when no partner was proposed
    ...
```

## Results (mean ± sd over 5 synthetic replications, 870 x 870 records, 680 true pairs)
| System | Precision | Recall | F1 | False-link rate |
|---|---|---|---|---|
| **Proposed, auto (perfect/high only)** | .993 (.004) | .927 (.013) | .959 (.009) | .017 (.011) |
| Supervised gradient boosting, trained on labeled synthetic pairs | .978 (.007) | .944 (.008) | .961 (.007) | .063 (.008) |
| Fuzzy matching, tuned | .840 (.004) | .883 (.009) | .861 (.006) | .540 (.015) |
| Fellegi-Sunter, tuned | .951 (.014) | .739 (.004) | .832 (.004) | .106 (.016) |
| Splink 4.0.17, tuned | .814 (.013) | .776 (.012) | .794 (.012) | .425 (.020) |
| Exact matching | .999 (.002) | .439 (.009) | .609 (.009) | .001 (.002) |

Baseline thresholds were tuned on separate synthetic seeds; the proposed method's constants were not tuned. The
supervised model was trained on the same generator, so its score is likely optimistic for real data. Removing the
conflict guards raises the false-link rate from .017 to .185; names written in English script are recovered at only
.26; records with no address are not accepted automatically. See `results/` for ablations and per-noise-type recall.

## Using it on your own data
`real_eval.py` evaluates the method on **your** files using verified pairs as labels and prints only aggregate numbers
(counts, rates, intervals; small cells suppressed). It makes no network calls. Keep your data and its output under your
own governance, and obtain whatever permission your data owner requires before sharing even aggregate results.
```bash
python real_eval.py --d1 file_a.xlsx --d2 file_b.xlsx --verified verified_pairs.csv   # needs columns d1_id,d2_id
```

## Limitations
Synthetic evidence only; author-designed noise; hand-set thresholds; minimal Fellegi-Sunter and Splink specifications
(richer comparisons could do better); Nepali-specific rules; the multi-reviewer workflow described in the article is not
part of this repository.

## AI assistance
The code in this repository was written with the assistance of Claude (Anthropic) and reviewed by the author.

## Citation and license
Author: Mahesh Dahal (ORCID [0000-0002-8238-8082](https://orcid.org/0000-0002-8238-8082)). Cite the software with `CITATION.cff` (GitHub's
"Cite this repository" button); the article reference and a DOI will be added when available. Released under the MIT License (see `LICENSE`).
