"""
Rule-based record linkage for Nepali (Devanagari) names and administrative addresses, without unique identifiers.

Links records of file A (name, district, local level, ward) to records of file B (name and up to three address
fields: permanent, temporary, other). Deterministic and explainable: every score has a stated reason.

  1. Normalization and phonetic romanization of names
  2. District / local-level / ward knowledge base
  3. Candidate generation (character n-gram TF-IDF, global + within-district) and rule-based scoring
     - name cascade with first-name / surname conflict guards
     - address score from any one of the three address fields
  4. Global one-to-one assignment; reviewer decisions (confirm / reject) re-solve the assignment

Final confidence = mean of the name score and the address score.
"""
import re
import time
import hashlib
import threading
import unicodedata
from typing import Dict, List, Any, Optional, Tuple, Callable

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, distance
from sklearn.feature_extraction.text import TfidfVectorizer

# Status thresholds on the averaged (name + address) / 2 score
PERFECT_MIN = 80.0   # and the address must have matched and the name must score >= 85
HIGH_MIN = 60.0      # and the name must score >= 85
REVIEW_MIN = 50.0    # below this the best candidate is reported as "No Match"

STATUS_LABELS = {
    'perfect': "Perfect Match",
    'high': "High Confidence Match",
    'review': "Review Required",
    'none': "No Match",
    'verified': "✓ User Verified Match (100%)",
    'blocked': "✕ User Blocked (0%)",
}

# ==========================================
# 1. DEVANAGARI & NEPALI NORMALIZATION
# ==========================================

NEPALI_DIGITS_MAP = str.maketrans('०१२३४५६७८९', '0123456789')
EMPTY_MARKERS = {'', 'nan', 'none', 'null', 'nat', 'नखुलेको', 'नखुलेको।', '-', '--', 'undefined'}

# Whole-token honorifics. Matched per token (never inside a word), because Python's \b
# treats Devanagari vowel signs as non-word characters and would cut names apart.
HONORIFIC_TOKENS = {'स्व', 'श्री', 'श्रीमती', 'सुश्री', 'डा', 'स्वर्गीय', 'स्वर्गिय'}

# Multi-character spellings, applied BEFORE single-character folding so they can still match.
PHRASE_REPLACEMENTS = [
    ('गा.वि.स.', 'गाविस'), ('गा.बि.स.', 'गाविस'), ('गा.वि.स', 'गाविस'), ('गा.बि.स', 'गाविस'),
    ('न.पा.', 'नपा'), ('न.पा', 'नपा'),
    ('काठमाण्डौं', 'काठमाडौं'), ('काठमाण्डौ', 'काठमाडौं'), ('काठमाण्डू', 'काठमाडौं'), ('काठमान्डौ', 'काठमाडौं'),
    ('धादिङ्ग', 'धादिङ'),
    ('नगरपालिका', 'नपा'),
]

CHAR_FOLDS = [
    ('क्ष', 'छ'), ('ज्ञ', 'ग्य'),
    ('श', 'स'), ('ष', 'स'),
    ('ब', 'व'),
    ('ी', 'ि'), ('ू', 'ु'),
    ('ण', 'न'), ('ङ', 'न'), ('ञ', 'न'),
    ('ं', ''), ('ँ', ''), ('ः', ''),   # anusvara / chandrabindu / visarga are written inconsistently
    ('़', ''),                          # nukta
    ('्', ''),                          # halant
]


def is_empty(value: Any) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    return str(value).strip().lower() in EMPTY_MARKERS


def clean_cell(value: Any) -> str:
    """Cell -> display string. Excel floats like 5.0 become '5'; empty markers become ''."""
    if is_empty(value):
        return ''
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def to_english_digits(text: Any) -> str:
    """Converts Devanagari digits to 0-9."""
    if is_empty(text):
        return ""
    return str(text).strip().translate(NEPALI_DIGITS_MAP)


def _strip_honorifics(text: str) -> str:
    tokens = text.replace('.', '. ').split()
    return ' '.join(t for t in tokens if t.rstrip('.') not in HONORIFIC_TOKENS)


def normalize_nepali_text(text: Any) -> str:
    """
    Spelling-insensitive form of a Nepali name or address:
    - Devanagari digits -> 0-9, digits split from letters
    - honorifics (स्व., श्री, ...) removed as whole tokens
    - श/ष->स, ब->व, ी->ि, ू->ु, ण/ङ/ञ->न, क्ष->छ, ज्ञ->ग्य
    - anusvara, chandrabindu, visarga, nukta and halant removed
    - punctuation removed, whitespace collapsed
    """
    if is_empty(text):
        return ""
    t = unicodedata.normalize('NFC', str(text)).strip()
    t = to_english_digits(t)
    for old, new in PHRASE_REPLACEMENTS:
        t = t.replace(old, new)
    t = _strip_honorifics(t)
    for old, new in CHAR_FOLDS:
        t = t.replace(old, new)
    t = t.lower()
    t = re.sub(r'[^\w\sऀ-ॿ]', ' ', t)
    t = re.sub(r'(?<=\d)(?=[^\d\s])|(?<=[^\d\s])(?=\d)', ' ', t)
    return re.sub(r'\s+', ' ', t).strip()


DEV_CONSONANTS = {
    'क': 'k', 'ख': 'kh', 'ग': 'g', 'घ': 'gh', 'ङ': 'ng',
    'च': 'ch', 'छ': 'chh', 'ज': 'j', 'झ': 'jh', 'ञ': 'n',
    'ट': 't', 'ठ': 'th', 'ड': 'd', 'ढ': 'dh', 'ण': 'n',
    'त': 't', 'थ': 'th', 'द': 'd', 'ध': 'dh', 'न': 'n',
    'प': 'p', 'फ': 'ph', 'ब': 'b', 'भ': 'bh', 'म': 'm',
    'य': 'y', 'र': 'r', 'ल': 'l', 'व': 'v',
    'श': 's', 'ष': 's', 'स': 's', 'ह': 'h',
}
DEV_VOWELS = {
    'अ': 'a', 'आ': 'a', 'इ': 'i', 'ई': 'i', 'उ': 'u', 'ऊ': 'u',
    'ऋ': 'ri', 'ए': 'e', 'ऐ': 'ai', 'ओ': 'o', 'औ': 'au',
}
DEV_MATRAS = {
    'ा': 'a', 'ि': 'i', 'ी': 'i', 'ु': 'u', 'ू': 'u',
    'ृ': 'ri', 'े': 'e', 'ै': 'ai', 'ो': 'o', 'ौ': 'au',
}
DEV_SIGNS = {'ं': 'n', 'ँ': 'n', 'ः': 'h'}
HALANT = '्'
NUKTA = '़'


def _roman_folds(t: str) -> str:
    t = re.sub(r'\b(bdr|b)\b', 'bahadur', t)
    t = re.sub(r'\b(pd|p)\b', 'prasad', t)
    t = re.sub(r'\b(kr|k)\b', 'kumar', t)
    t = re.sub(r'\b(bk|bishwakarma|biswakarma|vishwakarma)\b', 'bk', t)
    t = re.sub(r'\b(kshetri|chhetri|chetri|kshatri)\b', 'chhetri', t)
    t = t.replace('sh', 's').replace('w', 'v').replace('ee', 'i').replace('oo', 'u')
    t = re.sub(r'([a-z])\1+', r'\1', t)
    return re.sub(r'\s+', ' ', t).strip()


def to_roman_phonetic(text: Any) -> str:
    """Devanagari or English -> rough phonetic Roman string, for cross-script comparison."""
    if is_empty(text):
        return ""
    s = unicodedata.normalize('NFC', str(text)).strip().lower()
    if not re.search(r'[ऀ-ॿ]', s):
        s = re.sub(r'[^\w\s]', ' ', s)
        return _roman_folds(s)

    s = _strip_honorifics(s.translate(NEPALI_DIGITS_MAP))
    out: List[str] = []
    n = len(s)
    for i, c in enumerate(s):
        if c in DEV_CONSONANTS:
            out.append(DEV_CONSONANTS[c])
            nxt = s[i + 1] if i + 1 < n else ''
            if nxt == NUKTA:
                nxt = s[i + 2] if i + 2 < n else ''
            if nxt in DEV_MATRAS or nxt == HALANT:
                continue
            # schwa deletion at the end of a word ("राम" -> "ram", not "rama")
            if nxt == '' or not ('ऀ' <= nxt <= 'ॿ') or nxt in ('।',):
                continue
            out.append('a')
        elif c in DEV_MATRAS:
            out.append(DEV_MATRAS[c])
        elif c in DEV_VOWELS:
            out.append(DEV_VOWELS[c])
        elif c in DEV_SIGNS:
            out.append(DEV_SIGNS[c])
        elif c in (HALANT, NUKTA):
            continue
        elif c.isspace():
            out.append(' ')
        elif c.isascii() and c.isalnum():
            out.append(c)
        else:
            out.append(' ')
    return _roman_folds(''.join(out))


# ==========================================
# 2. NEPAL DISTRICTS KNOWLEDGE BASE (77 DISTRICTS)
# ==========================================

NEPAL_DISTRICTS: Dict[str, List[str]] = {
    'achham': ['अछाम', 'achham'],
    'arghakhanchi': ['अर्घाखाँची', 'अर्घाखांची', 'arghakhanchi'],
    'baglung': ['बागलुङ', 'बाग्लुङ', 'baglung'],
    'baitadi': ['बैतडी', 'baitadi'],
    'bajhang': ['बझाङ', 'बझाङ्ग', 'bajhang'],
    'bajura': ['बाजुरा', 'bajura'],
    'banke': ['बाँके', 'बाके', 'banke'],
    'bara': ['बारा', 'bara'],
    'bardiya': ['बर्दिया', 'बर्दीया', 'bardia', 'bardiya'],
    'bhaktapur': ['भक्तपुर', 'bhaktapur'],
    'bhojpur': ['भोजपुर', 'bhojpur'],
    'chitwan': ['चितवन', 'चितौन', 'chitwan', 'chitawan'],
    'dadeldhura': ['डडेलधुरा', 'डडेल्धुरा', 'dadeldhura'],
    'dailekh': ['दैलेख', 'dailekh'],
    'dang': ['दाङ', 'दाङ्ग', 'dang'],
    'darchula': ['दार्चुला', 'darchula'],
    'dhading': ['धादिङ', 'धादिङ्ग', 'dhading'],
    'dhankuta': ['धनकुटा', 'dhankuta'],
    'dhanusha': ['धनुषा', 'धनुसा', 'dhanusa', 'dhanusha'],
    'dolakha': ['दोलखा', 'dolakha', 'dolkha'],
    'dolpa': ['डोल्पा', 'dolpa'],
    'doti': ['डोटी', 'doti'],
    'gorkha': ['गोरखा', 'गोर्खा', 'gorkha'],
    'gulmi': ['गुल्मी', 'gulmi'],
    'humla': ['हुम्ला', 'humla'],
    'ilam': ['इलाम', 'ईलाम', 'ilam'],
    'jajarkot': ['जाजरकोट', 'jajarkot'],
    'jhapa': ['झापा', 'jhapa'],
    'jumla': ['जुम्ला', 'jumla'],
    'kailali': ['कैलाली', 'kailali'],
    'kalikot': ['कालिकोट', 'kalikot'],
    'kanchanpur': ['कञ्चनपुर', 'कन्चनपुर', 'kanchanpur'],
    'kapilvastu': ['कपिलवस्तु', 'कपिलबस्तु', 'kapilvastu', 'kapilbastu'],
    'kaski': ['कास्की', 'kaski'],
    'kathmandu': ['काठमाडौँ', 'काठमाडौं', 'काठमाडौ', 'काठमाण्डौ', 'काठमाण्डू', 'काठमान्डौ', 'kathmandu', 'ktm'],
    'kavrepalanchok': ['काभ्रेपलाञ्चोक', 'काभ्रेपलान्चोक', 'काभ्रे', 'kavre', 'kavrepalanchok', 'kabhre'],
    'khotang': ['खोटाङ', 'खोटाङ्ग', 'khotang'],
    'lalitpur': ['ललितपुर', 'पाटन', 'lalitpur', 'patan'],
    'lamjung': ['लमजुङ', 'लमजुङ्ग', 'lamjung'],
    'mahottari': ['महोत्तरी', 'mahottari'],
    'makwanpur': ['मकवानपुर', 'makwanpur', 'makawanpur'],
    'manang': ['मनाङ', 'मनाङ्ग', 'manang'],
    'morang': ['मोरङ', 'मोरङ्ग', 'morang'],
    'mugu': ['मुगु', 'mugu'],
    'mustang': ['मुस्ताङ', 'मुस्ताङ्ग', 'mustang'],
    'myagdi': ['म्याग्दी', 'myagdi'],
    'nawalparasi_east': ['नवलपुर', 'नवलपरासी पूर्व', 'नवलपरासी (पूर्व)', 'nawalpur', 'nawalparasi east'],
    'nawalparasi_west': ['परासी', 'नवलपरासी पश्चिम', 'नवलपरासी (पश्चिम)', 'नवलपरासी', 'nawalparasi', 'parasi', 'nawalparasi west'],
    'nuwakot': ['नुवाकोट', 'nuwakot'],
    'okhaldhunga': ['ओखलढुङ्गा', 'ओखलढुंगा', 'okhaldhunga'],
    'palpa': ['पाल्पा', 'palpa'],
    'panchthar': ['पाँचथर', 'पांचथर', 'panchthar'],
    'parbat': ['पर्वत', 'parbat'],
    'parsa': ['पर्सा', 'parsa'],
    'pyuthan': ['प्युठान', 'प्यूठान', 'pyuthan'],
    'ramechhap': ['रामेछाप', 'ramechhap'],
    'rasuwa': ['रसुवा', 'rasuwa'],
    'rautahat': ['रौतहट', 'rautahat'],
    'rolpa': ['रोल्पा', 'rolpa'],
    'rukum_east': ['पूर्वी रुकुम', 'रुकुम पूर्व', 'rukum east'],
    'rukum_west': ['पश्चिम रुकुम', 'रुकुम पश्चिम', 'रुकुम', 'rukum', 'rukum west'],
    'rupandehi': ['रुपन्देही', 'रुपन्देहि', 'rupandehi'],
    'salyan': ['सल्यान', 'salyan'],
    'sankhuwasabha': ['सङ्खुवासभा', 'संखुवासभा', 'sankhuwasabha'],
    'saptari': ['सप्तरी', 'saptari'],
    'sarlahi': ['सर्लाही', 'sarlahi'],
    'sindhuli': ['सिन्धुली', 'sindhuli'],
    'sindhupalchok': ['सिन्धुपाल्चोक', 'sindhupalchok', 'sindhupalchowk'],
    'siraha': ['सिराहा', 'सिरहा', 'siraha'],
    'solukhumbu': ['सोलुखुम्बु', 'सोलुखुम्वु', 'solukhumbu'],
    'sunsari': ['सुनसरी', 'sunsari'],
    'surkhet': ['सुर्खेत', 'surkhet'],
    'syangja': ['स्याङ्जा', 'स्याङ्गजा', 'syangja'],
    'tanahun': ['तनहुँ', 'तनहु', 'tanahun', 'tanahu'],
    'taplejung': ['ताप्लेजुङ', 'ताप्लेजुङ्ग', 'taplejung'],
    'terhathum': ['तेह्रथुम', 'terhathum'],
    'udayapur': ['उदयपुर', 'udayapur', 'udaypur'],
}

# normalized form -> district code; longest forms first so "नवलपरासी पूर्व" wins over "नवलपरासी"
_DISTRICT_FORMS: Dict[str, str] = {}
for _code, _names in NEPAL_DISTRICTS.items():
    for _n in _names:
        _norm = normalize_nepali_text(_n)
        if _norm:
            _DISTRICT_FORMS.setdefault(_norm, _code)
_DISTRICT_FORMS_SORTED: List[Tuple[str, str]] = sorted(_DISTRICT_FORMS.items(), key=lambda kv: -len(kv[0]))


def find_district_keys(norm_text: str) -> List[str]:
    """All district codes named (as whole tokens) in an already-normalized address string."""
    if not norm_text:
        return []
    padded = f' {norm_text} '
    found: List[str] = []
    for form, code in _DISTRICT_FORMS_SORTED:
        needle = f' {form} '
        if needle in padded:
            if code not in found:
                found.append(code)
            padded = padded.replace(needle, '  ')
    return found


def resolve_district_key(text: Any) -> Optional[str]:
    """Resolves a Nepali or English district string to a canonical district code."""
    norm = normalize_nepali_text(text)
    if not norm:
        return None
    if norm in _DISTRICT_FORMS:
        return _DISTRICT_FORMS[norm]
    keys = find_district_keys(norm)
    return keys[0] if keys else None


LOCAL_LEVEL_SUFFIX_TOKENS = {
    normalize_nepali_text(s) for s in [
        'गाउँपालिका', 'गाउपालिका', 'गाउपालीका', 'गाउँपालीका', 'गापा', 'गा.पा.',
        'नगरपालिका', 'नगरपालीका', 'नपा', 'महानगरपालिका', 'उपमहानगरपालिका', 'उपमहानपा', 'महानपा',
        'गाविस', 'पालिका',
    ]
}
LOCAL_LEVEL_GLUED_SUFFIXES = ('गाउपालिका', 'गाविस')


def clean_local_level(text: Any) -> str:
    """Normalized local-level / VDC name with the administrative suffix removed."""
    norm = normalize_nepali_text(text)
    if not norm:
        return ""
    kept = []
    for tok in norm.split():
        if tok in LOCAL_LEVEL_SUFFIX_TOKENS:
            continue
        for suf in LOCAL_LEVEL_GLUED_SUFFIXES:
            if tok.endswith(suf) and len(tok) > len(suf) + 1:
                tok = tok[:-len(suf)]
                break
        kept.append(tok)
    return ' '.join(kept)


def ward_digits(value: Any) -> str:
    """'५', '5', 5.0, 'वडा नं ५' -> '5'."""
    s = to_english_digits(clean_cell(value))
    m = re.search(r'\d+', s)
    return str(int(m.group(0))) if m else ''


def _token_in(needle: str, hay: str) -> bool:
    return bool(needle) and f' {needle} ' in f' {hay} '


def check_ward_match(ward: str, norm_addr: str, norm_loc: str = '') -> bool:
    """
    Ward number found in an address. A bare number is only trusted when it is written next to a
    ward marker, directly after the local-level name, or is the only small number in the address.
    """
    if not ward or not norm_addr:
        return False
    numbers = [str(int(n)) for n in re.findall(r'\d+', norm_addr)]
    if ward not in numbers:
        return False
    w = re.escape(ward)
    if re.search(r'(वडा|वारड|ward|wd)\s*(न|no|नमवर)?\s*0*' + w + r'(?!\d)', norm_addr):
        return True
    if norm_loc and re.search(re.escape(norm_loc) + r'\s*0*' + w + r'(?!\d)', norm_addr):
        return True
    small = {n for n in numbers if len(n) <= 2}
    return small == {ward}


def _hash(text: str) -> str:
    return hashlib.sha1(text.encode('utf-8')).hexdigest()[:20]


def row_key(summary: Any) -> str:
    """Identity of a whole source row (all its non-empty cells)."""
    s = re.sub(r'\s+', ' ', unicodedata.normalize('NFC', str(summary or ''))).strip()
    return _hash('row|' + s) if s else ''


def d1_field_key(name: Any, district: Any, locallevel: Any, ward: Any) -> str:
    parts = [normalize_nepali_text(name), normalize_nepali_text(district), clean_local_level(locallevel), ward_digits(ward)]
    return _hash('d1|' + '|'.join(parts)) if parts[0] else ''


def d2_field_key(name: Any, per: Any, tem: Any, other: Any) -> str:
    parts = [normalize_nepali_text(name), normalize_nepali_text(per), normalize_nepali_text(tem), normalize_nepali_text(other)]
    return _hash('d2|' + '|'.join(parts)) if parts[0] else ''


# ==========================================
# 3. REVIEWER DECISIONS (in memory)
# ==========================================

class FeedbackStore:
    """
    Remembers reviewer decisions per record pair, plus learned spelling aliases (kept in memory only).

    A pair is identified twice: by the full source rows and by the matched fields (name + address).
    """

    def __init__(self):
        self._lock = threading.RLock()
        self.pairs: Dict[str, Dict[str, Any]] = {}
        self.name_aliases: Dict[str, List[str]] = {}
        self._index: Dict[Tuple[str, str], str] = {}
        self._confirmed_by_d1: Dict[str, set] = {}

    def _rebuild_index(self):
        self._index = {}
        self._confirmed_by_d1 = {}
        for pid, p in self.pairs.items():
            for k1, k2 in ((p.get('d1_row_key'), p.get('d2_row_key')), (p.get('d1_field_key'), p.get('d2_field_key'))):
                if k1 and k2:
                    self._index[(k1, k2)] = pid
                    if p.get('decision') == 'confirmed':
                        self._confirmed_by_d1.setdefault(k1, set()).add(k2)

    def _find_pid(self, d1_row, d2_row, d1_field, d2_field) -> Optional[str]:
        return self._index.get((d1_row, d2_row)) or self._index.get((d1_field, d2_field))

    def record(self, decision: str, d1_row_key: str, d2_row_key: str, d1_field_key: str = '', d2_field_key: str = '',
               d1_name: str = '', d2_name: str = '') -> str:
        if decision not in ('confirmed', 'rejected'):
            raise ValueError("decision must be 'confirmed' or 'rejected'")
        if not ((d1_row_key and d2_row_key) or (d1_field_key and d2_field_key)):
            raise ValueError("record keys are missing")
        with self._lock:
            existing = self._find_pid(d1_row_key, d2_row_key, d1_field_key, d2_field_key)
            if existing:
                self.pairs.pop(existing, None)
            if decision == 'confirmed':
                # one-to-one: a record can be verified with only one partner
                d1_keys = {k for k in (d1_row_key, d1_field_key) if k}
                d2_keys = {k for k in (d2_row_key, d2_field_key) if k}
                for other in [q for q, p in self.pairs.items() if p['decision'] == 'confirmed' and (
                        {p.get('d1_row_key'), p.get('d1_field_key')} & d1_keys or {p.get('d2_row_key'), p.get('d2_field_key')} & d2_keys)]:
                    del self.pairs[other]
            pid = _hash(f"{d1_row_key or d1_field_key}#{d2_row_key or d2_field_key}")
            self.pairs[pid] = {'decision': decision, 'd1_row_key': d1_row_key, 'd2_row_key': d2_row_key,
                               'd1_field_key': d1_field_key, 'd2_field_key': d2_field_key,
                               'd1_name': d1_name, 'd2_name': d2_name, 'timestamp': time.time()}
            if decision == 'confirmed':
                self._maybe_learn_alias(d1_name, d2_name)
            self._rebuild_index()
            return pid

    def _maybe_learn_alias(self, name1: str, name2: str):
        """Only spelling variants become global aliases; a confirmed pair of clearly different names stays pair-specific."""
        n1, n2 = normalize_nepali_text(name1), normalize_nepali_text(name2)
        if not n1 or not n2 or n1 == n2 or len(n1) < 2 or len(n2) < 2:
            return
        if evaluate_name_pair(name1, name2, None)['score'] < 85:
            return
        with self._lock:
            for a, b in ((n1, n2), (n2, n1)):
                lst = self.name_aliases.setdefault(a, [])
                if b not in lst:
                    lst.append(b)

    def are_aliases(self, norm1: str, norm2: str) -> bool:
        return bool(norm1) and bool(norm2) and (norm1 == norm2 or norm2 in self.name_aliases.get(norm1, ()))

    def decision_for(self, d1_row, d2_row, d1_field, d2_field) -> Optional[str]:
        pid = self._find_pid(d1_row, d2_row, d1_field, d2_field)
        return self.pairs[pid]['decision'] if pid else None

    def confirmed_targets(self, d1_row: str, d1_field: str) -> set:
        return self._confirmed_by_d1.get(d1_row, set()) | self._confirmed_by_d1.get(d1_field, set())


# ==========================================
# 4. NAME MATCHING
# ==========================================

def evaluate_names(raw1: str, norm1: str, rom1: str, raw2: str, norm2: str, rom2: str,
                   feedback_store: Optional[FeedbackStore] = None) -> Dict[str, Any]:
    """Name (D1) vs name (D2), on precomputed normalized and romanized forms."""
    if not norm1 or not norm2:
        return {'score': 0.0, 'reason': "नाम विवरण नखुलेको (Name missing)", 'status': "Mismatch"}

    if feedback_store and norm1 != norm2 and feedback_store.are_aliases(norm1, norm2):
        return {'score': 100.0, 'reason': f"✓ सिकाईएको उपनाम/उस्तै नाम ('{raw1}' = '{raw2}')", 'status': "Exact"}

    if re.sub(r'\s+', ' ', raw1).strip().lower() == re.sub(r'\s+', ' ', raw2).strip().lower():
        return {'score': 100.0, 'reason': f"हुबहु नाम मिलान ('{raw1}')", 'status': "Exact"}

    if norm1 == norm2:
        return {'score': 99.0, 'reason': f"नेपाली हिज्जे तथा ध्वन्यात्मक पूर्ण समानता ('{raw1}' ≈ '{raw2}')", 'status': "Exact"}

    if rom1 and rom1 == rom2:
        return {'score': 98.0, 'reason': f"अंग्रेजी/नेपाली लिप्यन्तरण पूर्ण समानता ('{raw1}' ≈ '{raw2}')", 'status': "Exact"}

    t1, t2 = norm1.split(), norm2.split()
    if len(t1) >= 2 and len(t2) >= 2 and fuzz.ratio(t1[0], t2[0]) >= 85 and fuzz.ratio(t1[-1], t2[-1]) >= 85:
        return {'score': 93.0, 'reason': f"पहिलो नाम र थर मिलान, बीचको नाम थपघट ('{raw1}' ↔ '{raw2}')", 'status': "High"}

    rt1, rt2 = rom1.split(), rom2.split()
    if len(rt1) >= 2 and len(rt2) >= 2 and fuzz.ratio(rt1[0], rt2[0]) >= 85 and fuzz.ratio(rt1[-1], rt2[-1]) >= 85:
        return {'score': 92.0, 'reason': f"अंग्रेजी/नेपाली नाम-थर मिलान ('{raw1}' ↔ '{raw2}')", 'status': "High"}

    scores = [fuzz.token_sort_ratio(norm1, norm2), fuzz.token_sort_ratio(rom1, rom2),
              distance.JaroWinkler.similarity(norm1, norm2) * 100.0,
              distance.JaroWinkler.similarity(rom1, rom2) * 100.0]
    # token_set treats "थापा" as fully contained in "राम थापा"; only use it when both names are multi-word
    if len(t1) >= 2 and len(t2) >= 2:
        scores += [fuzz.token_set_ratio(norm1, norm2), fuzz.token_set_ratio(rom1, rom2)]
    sc = round(max(scores), 1)

    def similar(a: str, b: str, ra: str = '', rb: str = '') -> float:
        return max(fuzz.ratio(a, b), fuzz.ratio(ra, rb) if ra and rb else 0)

    multi = len(t1) >= 2 and len(t2) >= 2
    # A clearly different first name (same surname / middle name) is usually a relative, not the same person.
    # Initials (one letter, e.g. "र. ब. थापा") are not judged here.
    first_differs = (multi and len(t1[0]) > 1 and len(t2[0]) > 1
                     and similar(t1[0], t2[0], rt1[0] if rt1 else '', rt2[0] if rt2 else '') < 70)
    if first_differs and sc > 75.0:
        return {'score': 75.0, 'reason': f"पहिलो नाम फरक ('{t1[0]}' ≠ '{t2[0]}') - समीक्षा आवश्यक", 'status': "Review"}
    # A surname that appears nowhere in the other name means a different family; an added clan word
    # ("थापा" -> "थापा मगर") is fine because the surname is still present.
    surname_differs = (multi
                       and not any(similar(t1[-1], x) >= 85 for x in t2)
                       and not any(similar(t2[-1], x) >= 85 for x in t1)
                       and not (rt1 and rt2 and (any(fuzz.ratio(rt1[-1], x) >= 85 for x in rt2) or any(fuzz.ratio(rt2[-1], x) >= 85 for x in rt1))))
    if surname_differs and sc > 75.0:
        return {'score': 75.0, 'reason': f"थर फरक ('{t1[-1]}' ≠ '{t2[-1]}') - समीक्षा आवश्यक", 'status': "Review"}

    if sc >= 85.0:
        return {'score': sc, 'reason': f"उच्च अक्षर तथा हिज्जे समानता ({sc}%)", 'status': "High"}
    if sc >= 50.0:
        return {'score': sc, 'reason': f"आंशिक नाम समानता ({sc}%) - समीक्षा आवश्यक", 'status': "Review"}
    return {'score': round(sc * 0.35, 1), 'reason': f"नाम फरक परेको ('{raw1}' ≠ '{raw2}')", 'status': "Mismatch"}


def evaluate_name_pair(v1: Any, v2: Any, feedback_store: Optional[FeedbackStore] = None) -> Dict[str, Any]:
    s1, s2 = clean_cell(v1), clean_cell(v2)
    return evaluate_names(s1, normalize_nepali_text(s1), to_roman_phonetic(s1),
                          s2, normalize_nepali_text(s2), to_roman_phonetic(s2), feedback_store)


GLOBAL_FEEDBACK_STORE: Optional[FeedbackStore] = None


def get_feedback_store() -> FeedbackStore:
    global GLOBAL_FEEDBACK_STORE
    if GLOBAL_FEEDBACK_STORE is None:
        GLOBAL_FEEDBACK_STORE = FeedbackStore()
    return GLOBAL_FEEDBACK_STORE


# ==========================================
# 5. ADDRESS MATCHING
# ==========================================

ADDRESS_COLUMNS = ('PerAddress', 'TemAddress', 'OtherAddress')
# how the three address columns are named in reasons shown to the reviewer
ADDRESS_LABELS = {'PerAddress': 'स्थायी ठेगाना', 'TemAddress': 'अस्थायी ठेगाना', 'OtherAddress': 'अन्य ठेगाना'}


def score_address(d1: Dict[str, Any], d2: Dict[str, Any]) -> Dict[str, Any]:
    """district / locallevel / wardno (D1) against each of PerAddress, TemAddress, OtherAddress (D2); best one wins."""
    raw_dist, raw_loc, raw_ward = d1['district'], d1['locallevel'], d1['wardno']
    candidates = d2['addresses']   # [(col, raw, norm, district_codes)]

    if not candidates:
        return {'matched': False, 'matched_col': '', 'matched_val': '', 'score': 50.0,
                'reason': "फाइल B मा ठेगाना (स्थायी, अस्थायी, अन्य) खाली रहेको",
                'matched_district': False, 'matched_locallevel': False, 'matched_ward': False}

    best = None
    for col, raw_addr, norm_addr, codes in candidates:
        dist_matched = False
        if d1['norm_dist']:
            if d1['district_code'] and d1['district_code'] in codes:
                dist_matched = True
            elif _token_in(d1['norm_dist'], norm_addr):
                dist_matched = True
            elif len(d1['norm_dist']) >= 4 and fuzz.partial_ratio(d1['norm_dist'], norm_addr) >= 85:
                dist_matched = True

        loc_matched = False
        if d1['norm_loc']:
            if _token_in(d1['norm_loc'], norm_addr):
                loc_matched = True
            elif len(d1['norm_loc']) >= 3 and fuzz.partial_ratio(d1['norm_loc'], norm_addr) >= 85:
                loc_matched = True

        ward_matched = check_ward_match(d1['ward'], norm_addr, d1['norm_loc'] if loc_matched else '')

        parts = []
        if dist_matched:
            parts.append(f"जिल्ला '{raw_dist}'")
            if loc_matched and ward_matched:
                score = 100.0
                parts += [f"स्थानीय तह '{raw_loc}'", f"वडा नं. '{raw_ward}'"]
            elif loc_matched:
                score = 95.0
                parts.append(f"स्थानीय तह '{raw_loc}'")
            elif ward_matched:
                score = 90.0
                parts.append(f"वडा नं. '{raw_ward}'")
            else:
                score = 80.0
        else:
            if loc_matched and ward_matched:
                score = 85.0
                parts += [f"स्थानीय तह '{raw_loc}'", f"वडा नं. '{raw_ward}' (जिल्ला भिन्न वा नखुलेको)"]
            elif loc_matched:
                score = 65.0
                parts.append(f"स्थानीय तह '{raw_loc}'")
            elif ward_matched:
                score = 35.0
                parts.append(f"वडा नं. '{raw_ward}'")
            else:
                score = 0.0

        label = ADDRESS_LABELS.get(col, col)
        reason = (f"{', '.join(parts)} {label} ('{raw_addr}') मा मिलान" if parts
                  else f"{label} ('{raw_addr}') मा ठेगाना मेल खाएन")
        if best is None or score > best['score']:
            best = {'matched': score >= 70.0, 'matched_col': col, 'matched_val': raw_addr, 'score': score,
                    'reason': reason, 'matched_district': dist_matched, 'matched_locallevel': loc_matched,
                    'matched_ward': ward_matched}

    if best['score'] == 0.0:
        present = ", ".join(f"{ADDRESS_LABELS.get(c, c)}: '{v}'" for c, v, _, _ in candidates)
        best['reason'] = f"जिल्ला/स्थानीय तह/वडा स्थायी, अस्थायी वा अन्य ठेगाना कुनैमा पनि फेला परेन ({present})"
        best['matched_col'] = ''
        best['matched_val'] = ''
    return best


# ==========================================
# 6. COLUMN DETECTION
# ==========================================

def _col_norm(c: Any) -> str:
    return re.sub(r'[\s_.\-]+', '', str(c)).lower()


# field -> accepted normalized header names, in priority order
COLUMN_RULES_A: Dict[str, List[str]] = {
    'name': ['fullname', 'name', 'नाम', 'नामथर'],
    'district': ['district', 'जिल्ला'],
    'locallevel': ['locallevel', 'locallevelname', 'स्थानीयतह', 'गाउँपालिका', 'नगरपालिका', 'पालिका'],
    'wardno': ['wardno', 'ward', 'वडा', 'वडानं', 'वडानम्बर'],
    'id': ['id', 'recid', 'sn', 'सिनं'],
}
COLUMN_RULES_B: Dict[str, List[str]] = {
    'name': ['fullname', 'name', 'नाम', 'नामथर'],
    'per_address': ['peraddress', 'permanentaddress', 'स्थायीठेगाना'],
    'tem_address': ['temaddress', 'temporaryaddress', 'अस्थायीठेगाना'],
    'other_address': ['otheraddress', 'अन्यठेगाना', 'address', 'ठेगाना'],
    'id': ['id', 'recid', 'regno', 'regdno', 'दर्तानं'],
}


def detect_columns(columns: List[Any], dataset: int, mapping: Optional[Dict[str, str]] = None) -> Dict[str, Optional[str]]:
    """
    Column for each field of file A (dataset 1) or file B (dataset 2). An explicit mapping {field: column} wins
    when the column exists; otherwise the accepted header names are tried in priority order.
    """
    rules = COLUMN_RULES_A if dataset == 1 else COLUMN_RULES_B
    mapping = mapping or {}
    cols = list(columns)
    norms = {c: _col_norm(c) for c in cols}
    used: set = set()
    result: Dict[str, Optional[str]] = {}
    for field in rules:
        m = mapping.get(field)
        if m and m in cols:
            result[field] = m
            used.add(m)
    for field, names in rules.items():
        if field in result:
            continue
        found = None
        for name in names:
            found = next((c for c in cols if c not in used and norms[c] == name), None)
            if found is not None:
                break
        result[field] = found
        if found is not None:
            used.add(found)
    return result


def _status(category: str) -> str:
    return STATUS_LABELS[category]


ASSIGNABLE = ('verified', 'perfect', 'high', 'review')
CANDIDATES_KEPT = 8     # scored candidates remembered per file A row for live re-assignment
RECOMMENDATIONS = 5     # candidates shown to the reviewer per file A row, best name match first
REC_NAME_MIN = 50.0     # a recommendation needs a name score above this; an address match alone means nothing


class RecordMatcher:
    def __init__(self, feedback_store: Optional[FeedbackStore] = None, k_candidates: int = 20):
        self.feedback_store = feedback_store or get_feedback_store()
        self.k_candidates = k_candidates

    # ---------- preprocessing ----------
    def _prepare(self, df1: pd.DataFrame, df2: pd.DataFrame,
                 mapping1: Optional[Dict[str, str]], mapping2: Optional[Dict[str, str]]):
        c1 = detect_columns(list(df1.columns), 1, mapping1)
        c2 = detect_columns(list(df2.columns), 2, mapping2)
        if not c1.get('name'):
            raise ValueError("File A: name column not found (pass an explicit column mapping).")
        if not c2.get('name'):
            raise ValueError("File B: name column not found (pass an explicit column mapping).")
        if not (c2.get('per_address') or c2.get('tem_address') or c2.get('other_address')):
            raise ValueError("File B: no address column (PerAddress / TemAddress / OtherAddress) found.")

        def get(row, col):
            return clean_cell(row.get(col)) if col else ''

        d1_records, d1_texts = [], []
        for pos, (idx, row) in enumerate(df1.iterrows()):
            row_dict = row.dropna().to_dict()
            items = [f"{k}: {v}" for k, v in row_dict.items() if str(v).strip()]
            raw_summary = " | ".join(items) if items else f"Row {idx + 1}"
            name, dist, loc, ward_raw = get(row, c1['name']), get(row, c1['district']), get(row, c1['locallevel']), get(row, c1['wardno'])
            rec = {
                'orig_index': idx,
                'full_name': name, 'district': dist, 'locallevel': loc, 'wardno': ward_raw,
                'id': get(row, c1['id']) or f"D1_row_{pos + 1}",
                'norm_name': normalize_nepali_text(name), 'roman_name': to_roman_phonetic(name),
                'norm_dist': normalize_nepali_text(dist), 'district_code': resolve_district_key(dist),
                'norm_loc': clean_local_level(loc), 'ward': ward_digits(ward_raw),
                'raw_summary': raw_summary,
                'row_key': row_key(raw_summary),
                'field_key': d1_field_key(name, dist, loc, ward_raw),
            }
            d1_records.append(rec)
            d1_texts.append(" ".join(p for p in [rec['norm_name'], rec['norm_name'], rec['norm_dist'], rec['norm_loc'], rec['ward']] if p) or "खाली")

        d2_records, d2_texts = [], []
        for pos, (idx, row) in enumerate(df2.iterrows()):
            row_dict = row.dropna().to_dict()
            items = [f"{k}: {v}" for k, v in row_dict.items() if str(v).strip()]
            raw_summary = " | ".join(items) if items else f"Row {idx + 1}"
            name = get(row, c2['name'])
            per, tem, other = get(row, c2['per_address']), get(row, c2['tem_address']), get(row, c2['other_address'])
            addresses = []
            for col, raw in zip(ADDRESS_COLUMNS, (per, tem, other)):
                if raw:
                    norm = normalize_nepali_text(raw)
                    addresses.append((col, raw, norm, find_district_keys(norm)))
            rec = {
                'orig_index': idx,
                'full_name': name, 'PerAddress': per, 'TemAddress': tem, 'OtherAddress': other,
                'id': get(row, c2['id']) or f"D2_row_{pos + 1}",
                'norm_name': normalize_nepali_text(name), 'roman_name': to_roman_phonetic(name),
                'addresses': addresses,
                'district_codes': {code for a in addresses for code in a[3]},
                'raw_summary': raw_summary,
                'row_key': row_key(raw_summary),
                'field_key': d2_field_key(name, per, tem, other),
            }
            d2_records.append(rec)
            d2_texts.append(" ".join([rec['norm_name'], rec['norm_name']] + [a[2] for a in addresses]).strip() or "खाली")

        id_columns = {'d1': c1.get('id'), 'd2': c2.get('id')}
        return d1_records, d2_records, d1_texts, d2_texts, id_columns

    # ---------- pair scoring ----------
    def compute_similarity(self, d1: Dict[str, Any], d2: Dict[str, Any]) -> Dict[str, Any]:
        decision = self.feedback_store.decision_for(d1['row_key'], d2['row_key'], d1['field_key'], d2['field_key']) if self.feedback_store else None

        name_info = evaluate_names(d1['full_name'], d1['norm_name'], d1['roman_name'],
                                   d2['full_name'], d2['norm_name'], d2['roman_name'], self.feedback_store)
        addr_info = score_address(d1, d2)
        name_score = round(name_info['score'], 1)
        addr_score = round(addr_info['score'], 1)
        avg = round((name_score + addr_score) / 2.0, 1)

        if decision == 'rejected':
            category, avg = 'blocked', 0.0
            overall = "प्रयोगकर्ताद्वारा अमान्य गरिएको जोडी (User blocked this pair)"
        elif decision == 'confirmed':
            category, avg = 'verified', 100.0
            overall = "प्रयोगकर्ताद्वारा प्रमाणीकरण गरिएको मिलान (User verified this pair)"
        else:
            if avg >= PERFECT_MIN and addr_info['matched'] and name_score >= 85.0:
                category = 'perfect'
            elif avg >= HIGH_MIN and name_score >= 85.0:
                # a good address alone never makes a High match: the name itself must be convincing
                category = 'high'
            elif avg >= REVIEW_MIN:
                category = 'review'
            else:
                category = 'none'
            overall = f"नाम मिलान: {name_score}% + ठेगाना मिलान: {addr_score}% = औसत मिलान: {avg}%"

        return {
            'confidence_score': avg,
            'name_score': name_score,
            'name_reason': name_info['reason'],
            'name_status': name_info['status'],
            'district_score': addr_score,
            'district_reason': addr_info['reason'],
            'district_status': "Exact" if addr_info['matched'] else "Mismatch",
            'matched_address_col': addr_info['matched_col'],
            'matched_address_val': addr_info['matched_val'],
            'matched_district': addr_info['matched_district'],
            'matched_locallevel': addr_info['matched_locallevel'],
            'matched_ward': addr_info['matched_ward'],
            'match_category': category,
            'match_status': _status(category),
            'overall_reason': overall,
        }

    # ---------- dataset matching ----------
    def match_datasets(self, df1: pd.DataFrame, df2: pd.DataFrame,
                       mapping1: Optional[Dict[str, str]] = None, mapping2: Optional[Dict[str, str]] = None
                       ) -> 'MatchSession':
        d1_records, d2_records, d1_texts, d2_texts, id_columns = self._prepare(df1, df2, mapping1, mapping2)
        total_rows, n2 = len(d1_records), len(d2_records)
        candidates: List[List[Tuple[int, Dict[str, Any]]]] = [[] for _ in range(total_rows)]
        if total_rows == 0 or n2 == 0:
            return MatchSession(self, d1_records, d2_records, candidates, id_columns)

        district_index: Dict[str, List[int]] = {}
        row_key_index: Dict[str, int] = {}
        field_key_index: Dict[str, int] = {}
        for j, r2 in enumerate(d2_records):
            for code in r2['district_codes']:
                district_index.setdefault(code, []).append(j)
            if r2['row_key']:
                row_key_index.setdefault(r2['row_key'], j)
            if r2['field_key']:
                field_key_index.setdefault(r2['field_key'], j)
        district_index_np = {code: np.array(idx, dtype=np.int64) for code, idx in district_index.items()}

        vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 3), min_df=1, sublinear_tf=True, dtype=np.float32)
        tfidf_d2 = vectorizer.fit_transform(d2_texts)
        tfidf_d1 = vectorizer.transform(d1_texts)
        tfidf_d2_t = tfidf_d2.T.tocsr()

        k = min(self.k_candidates, n2)
        # cap the dense similarity block at ~20M float32 (~80 MB) regardless of dataset size
        batch_size = max(1, min(500, 20_000_000 // n2))

        def top_k(sims: np.ndarray, pool: Optional[np.ndarray] = None) -> List[int]:
            if pool is None:
                if n2 <= k:
                    return [int(x) for x in np.argsort(sims)[::-1]]
                part = np.argpartition(sims, -k)[-k:]
                return [int(x) for x in part[np.argsort(sims[part])[::-1]]]
            sub = sims[pool]
            if len(pool) <= k:
                return [int(x) for x in pool[np.argsort(sub)[::-1]]]
            part = np.argpartition(sub, -k)[-k:]
            return [int(x) for x in pool[part[np.argsort(sub[part])[::-1]]]]

        for b_start in range(0, total_rows, batch_size):
            b_end = min(b_start + batch_size, total_rows)
            chunk = (tfidf_d1[b_start:b_end] @ tfidf_d2_t).toarray()

            for bi in range(b_end - b_start):
                i = b_start + bi
                r1 = d1_records[i]
                sims = chunk[bi]

                pool: List[int] = []
                for key in self.feedback_store.confirmed_targets(r1['row_key'], r1['field_key']) if self.feedback_store else ():
                    j = row_key_index.get(key, field_key_index.get(key))
                    if j is not None:
                        pool.append(j)
                if r1['district_code'] in district_index_np:
                    pool += top_k(sims, district_index_np[r1['district_code']])
                pool += top_k(sims)
                seen: set = set()
                pool = [c for c in pool if not (c in seen or seen.add(c))]

                scored = [(j, self.compute_similarity(r1, d2_records[j])) for j in pool]
                plausible = MatchSession.rank(scored)[:CANDIDATES_KEPT]
                # low scorers with a real name match are kept only for the reviewer's recommendation list; never assigned
                low = sorted((c for c in scored if c[1]['match_category'] == 'none' and c[1]['name_score'] > REC_NAME_MIN),
                             key=lambda c: (-c[1]['name_score'], -c[1]['confidence_score']))[:RECOMMENDATIONS]
                candidates[i] = plausible + low + [c for c in scored if c[1]['match_category'] == 'blocked']

        session = MatchSession(self, d1_records, d2_records, candidates, id_columns)
        session.assign_all()
        return session


class MatchSession:
    """
    Result of one matching run.

    One-to-one: every record of file B is suggested for at most one record of file A.
    Reviewer-verified pairs are fixed first; the rest are assigned greedily from the highest score down,
    so a row whose best candidate went to a stronger pair falls back to its next-best free candidate.
    """

    def __init__(self, matcher: 'RecordMatcher', d1_records, d2_records, candidates, id_columns):
        self.matcher = matcher
        self.store = matcher.feedback_store
        self.d1 = d1_records
        self.d2 = d2_records
        self.candidates: List[List[Tuple[int, Dict[str, Any]]]] = candidates
        self.id_columns = id_columns
        self.assign: List[Optional[int]] = [None] * len(d1_records)
        self.claimed: Dict[int, int] = {}
        self.d1_pos = {r['orig_index']: i for i, r in enumerate(d1_records)}
        self.d2_pos = {r['orig_index']: j for j, r in enumerate(d2_records)}
        self.lock = threading.RLock()
        self._wanted_by: Dict[int, set] = {}
        for i, cands in enumerate(candidates):
            for j, _ in cands:
                self._wanted_by.setdefault(j, set()).add(i)

    # ---------- ranking & assignment ----------
    @staticmethod
    def rank(cands: List[Tuple[int, Dict[str, Any]]]) -> List[Tuple[int, Dict[str, Any]]]:
        order = {'verified': 0, 'perfect': 1, 'high': 2, 'review': 3}
        usable = [c for c in cands if c[1]['match_category'] in order]
        return sorted(usable, key=lambda c: (0 if c[1]['match_category'] == 'verified' else 1, -c[1]['confidence_score']))

    def _set(self, i: int, j: Optional[int]):
        old = self.assign[i]
        if old is not None and self.claimed.get(old) == i:
            del self.claimed[old]
        self.assign[i] = j
        if j is not None:
            self.claimed[j] = i

    def _info(self, i: int, j: int) -> Optional[Dict[str, Any]]:
        return next((inf for jj, inf in self.candidates[i] if jj == j), None)

    def assign_all(self):
        with self.lock:
            self.assign = [None] * len(self.d1)
            self.claimed = {}
            for i, cands in enumerate(self.candidates):
                for j, inf in cands:
                    if inf['match_category'] == 'verified' and j not in self.claimed and self.assign[i] is None:
                        self._set(i, j)
            pairs = [(inf['confidence_score'], i, j) for i, cands in enumerate(self.candidates)
                     for j, inf in cands if inf['match_category'] in ('perfect', 'high', 'review')]
            pairs.sort(key=lambda p: (-p[0], p[1]))
            for _, i, j in pairs:
                if self.assign[i] is None and j not in self.claimed:
                    self._set(i, j)

    def _fill(self, rows: List[int]) -> set:
        """Give each listed unassigned row its best still-free candidate (strongest rows first)."""
        changed = set()
        options = []
        for i in set(rows):
            if self.assign[i] is not None:
                continue
            for j, inf in self.rank(self.candidates[i]):
                if j not in self.claimed:
                    options.append((inf['confidence_score'], i, j))
        options.sort(key=lambda p: (-p[0], p[1]))
        for _, i, j in options:
            if self.assign[i] is None and j not in self.claimed:
                self._set(i, j)
                changed.add(i)
        return changed

    def apply_decision(self, d1_orig: Any, d2_orig: Any, action: str) -> List[int]:
        """Called after the FeedbackStore recorded the decision. Returns the D1 positions whose rows changed."""
        with self.lock:
            i, j = self.d1_pos.get(d1_orig), self.d2_pos.get(d2_orig)
            if i is None or j is None:
                return []
            info = self.matcher.compute_similarity(self.d1[i], self.d2[j])
            self.candidates[i] = [c for c in self.candidates[i] if c[0] != j] + [(j, info)]
            self._wanted_by.setdefault(j, set()).add(i)
            changed = {i}
            freed: List[int] = []

            if action == 'confirm':
                old = self.assign[i]
                other = self.claimed.get(j)
                if other is not None and other != i:
                    self._set(other, None)
                    changed.add(other)
                self._set(i, j)
                if old is not None and old != j:
                    freed.append(old)
                if other is not None and other != i:
                    changed |= self._fill([other])
            elif action == 'reject':
                if self.assign[i] == j:
                    self._set(i, None)
                    freed.append(j)
                changed |= self._fill([i])
            elif action == 'undo':
                if self.assign[i] is None and j not in self.claimed and info['match_category'] in ASSIGNABLE:
                    self._set(i, j)

            for fj in freed:
                if fj not in self.claimed:
                    changed |= self._fill(sorted(self._wanted_by.get(fj, ())))
            return sorted(changed)

    def pair_key(self, i: int, j: int) -> str:
        return f"{self.d1[i]['row_key']}|{self.d2[j]['row_key']}"


    def verified_elsewhere(self, i: int, j: int) -> bool:
        """file B record j is already verified (matched by the reviewer) with another file A row."""
        other = self.claimed.get(j)
        if other is None or other == i:
            return False
        inf = self._info(other, j)
        return bool(inf) and inf['match_category'] == 'verified'

    def recommendations(self, i: int) -> List[Tuple[int, Dict[str, Any]]]:
        """
        Up to RECOMMENDATIONS candidates of a record whose NAME matches (name score above REC_NAME_MIN),
        best name match first (overall score breaks ties), plus the current suggestion if it is not among them.
        Left out: weak names (however good the address), blocked pairs, and file B records already
        verified with another row. Reviewer-verified pairs are always kept.
        """
        usable = [c for c in self.candidates[i]
                  if c[1]['match_category'] != 'blocked' and not self.verified_elsewhere(i, c[0])
                  and (c[1]['name_score'] > REC_NAME_MIN or c[1]['match_category'] == 'verified')]
        j = self.assign[i]
        usable.sort(key=lambda c: (c[1]['match_category'] != 'verified', -c[1]['name_score'],
                                   -c[1]['confidence_score'], c[0] != j))
        recs = usable[:RECOMMENDATIONS]
        if j is not None and all(c[0] != j for c in recs):
            inf = self._info(i, j)
            if inf:
                recs.append((j, inf))
        return recs

