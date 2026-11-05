"""
=============================================================================
  MODUL BERSAMA: PRAPROSES & PELABELAN INSET LEXICON
  Dipakai oleh main.py (pipeline) DAN dashboard.py (prediksi komentar baru)
  agar aturan pelabelan di dasbor selalu identik dengan pipeline penelitian.
=============================================================================
  Rancangan (sesuai proposal):
  - Praproses tahap 1 : pembersihan tautan/mention/tagar/angka, case folding,
                        pengurangan huruf berulang, normalisasi kata tidak baku,
                        tokenisasi dengan tanda baca sebagai batas klausa.
  - Pelabelan         : per klausa; negasi (jendela 3 token, klausa yang sama)
                        dikalikan NEGASI_FAKTOR = -0,74 (Hutto & Gilbert, 2014);
                        klausa sebelum/sesudah kata kontras dibobot 0,5 / 1,5.
  - Kamus             : kata tugas (stopword) dikeluarkan, koreksi domain
                        ekonomi diterapkan, lalu kata yang ada di kamus positif
                        dan negatif sekaligus diberi bobot |terbesar| (seri -> abaikan).
  - Praproses tahap 2 : stopword selektif, buang token < 4 huruf (kecuali
                        negasi), stemming ECS (PySastrawi).
=============================================================================
"""
import os
import re
import unicodedata

import pandas as pd

# =============================================================================
# PARAMETER PELABELAN (ditetapkan a priori, bukan di-tuning pada data)
# =============================================================================
NEGASI_FAKTOR = -0.74      # faktor pelemahan negasi VADER (Hutto & Gilbert, 2014)
JENDELA_NEGASI = 3         # kata negasi memengaruhi 3 token sesudahnya (klausa yang sama)
KONTRAS_PRE   = 0.5        # bobot klausa SEBELUM kata kontras
KONTRAS_POST  = 1.5        # bobot klausa SESUDAH kata kontras
OVERLAP_RULE  = 'maks_abs' # kata di kamus positif & negatif: bobot |terbesar| menang, seri -> abaikan

KONTRAS = {'tapi', 'tetapi', 'namun', 'melainkan', 'padahal', 'sayangnya'}

# Varian tidak baku (gak, ga, jgn, engga, ...) dinormalisasi oleh SLANG_DICT
# SEBELUM pelabelan, sehingga negasi efektif: tidak, bukan, belum, jangan,
# tanpa, kurang, tiada (sesuai Batasan Masalah pada proposal).
KATA_NEGASI = {
    'tidak', 'bukan', 'belum', 'jangan', 'tanpa', 'kurang',
    'tiada', 'tak', 'gak', 'ga', 'nggak', 'bkn', 'tdk'
}

# kata sentimen yang TIDAK boleh terhapus oleh stopword (praproses tahap 2)
KATA_SENTIMEN_WHITELIST = {
    'baik', 'bagus', 'tepat', 'yakin', 'benar', 'jelas',
    'cukup', 'hebat', 'mantap', 'keren'
}

# stopword Sastrawi yang TETAP dipertahankan di kamus karena memang bermuatan sentimen
STOPWORD_SENTIMEN_TETAP = {
    'masalah', 'persoalan', 'mempersoalkan', 'penting', 'enak', 'keterlaluan', 'percuma',
    'mampu', 'pantas', 'betul', 'baiklah', 'seenaknya', 'kelamaan', 'sepihak', 'memihak',
    'meyakinkan', 'siap'
}

SLANG_DICT = {
    # Negasi
    'gak': 'tidak', 'ga': 'tidak', 'nggak': 'tidak', 'ngga': 'tidak', 'nggk': 'tidak',
    'kagak': 'tidak', 'tdk': 'tidak', 'tak': 'tidak', 'g': 'tidak', 'kaga': 'tidak',
    'gk': 'tidak', 'gx': 'tidak', 'engga': 'tidak', 'enggak': 'tidak', 'ngak': 'tidak',
    'ndak': 'tidak', 'nda': 'tidak', 'tdak': 'tidak',
    'bkn': 'bukan', 'blm': 'belum', 'blum': 'belum',
    'jgn': 'jangan', 'jngn': 'jangan', 'jangn': 'jangan',
    # Positif / Apresiasi
    'bgt': 'banget', 'bener': 'benar', 'betull': 'betul', 'btul': 'betul',
    'mantapp': 'mantap', 'mantabb': 'mantap', 'mantul': 'mantap', 'mantep': 'mantap',
    'kerenn': 'keren', 'kereen': 'keren', 'baguss': 'bagus', 'baguuus': 'bagus',
    'tks': 'terima kasih', 'makasih': 'terima kasih', 'makasi': 'terima kasih',
    'thanks': 'terima kasih', 'thank': 'terima kasih', 'thx': 'terima kasih',
    'trims': 'terima kasih', 'terimakasih': 'terima kasih',
    'jos': 'bagus', 'joss': 'bagus', 'top': 'hebat', 'oke': 'baik', 'ok': 'baik',
    'sip': 'baik', 'sipp': 'baik', 'cuan': 'untung', 'beneran': 'benar', 'juoss': 'bagus',
    # Negatif / Keluhan
    'boncos': 'rugi', 'anjlokk': 'anjlok', 'nyungsep': 'jatuh', 'drop': 'turun',
    'ancur': 'hancur', 'bobrok': 'rusak', 'kaco': 'kacau', 'amblas': 'hilang',
    'bego': 'bodoh', 'tolol': 'bodoh', 'goblok': 'bodoh', 'ngawur': 'salah',
    'korup': 'korupsi', 'koruptor': 'korupsi',
    # Umum
    'tp': 'tapi', 'tpi': 'tapi', 'klo': 'kalau', 'kalo': 'kalau', 'dr': 'dari', 'yg': 'yang',
    'dg': 'dengan', 'dgn': 'dengan', 'krn': 'karena', 'utk': 'untuk', 'sy': 'saya',
    'gw': 'saya', 'gue': 'saya', 'lu': 'kamu', 'loe': 'kamu', 'sdh': 'sudah',
    'udh': 'sudah', 'udah': 'sudah', 'aja': 'saja', 'org': 'orang'
}

# =============================================================================
# PEMBERSIHAN DASAR (dipakai praproses tahap 1 dan pemecah klausa)
# =============================================================================
_RE_LEBAR_NOL = re.compile('[​‌‍⁠﻿]')
_RE_TAUTAN    = re.compile(r'http\S+|www\.\S+')
_RE_MENTION   = re.compile(r'@[\w.\-]+')        # handle YouTube boleh memuat '-' dan '.'
_RE_TAGAR     = re.compile(r'#\w+')
_RE_ANGKA     = re.compile(r'\d+')
_RE_TITIK_SPASI = re.compile(r'(?<=[a-z])\.(?=[a-z])')   # "kalau.kita.tidak" -> "kalau kita tidak"
_RE_ULANG     = re.compile(r'(.)\1{2,}')                  # "baguuus" -> "bagus"


def bersihkan_dasar(text: str) -> str:
    """NFKC + hapus karakter lebar nol, tautan, mention, tagar, angka; huruf kecil;
    titik di antara huruf dianggap spasi; huruf berulang >= 3 dipendekkan."""
    t = unicodedata.normalize('NFKC', text)
    t = _RE_LEBAR_NOL.sub('', t).lower()
    t = _RE_TAUTAN.sub(' ', t)
    t = _RE_MENTION.sub(' ', t)
    t = _RE_TAGAR.sub(' ', t)
    t = _RE_ANGKA.sub(' ', t)
    t = _RE_TITIK_SPASI.sub(' ', t)
    t = _RE_ULANG.sub(r'\1', t)
    return t


def tokenize_pra_stem(text: str) -> list:
    """Praproses TAHAP 1 tanpa batas klausa (dipakai untuk fitur & cakupan leksikon)."""
    if not isinstance(text, str):
        return []
    t = re.sub(r'[^\w\s]', ' ', bersihkan_dasar(text))
    flat = []
    for tk in t.split():
        flat.extend(SLANG_DICT.get(tk, tk).split())
    return flat


def parse_klausa(text: str) -> list:
    """
    Memecah komentar menjadi KALIMAT -> SEGMEN (klausa) -> token.
    Batas kalimat : . ! ? ; baris baru
    Batas klausa  : , : ( ) " / dan kata kontras (tapi/namun/tetapi/...)
    Hasil: list of (segmen_list, indeks_segmen_pertama_setelah_kontras | None)
    """
    if not isinstance(text, str):
        return []
    t = bersihkan_dasar(text)
    t = re.sub(r'[.!?;\n\r]+', ' __S__ ', t)
    t = re.sub(r'[,:()\[\]"/–—]+', ' __K__ ', t)
    t = re.sub(r'[^\w\s]', ' ', t)

    hasil, segs, kidx = [], [[]], None
    for tk in t.split():
        if tk == '__S__':
            hasil.append((segs, kidx))
            segs, kidx = [[]], None
        elif tk == '__K__':
            if segs[-1]:
                segs.append([])
        else:
            for sub_t in SLANG_DICT.get(tk, tk).split():
                if sub_t in KONTRAS:
                    if segs[-1]:
                        segs.append([])
                    kidx = len(segs) - 1
                else:
                    segs[-1].append(sub_t)
    hasil.append((segs, kidx))
    return [(sg, k) for sg, k in hasil if any(sg)]


# =============================================================================
# PELABELAN
# =============================================================================
def label_klausa(kalimat_list: list, pdict: dict, ndict: dict, neg_faktor=None,
                 pre=None, post=None, window: int = JENDELA_NEGASI):
    """
    Pelabelan InSet sadar klausa (Persamaan (1)-(3) pada proposal).
    Mengembalikan (label, skor, ada_kontras, campuran).
    """
    neg_faktor = NEGASI_FAKTOR if neg_faktor is None else neg_faktor
    pre  = KONTRAS_PRE  if pre  is None else pre
    post = KONTRAS_POST if post is None else post
    total, ada_kontras, ada_pos, ada_neg = 0.0, False, False, False
    for segs, kidx in kalimat_list:
        if kidx is not None:
            ada_kontras = True
        for j, seg in enumerate(segs):
            bobot = 1.0 if kidx is None else (pre if j < kidx else post)
            skor_seg, neg_until = 0.0, -1
            for i, t in enumerate(seg):
                if t in KATA_NEGASI:
                    neg_until = i + window
                    continue
                if t in pdict:
                    w = pdict[t]
                elif t in ndict:
                    w = -abs(ndict[t])
                else:
                    continue
                skor_seg += neg_faktor * w if i <= neg_until else w
            if skor_seg > 0:
                ada_pos = True
            elif skor_seg < 0:
                ada_neg = True
            total += bobot * skor_seg
    label = 'positif' if total > 0 else ('negatif' if total < 0 else 'netral')
    return label, total, ada_kontras, (ada_pos and ada_neg)


def label_baseline(tokens: list, pdict: dict, ndict: dict, window: int = JENDELA_NEGASI):
    """Aturan v2 (tanpa klausa, negasi membalik penuh) -- hanya untuk ablasi.
    Mengembalikan (label, skor, jumlah_kata_cocok)."""
    score, n_cocok, neg_until = 0.0, 0, -1
    for i, t in enumerate(tokens):
        if t in KATA_NEGASI:
            neg_until = i + window
            continue
        negated = (i <= neg_until)
        if t in pdict:
            n_cocok += 1
            w = pdict[t]
            score += -w if negated else w
        elif t in ndict:
            n_cocok += 1
            w = abs(ndict[t])
            score += w if negated else -w
    lab = 'positif' if score > 0 else ('negatif' if score < 0 else 'netral')
    return lab, score, n_cocok


# =============================================================================
# KAMUS
# =============================================================================
def load_lexicon_dict(file_path: str) -> dict:
    """Memuat kamus InSet (format: kata<TAB>bobot). Gagal keras jika file hilang."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Kamus leksikon tidak ditemukan: {file_path}")
    d = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.lower().startswith('word'):
                continue
            parts = line.split('\t')
            try:
                d[parts[0].strip().lower()] = float(parts[1])
            except (IndexError, ValueError):
                continue
    if not d:
        raise ValueError(f"Kamus kosong / format tidak terbaca: {file_path}")
    return d


def terapkan_aturan_overlap(pos0: dict, neg0: dict, aturan: str):
    """Selesaikan kata yang muncul di kedua kamus. Mengembalikan (pos, neg) baru."""
    P, N = dict(pos0), dict(neg0)
    for w in set(pos0) & set(neg0):
        wp, wn = abs(pos0[w]), -abs(neg0[w])
        if aturan == 'positif':
            N.pop(w)
        elif aturan == 'negatif':
            P.pop(w)
        elif aturan == 'maks_abs':
            if wp > abs(wn):
                N.pop(w)
            elif wp < abs(wn):
                P.pop(w)
            else:
                P.pop(w)
                N.pop(w)
        elif aturan == 'jumlah':
            P.pop(w)
            N.pop(w)
            net = wp + wn
            if net > 0:
                P[w] = net
            elif net < 0:
                N[w] = net
        elif aturan == 'abaikan':
            P.pop(w)
            N.pop(w)
        else:
            raise ValueError(f"OVERLAP_RULE tidak dikenal: {aturan}")
    return P, N


def bangun_kamus(pos_txt: str, neg_txt: str, stop_words: set, koreksi_csv: str = None,
                 overlap_rule: str = OVERLAP_RULE, buang_stopword: bool = True) -> dict:
    """
    Membangun kamus aktif dengan urutan tetap:
      1) muat kamus asli  2) buang kata tugas (stopword)  3) koreksi domain
      4) aturan kata ganda (overlap)
    Mengembalikan dict berisi kamus aktif beserta kamus antara (untuk audit).
    """
    pos_raw = load_lexicon_dict(pos_txt)
    neg_raw = load_lexicon_dict(neg_txt)
    pos, neg = dict(pos_raw), dict(neg_raw)

    kata_tugas = set()
    if buang_stopword:
        kata_tugas = (set(stop_words) - STOPWORD_SENTIMEN_TETAP) & (set(pos) | set(neg))
        for w in kata_tugas:
            pos.pop(w, None)
            neg.pop(w, None)

    koreksi = []
    if koreksi_csv and os.path.exists(koreksi_csv):
        kor = pd.read_csv(koreksi_csv)
        for _, r in kor.iterrows():
            w = str(r['kata']).strip().lower()
            b = float(r['bobot'])
            pos.pop(w, None)
            neg.pop(w, None)
            if b > 0:
                pos[w] = b
            elif b < 0:
                neg[w] = b
            koreksi.append(w)

    pos_k, neg_k = dict(pos), dict(neg)          # setelah koreksi, sebelum aturan overlap
    pos_akt, neg_akt = terapkan_aturan_overlap(pos_k, neg_k, overlap_rule)
    return {'pos': pos_akt, 'neg': neg_akt, 'pos_raw': pos_raw, 'neg_raw': neg_raw,
            'pos_k': pos_k, 'neg_k': neg_k, 'kata_tugas_dibuang': kata_tugas,
            'koreksi': koreksi, 'overlap_rule': overlap_rule}


# =============================================================================
# PRAPROSES TAHAP 2 (fitur TF-IDF)
# =============================================================================
def siapkan_stopword(raw_stopwords: set) -> set:
    """Stopword aktif = stopword Sastrawi - kata negasi - kata sentimen penting."""
    return set(raw_stopwords) - KATA_NEGASI - KATA_SENTIMEN_WHITELIST


def clean_post_stem(tokens: list, stop_words: set, stem_fn) -> str:
    """stopword selektif -> filter panjang >= 4 (negasi dikecualikan) -> stemming"""
    kept = [t for t in tokens if (len(t) >= 4 or t in KATA_NEGASI) and t not in stop_words]
    return ' '.join(t if t in KATA_NEGASI else stem_fn(t) for t in kept)
