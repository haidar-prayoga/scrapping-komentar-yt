"""
=============================================================================
  PIPELINE ANALISIS SENTIMEN KOMENTAR YOUTUBE  --  REVISI v4
  Topik   : Kondisi Ekonomi Indonesia Akibat Pelemahan Nilai Tukar Rupiah
  Model   : Naive Bayes Classifier (NBC) & Random Forest Classifier (RFC)
  Pelabel : InSet Lexicon (pra-stemming + penanganan negasi)
-----------------------------------------------------------------------------
  PERUBAHAN DARIPADA main.py LAMA
  1. [BUG]  Regex reduksi huruf berulang diperbaiki: (.)\\1{2,} -> \\1
  2. [LEAK] Split data dilakukan SEBELUM TF-IDF; vektorizer hanya fit di data
            latih (sebelumnya fit pada seluruh data).
  3. [LEAK] Komentar duplikat dibuang SEBELUM split.
  4. Metrik utama macro-average (weighted tetap dilaporkan sebagai pembanding).
  5. Cakupan leksikon (jumlah kata cocok per komentar) dihitung & dilaporkan.
  6. 5-fold stratified CV (TF-IDF di-fit ulang di tiap fold, tanpa kebocoran).
  7. Uji McNemar NBC vs RFC pada data uji.
  8. Analisis probabilitas keyakinan: confidence per kelas, benar vs salah,
     Multi-class Brier Score, ECE (10 bin), reliability diagram.
  9. Ekspor sampel anotasi manual (blind) + Cohen's Kappa + evaluasi gold
     standard (otomatis jalan jika file anotasi sudah diisi).
 10. Semua angka hasil disimpan ke output/*.csv & output/hasil_ringkasan.json
-----------------------------------------------------------------------------
  TAMBAHAN v3 (perbaikan kualitas pelabelan)
 11. Pelabelan SADAR KLAUSA: cakupan negasi berhenti di batas klausa/kalimat
     (tanda baca kini dipakai sebagai batas), tidak bocor lintas kalimat.
 12. Penanganan KONTRAS (tapi/namun/tetapi/...): klausa sebelum kontras
     diberi bobot 0,5 dan sesudahnya 1,5. Negasi membalik polaritas penuh
     (faktor -1,0). Catatan: faktor -0,74 (VADER) sempat dipakai, tetapi pada
     InSet melemahkan negasi sehingga 'tidak suka' kalah oleh kata bobot
     ambigu; keputusan ini diambil SETELAH memeriksa contoh keluaran, dan
     wajib diungkapkan; sensitivitasnya dilaporkan terhadap gold.
 13. LABEL_MODE: 'baseline' (aturan v2) vs 'klausa' (v3) -> ablasi; keduanya
     dihitung dan dibandingkan; pemenangnya ditentukan oleh GOLD STANDARD
     manual, bukan oleh akurasi classifier (tidak sebanding antarlabel).
 14. Audit kamus domain: output/audit_kata_kamus.csv + file opsional
     data/lexicon/koreksi_domain.csv (kata,bobot,alasan) untuk koreksi
     polaritas kata yang keliru pada konteks ekonomi.
 15. Penanda diagnostik: kolom kontras, campuran, fokus_kreator.
 16. Templat anotasi + kolom target_opini (ekonomi/kreator/lainnya) dan
     evaluasi pada subset target 'ekonomi'.
 17. Analisis sensitivitas parameter terhadap gold (hanya robustness).
 23. [v4] KATA TUGAS DI KAMUS: InSet memberi bobot sentimen pada 226 kata tugas
     (yang -5, dari -3, itu -2, karena -3, saya -3, harus -5, ...). Kini dibuang
     (BUANG_STOPWORD_DARI_KAMUS), kecuali yang benar-benar bersentimen
     (STOPWORD_SENTIMEN_TETAP). Daftar dibuang: kata_tugas_dibuang_dari_kamus.csv.
 19. [v4] ATURAN KATA GANDA: kata yang ada di kamus positif DAN negatif
     (overlap) tidak lagi otomatis dihitung positif. OVERLAP_RULE:
     'abaikan' (default, a priori) | 'positif' | 'negatif' | 'maks_abs' | 'jumlah'.
     Dampaknya pada distribusi label & kesesuaian gold dilaporkan.
 20. [v4] Audit kamus menampilkan bobot asli (pos/neg), bobot dipakai, penanda
     overlap & stopword; output/overlap_kamus.csv.
 21. [v4] Bootstrap 95% CI untuk selisih F1-macro RFC - NBC (McNemar hanya
     menguji akurasi, tidak menguji F1-macro).
 22. [v4] Ringkasan metadata video (kanal, tanggal unggah) bila
     data/raw/metadata_video.csv tersedia; komentar yang dibuang karena kosong
     setelah praproses diekspor untuk diperiksa.
 18. Split train/test DIBEKUKAN di data/processed/split_ids.csv (tidak bergeser
     saat aturan pelabelan/kamus berubah; sampel anotasi tetap valid).=============================================================================
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import os
import re
import json
import time
import pickle
import shutil
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from scipy.stats import chi2 as chi2_dist

try:
    from scipy.stats import binomtest
except Exception:                       # scipy lama
    binomtest = None

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, cohen_kappa_score
)

warnings.filterwarnings('ignore')

# =============================================================================
# PARAMETER EKSPERIMEN
# =============================================================================
RANDOM_STATE = 42
TEST_SIZE    = 0.2
N_FOLDS      = 5
N_BINS       = 10        # jumlah bin ECE
ANOTASI_N    = 400       # ukuran sampel anotasi manual (300-500)
TFIDF_PARAMS = dict(max_features=5000, ngram_range=(1, 2), min_df=2)

# --- parameter pelabelan (ditetapkan a priori, bukan di-tuning pada data) -----
LABEL_MODE    = 'klausa'   # 'baseline' = aturan v2 | 'klausa' = aturan v3
NEGASI_FAKTOR = -1.0       # pembalikan penuh. (-0.74 pernah dicoba; lihat sensitivitas_parameter_gold.csv)
KONTRAS_PRE   = 0.5        # bobot klausa SEBELUM kata kontras (tapi/namun/...)
KONTRAS_POST  = 1.5        # bobot klausa SESUDAH kata kontras
KONTRAS = {'tapi', 'tetapi', 'namun', 'melainkan', 'padahal', 'sayangnya'}
POLA_FOKUS_KREATOR = (r'terima ?kasih|makasih|thanks?|thank you|\bvideo\b|channel|'
                      r'konten|subscriber|kreator|penjelasan|edukasi')
OVERLAP_RULE = 'maks_abs'  # kata di kamus positif DAN negatif: 'maks_abs' | 'jumlah' | 'abaikan' | 'positif' | 'negatif'
BUANG_STOPWORD_DARI_KAMUS = True   # buang kata tugas (stopword Sastrawi) dari kamus sentimen
# stopword yang tetap dipertahankan karena memang bermuatan sentimen (tinjau & sesuaikan)
STOPWORD_SENTIMEN_TETAP = {
    'masalah', 'persoalan', 'mempersoalkan', 'penting', 'enak', 'keterlaluan', 'percuma',
    'mampu', 'pantas', 'betul', 'baiklah', 'seenaknya', 'kelamaan', 'sepihak', 'memihak',
    'meyakinkan', 'siap'
}
# =============================================================================
# KONFIGURASI DIREKTORI
# =============================================================================
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
DIR_RAW       = os.path.join(BASE_DIR, 'data', 'raw')
DIR_PROCESSED = os.path.join(BASE_DIR, 'data', 'processed')
DIR_LEXICON   = os.path.join(BASE_DIR, 'data', 'lexicon')
DIR_ANOTASI   = os.path.join(BASE_DIR, 'data', 'annotation')
DIR_OUTPUT    = os.path.join(BASE_DIR, 'output')

for d in (DIR_PROCESSED, DIR_ANOTASI, DIR_OUTPUT):
    os.makedirs(d, exist_ok=True)

RAW_CSV      = os.path.join(DIR_RAW,       'komentar_raw.csv')
CLEAN_CSV    = os.path.join(DIR_PROCESSED, 'komentar_clean.csv')
POS_TXT      = os.path.join(DIR_LEXICON,   'inset_positif.txt')
NEG_TXT      = os.path.join(DIR_LEXICON,   'inset_negatif.txt')
ANOTASI_BLIND = os.path.join(DIR_ANOTASI,  'anotasi_sampel_blind.csv')
ANOTASI_DONE  = os.path.join(DIR_ANOTASI,  'anotasi_manual.csv')
KOREKSI_CSV   = os.path.join(DIR_LEXICON,   'koreksi_domain.csv')

RINGKASAN = {}   # dikumpulkan lalu ditulis ke output/hasil_ringkasan.json


def to_py(o):
    """Konversi tipe numpy agar bisa diserialisasi ke JSON."""
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


# =============================================================================
# DEFINISI KATA NEGASI, WHITELIST SENTIMEN & NORMALISASI SLANG
# =============================================================================
# Catatan: varian (gak, ga, tak, tdk, nggak, bkn) dinormalisasi oleh SLANG_DICT
# menjadi 'tidak'/'bukan' SEBELUM pelabelan, sehingga negasi efektif hanya:
# tidak, bukan, belum, jangan, tanpa, kurang, tiada.
KATA_NEGASI = {
    'tidak', 'bukan', 'belum', 'jangan', 'tanpa', 'kurang',
    'tiada', 'tak', 'gak', 'ga', 'nggak', 'bkn', 'tdk'
}

KATA_SENTIMEN_WHITELIST = {
    'baik', 'bagus', 'tepat', 'yakin', 'benar', 'jelas',
    'cukup', 'hebat', 'mantap', 'keren'
}

SLANG_DICT = {
    # Negasi
    'gak': 'tidak', 'ga': 'tidak', 'nggak': 'tidak', 'ngga': 'tidak',
    'kagak': 'tidak', 'tdk': 'tidak', 'tak': 'tidak', 'bkn': 'bukan',
    'blm': 'belum', 'g': 'tidak', 'kaga': 'tidak',
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
    'tp': 'tapi', 'klo': 'kalau', 'kalo': 'kalau', 'dr': 'dari', 'yg': 'yang',
    'dg': 'dengan', 'dgn': 'dengan', 'krn': 'karena', 'utk': 'untuk', 'sy': 'saya',
    'gw': 'saya', 'gue': 'saya', 'lu': 'kamu', 'loe': 'kamu', 'sdh': 'sudah',
    'udh': 'sudah', 'udah': 'sudah', 'aja': 'saja'
}
# CATATAN: kunci 'matur nuwun' dihapus karena tidak pernah cocok (token tunggal),
#          dan KATA_PENDEK_POSITIF dihapus karena tak terjangkau (sudah dipetakan
#          SLANG_DICT: top->hebat, jos->bagus, oke/ok/sip->baik).


# =============================================================================
# 1. INISIALISASI SASTRAWI & KAMUS LEKSIKON
# =============================================================================
print("=" * 70)
print("1. MEMUAT SASTRAWI & KAMUS INSET LEXICON")
print("=" * 70)

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

stemmer = StemmerFactory().create_stemmer()
raw_stopwords = set(StopWordRemoverFactory().get_stop_words())
# Whitelist: negasi & kata sentimen penting tidak boleh terhapus
stop_words = raw_stopwords - KATA_NEGASI - KATA_SENTIMEN_WHITELIST

stem_cache = {}


def fast_stem(word: str) -> str:
    """Stemming dengan cache dalam memori."""
    if word in KATA_NEGASI:
        return word
    if word not in stem_cache:
        stem_cache[word] = stemmer.stem(word)
    return stem_cache[word]


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
            w = parts[0].strip().lower()
            try:
                d[w] = float(parts[1])
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
        elif aturan == 'maks_abs':      # entri dengan |bobot| terbesar menang; seri -> abaikan
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


pos_dict = load_lexicon_dict(POS_TXT)
neg_dict = load_lexicon_dict(NEG_TXT)
overlap = set(pos_dict) & set(neg_dict)
pendek = sum(1 for w in list(pos_dict) + list(neg_dict) if len(w) < 4)

print(f"Kamus Positif : {len(pos_dict):,} kata")
print(f"Kamus Negatif : {len(neg_dict):,} kata")
print(f"Kata ada di KEDUA kamus (positif diprioritaskan): {len(overlap):,}")
print(f"Entri kamus dengan panjang < 4 huruf (tidak lolos filter fitur): {pendek:,}")
RINGKASAN['leksikon'] = {
    'positif': len(pos_dict), 'negatif': len(neg_dict),
    'overlap': len(overlap), 'entri_kurang_4_huruf': pendek
}

# --- koreksi polaritas domain (opsional; hasil audit manual) -------------------
POS_RAW, NEG_RAW = dict(pos_dict), dict(neg_dict)   # kamus asli (sebelum koreksi)
n_stop_buang = 0
KATA_TUGAS_DIBUANG = set()
if BUANG_STOPWORD_DARI_KAMUS:
    kata_buang = (stop_words - STOPWORD_SENTIMEN_TETAP) & (set(pos_dict) | set(neg_dict))
    for w in kata_buang:
        pos_dict.pop(w, None)
        neg_dict.pop(w, None)
    n_stop_buang = len(kata_buang)
    KATA_TUGAS_DIBUANG = set(kata_buang)
    print(f"Kata tugas dibuang dari kamus : {n_stop_buang:,} (stopword Sastrawi, kecuali "
          f"{len(STOPWORD_SENTIMEN_TETAP)} yang dipertahankan)")
    pd.DataFrame({'kata': sorted(kata_buang),
                  'bobot_pos_asli': [POS_RAW.get(w) for w in sorted(kata_buang)],
                  'bobot_neg_asli': [NEG_RAW.get(w) for w in sorted(kata_buang)]}
                 ).to_csv(os.path.join(DIR_OUTPUT, 'kata_tugas_dibuang_dari_kamus.csv'), index=False)
RINGKASAN['leksikon']['kata_tugas_dibuang'] = n_stop_buang
n_koreksi = 0
if os.path.exists(KOREKSI_CSV):
    kor = pd.read_csv(KOREKSI_CSV)
    for _, r in kor.iterrows():
        w = str(r['kata']).strip().lower()
        b = float(r['bobot'])
        pos_dict.pop(w, None)
        neg_dict.pop(w, None)
        if b > 0:
            pos_dict[w] = b
        elif b < 0:
            neg_dict[w] = b
        n_koreksi += 1
    print(f"Koreksi domain diterapkan     : {n_koreksi} kata ({KOREKSI_CSV})")
else:
    print("Koreksi domain                : tidak ada (data/lexicon/koreksi_domain.csv)")
RINGKASAN['koreksi_domain'] = n_koreksi

POS_K, NEG_K = dict(pos_dict), dict(neg_dict)          # setelah koreksi, sebelum aturan overlap
pos_dict, neg_dict = terapkan_aturan_overlap(POS_K, NEG_K, OVERLAP_RULE)
print(f"Aturan kata ganda (overlap)   : '{OVERLAP_RULE}' -> kamus aktif "
      f"+{len(pos_dict):,} / -{len(neg_dict):,}")
RINGKASAN['leksikon']['aturan_overlap'] = OVERLAP_RULE
RINGKASAN['leksikon']['aktif_positif'] = len(pos_dict)
RINGKASAN['leksikon']['aktif_negatif'] = len(neg_dict)

# =============================================================================
# 2. FUNGSI PRAPROSES & PELABELAN
# =============================================================================
def tokenize_pra_stem(text: str) -> list:
    """
    Praproses TAHAP 1 (sebelum pelabelan):
    cleaning -> case folding -> reduksi huruf berulang -> normalisasi slang -> tokenisasi
    """
    if not isinstance(text, str):
        return []
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+|@\w+|#\w+|\d+|[^\w\s]', ' ', text)
    text = re.sub(r'(.)\1{2,}', r'\1', text)      # baguuus -> bagus  (BUG DIPERBAIKI)
    text = re.sub(r'\s+', ' ', text).strip()

    flat = []
    for t in text.split():
        flat.extend(SLANG_DICT.get(t, t).split())
    return flat


def label_sentimen_lexicon(tokens: list, window: int = 3):
    """
    Pelabelan InSet pada token pra-stemming dengan penanganan negasi (jendela 3 token).
    Mengembalikan (label, skor, jumlah_kata_cocok).
    """
    score, n_cocok = 0.0, 0
    neg_active_until = -1
    for i, t in enumerate(tokens):
        if t in KATA_NEGASI:
            neg_active_until = i + window
            continue
        negated = (i <= neg_active_until)
        if t in pos_dict:
            n_cocok += 1
            w = pos_dict[t]
            score += -w if negated else w
        elif t in neg_dict:
            n_cocok += 1
            w = abs(neg_dict[t])
            score += w if negated else -w
    if score > 0:
        return 'positif', score, n_cocok
    if score < 0:
        return 'negatif', score, n_cocok
    return 'netral', score, n_cocok


def clean_post_stem(tokens: list) -> str:
    """
    Praproses TAHAP 2 (setelah pelabelan, untuk fitur TF-IDF):
    stopword selektif -> filter panjang >= 4 (negasi dikecualikan) -> stemming
    """
    kept = [t for t in tokens
            if (len(t) >= 4 or t in KATA_NEGASI) and t not in stop_words]
    return ' '.join(fast_stem(t) for t in kept)


# =============================================================================
# 2b. PELABELAN SADAR KLAUSA (v3)
# =============================================================================
def parse_klausa(text: str) -> list:
    """
    Memecah komentar menjadi KALIMAT -> SEGMEN (klausa) -> token.
    Batas kalimat : . ! ? ; baris baru
    Batas klausa  : , : ( ) " / dan kata kontras (tapi/namun/tetapi/...)
    Hasil: list of (segmen_list, indeks_segmen_pertama_setelah_kontras | None)
    """
    if not isinstance(text, str):
        return []
    t = text.lower()
    t = re.sub(r'http\S+|www\.\S+|@\w+|#\w+|\d+', ' ', t)
    t = re.sub(r'(.)\1{2,}', r'\1', t)
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


def label_klausa(kalimat_list: list, neg_faktor=None, pre=None, post=None, window: int = 3,
                 pdict=None, ndict=None):
    """
    Pelabelan InSet sadar klausa. Mengembalikan (label, skor, ada_kontras, campuran).
    - negasi berlaku hanya dalam satu segmen (jendela 3 token)
    - kata bersentimen yang dinegasi dikali neg_faktor (<0)
    - klausa sebelum/sesudah kata kontras dikali pre/post
    - campuran = ada segmen positif DAN segmen negatif dalam satu komentar
    """
    neg_faktor = NEGASI_FAKTOR if neg_faktor is None else neg_faktor
    P = pos_dict if pdict is None else pdict
    N = neg_dict if ndict is None else ndict
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
                if t in P:
                    w = P[t]
                elif t in N:
                    w = -abs(N[t])
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


def jelaskan_komentar(teks: str):
    """Diagnosis: tampilkan kontribusi tiap kata pada skor (aturan v3)."""
    print("\nKOMENTAR:", teks[:140].replace("\n", " "))
    print(f"  [overlap='{OVERLAP_RULE}' | faktor negasi={NEGASI_FAKTOR} | "
          f"kontras={KONTRAS_PRE}/{KONTRAS_POST}]")
    skor_total = 0.0
    for segs, kidx in parse_klausa(teks):
        for j, seg in enumerate(segs):
            bobot = 1.0 if kidx is None else (KONTRAS_PRE if j < kidx else KONTRAS_POST)
            print(f"  segmen {j} (bobot {bobot}): {seg}")
            neg_until = -1
            for i, t in enumerate(seg):
                if t in KATA_NEGASI:
                    neg_until = i + 3
                    print(f"     [{i}] {t:<12} NEGASI")
                    continue
                if t in pos_dict:
                    w = pos_dict[t]
                elif t in neg_dict:
                    w = -abs(neg_dict[t])
                else:
                    if t in POS_RAW or t in NEG_RAW:
                        alasan_ = ("kata tugas dibuang dari kamus" if t in KATA_TUGAS_DIBUANG
                                   else f"aturan overlap='{OVERLAP_RULE}'")
                        print(f"     [{i}] {t:<12} DIABAIKAN (kamus asli: pos={POS_RAW.get(t)}, "
                              f"neg={NEG_RAW.get(t)}; {alasan_})")
                    continue
                v = (NEGASI_FAKTOR * w if i <= neg_until else w) * bobot
                skor_total += v
                print(f"     [{i}] {t:<12} bobot={w:+.0f}{' (dinegasi)' if i <= neg_until else ''} -> {v:+.2f}"
                      f"   [kamus asli: pos={POS_RAW.get(t)}, neg={NEG_RAW.get(t)}]")
    print(f"  TOTAL = {skor_total:+.2f}")


# Mode diagnosis cepat:  python main_v4.py --jelaskan "teks komentar" ["teks lain" ...]
if '--jelaskan' in sys.argv:
    for teks_ in sys.argv[sys.argv.index('--jelaskan') + 1:]:
        jelaskan_komentar(teks_)
    sys.exit(0)


# =============================================================================
# 3. MEMUAT DATA, DEDUPLIKASI, PRAPROSES & PELABELAN
# =============================================================================
print("\n" + "=" * 70)
print("3. MEMPROSES DATA RAW DARI YOUTUBE")
print("=" * 70)

if not os.path.exists(RAW_CSV):
    raise FileNotFoundError(f"File raw data tidak ditemukan di: {RAW_CSV}")

df = pd.read_csv(RAW_CSV)
n_raw = len(df)
print(f"Data raw dimuat                : {n_raw:,} baris")

# --- ringkasan sumber (jumlah video/kanal) jika kolomnya tersedia ----------
for kol in ['video_id', 'video', 'judul_video', 'kanal', 'channel', 'nama_kanal']:
    if kol in df.columns:
        print(f"Jumlah {kol:<12} unik        : {df[kol].nunique():,}")
        RINGKASAN.setdefault('sumber', {})[kol] = int(df[kol].nunique())
        df[kol].value_counts().to_csv(os.path.join(DIR_OUTPUT, f'sumber_{kol}.csv'))

# --- metadata video (opsional): kanal & tanggal unggah dari data/raw/metadata_video.csv
META_CSV = os.path.join(DIR_RAW, 'metadata_video.csv')
if os.path.exists(META_CSV) and 'video' in df.columns:
    meta = pd.read_csv(META_CSV)
    kunci = next((k for k in ('video_id', 'judul', 'url')
                  if k in meta.columns and df['video'].isin(meta[k]).mean() > 0.9), None)
    if kunci is None:
        print("[PERINGATAN] metadata_video.csv tidak cocok dengan kolom 'video' pada data raw.")
    else:
        meta = meta.drop_duplicates(kunci)
        jml = df.groupby('video').size().rename('komentar_terkumpul')
        rv = meta.merge(jml, left_on=kunci, right_index=True, how='left')
        rv.to_csv(os.path.join(DIR_OUTPUT, 'ringkasan_video.csv'), index=False)
        info = {'jumlah_video': int(len(rv))}
        if 'kanal' in rv.columns:
            info['jumlah_kanal'] = int(rv['kanal'].nunique())
        if 'tanggal_unggah' in rv.columns:
            tg = pd.to_datetime(rv['tanggal_unggah'], errors='coerce')
            info['tanggal_unggah_terlama'] = str(tg.min().date())
            info['tanggal_unggah_terbaru'] = str(tg.max().date())
        print(f"Metadata video                 : {info}")
        RINGKASAN['video'] = info
else:
    print("Metadata video                 : tidak ada (data/raw/metadata_video.csv)")

df = df.dropna(subset=['komentar']).copy()
df['_kunci'] = (df['komentar'].astype(str).str.lower()
                .str.replace(r'\s+', ' ', regex=True).str.strip())
df = df[df['_kunci'] != '']
n_nonkosong = len(df)
df = df.drop_duplicates(subset='_kunci').drop(columns='_kunci').reset_index(drop=True)
n_dedup = len(df)
print(f"Setelah buang kosong           : {n_nonkosong:,}")
print(f"Setelah deduplikasi            : {n_dedup:,}  (duplikat dibuang: {n_nonkosong - n_dedup:,})")

df.insert(0, 'id_komentar', np.arange(len(df)))

t0 = time.time()
tokens_list = df['komentar'].apply(tokenize_pra_stem)
hasil_label = [label_sentimen_lexicon(tk) for tk in tokens_list]          # aturan v2
df['label_baseline'] = [h[0] for h in hasil_label]
df['skor_baseline']  = [h[1] for h in hasil_label]
df['n_cocok']        = [h[2] for h in hasil_label]

parsed = df['komentar'].apply(parse_klausa)                                  # aturan v3
PARSED = dict(zip(df['id_komentar'], parsed))
TOKENS = dict(zip(df['id_komentar'], tokens_list))
hasil_k = [label_klausa(p) for p in parsed]
df['label_klausa'] = [h[0] for h in hasil_k]
df['skor_klausa']  = [h[1] for h in hasil_k]
df['kontras']      = [h[2] for h in hasil_k]
df['campuran']     = [h[3] for h in hasil_k]
df['fokus_kreator'] = df['komentar'].astype(str).str.lower().str.contains(
    POLA_FOKUS_KREATOR, regex=True)

# label yang dipakai untuk melatih model
if LABEL_MODE == 'baseline':
    df['label'], df['skor'] = df['label_baseline'], df['skor_baseline']
else:
    df['label'], df['skor'] = df['label_klausa'], df['skor_klausa']
print(f"LABEL_MODE aktif: {LABEL_MODE}")
df['komentar_bersih'] = tokens_list.apply(clean_post_stem)

df[df['komentar_bersih'].str.strip() == ''][['id_komentar', 'komentar']].to_csv(
    os.path.join(DIR_OUTPUT, 'komentar_dibuang_kosong.csv'), index=False)
df_clean = df[df['komentar_bersih'].str.strip() != ''].copy().reset_index(drop=True)
print(f"Dibuang karena kosong setelah praproses: {(df['komentar_bersih'].str.strip() == '').sum():,} "
      f"(lihat output/komentar_dibuang_kosong.csv)")
print(f"Praproses & pelabelan selesai ({time.time() - t0:.1f} dtk). Total data bersih: {len(df_clean):,}")

df_clean.to_csv(CLEAN_CSV, index=False, encoding='utf-8')
extra_processed = os.path.join(BASE_DIR, 'data', 'data', 'processed', 'komentar_clean.csv')
if os.path.exists(os.path.dirname(extra_processed)):
    df_clean.to_csv(extra_processed, index=False, encoding='utf-8')

dist = df_clean['label'].value_counts()
dist_pct = df_clean['label'].value_counts(normalize=True) * 100
print("\nDistribusi sentimen:")
for k in dist.index:
    print(f"  {k:<8}: {dist[k]:>6,}  ({dist_pct[k]:.2f}%)")

# --- cakupan leksikon --------------------------------------------------------
cakupan = (df_clean['n_cocok'] > 0).mean() * 100
net = df_clean[df_clean['label'] == 'netral']
net_tanpa_kata = (net['n_cocok'] == 0).sum()
net_saling_tiadakan = (net['n_cocok'] > 0).sum()
print(f"\nCakupan leksikon (komentar >=1 kata cocok): {cakupan:.2f}%")
print(f"Netral tanpa kata cocok       : {net_tanpa_kata:,}")
print(f"Netral karena skor saling meniadakan: {net_saling_tiadakan:,}")

RINGKASAN['data'] = {
    'raw': n_raw, 'setelah_buang_kosong': n_nonkosong, 'setelah_dedup': n_dedup,
    'bersih': len(df_clean),
    'distribusi': {k: int(v) for k, v in dist.items()},
    'distribusi_persen': {k: round(float(v), 2) for k, v in dist_pct.items()},
    'cakupan_leksikon_persen': round(float(cakupan), 2),
    'netral_tanpa_kata_cocok': int(net_tanpa_kata),
    'netral_skor_saling_meniadakan': int(net_saling_tiadakan),
}

# --- ablasi: aturan v2 (baseline) vs v3 (klausa) --------------------------------
print("\n" + "-" * 70)
print("ABLASI PELABELAN: baseline (v2) vs klausa (v3)")
print("-" * 70)
ct = pd.crosstab(df_clean['label_baseline'], df_clean['label_klausa'],
                 rownames=['baseline'], colnames=['klausa'])
print(ct)
ct.to_csv(os.path.join(DIR_OUTPUT, 'ablasi_label_crosstab.csv'))
berubah = df_clean['label_baseline'] != df_clean['label_klausa']
kappa_mode = cohen_kappa_score(df_clean['label_baseline'], df_clean['label_klausa'])
print(f"Label berubah: {int(berubah.sum()):,} komentar ({berubah.mean()*100:.2f}%) | "
      f"Kappa antar-aturan = {kappa_mode:.4f}")
print(f"Komentar dengan kata kontras  : {df_clean['kontras'].mean()*100:.2f}%")
print(f"Komentar bersentimen campuran : {df_clean['campuran'].mean()*100:.2f}%")
contoh = df_clean[berubah]
if len(contoh) > 0:
    contoh.sample(min(150, len(contoh)), random_state=RANDOM_STATE)[
        ['id_komentar', 'komentar', 'label_baseline', 'label_klausa',
         'skor_baseline', 'skor_klausa', 'kontras', 'campuran']
    ].to_csv(os.path.join(DIR_OUTPUT, 'contoh_perbedaan_label.csv'), index=False)

# --- fokus kreator (diagnostik target opini; tidak memengaruhi label) -----------
fk = df_clean.groupby('label')['fokus_kreator'].mean() * 100
print("Persentase komentar berfokus apresiasi video/kreator (heuristik) per label:")
for k, v in fk.items():
    print(f"  {k:<8}: {v:.2f}%")

# --- audit kata kamus: bobot yang BENAR-BENAR dipakai + bobot asli ----------------
kontrib = {}
for i_d in df_clean['id_komentar']:
    for t in TOKENS[i_d]:
        if t in pos_dict or t in neg_dict:
            kontrib[t] = kontrib.get(t, 0) + 1
rows_a = []
for w, n in kontrib.items():
    wd = pos_dict[w] if w in pos_dict else -abs(neg_dict[w])
    rows_a.append({'kata': w, 'frekuensi': n, 'bobot_dipakai': wd,
                   'bobot_pos_asli': POS_RAW.get(w), 'bobot_neg_asli': NEG_RAW.get(w),
                   'overlap': (w in POS_RAW and w in NEG_RAW), 'stopword': w in stop_words,
                   'pengaruh_total': n * abs(wd)})
audit = pd.DataFrame(rows_a).sort_values('pengaruh_total', ascending=False)
audit.head(200).to_csv(os.path.join(DIR_OUTPUT, 'audit_kata_kamus.csv'), index=False)
pangsa_stop = audit.head(200).loc[audit.head(200)['stopword'], 'pengaruh_total'].sum() \
    / max(audit.head(200)['pengaruh_total'].sum(), 1e-9) * 100
print(f"\nPangsa pengaruh stopword pada 200 kata teratas: {pangsa_stop:.1f}%")
print("25 kata kamus paling berpengaruh (kata, frekuensi, bobot dipakai, overlap, stopword):")
for _, r in audit.head(25).iterrows():
    print(f"  {r['kata']:<14}{int(r['frekuensi']):>6}   bobot={r['bobot_dipakai']:+.0f}   "
          f"overlap={'Y' if r['overlap'] else '-'}  stop={'Y' if r['stopword'] else '-'}")

# --- kata yang ada di KEDUA kamus (overlap), diurutkan menurut frekuensi korpus ---
frek_all = {}
for i_d in df_clean['id_komentar']:
    for t in TOKENS[i_d]:
        frek_all[t] = frek_all.get(t, 0) + 1
ov = sorted(set(POS_RAW) & set(NEG_RAW))
df_ov = pd.DataFrame({'kata': ov, 'bobot_pos': [POS_RAW[w] for w in ov],
                      'bobot_neg': [NEG_RAW[w] for w in ov],
                      'frekuensi': [frek_all.get(w, 0) for w in ov]}
                     ).sort_values('frekuensi', ascending=False)
df_ov.to_csv(os.path.join(DIR_OUTPUT, 'overlap_kamus.csv'), index=False)
ada_ov = df_clean['id_komentar'].map(
    lambda i_: any((t in POS_RAW and t in NEG_RAW) for t in TOKENS[i_])).mean() * 100
print(f"\nKata di kedua kamus: {len(ov):,} | komentar memuat >=1 kata overlap: {ada_ov:.1f}%")
for _, r in df_ov.head(15).iterrows():
    print(f"  {r['kata']:<14}{int(r['frekuensi']):>6}   pos={r['bobot_pos']:+.0f}  neg={r['bobot_neg']:+.0f}")

# --- sensitivitas distribusi label terhadap aturan overlap ------------------------
baris_ol = []
lab_aktif = df_clean['label_klausa']
for rule in ('positif', 'negatif', 'maks_abs', 'jumlah', 'abaikan'):
    P_, N_ = terapkan_aturan_overlap(POS_K, NEG_K, rule)
    labs = pd.Series([label_klausa(PARSED[i_], pdict=P_, ndict=N_)[0]
                      for i_ in df_clean['id_komentar']])
    vc = labs.value_counts(normalize=True) * 100
    baris_ol.append({'aturan_overlap': rule,
                     'persen_positif': vc.get('positif', 0.0),
                     'persen_negatif': vc.get('negatif', 0.0),
                     'persen_netral': vc.get('netral', 0.0),
                     'kappa_vs_aturan_aktif': cohen_kappa_score(lab_aktif.values, labs.values)})
df_ol = pd.DataFrame(baris_ol)
df_ol.to_csv(os.path.join(DIR_OUTPUT, 'sensitivitas_overlap_label.csv'), index=False,
             float_format='%.4f')
print("\nDistribusi label menurut aturan kata ganda (%):")
print(df_ol.round(2).to_string(index=False))
RINGKASAN['sensitivitas_overlap_label'] = baris_ol

RINGKASAN['ablasi_label'] = {
    'mode_aktif': LABEL_MODE, 'persen_berubah': round(float(berubah.mean() * 100), 2),
    'kappa_antar_aturan': float(kappa_mode),
    'persen_kontras': round(float(df_clean['kontras'].mean() * 100), 2),
    'persen_campuran': round(float(df_clean['campuran'].mean() * 100), 2),
    'persen_fokus_kreator_per_label': {k: round(float(v), 2) for k, v in fk.items()},
}


# =============================================================================
# 4. SPLIT DATA (80:20, STRATIFIED) -> BARU TF-IDF (FIT HANYA DI DATA LATIH)
# =============================================================================
print("\n" + "=" * 70)
print("4. SPLIT DATA 80:20 LALU TF-IDF (tanpa kebocoran)")
print("=" * 70)

# Split DIBEKUKAN: dibuat sekali lalu disimpan, agar perubahan aturan pelabelan /
# kamus tidak menggeser data uji (yang juga dipakai sebagai sumber sampel anotasi).
# Hapus data/processed/split_ids.csv hanya bila data mentah berubah.
SPLIT_CSV = os.path.join(DIR_PROCESSED, 'split_ids.csv')
sp = pd.read_csv(SPLIT_CSV) if os.path.exists(SPLIT_CSV) else None
if sp is not None and set(sp['id_komentar']) == set(df_clean['id_komentar']):
    peta = dict(zip(sp['id_komentar'], sp['subset']))
    subset = df_clean['id_komentar'].map(peta)
    idx_train = df_clean.index[subset == 'latih']
    idx_test  = df_clean.index[subset == 'uji']
    print(f"Split dipakai ulang dari {SPLIT_CSV}")
else:
    if sp is not None:
        print("[PERINGATAN] Data berubah -> split dibuat ulang. Sampel anotasi lama "
              "TIDAK valid lagi; buat ulang dan anotasi ulang.")
    idx_train, idx_test = train_test_split(
        df_clean.index, test_size=TEST_SIZE,
        random_state=RANDOM_STATE, stratify=df_clean['label']
    )
    pd.DataFrame({'id_komentar': df_clean['id_komentar'],
                  'subset': np.where(df_clean.index.isin(idx_test), 'uji', 'latih')}
                 ).to_csv(SPLIT_CSV, index=False)
    print(f"Split baru dibuat & disimpan: {SPLIT_CSV}")
X_tr_txt = df_clean.loc[idx_train, 'komentar_bersih']
X_te_txt = df_clean.loc[idx_test,  'komentar_bersih']
y_train  = df_clean.loc[idx_train, 'label']
y_test   = df_clean.loc[idx_test,  'label']

tfidf = TfidfVectorizer(**TFIDF_PARAMS)
X_train = tfidf.fit_transform(X_tr_txt)      # fit HANYA data latih
X_test  = tfidf.transform(X_te_txt)          # data uji hanya di-transform

print(f"Dimensi fitur TF-IDF : {X_train.shape[1]:,}")
print(f"Data latih           : {X_train.shape[0]:,}")
print(f"Data uji             : {X_test.shape[0]:,}")
RINGKASAN['split'] = {'latih': int(X_train.shape[0]), 'uji': int(X_test.shape[0]),
                      'fitur': int(X_train.shape[1])}


# =============================================================================
# 5. PELATIHAN MODEL
# =============================================================================
print("\n" + "=" * 70)
print("5. MELATIH MODEL (NBC & RFC)")
print("=" * 70)

nbc = MultinomialNB()                                   # alpha=1.0 (Laplace)
nbc.fit(X_train, y_train)

rfc = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE,
                             n_jobs=-1, class_weight='balanced')
rfc.fit(X_train, y_train)

# Analisis sensitivitas: NB dengan prior seragam (pembanding adil thd class_weight RF)
nbc_seragam = MultinomialNB(fit_prior=False)
nbc_seragam.fit(X_train, y_train)

labels_order = [str(c) for c in nbc.classes_]       # urutan alfabet: negatif, netral, positif
print(f"Urutan kelas model: {labels_order}")


# =============================================================================
# 6. EVALUASI MODEL (MACRO UTAMA, WEIGHTED PEMBANDING)
# =============================================================================
print("\n" + "=" * 70)
print("6. EVALUASI MODEL")
print("=" * 70)


def hitung_metrik(y_true, y_pred) -> dict:
    return {
        'Akurasi':            accuracy_score(y_true, y_pred),
        'Precision_macro':    precision_score(y_true, y_pred, average='macro',    zero_division=0),
        'Recall_macro':       recall_score(y_true, y_pred, average='macro',       zero_division=0),
        'F1_macro':           f1_score(y_true, y_pred, average='macro',           zero_division=0),
        'Precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'Recall_weighted':    recall_score(y_true, y_pred, average='weighted',    zero_division=0),
        'F1_weighted':        f1_score(y_true, y_pred, average='weighted',        zero_division=0),
    }


model_eval = [('Naive Bayes', nbc), ('Random Forest', rfc),
              ('Naive Bayes (prior seragam)', nbc_seragam)]
hasil_evaluasi = {}
for nama, model in model_eval:
    y_pred = model.predict(X_test)
    m = hitung_metrik(y_test, y_pred)
    m['y_pred'] = y_pred
    hasil_evaluasi[nama] = m
    print(f"\n--- {nama} ---")
    for k, v in m.items():
        if k != 'y_pred':
            print(f"  {k:<19}: {v * 100:6.2f}%")
    print(classification_report(y_test, y_pred, labels=labels_order, zero_division=0))

tabel_metrik = pd.DataFrame({n: {k: v for k, v in m.items() if k != 'y_pred'}
                             for n, m in hasil_evaluasi.items()}).T
tabel_metrik.to_csv(os.path.join(DIR_OUTPUT, 'metrik_model.csv'), float_format='%.6f')

# laporan per kelas
rows = []
for nama, model in model_eval:
    rep = classification_report(y_test, hasil_evaluasi[nama]['y_pred'],
                                labels=labels_order, output_dict=True, zero_division=0)
    for k in labels_order:
        rows.append({'model': nama, 'kelas': k, **{a: rep[k][a] for a in
                     ('precision', 'recall', 'f1-score', 'support')}})
pd.DataFrame(rows).to_csv(os.path.join(DIR_OUTPUT, 'metrik_per_kelas.csv'),
                          index=False, float_format='%.6f')

RINGKASAN['metrik'] = {n: {k: float(v) for k, v in m.items() if k != 'y_pred'}
                       for n, m in hasil_evaluasi.items()}
RINGKASAN['confusion_matrix'] = {
    'urutan_kelas': labels_order,
    **{n: confusion_matrix(y_test, hasil_evaluasi[n]['y_pred'], labels=labels_order).tolist()
       for n, _ in model_eval}
}


# =============================================================================
# 7. VALIDASI SILANG 5-FOLD (TF-IDF DI-FIT ULANG DI TIAP FOLD)
# =============================================================================
print("\n" + "=" * 70)
print(f"7. {N_FOLDS}-FOLD STRATIFIED CROSS-VALIDATION (data latih)")
print("=" * 70)

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
rows_cv = []
for nama, model in [('Naive Bayes', MultinomialNB()),
                    ('Random Forest', RandomForestClassifier(
                        n_estimators=100, random_state=RANDOM_STATE,
                        n_jobs=-1, class_weight='balanced'))]:
    pipe = Pipeline([('tfidf', TfidfVectorizer(**TFIDF_PARAMS)), ('clf', model)])
    cv = cross_validate(pipe, X_tr_txt, y_train, cv=skf,
                        scoring=['accuracy', 'f1_macro'], n_jobs=1)
    rows_cv.append({
        'model': nama,
        'akurasi_mean': cv['test_accuracy'].mean(), 'akurasi_std': cv['test_accuracy'].std(),
        'f1_macro_mean': cv['test_f1_macro'].mean(), 'f1_macro_std': cv['test_f1_macro'].std(),
    })
    print(f"{nama:<14} akurasi {cv['test_accuracy'].mean()*100:.2f}% ± {cv['test_accuracy'].std()*100:.2f} | "
          f"F1 macro {cv['test_f1_macro'].mean()*100:.2f}% ± {cv['test_f1_macro'].std()*100:.2f}")
df_cv = pd.DataFrame(rows_cv)
df_cv.to_csv(os.path.join(DIR_OUTPUT, 'cv_hasil.csv'), index=False, float_format='%.6f')
RINGKASAN['cv'] = rows_cv


# =============================================================================
# 8. UJI McNEMAR (NBC vs RFC pada data uji yang sama)
# =============================================================================
print("\n" + "=" * 70)
print("8. UJI McNEMAR")
print("=" * 70)

benar_nb = (hasil_evaluasi['Naive Bayes']['y_pred']   == y_test.values)
benar_rf = (hasil_evaluasi['Random Forest']['y_pred'] == y_test.values)
b = int(np.sum(benar_nb & ~benar_rf))      # hanya NB benar
c = int(np.sum(~benar_nb & benar_rf))      # hanya RF benar
if b + c > 0:
    chi2_stat = (abs(b - c) - 1) ** 2 / (b + c)           # dengan koreksi kontinuitas
    p_chi2 = float(chi2_dist.sf(chi2_stat, 1))
    p_exact = float(binomtest(min(b, c), b + c, 0.5).pvalue) if binomtest else None
else:
    chi2_stat, p_chi2, p_exact = 0.0, 1.0, 1.0
p_pakai = p_exact if (p_exact is not None and (b + c) < 25) else p_chi2
print(f"Hanya NB benar (b): {b:,} | Hanya RF benar (c): {c:,}")
print(f"chi2 = {chi2_stat:.4f} | p (chi2) = {p_chi2:.4f} | p (exact) = {p_exact}")
print("Kesimpulan (alpha=0.05):",
      "perbedaan SIGNIFIKAN" if p_pakai < 0.05 else "perbedaan TIDAK signifikan")
RINGKASAN['mcnemar'] = {'b_hanya_NB_benar': b, 'c_hanya_RF_benar': c,
                        'chi2': float(chi2_stat), 'p_chi2': p_chi2, 'p_exact': p_exact,
                        'signifikan_alpha_0_05': bool(p_pakai < 0.05)}

# --- bootstrap 95% CI selisih F1-macro (McNemar hanya menguji akurasi) -------------
print("\nBootstrap 95% CI selisih F1-macro (1000 resampel data uji):")
rng_b = np.random.RandomState(RANDOM_STATE)
yt = y_test.values
n_uji = len(yt)
pred_b = {n: hasil_evaluasi[n]['y_pred'] for n in
          ('Naive Bayes', 'Random Forest', 'Naive Bayes (prior seragam)')}
sel_nb, sel_nbu = [], []
for _ in range(1000):
    ib = rng_b.randint(0, n_uji, n_uji)
    f_rf = f1_score(yt[ib], pred_b['Random Forest'][ib], average='macro')
    sel_nb.append(f_rf - f1_score(yt[ib], pred_b['Naive Bayes'][ib], average='macro'))
    sel_nbu.append(f_rf - f1_score(yt[ib], pred_b['Naive Bayes (prior seragam)'][ib], average='macro'))
boot = {}
for nama_b, arr in (('RF_minus_NB', sel_nb), ('RF_minus_NB_prior_seragam', sel_nbu)):
    lo, hi = np.percentile(arr, [2.5, 97.5])
    boot[nama_b] = {'rata2': float(np.mean(arr)), 'ci95_bawah': float(lo), 'ci95_atas': float(hi),
                    'signifikan': bool(lo > 0 or hi < 0)}
    print(f"  {nama_b:<28}: {np.mean(arr)*100:+.2f} poin  CI95 [{lo*100:+.2f}, {hi*100:+.2f}]"
          f"  -> {'signifikan' if boot[nama_b]['signifikan'] else 'TIDAK signifikan'}")
RINGKASAN['bootstrap_f1_macro'] = boot


# =============================================================================
# 9. ANALISIS PROBABILITAS KEYAKINAN & KALIBRASI  (NOVELTY)
# =============================================================================
print("\n" + "=" * 70)
print("9. PROBABILITAS KEYAKINAN & KALIBRASI")
print("=" * 70)


def brier_multiclass(P, y_true, classes) -> float:
    Y = (np.asarray(y_true)[:, None] == np.asarray(classes)[None, :]).astype(float)
    return float(np.mean(np.sum((P - Y) ** 2, axis=1)))


def ece_top_label(conf, benar, n_bins=N_BINS):
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    n, ece, tabel = len(conf), 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi)
        cnt = int(mask.sum())
        if cnt == 0:
            tabel.append({'bin': f'({lo:.1f},{hi:.1f}]', 'n': 0,
                          'conf': np.nan, 'acc': np.nan})
            continue
        acc_b, conf_b = float(benar[mask].mean()), float(conf[mask].mean())
        ece += cnt / n * abs(acc_b - conf_b)
        tabel.append({'bin': f'({lo:.1f},{hi:.1f}]', 'n': cnt, 'conf': conf_b, 'acc': acc_b})
    return float(ece), pd.DataFrame(tabel)


analisis = {}
proba_out = pd.DataFrame({'id_komentar': df_clean.loc[idx_test, 'id_komentar'].values,
                          'label_aktual': y_test.values})
rows_kelas = []
for nama, model in [('Naive Bayes', nbc), ('Random Forest', rfc)]:
    P = model.predict_proba(X_test)
    classes = model.classes_
    pred = classes[P.argmax(axis=1)]
    conf = P.max(axis=1)
    benar = (pred == y_test.values)

    brier = brier_multiclass(P, y_test.values, classes)
    ece, tabel_bin = ece_top_label(conf, benar)
    tabel_bin.to_csv(os.path.join(DIR_OUTPUT, f'kalibrasi_bin_{"nbc" if nama == "Naive Bayes" else "rfc"}.csv'),
                     index=False, float_format='%.6f')

    ringkas = {
        'akurasi': float(benar.mean()),
        'confidence_rata2': float(conf.mean()),
        'confidence_rata2_benar': float(conf[benar].mean()) if benar.any() else None,
        'confidence_rata2_salah': float(conf[~benar].mean()) if (~benar).any() else None,
        'gap_overconfidence': float(conf.mean() - benar.mean()),
        'brier_multiclass': brier,
        'ece_10bin': ece,
    }
    analisis[nama] = {**ringkas, 'tabel_bin': tabel_bin, 'conf': conf, 'benar': benar}
    RINGKASAN.setdefault('keyakinan', {})[nama] = ringkas

    print(f"\n--- {nama} ---")
    for k, v in ringkas.items():
        print(f"  {k:<26}: {v:.4f}" if v is not None else f"  {k:<26}: -")
    for k in classes:                                   # per kelas PREDIKSI
        m = (pred == k)
        if m.any():
            rows_kelas.append({'model': nama, 'kelas_prediksi': k, 'n': int(m.sum()),
                               'confidence_rata2': float(conf[m].mean()),
                               'akurasi_pada_kelas': float(benar[m].mean())})
    tag = 'nbc' if nama == 'Naive Bayes' else 'rfc'
    proba_out[f'pred_{tag}'] = pred
    proba_out[f'conf_{tag}'] = conf
    for j, k in enumerate(classes):
        proba_out[f'P_{k}_{tag}'] = P[:, j]

pd.DataFrame(rows_kelas).to_csv(os.path.join(DIR_OUTPUT, 'keyakinan_per_kelas.csv'),
                                index=False, float_format='%.6f')
proba_out.to_csv(os.path.join(DIR_OUTPUT, 'proba_data_uji.csv'), index=False, float_format='%.6f')
pd.DataFrame([{**{'model': n}, **RINGKASAN['keyakinan'][n]} for n in analisis]).to_csv(
    os.path.join(DIR_OUTPUT, 'keyakinan_ringkas.csv'), index=False, float_format='%.6f')
print("\n[OK] keyakinan_ringkas.csv, keyakinan_per_kelas.csv, proba_data_uji.csv tersimpan.")


# =============================================================================
# 10. ANOTASI MANUAL: EKSPOR SAMPEL, KAPPA & GOLD STANDARD
# =============================================================================
print("\n" + "=" * 70)
print("10. ANOTASI MANUAL (Cohen's Kappa & Gold Standard)")
print("=" * 70)

if not os.path.exists(ANOTASI_BLIND):
    rng = np.random.RandomState(RANDOM_STATE)
    n_s = min(ANOTASI_N, len(idx_test))
    pilih = rng.choice(np.array(idx_test), size=n_s, replace=False)
    blind = df_clean.loc[pilih, ['id_komentar', 'komentar']].copy()
    blind['anotator_1'] = ''
    blind['anotator_2'] = ''
    blind['label_konsensus'] = ''
    blind['target_opini'] = ''   # ekonomi / kreator / lainnya
    blind.to_csv(ANOTASI_BLIND, index=False, encoding='utf-8-sig')
    print(f"[OK] Sampel anotasi (n={n_s}) diekspor: {ANOTASI_BLIND}")
    print("     Isi anotator_1 & anotator_2 (positif/negatif/netral) tanpa melihat label model,")
    print("     lalu isi label_konsensus setelah diskusi, simpan sebagai 'anotasi_manual.csv'.")
else:
    print(f"Sampel anotasi sudah ada (tidak ditimpa): {ANOTASI_BLIND}")

if os.path.exists(ANOTASI_DONE):
    gold = pd.read_csv(ANOTASI_DONE)
    for kol in ('anotator_1', 'anotator_2', 'label_konsensus'):
        gold[kol] = gold[kol].astype(str).str.strip().str.lower()
    gold = gold[gold['label_konsensus'].isin(labels_order)]
    ids_uji = set(df_clean.loc[idx_test, 'id_komentar'])
    if not set(gold['id_komentar']).issubset(ids_uji):
        print("[PERINGATAN] Ada id anotasi yang bukan data uji -> berpotensi bocor. "
              "Regenerate sampel jika data berubah.")
    gold = gold[gold['id_komentar'].isin(ids_uji)]
    if len(gold) > 0:
        kappa_antar = cohen_kappa_score(gold['anotator_1'], gold['anotator_2'])
        pos_uji = pd.Series(np.arange(len(idx_test)),
                            index=df_clean.loc[idx_test, 'id_komentar'].values)
        ix = pos_uji.loc[gold['id_komentar']].values
        lab_leks = df_clean.set_index('id_komentar').loc[gold['id_komentar'], 'label'].values
        gold_y = gold['label_konsensus'].values
        out_gold = {'n': int(len(gold)), 'kappa_antar_anotator': float(kappa_antar),
                    'kappa_leksikon_vs_konsensus': float(cohen_kappa_score(lab_leks, gold_y)),
                    'akurasi_leksikon_vs_gold': float(accuracy_score(gold_y, lab_leks)),
                    'f1_macro_leksikon_vs_gold': float(f1_score(gold_y, lab_leks, average='macro'))}
        for nama in ('Naive Bayes', 'Random Forest'):
            yp = hasil_evaluasi[nama]['y_pred'][ix]
            out_gold[f'akurasi_{nama}_vs_gold'] = float(accuracy_score(gold_y, yp))
            out_gold[f'f1_macro_{nama}_vs_gold'] = float(f1_score(gold_y, yp, average='macro'))
        # --- dua aturan pelabelan vs gold (penentu aturan terbaik) --------------
        ref = df_clean.set_index('id_komentar')
        idxs = gold['id_komentar'].values
        for nm in ('baseline', 'klausa'):
            lab = ref.loc[idxs, f'label_{nm}'].values
            out_gold[f'akurasi_leksikon_{nm}_vs_gold'] = float(accuracy_score(gold_y, lab))
            out_gold[f'f1_macro_leksikon_{nm}_vs_gold'] = float(f1_score(gold_y, lab, average='macro'))
            out_gold[f'kappa_leksikon_{nm}_vs_gold'] = float(cohen_kappa_score(gold_y, lab))
        # --- subset target opini 'ekonomi' ---------------------------------------
        if 'target_opini' in gold.columns:
            tg = gold['target_opini'].astype(str).str.strip().str.lower().values
            out_gold['target_distribusi'] = pd.Series(tg).value_counts().to_dict()
            m_eko = (tg == 'ekonomi')
            if m_eko.sum() >= 30:
                out_gold['n_target_ekonomi'] = int(m_eko.sum())
                out_gold['f1_macro_leksikon_vs_gold_ekonomi'] = float(
                    f1_score(gold_y[m_eko], lab_leks[m_eko], average='macro'))
                for nama in ('Naive Bayes', 'Random Forest'):
                    yp = hasil_evaluasi[nama]['y_pred'][ix]
                    out_gold[f'f1_macro_{nama}_vs_gold_ekonomi'] = float(
                        f1_score(gold_y[m_eko], yp[m_eko], average='macro'))
        # --- sensitivitas parameter (robustness, BUKAN untuk memilih parameter) --
        baris_s = []
        for nf in (-1.0, -0.74, -0.5):
            for po in (1.0, 1.5, 2.0):
                labs = [label_klausa(PARSED[i], nf, KONTRAS_PRE, po)[0] for i in idxs]
                baris_s.append({'neg_faktor': nf, 'kontras_post': po,
                                'f1_macro_vs_gold': f1_score(gold_y, labs, average='macro'),
                                'akurasi_vs_gold': accuracy_score(gold_y, labs)})
        pd.DataFrame(baris_s).to_csv(os.path.join(DIR_OUTPUT, 'sensitivitas_parameter_gold.csv'),
                                     index=False, float_format='%.6f')
        baris_o = []
        for rule in ('positif', 'negatif', 'maks_abs', 'jumlah', 'abaikan'):
            P_, N_ = terapkan_aturan_overlap(POS_K, NEG_K, rule)
            labs = [label_klausa(PARSED[i_], pdict=P_, ndict=N_)[0] for i_ in idxs]
            baris_o.append({'aturan_overlap': rule,
                            'akurasi_vs_gold': accuracy_score(gold_y, labs),
                            'f1_macro_vs_gold': f1_score(gold_y, labs, average='macro'),
                            'kappa_vs_gold': cohen_kappa_score(gold_y, labs)})
        pd.DataFrame(baris_o).to_csv(os.path.join(DIR_OUTPUT, 'sensitivitas_overlap_gold.csv'),
                                     index=False, float_format='%.6f')
        RINGKASAN['gold_standard'] = out_gold
        for k, v in out_gold.items():
            print(f"  {k:<34}: {v:.4f}" if isinstance(v, float) else f"  {k:<34}: {v}")
    else:
        print("File anotasi ada tetapi belum terisi/valid.")
else:
    print("anotasi_manual.csv belum ada -> tahap Kappa & gold standard dilewati.")


# =============================================================================
# 11. VISUALISASI
# =============================================================================
print("\n" + "=" * 70)
print("11. VISUALISASI -> output/")
print("=" * 70)

sns.set_theme(style='whitegrid')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'

# 1. Distribusi label
plt.figure(figsize=(7, 6))
lc = df_clean['label'].value_counts()
cp = {'positif': '#2ecc71', 'negatif': '#e74c3c', 'netral': '#3498db'}
plt.pie(lc.values, labels=[f"{k.capitalize()} ({v:,})" for k, v in zip(lc.index, lc.values)],
        autopct='%1.1f%%', colors=[cp.get(c, '#95a5a6') for c in lc.index],
        startangle=140, explode=[0.04 if c == 'positif' else 0 for c in lc.index], shadow=True)
plt.title('Distribusi Sentimen Opini Publik YouTube (InSet Lexicon)',
          fontsize=13, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig(os.path.join(DIR_OUTPUT, 'distribusi_label.png'), dpi=300)
plt.close()

# 2. Perbandingan performa (macro)
metrik_kolom = ['Akurasi', 'Precision_macro', 'Recall_macro', 'F1_macro']
metrik_label = ['Akurasi', 'Precision\n(macro)', 'Recall\n(macro)', 'F1-Score\n(macro)']
nb_s = [hasil_evaluasi['Naive Bayes'][m] * 100 for m in metrik_kolom]
rf_s = [hasil_evaluasi['Random Forest'][m] * 100 for m in metrik_kolom]
x, w = np.arange(len(metrik_kolom)), 0.35
fig, ax = plt.subplots(figsize=(9, 5.5))
r1 = ax.bar(x - w / 2, nb_s, w, label='Naive Bayes Classifier', color='#F39C12')
r2 = ax.bar(x + w / 2, rf_s, w, label='Random Forest Classifier', color='#8E44AD')
ax.set_ylabel('Persentase (%)', fontsize=11, fontweight='bold')
ax.set_title('Perbandingan Kinerja Model: NBC vs Random Forest (macro-average)',
             fontsize=13, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(metrik_label, fontsize=11)
ax.set_ylim(0, 105)
ax.legend(frameon=True, facecolor='white', framealpha=0.9)
for rs in (r1, r2):
    for r in rs:
        ax.annotate(f'{r.get_height():.2f}%', xy=(r.get_x() + r.get_width() / 2, r.get_height()),
                    xytext=(0, 3), textcoords='offset points', ha='center', va='bottom',
                    fontsize=9, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(DIR_OUTPUT, 'perbandingan_performa.png'), dpi=300)
plt.close()

# 3. Confusion matrix
for model_name, suf in [('Naive Bayes', 'nbc'), ('Random Forest', 'rfc')]:
    cm = confusion_matrix(y_test, hasil_evaluasi[model_name]['y_pred'], labels=labels_order)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[l.capitalize() for l in labels_order],
                yticklabels=[l.capitalize() for l in labels_order])
    plt.title(f'Confusion Matrix — {model_name}', fontsize=12, fontweight='bold', pad=12)
    plt.xlabel('Prediksi Model', fontsize=10, fontweight='bold')
    plt.ylabel('Label Aktual', fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(DIR_OUTPUT, f'confusion_matrix_{suf}.png'), dpi=300)
    plt.close()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, (model_name, cm_name) in zip(axes, [('Naive Bayes', 'Oranges'), ('Random Forest', 'Purples')]):
    cm = confusion_matrix(y_test, hasil_evaluasi[model_name]['y_pred'], labels=labels_order)
    sns.heatmap(cm, annot=True, fmt='d', cmap=cm_name, ax=ax,
                xticklabels=[l.capitalize() for l in labels_order],
                yticklabels=[l.capitalize() for l in labels_order])
    ax.set_title(f'Confusion Matrix: {model_name}', fontsize=12, fontweight='bold')
    ax.set_xlabel('Prediksi Model', fontsize=10)
    ax.set_ylabel('Label Aktual', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(DIR_OUTPUT, 'confusion_matrix.png'), dpi=300)
plt.close()

# 4. Reliability diagram (NOVELTY)
fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
for ax, (nama, warna) in zip(axes, [('Naive Bayes', '#F39C12'), ('Random Forest', '#8E44AD')]):
    tb = analisis[nama]['tabel_bin'].dropna()
    ax.plot([0, 1], [0, 1], '--', color='gray', label='Kalibrasi sempurna')
    ax.plot(tb['conf'], tb['acc'], marker='o', color=warna, label=nama)
    for _, r in tb.iterrows():
        ax.annotate(f"{int(r['n'])}", (r['conf'], r['acc']), textcoords='offset points',
                    xytext=(0, 7), ha='center', fontsize=8)
    ax.set_title(f"{nama}\nECE={analisis[nama]['ece_10bin']:.3f} | "
                 f"Brier={analisis[nama]['brier_multiclass']:.3f}", fontsize=11, fontweight='bold')
    ax.set_xlabel('Rata-rata keyakinan per bin')
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.05)
    ax.legend(loc='upper left')
axes[0].set_ylabel('Akurasi aktual per bin')
plt.suptitle('Reliability Diagram (angka = jumlah sampel per bin)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(DIR_OUTPUT, 'reliability_diagram.png'), dpi=300)
plt.close()

# 5. Distribusi confidence: prediksi benar vs salah
fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=False)
bins_h = np.linspace(0.3, 1.0, 15)
for ax, nama in zip(axes, ['Naive Bayes', 'Random Forest']):
    cf, bn = analisis[nama]['conf'], analisis[nama]['benar']
    ax.hist(cf[bn], bins=bins_h, alpha=0.65, label='Prediksi benar', color='#2ecc71')
    ax.hist(cf[~bn], bins=bins_h, alpha=0.65, label='Prediksi salah', color='#e74c3c')
    ax.set_title(nama, fontsize=12, fontweight='bold')
    ax.set_xlabel('Confidence = max P(y|x)')
    ax.set_ylabel('Jumlah komentar')
    ax.legend()
plt.suptitle('Distribusi Probabilitas Keyakinan: Prediksi Benar vs Salah',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(DIR_OUTPUT, 'distribusi_confidence.png'), dpi=300)
plt.close()

# 6. WordCloud
for sentimen, cmap_wc in [('positif', 'Greens'), ('negatif', 'Reds'), ('netral', 'Blues')]:
    teks = ' '.join(df_clean[df_clean['label'] == sentimen]['komentar_bersih'])
    if teks.strip():
        wc = WordCloud(width=800, height=450, background_color='white', colormap=cmap_wc,
                       max_words=150, random_state=RANDOM_STATE).generate(teks)
        plt.figure(figsize=(10, 5.5))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'WordCloud Sentimen {sentimen.capitalize()}', fontsize=14, fontweight='bold', pad=12)
        plt.tight_layout()
        plt.savefig(os.path.join(DIR_OUTPUT, f'wordcloud_{sentimen}.png'), dpi=300)
        plt.close()
print("[OK] Semua gambar tersimpan di output/.")


# =============================================================================
# 12. SIMPAN RINGKASAN, MODEL (.pkl) & SINKRONISASI
# =============================================================================
print("\n" + "=" * 70)
print("12. MENYIMPAN RINGKASAN & MODEL")
print("=" * 70)

with open(os.path.join(DIR_OUTPUT, 'hasil_ringkasan.json'), 'w', encoding='utf-8') as f:
    json.dump(RINGKASAN, f, indent=2, ensure_ascii=False, default=to_py)

with open(os.path.join(DIR_OUTPUT, 'model_nbc.pkl'),        'wb') as f: pickle.dump(nbc,   f)
with open(os.path.join(DIR_OUTPUT, 'model_rfc.pkl'),        'wb') as f: pickle.dump(rfc,   f)
with open(os.path.join(DIR_OUTPUT, 'tfidf_vectorizer.pkl'), 'wb') as f: pickle.dump(tfidf, f)

sub_output = os.path.join(DIR_OUTPUT, 'output')
if os.path.exists(sub_output):
    for fn in os.listdir(DIR_OUTPUT):
        src = os.path.join(DIR_OUTPUT, fn)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(sub_output, fn))

print("[OK] output/hasil_ringkasan.json, model_nbc.pkl, model_rfc.pkl, tfidf_vectorizer.pkl")
print("\n" + "=" * 70)
print("PIPELINE SELESAI. Jalankan dasbor: streamlit run dashboard.py")
print("=" * 70)

# --- diagnosis: jelaskan 5 komentar yang labelnya berubah antara v2 dan v3 ------
if berubah.any():
    print("\nDIAGNOSIS 5 KOMENTAR YANG LABELNYA BERUBAH (aturan v3):")
    for _, r in df_clean[berubah].sample(min(5, int(berubah.sum())),
                                         random_state=RANDOM_STATE).iterrows():
        print(f"  label v2={r['label_baseline']} | label v3={r['label_klausa']}")
        jelaskan_komentar(str(r['komentar']))