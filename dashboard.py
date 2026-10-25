"""
=============================================================================
  DASHBOARD ANALISIS SENTIMEN — STREAMLIT
  Kondisi Ekonomi Indonesia: Pelemahan Rupiah (YouTube)
  
  Cara menjalankan:
      streamlit run dashboard.py
=============================================================================
"""

import os
import sys
import pickle
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import streamlit as st

warnings.filterwarnings('ignore')

# =============================================================================
# KONFIGURASI HALAMAN
# =============================================================================
st.set_page_config(
    page_title="Dashboard Analisis Sentimen",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CSS KUSTOM
# =============================================================================
st.markdown("""
<style>
    /* Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"]  { font-family: 'Inter', sans-serif; font-size: 15px; }

    /* Header utama */
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    .main-header h1 {
        color: #f8fafc !important;
        font-size: 2.2rem !important;
        font-weight: 700;
        margin: 0;
        text-shadow: 0 2px 8px rgba(0,0,0,0.4);
    }
    .main-header p {
        color: #cbd5e1 !important;
        font-size: 1.05rem !important;
        margin: 0.5rem 0 0 0;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1e293b, #0f172a);
        border: 1px solid #475569;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        box-shadow: 0 4px 16px rgba(0,0,0,0.2);
    }
    .metric-card .value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #38bdf8;
    }
    .metric-card .label {
        font-size: 0.95rem;
        font-weight: 500;
        color: #cbd5e1;
        margin-top: 0.3rem;
    }

    /* Sentimen badge */
    .badge-positif { background:#166534; color:#bbf7d0; padding:3px 10px; border-radius:20px; font-size:0.85rem; font-weight:600; }
    .badge-negatif { background:#7f1d1d; color:#fecaca; padding:3px 10px; border-radius:20px; font-size:0.85rem; font-weight:600; }
    .badge-netral  { background:#1e3a5f; color:#bae6fd; padding:3px 10px; border-radius:20px; font-size:0.85rem; font-weight:600; }

    /* Prediksi result box */
    .prediksi-box {
        border-radius: 12px;
        padding: 1.5rem;
        margin-top: 1rem;
        text-align: center;
        font-weight: 700;
        font-size: 1.2rem;
    }
    .prediksi-positif { background: #14532d; border: 2px solid #22c55e; color: #86efac; }
    .prediksi-negatif { background: #450a0a; border: 2px solid #ef4444; color: #fca5a5; }
    .prediksi-netral  { background: #0c2a4a; border: 2px solid #3b82f6; color: #93c5fd; }

    /* Section title */
    .section-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #f8fafc;
        border-left: 5px solid #38bdf8;
        padding-left: 0.9rem;
        margin: 1.8rem 0 1rem 0;
    }

    /* Perbaikan khusus Sidebar & Code Block */
    [data-testid="stSidebar"] pre {
        background-color: #0f172a !important;
        border: 1px solid #334155;
    }
    [data-testid="stSidebar"] code {
        color: #38bdf8 !important;
    }

    /* Penyesuaian font teks markdown */
    .stMarkdown p, .stMarkdown li { 
        font-size: 1.05rem; 
        line-height: 1.7; 
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# PATH & KONSTANTA
# =============================================================================
DIR_OUTPUT    = 'output'
DIR_PROCESSED = os.path.join('data', 'processed')
DIR_LEXICON   = os.path.join('data', 'lexicon')

CLEAN_CSV = os.path.join(DIR_PROCESSED, 'komentar_clean.csv')
NBC_PKL   = os.path.join(DIR_OUTPUT, 'model_nbc.pkl')
RFC_PKL   = os.path.join(DIR_OUTPUT, 'model_rfc.pkl')
TFIDF_PKL = os.path.join(DIR_OUTPUT, 'tfidf_vectorizer.pkl')

COLORS      = {'positif': '#22c55e', 'negatif': '#ef4444', 'netral': '#3b82f6'}
PALETTE_NBC = '#F39C12'
PALETTE_RFC = '#8E44AD'

# =============================================================================
# FUNGSI HELPER
# =============================================================================

@st.cache_data
def load_data():
    if os.path.exists(CLEAN_CSV):
        return pd.read_csv(CLEAN_CSV)
    return None


@st.cache_resource
def load_models():
    if all(os.path.exists(p) for p in [NBC_PKL, RFC_PKL, TFIDF_PKL]):
        with open(NBC_PKL,   'rb') as f: nbc   = pickle.load(f)
        with open(RFC_PKL,   'rb') as f: rfc   = pickle.load(f)
        with open(TFIDF_PKL, 'rb') as f: tfidf = pickle.load(f)
        return nbc, rfc, tfidf
    return None, None, None


@st.cache_resource
def load_lexicon():
    path_pos = os.path.join(DIR_LEXICON, 'inset_positif.txt')
    path_neg = os.path.join(DIR_LEXICON, 'inset_negatif.txt')
    pos_dict, neg_dict = {}, {}
    if os.path.exists(path_pos):
        with open(path_pos, 'r', encoding='utf-8') as f:
            for l in f:
                l = l.strip()
                if not l or l.startswith('word\t'):
                    continue
                parts = l.split('\t')
                pos_dict[parts[0].strip().lower()] = float(parts[1]) if len(parts) > 1 else 1.0
    if os.path.exists(path_neg):
        with open(path_neg, 'r', encoding='utf-8') as f:
            for l in f:
                l = l.strip()
                if not l or l.startswith('word\t'):
                    continue
                parts = l.split('\t')
                neg_dict[parts[0].strip().lower()] = float(parts[1]) if len(parts) > 1 else -1.0
    return pos_dict, neg_dict


KATA_NEGASI = {
    'tidak', 'bukan', 'belum', 'jangan', 'tanpa', 'kurang',
    'tiada', 'tak', 'gak', 'ga', 'nggak', 'bkn', 'tdk'
}

KATA_PENDEK_POSITIF = {'top', 'jos', 'oke', 'sip', 'ok'}

KATA_SENTIMEN_WHITELIST = {
    'baik', 'bagus', 'tepat', 'yakin', 'benar', 'jelas', 'cukup', 'hebat', 'mantap', 'keren'
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
    'trims': 'terima kasih', 'terimakasih': 'terima kasih', 'matur nuwun': 'terima kasih',
    'jos': 'bagus', 'joss': 'bagus', 'top': 'hebat', 'oke': 'baik', 'ok': 'baik', 'sip': 'baik',
    'sipp': 'baik', 'cuan': 'untung', 'beneran': 'benar', 'juoss': 'bagus',
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


@st.cache_resource
def load_sastrawi():
    try:
        from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
        from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
        stemmer    = StemmerFactory().create_stemmer()
        # Whitelist kata negasi dan sentimen agar tidak terhapus
        stop_words = set(StopWordRemoverFactory().get_stop_words()) - KATA_NEGASI - KATA_SENTIMEN_WHITELIST
        return stemmer, stop_words
    except Exception:
        return None, None


def tokenize_pra_stem(text: str) -> list:
    """Pembersihan teks tahap 1 (Pra-stemming untuk pelabelan InSet Lexicon)."""
    if not isinstance(text, str):
        return []
    text = re.sub(r'http\S+|@\w+|#\w+|\d+|[^\w\s]', ' ', text)
    text = re.sub(r'(.)\1{2,}', r'\1', text)
    text = re.sub(r'\s+', ' ', text).strip().lower()
    tokens = text.split()
    flat = []
    for t in tokens:
        expanded = SLANG_DICT.get(t, t)
        flat.extend(expanded.split())
    return flat


def label_sentimen_lexicon(tokens, pos_dict: dict, neg_dict: dict, window: int = 3) -> str:
    """Pelabelan sentimen otomatis berbasis InSet Lexicon dengan jangkauan negasi (pra-stemming)."""
    if isinstance(tokens, str):
        tokens = tokenize_pra_stem(tokens)
    score = 0.0
    neg_active_until = -1

    for i, t in enumerate(tokens):
        if t in KATA_NEGASI:
            neg_active_until = i + window
            continue

        is_negated = (i <= neg_active_until)

        if t in pos_dict:
            w = pos_dict[t]
            val = -w if is_negated else w
            score += val
        elif t in neg_dict:
            w = neg_dict[t]
            val = abs(w) if is_negated else -abs(w)
            score += val

    if score > 0:
        return 'positif'
    elif score < 0:
        return 'negatif'
    else:
        return 'netral'


def preprocess_text(text: str, stemmer, stop_words: set) -> str:
    """Pembersihan teks tahap 2 (Stopwords removal selektif & stemming Sastrawi untuk TF-IDF)."""
    tokens = tokenize_pra_stem(text)
    filtered = []
    for t in tokens:
        if (len(t) >= 4 or t in KATA_NEGASI or t in KATA_PENDEK_POSITIF) and t not in stop_words:
            filtered.append(t)
    if stemmer:
        filtered = [t if (t in KATA_NEGASI or t in KATA_PENDEK_POSITIF) else stemmer.stem(t) for t in filtered]
    return ' '.join(filtered)


def show_image(path: str, caption: str = ''):
    if os.path.exists(path):
        img = Image.open(path)
        st.image(img, caption=caption, use_container_width=True)
    else:
        st.warning(f"Gambar belum tersedia: `{path}`. Jalankan `python main.py` terlebih dahulu.")


# =============================================================================
# HEADER
# =============================================================================
st.markdown("""
<div class="main-header">
    <h1>📊 Dashboard Analisis Sentimen YouTube</h1>
    <p>Kondisi Ekonomi Indonesia &mdash; Pelemahan Nilai Tukar Rupiah</p>
    <p style="font-size:0.8rem; color:#64748b; margin-top:0.5rem;">
        Algoritma: Naive Bayes Classifier &nbsp;|&nbsp; Random Forest Classifier &nbsp;|&nbsp;
        Labeling: InSet Lexicon Bahasa Indonesia
    </p>
</div>
""", unsafe_allow_html=True)

# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.markdown("## ⚙️ Navigasi")
    tab_sel = st.radio(
        "Pilih Halaman",
        ["🏠 Gambaran Umum", "📈 Visualisasi", "📋 Data Komentar", "🔮 Prediksi Sentimen"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("### 📌 Tentang Proyek")
    st.markdown("""
    **Topik:** Analisis Sentimen  
    **Data:** Komentar YouTube  
    **Sumber:** Channel berita & influencer ekonomi  
    **Metode:** NBC & RFC  
    **Labeling:** InSet Lexicon  
    """)
    st.markdown("---")
    st.markdown("### 🚀 Cara Menjalankan")
    st.code(".\\.venv\\Scripts\\streamlit.exe run dashboard.py", language="powershell")

# Muat data & model
df = load_data()
nbc_model, rfc_model, tfidf_model = load_models()
kata_positif, kata_negatif = load_lexicon()

# =============================================================================
# TAB 1 — GAMBARAN UMUM (OVERVIEW)
# =============================================================================
if tab_sel in ["🏠 Gambaran Umum", "🏠 Overview"]:
    st.markdown('<div class="section-title">Ringkasan Dataset</div>', unsafe_allow_html=True)

    if df is not None:
        total       = len(df)
        n_positif   = int((df['label'] == 'positif').sum()) if 'label' in df.columns else 0
        n_negatif   = int((df['label'] == 'negatif').sum()) if 'label' in df.columns else 0
        n_netral    = int((df['label'] == 'netral').sum())  if 'label' in df.columns else 0

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="value">{total:,}</div>
                <div class="label">Total Komentar</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="value" style="color:#22c55e">{n_positif:,}</div>
                <div class="label">Sentimen Positif</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="value" style="color:#ef4444">{n_negatif:,}</div>
                <div class="label">Sentimen Negatif</div>
            </div>""", unsafe_allow_html=True)
        with c4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="value" style="color:#3b82f6">{n_netral:,}</div>
                <div class="label">Sentimen Netral</div>
            </div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-title">Alur Pipeline Metodologi</div>', unsafe_allow_html=True)
        steps = [
            ("1", "Pengumpulan Data", "Crawling komentar dari 12 video YouTube berita ekonomi menggunakan youtube-comment-downloader (tanpa API key)"),
            ("2", "Preprocessing Teks", "Cleaning URL/mention → Lowercase → Tokenisasi → Filter pendek → Stopword Removal → Stemming (PySastrawi)"),
            ("3", "Pelabelan Sentimen", "Klasifikasi otomatis: Positif / Negatif / Netral menggunakan InSet Lexicon Bahasa Indonesia"),
            ("4", "Ekstraksi Fitur TF-IDF", "Pembobotan kata menggunakan TF-IDF (max 5000 fitur, n-gram 1-2) + Split 80% training / 20% testing"),
            ("5", "Pelatihan Model", "Melatih Naive Bayes Classifier (NBC) dan Random Forest Classifier (RFC) secara paralel"),
            ("6", "Evaluasi Model", "Mengukur Akurasi, Presisi, Recall, dan F1-Score pada data testing"),
            ("7", "Visualisasi Hasil", "Pie Chart, Bar Chart perbandingan, Confusion Matrix, dan WordCloud per kelas"),
            ("8", "Simpan Model", "Ekspor model (.pkl) untuk digunakan kembali tanpa perlu melatih ulang"),
        ]
        for num, title, desc in steps:
            col_num, col_content = st.columns([1, 10])
            with col_num:
                st.markdown(f"""
                <div style='background:#0f3460;color:#38bdf8;font-weight:700;
                            font-size:1.1rem;border-radius:50%;width:40px;height:40px;
                            display:flex;align-items:center;justify-content:center;
                            margin-top:6px;'>
                    {num}
                </div>""", unsafe_allow_html=True)
            with col_content:
                st.markdown(f"**{title}**  \n{desc}")

        if 'label' in df.columns:
            st.markdown('<div class="section-title">Distribusi Sentimen</div>', unsafe_allow_html=True)
            label_counts = df['label'].value_counts()
            fig, ax = plt.subplots(figsize=(6, 3))
            colors  = [COLORS.get(l, '#64748b') for l in label_counts.index]
            bars    = ax.barh(label_counts.index, label_counts.values, color=colors, edgecolor='none')
            for bar, val in zip(bars, label_counts.values):
                pct = val / total * 100
                ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                        f'{val:,} ({pct:.1f}%)', va='center', fontsize=11, fontweight='bold', color='white')
            ax.set_facecolor('#0f172a')
            fig.patch.set_facecolor('#0f172a')
            ax.tick_params(colors='white')
            ax.spines[:].set_visible(False)
            ax.set_xlim(0, label_counts.max() * 1.3)
            st.pyplot(fig)
            plt.close()
    else:
        st.warning("Data belum tersedia. Jalankan `python main.py` terlebih dahulu.")

    if nbc_model and rfc_model:
        st.markdown('<div class="section-title">Performa Model (Ringkasan)</div>', unsafe_allow_html=True)
        st.info("Lihat detail metrik dan confusion matrix di tab **Visualisasi**.")


# =============================================================================
# TAB 2 — VISUALISASI
# =============================================================================
elif tab_sel == "📈 Visualisasi":
    st.markdown('<div class="section-title">Visualisasi Hasil Analisis Sentimen</div>', unsafe_allow_html=True)

    if df is None:
        st.warning("Data belum tersedia. Jalankan `python main.py` terlebih dahulu.")
    else:
        # ── Tabel Perbandingan NBC vs RFC ────────────────────────────────────
        st.markdown("#### 📋 Tabel Perbandingan Performa: Naive Bayes vs Random Forest")

        if nbc_model and rfc_model and tfidf_model:
            import pickle
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

            # Hitung ulang metrik dari test data yang tersimpan di CSV
            # Gunakan seluruh data untuk menampilkan gambaran umum
            try:
                from sklearn.feature_extraction.text import TfidfVectorizer
                from sklearn.model_selection import train_test_split

                X_all = tfidf_model.transform(df['komentar_bersih'].fillna(''))
                y_all = df['label']
                _, X_test, _, y_test = train_test_split(
                    X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
                )
                y_pred_nbc = nbc_model.predict(X_test)
                y_pred_rfc = rfc_model.predict(X_test)

                metrik_rows = []
                for metrik, fn in [
                    ("Akurasi (%)",  lambda yp: accuracy_score(y_test, yp) * 100),
                    ("Presisi (%)",  lambda yp: precision_score(y_test, yp, average='weighted', zero_division=0) * 100),
                    ("Recall (%)",   lambda yp: recall_score(y_test, yp, average='weighted', zero_division=0) * 100),
                    ("F1-Score (%)", lambda yp: f1_score(y_test, yp, average='weighted', zero_division=0) * 100),
                ]:
                    nbc_val = fn(y_pred_nbc)
                    rfc_val = fn(y_pred_rfc)
                    selisih = rfc_val - nbc_val
                    pemenang = "🟡 NBC" if nbc_val > rfc_val else ("🟣 RFC" if rfc_val > nbc_val else "🟰 Seri")
                    metrik_rows.append({
                        "Metrik"         : metrik,
                        "Naive Bayes (NBC) 🟡" : f"{nbc_val:.2f}%",
                        "Random Forest (RFC) 🟣": f"{rfc_val:.2f}%",
                        "Selisih"        : f"{'+' if selisih >= 0 else ''}{selisih:.2f}%",
                        "Unggul"         : pemenang,
                    })

                df_metrik = pd.DataFrame(metrik_rows)
                st.dataframe(df_metrik, use_container_width=True, hide_index=True)

                # Grafik perbandingan interaktif (live, bukan dari file PNG)
                st.markdown("#### 📊 Grafik Perbandingan Performa (NBC vs RFC)")
                metrik_names = ["Akurasi", "Presisi", "Recall", "F1-Score"]
                nbc_vals_plot = [
                    accuracy_score(y_test, y_pred_nbc) * 100,
                    precision_score(y_test, y_pred_nbc, average='weighted', zero_division=0) * 100,
                    recall_score(y_test, y_pred_nbc, average='weighted', zero_division=0) * 100,
                    f1_score(y_test, y_pred_nbc, average='weighted', zero_division=0) * 100,
                ]
                rfc_vals_plot = [
                    accuracy_score(y_test, y_pred_rfc) * 100,
                    precision_score(y_test, y_pred_rfc, average='weighted', zero_division=0) * 100,
                    recall_score(y_test, y_pred_rfc, average='weighted', zero_division=0) * 100,
                    f1_score(y_test, y_pred_rfc, average='weighted', zero_division=0) * 100,
                ]

                x     = np.arange(len(metrik_names))
                width = 0.35
                fig, ax = plt.subplots(figsize=(10, 5))
                fig.patch.set_facecolor('#0f172a')
                ax.set_facecolor('#0f172a')

                bars1 = ax.bar(x - width/2, nbc_vals_plot, width,
                               label='Naive Bayes (NBC)', color=PALETTE_NBC,
                               edgecolor='none', zorder=3)
                bars2 = ax.bar(x + width/2, rfc_vals_plot, width,
                               label='Random Forest (RFC)', color=PALETTE_RFC,
                               edgecolor='none', zorder=3)

                for bar in bars1:
                    h = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2, h + 0.8,
                            f'{h:.1f}%', ha='center', va='bottom',
                            fontsize=10, fontweight='bold', color=PALETTE_NBC)
                for bar in bars2:
                    h = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2, h + 0.8,
                            f'{h:.1f}%', ha='center', va='bottom',
                            fontsize=10, fontweight='bold', color=PALETTE_RFC)

                ax.set_xticks(x)
                ax.set_xticklabels(metrik_names, fontsize=12, color='white')
                ax.set_ylim(0, 115)
                ax.set_ylabel('Nilai (%)', fontsize=11, color='white')
                ax.set_title('Perbandingan Performa NBC vs RFC\n(Analisis Sentimen Komentar YouTube - Pelemahan Rupiah)',
                             fontsize=12, fontweight='bold', color='white', pad=12)
                ax.tick_params(colors='white')
                ax.spines[:].set_visible(False)
                ax.grid(axis='y', linestyle='--', alpha=0.2, color='white', zorder=0)
                ax.legend(fontsize=11, facecolor='#1e293b', labelcolor='white',
                          edgecolor='#334155', loc='upper right')
                st.pyplot(fig)
                plt.close()

                # Kesimpulan otomatis
                st.markdown("---")
                best_acc = "Random Forest" if rfc_vals_plot[0] > nbc_vals_plot[0] else "Naive Bayes"
                best_f1  = "Random Forest" if rfc_vals_plot[3] > nbc_vals_plot[3] else "Naive Bayes"
                st.success(f"""
**Kesimpulan Perbandingan Model:**
- 🏆 **{best_acc}** memiliki **Akurasi tertinggi** ({max(nbc_vals_plot[0], rfc_vals_plot[0]):.2f}%)
- 🏆 **{best_f1}** memiliki **F1-Score tertinggi** ({max(nbc_vals_plot[3], rfc_vals_plot[3]):.2f}%)
- Random Forest cenderung lebih baik karena mampu menangkap pola yang lebih kompleks dari data teks.
                """)

            except Exception as e:
                st.warning(f"Tidak dapat menghitung metrik: {e}")
                show_image(os.path.join(DIR_OUTPUT, 'perbandingan_performa.png'))
        else:
            st.warning("Model belum tersedia. Jalankan `python main.py` terlebih dahulu.")

        st.markdown("---")

        # ── Grafik lainnya ───────────────────────────────────────────────────
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 🥧 Distribusi Label Sentimen")
            show_image(os.path.join(DIR_OUTPUT, 'distribusi_label.png'))
        with col2:
            st.markdown("#### 📊 Perbandingan Performa (dari file)")
            show_image(os.path.join(DIR_OUTPUT, 'perbandingan_performa.png'))

        st.markdown("---")
        st.markdown("#### 🔲 Confusion Matrix")
        show_image(os.path.join(DIR_OUTPUT, 'confusion_matrix.png'))

        st.markdown("---")
        st.markdown("#### ☁️ WordCloud per Kelas Sentimen")
        wc1, wc2, wc3 = st.columns(3)
        with wc1:
            st.markdown("**Positif**")
            show_image(os.path.join(DIR_OUTPUT, 'wordcloud_positif.png'))
        with wc2:
            st.markdown("**Negatif**")
            show_image(os.path.join(DIR_OUTPUT, 'wordcloud_negatif.png'))
        with wc3:
            st.markdown("**Netral**")
            show_image(os.path.join(DIR_OUTPUT, 'wordcloud_netral.png'))


# =============================================================================
# TAB 3 — DATA KOMENTAR
# =============================================================================
elif tab_sel == "📋 Data Komentar":
    st.markdown('<div class="section-title">Tabel Data Komentar</div>', unsafe_allow_html=True)

    if df is None:
        st.warning("Data belum tersedia. Jalankan `python main.py` terlebih dahulu.")
    else:
        # Filter
        col_f1, col_f2, col_f3 = st.columns([2, 2, 3])
        with col_f1:
            filter_label = st.selectbox(
                "Filter Sentimen",
                ["Semua", "positif", "negatif", "netral"]
            )
        with col_f2:
            n_tampil = st.selectbox("Tampilkan baris", [50, 100, 200, 500], index=0)
        with col_f3:
            cari = st.text_input("Cari kata dalam komentar...", placeholder="Ketik kata kunci...")

        df_show = df.copy()
        if filter_label != "Semua" and 'label' in df_show.columns:
            df_show = df_show[df_show['label'] == filter_label]
        if cari and 'komentar' in df_show.columns:
            df_show = df_show[df_show['komentar'].str.contains(cari, case=False, na=False)]

        st.markdown(f"**Menampilkan {min(n_tampil, len(df_show)):,} dari {len(df_show):,} data**")

        cols_display = [c for c in ['komentar', 'komentar_bersih', 'label', 'likes', 'tanggal']
                        if c in df_show.columns]
        st.dataframe(
            df_show[cols_display].head(n_tampil),
            use_container_width=True,
            height=500
        )

        # Download
        csv_data = df_show[cols_display].to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
        st.download_button(
            label="📥 Download Data (CSV)",
            data=csv_data,
            file_name='komentar_sentimen.csv',
            mime='text/csv'
        )


# =============================================================================
# TAB 4 — PREDIKSI SENTIMEN
# =============================================================================
elif tab_sel == "🔮 Prediksi Sentimen":
    st.markdown('<div class="section-title">Prediksi Sentimen Komentar Baru</div>', unsafe_allow_html=True)

    if nbc_model is None or rfc_model is None:
        st.warning("Model belum tersedia. Jalankan `python main.py` terlebih dahulu.")
    else:
        st.markdown("""
        Masukkan komentar berbahasa Indonesia tentang kondisi ekonomi/rupiah,
        lalu klik **Prediksi** untuk melihat hasil analisis sentimen dari kedua model.
        """)

        # Contoh komentar
        contoh = st.selectbox("📝 Atau pilih contoh komentar:", [
            "(pilih contoh...)",
            "Rupiah semakin anjlok tidak ada harapan, harga barang mahal semua susah",
            "Alhamdulillah pemerintah berhasil menstabilkan nilai rupiah, ekonomi kita menguat",
            "Nilai tukar rupiah terhadap dolar hari ini bergerak di kisaran normal",
            "PHK massal terjadi dimana-mana karena rupiah melemah bangkrut semua",
            "Cadangan devisa Indonesia masih aman dan kuat, optimis ekonomi pulih",
        ])
        if contoh != "(pilih contoh...)":
            input_default = contoh
        else:
            input_default = ""

        komentar_input = st.text_area(
            "Masukkan komentar:",
            value=input_default,
            height=120,
            placeholder="Contoh: Rupiah terus anjlok, harga bahan pokok naik, susah hidup rakyat kecil..."
        )

        if st.button("🔮 Prediksi Sentimen", type="primary", use_container_width=True):
            if not komentar_input.strip():
                st.error("Harap masukkan teks komentar terlebih dahulu.")
            else:
                stemmer, stop_words = load_sastrawi()

                with st.spinner("Memproses..."):
                    # Preprocessing Pra-stemming & InSet Lexicon
                    tokens_pra  = tokenize_pra_stem(komentar_input)
                    pred_lex    = label_sentimen_lexicon(tokens_pra, kata_positif, kata_negatif)
                    teks_bersih = preprocess_text(komentar_input, stemmer, stop_words or set())

                    # Prediksi model (TF-IDF)
                    vektor      = tfidf_model.transform([teks_bersih])
                    pred_nbc    = nbc_model.predict(vektor)[0]
                    pred_rfc    = rfc_model.predict(vektor)[0]
                    prob_nbc    = nbc_model.predict_proba(vektor)[0]
                    prob_rfc    = rfc_model.predict_proba(vektor)[0]
                    classes     = nbc_model.classes_

                # Tampilkan hasil
                st.markdown("---")
                st.markdown("#### Hasil Prediksi")

                res_cols = st.columns(3)
                for col, (nama, pred) in zip(res_cols, [
                    ("Lexicon (InSet)", pred_lex),
                    ("Naive Bayes",     pred_nbc),
                    ("Random Forest",   pred_rfc),
                ]):
                    emoji = {"positif": "😊", "negatif": "😟", "netral": "😐"}.get(pred, "")
                    warna = {"positif": "prediksi-positif", "negatif": "prediksi-negatif", "netral": "prediksi-netral"}.get(pred, "")
                    with col:
                        st.markdown(f"""
                        <div class="prediksi-box {warna}">
                            {emoji} {pred.upper()}<br>
                            <small style="font-weight:400;font-size:0.8rem;">{nama}</small>
                        </div>""", unsafe_allow_html=True)

                # Teks bersih
                st.markdown("---")
                with st.expander("🔍 Detail Preprocessing"):
                    st.markdown(f"**Teks asli:** {komentar_input}")
                    st.markdown(f"**Setelah preprocessing:** `{teks_bersih}`")

                # Probabilitas
                st.markdown("#### Probabilitas Keyakinan Model")
                prob_col1, prob_col2 = st.columns(2)

                def show_prob_chart(probs, classes, title, color):
                    fig, ax = plt.subplots(figsize=(5, 2.5))
                    bar_colors = ['#22c55e' if c == 'positif' else '#ef4444' if c == 'negatif' else '#3b82f6'
                                  for c in classes]
                    bars = ax.barh(classes, probs * 100, color=bar_colors, edgecolor='none')
                    for bar, val in zip(bars, probs * 100):
                        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                                f'{val:.1f}%', va='center', fontsize=10, color='white', fontweight='bold')
                    ax.set_xlim(0, 115)
                    ax.set_facecolor('#0f172a')
                    fig.patch.set_facecolor('#0f172a')
                    ax.tick_params(colors='white')
                    ax.spines[:].set_visible(False)
                    ax.set_title(title, color='white', fontsize=11, fontweight='bold', pad=8)
                    return fig

                with prob_col1:
                    fig = show_prob_chart(prob_nbc, classes, "Naive Bayes - Probabilitas", PALETTE_NBC)
                    st.pyplot(fig); plt.close()
                with prob_col2:
                    fig = show_prob_chart(prob_rfc, classes, "Random Forest - Probabilitas", PALETTE_RFC)
                    st.pyplot(fig); plt.close()


# =============================================================================
# FOOTER
# =============================================================================
st.markdown("---")
st.markdown("""
<div style="text-align:center;color:#475569;font-size:0.8rem;padding:1rem 0;">
    Dashboard Analisis Sentimen &mdash; Kondisi Ekonomi Indonesia &bull;
    Tugas Proposal Program Studi Teknik Informatika S1 &bull;
    Naive Bayes &amp; Random Forest + InSet Lexicon
</div>
""", unsafe_allow_html=True)


