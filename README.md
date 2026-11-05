# 📊 Sistem Analisis Sentimen Komentar YouTube: Kondisi Ekonomi Indonesia Terkait Pelemahan Nilai Tukar Rupiah

Sistem analisis sentimen komprehensif untuk pengumpulan data (*crawling*), pembersihan teks (praproses dua tahap), pelabelan sentimen otomatis berbasis **InSet Lexicon** (*Indonesian Sentiment Lexicon* pada tahap *pra-stemming* dengan *Negation Handling* & pembobotan klausa), ekstraksi fitur **TF-IDF**, serta klasifikasi dan komparasi algoritma **Naïve Bayes Classifier (NBC)** dan **Random Forest Classifier (RFC)** yang dilengkapi validasi statistik (5-Fold CV & Uji McNemar) dan dashboard interaktif berbasis **Streamlit**.

---

## 📑 Daftar Isi
1. [Latar Belakang & Tujuan](#-latar-belakang--tujuan)
2. [Alat, Bahan, & Teknologi](#-alat-bahan--teknologi)
3. [Arsitektur & Alur Kerja Sistem (Pipeline)](#-arsitektur--alur-kerja-sistem-pipeline)
4. [Detail Metodologi & Preprocessing](#-detail-metodologi--preprocessing)
5. [Hasil Evaluasi & Rangkuman Analisis Sistem](#-hasil-evaluasi--rangkuman-analisis-sistem)
   - [Sebaran Distribusi Sentimen Publik](#1-sebaran-distribusi-sentimen-opini-publik)
   - [Kinerja Model Machine Learning](#2-kinerja-model-machine-learning-data-uji-20--1107-komentar)
   - [Matriks Kebingungan (Confusion Matrix)](#3-detail-matriks-kebingungan-confusion-matrix)
   - [Validasi 5-Fold Stratified Cross-Validation](#4-validasi-stabilitas-5-fold-stratified-cross-validation)
   - [Uji Signifikansi Statistik (Uji McNemar)](#5-uji-signifikansi-statistik-uji-mcnemar)
   - [Analisis Keyakinan & Kalibrasi Probabilitas](#6-analisis-keyakinan--kalibrasi-probabilitas-model)
6. [Fitur Dashboard Interaktif (Streamlit)](#-fitur-dashboard-interaktif-streamlit)
7. [Kegunaan & Manfaat Sistem](#-kegunaan--manfaat-sistem)
8. [Struktur Direktori Proyek](#-struktur-direktori-proyek)
9. [Panduan Instalasi & Penggunaan](#-panduan-instalasi--penggunaan)
10. [Referensi Ilmiah](#-referensi-ilmiah)

---

## 🎯 Latar Belakang & Tujuan

### Latar Belakang
Fluktuasi dan tren pelemahan nilai tukar Rupiah terhadap Dolar Amerika Serikat memicu perhatian luas dan reaksi emosional dari masyarakat Indonesia. Media sosial, khususnya platform YouTube, menjadi kanal publik utama di mana masyarakat secara bebas menyampaikan opini, kritik ekonomi, kekhawatiran inflasi harga bahan pokok, maupun apresiasi terhadap konten edukasi finansial. Mengetahui polaritas sentimen masyarakat secara akurat, objektif, dan terukur sangat krusial bagi analisis respon publik terhadap stabilitas moneter nasional.

### Tujuan Sistem
1. **Pengumpulan Data Publik Terfokus**: Mengumpulkan komentar warganet dari video YouTube kanal berita nasional dan kreator edukasi finansial tentang pelemahan rupiah (saat ini 13 video).
2. **Standardisasi Teks Percakapan**: Membersihkan data teks media sosial yang sarat bahasa gaul (*slang*), singkatan, tanda baca tidak baku, dan elongasi huruf melalui praproses dua tahap.
3. **Pelabelan Otomatis Ilmiah & Akurat**: Menerapkan kamus resmi **InSet Lexicon** pada tahap *pra-stemming* yang dilengkapi penanganan negasi (*negation handling* jendela 3 kata) dan pembobotan klausa untuk mengelompokkan komentar ke dalam tiga kelas: **Positif**, **Negatif**, dan **Netral**.
4. **Komparasi Model Pembelajaran Mesin**: Membandingkan kinerja klasifikasi antara algoritma probabilistik (**Naïve Bayes Classifier**) dan algoritma *ensemble tree-based* (**Random Forest Classifier**) berbasis representasi fitur **TF-IDF**.
5. **Validasi Statistik Komprehensif**: Menguji stabilitas model melalui *5-Fold Stratified Cross-Validation*, uji signifikansi komparatif (*Uji McNemar*), serta analisis kalibrasi keyakinan (*Brier Score & Expected Calibration Error*).
6. **Diseminasi Interaktif Modern**: Menyediakan antarmuka dashboard berbasis web (**Streamlit**) yang memfasilitasi eksplorasi data komentar mentah → bersih, pemantauan visualisasi metrik, dan simulasi prediksi teks baru secara *real-time*.

---

## 🛠 Alat, Bahan, & Teknologi

### 1. Perangkat Lunak & Bahasa Pemrograman
* **Bahasa**: Python 3.10+ (Direkomendasikan Python 3.11 / 3.12 / 3.14)
* **Lingkungan Virtual**: Virtual Environment (`.venv`)

### 2. Pustaka & Library Utama
* **Machine Learning & NLP**:
  * `scikit-learn`: Implementasi TF-IDF Vectorizer, Multinomial Naive Bayes, Random Forest Classifier, Stratified Train-Test Split, 5-Fold Stratified CV, dan metrik evaluasi (Akurasi, Presisi, Recall, F1-Score, Confusion Matrix).
  * `PySastrawi`: Pembersihan stopword Bahasa Indonesia dan algoritma stemming *Enhanced Confix Stripping* (ECS).
  * `scipy`: Komputasi statistik uji signifikansi McNemar (Chi-Square & binomial exact test).
* **Data Processing & Scraping**:
  * `pandas` & `numpy`: Manipulasi data tabular terstruktur dan operasi matriks numerik berkecepatan tinggi.
  * `youtube-comment-downloader`: Ekstraksi komentar publik YouTube bebas limit kuota API.
* **Visualisasi Data**:
  * `matplotlib` & `seaborn`: Desain grafik pie chart, bar chart perbandingan performa, dan heatmap matriks kebingungan.
  * `wordcloud`: Visualisasi awan kata dominan per kelas sentimen.
* **Aplikasi Web**:
  * `streamlit`: Framework antarmuka dashboard interaktif responsif berbasis web.

### 3. Bahan Dataset
* **Sumber Data**: 13 video YouTube dari kanal berita nasional dan kreator edukasi finansial (daftar di `data/raw/daftar_video.txt`).
* **Dataset Mentah (`komentar_raw.csv`)**: **5.733 komentar**.
* **Setelah Deduplikasi**: **5.632 komentar**.
* **Dataset Bersih Valid (`komentar_clean.csv`)**: **5.530 komentar** (102 komentar kosong setelah praproses dibuang, lihat `output/komentar_dibuang_kosong.csv`).

### 4. Kamus Leksikon (InSet Lexicon)
* Kamus **InSet Lexicon** oleh **Koto & Rahmaningtyas (2017)**: **3.609 kata positif** dan **6.579 kata negatif**, bobot **−5** sampai **+5**.
* Kamus aktif dibangun dengan urutan tetap (`sentimen_inset.bangun_kamus`):
  1. **Buang kata tugas** (stopword Sastrawi, 181 kata) kecuali 17 kata yang memang bersentimen.
  2. **Koreksi kamus domain** (`data/lexicon/koreksi_domain.csv`, 126 kata, setiap kata diberi kategori & alasan):
     K1 kata fungsi/modalitas → 0; K2 kata topik ekonomi/pemerintahan (*nilai*, *harga*, *presiden*) → 0;
     K3 kata arah/derajat (*turun*, *tinggi*) → 0; K4 kata ganda yang terpilih di sisi keliru (*bagus*, *benar*, *lemah*) → bobot asli sisi yang benar;
     K5 polaritas keliru di konteks ekonomi (*mahal*, *mandiri*) → bobot manual.
  3. **Kata ganda** (ada di kamus positif & negatif): bobot dengan nilai mutlak terbesar dipakai; bila sama besar, kata diabaikan.

## 🔄 Arsitektur & Alur Kerja Sistem (Pipeline)

```
[ data/raw/daftar_video.txt ] ──> scraping_komentar.py (youtube-comment-downloader)
             │
             ▼
 1. PENGUMPULAN DATA      komentar_raw.csv (5.733) ─> buang kosong & duplikat (5.632)
 2. PRAPROSES TAHAP 1     NFKC, buang tautan/mention/tagar/angka, huruf kecil,
                          pengurangan huruf berulang, normalisasi kata tidak baku,
                          tokenisasi (tanda baca = batas klausa)
 3. PELABELAN INSET       per klausa; negasi jendela 3 token (× −0,74);
                          kontras 0,5 / 1,5; kamus aktif (kata tugas dibuang,
                          koreksi domain, aturan kata ganda)  ─> cakupan 77,02%
 4. PRAPROSES TAHAP 2     stopword selektif, token < 4 huruf dibuang (kecuali
                          negasi), stemming ECS  ─> komentar_clean.csv (5.530)
 5. SPLIT & TF-IDF        stratified 80:20 (4.424 / 1.106; dibekukan di
                          split_ids.csv) lalu TF-IDF fit di data latih saja
 6. PEMODELAN             MultinomialNB (α=1), NB prior seragam (pembanding),
                          Random Forest (100 pohon, class_weight='balanced')
 7. EVALUASI              accuracy + precision/recall/F1 macro, 5-fold CV,
                          uji McNemar, bootstrap CI F1-macro
 8. KEYAKINAN/KALIBRASI   predict_proba, benar vs salah, Brier multikelas,
                          ECE 10 bin, reliability diagram
 9. ANOTASI MANUAL        400 komentar acak dari data uji, 2 anotator, Cohen's Kappa
10. DASBOR STREAMLIT      streamlit run dashboard.py
```

Aturan praproses dan pelabelan berada di **`sentimen_inset.py`** dan dipakai bersama oleh `main.py` dan `dashboard.py`, sehingga label pada dasbor identik dengan label penelitian.

---

## 🔬 Detail Metodologi & Preprocessing

### 1. Landasan Ilmiah Pelabelan Pra-Stemming
Sesuai rancangan metodologi pada proposal penelitian:
* Kamus InSet Lexicon secara inheren memuat lema dan kata berimbuhan bahasa Indonesia (seperti *meningkat*, *membaik*, *bermanfaat*, *bersyukur*, *terpercaya*).
* Apabila pelabelan dilakukan setelah stemming, kata-kata tersebut akan tereduksi menjadi kata dasar (*tingkat*, *kembang*, *manfaat*, *syukur*) yang tidak terdaftar dalam leksikon, sehingga tingkat kecocokan (*match rate*) menurun drastis.
* Oleh karena itu, pelabelan diterapkan pada token **pra-stemming**, sedangkan proses stemming Sastrawi ECS diterapkan setelahnya khusus untuk pembentukan fitur representasi **TF-IDF**.

### 2. Mekanisme Penanganan Negasi (Negation Handling)
* Kata negasi: *tidak*, *bukan*, *belum*, *jangan*, *tanpa*, *kurang*, *tiada*. Varian tidak baku (*gak*, *ga*, *gk*, *engga*, *enggak*, *ndak*, *jgn*, *blm*, *bkn*, ...) dinormalisasi lebih dahulu.
* Negasi memengaruhi kata bersentimen dalam **jendela 3 token** pada **klausa yang sama**.
* Kata bersentimen yang dinegasi dikalikan **−0,74**, yaitu faktor pelemahan negasi VADER (Hutto & Gilbert, 2014):
  *"tidak suka"* → 3 × (−0,74) = −2,22; *"tidak panik"* → (−4) × (−0,74) = +2,96.
* Diagnosis per kata: `python main.py --jelaskan "teks komentar"`.

### 3. Stopword Removal Selektif (Whitelist Protection)
Stopword bawaan bahasa Indonesia sering kali menghapus kata berbobot opini penting:
* **Kata Negasi**: *tidak*, *bukan*, *belum*, *tanpa*, *kurang*
* **Kata Sentimen Positif Kunci**: *baik*, *tepat*, *yakin*, *benar*, *jelas*
* **Kata Pujian Pendek**: *top*, *jos*, *oke*, *sip*, *ok*

Sistem menerapkan aturan *whitelist* ketat:
`Stopword aktif = Stopword Sastrawi − KATA_NEGASI − KATA_SENTIMEN_WHITELIST`
Hal ini menjamin kata-kata sentimen dasar seperti *"baik"* tidak tereliminasi pada teks bersih pemodelan.

### 4. Normalisasi Slang & Reduksi Karakter Berulang (Elongasi)
* Reduksi huruf berulang ≥ 3 kali menjadi bentuk tunggal menggunakan *regular expression*:
  Contoh: `baguuus` → `bagus`, `kerennn` → `keren`, `njirrrr` → `njir`.
* Kamus slang komprehensif mengonversi ratusan istilah khas percakapan warganet (contoh: `bgt` → `banget`, `mantul` → `mantap`, `tks` / `thanks` → `terima kasih`, `boncos` → `rugi`, `cuan` → `untung`).

---

## 📊 Hasil Evaluasi & Rangkuman Analisis Sistem

> **Status:** hasil sementara. Label masih label otomatis InSet; validasi anotasi manual (Cohen's Kappa) belum dijalankan karena `data/annotation/anotasi_manual.csv` belum diisi. Semua angka dibaca dari `output/hasil_ringkasan.json`.

### 1. Distribusi Sentimen (5.530 komentar)

| Kelas | Jumlah | Persentase |
| :--- | :---: | :---: |
| Negatif | 2.354 | 42,57% |
| Positif | 1.777 | 32,13% |
| Netral | 1.399 | 25,30% |

Cakupan leksikon 77,02%. Dari 1.399 komentar netral, 1.271 tidak memuat kata kamus dan 128 memiliki skor yang saling meniadakan.

### 2. Kinerja Model (data uji 1.106 komentar)

| Metrik | Naïve Bayes | NB (prior seragam) | Random Forest |
| :--- | :---: | :---: | :---: |
| Accuracy | 57,05% | 60,76% | **62,39%** |
| Precision (macro) | 62,27% | 60,26% | **63,44%** |
| Recall (macro) | 50,74% | 57,80% | **63,55%** |
| F1-Score (macro) | 49,50% | 58,27% | **61,81%** |
| F1-Score (weighted) | 52,80% | 59,94% | **62,34%** |

### 3. Confusion Matrix (baris = aktual, kolom = prediksi; urutan negatif, netral, positif)

```
Random Forest            Naïve Bayes
[[294 121  56]           [[418  12  41]
 [ 35 221  24]            [205  46  29]
 [ 83  97 175]]           [174  14 167]]
```

### 4. 5-Fold Stratified Cross-Validation (data latih)
* Naïve Bayes: akurasi 55,43% ± 1,13; F1 macro 46,16% ± 1,88
* Random Forest: akurasi 63,54% ± 1,75; F1 macro 63,31% ± 1,74

### 5. Uji McNemar
* b (hanya NB benar) = 186; c (hanya RF benar) = 245; χ² = 7,8051; p = 0,0052 → perbedaan **signifikan** (α = 0,05).
* Bootstrap 95% CI selisih F1 macro (RF − NB): +12,38 poin [+8,66; +15,92].

### 6. Probabilitas Keyakinan & Kalibrasi

| Model | Keyakinan rata-rata | Benar | Salah | Brier multikelas | ECE (10 bin) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Naïve Bayes | 0,5485 | 0,5906 | 0,4925 | 0,5321 | 0,0573 |
| Random Forest | 0,6346 | 0,6682 | 0,5789 | 0,5084 | 0,0414 |

---

## 💻 Fitur Dashboard Interaktif (Streamlit)

Dashboard dirancang dengan konsep estetika modern bertema *dark mode* elegan menggunakan palet warna terkurasi:

1. **Tab 1 — 🏠 Gambaran Umum (Overview)**:
   * Kartu ringkasan metrik utama: Total Dataset (5.530 data), Total Positif, Total Negatif, Total Netral.
   * Visualisasi donat/pie chart proporsi sentimen publik berbasis InSet Lexicon.
   * Diagram ringkas tahapan alur penelitian 8 tahap.
2. **Tab 2 — 📈 Visualisasi & Komparasi Model**:
   * Tabel metrik NBC vs RFC pada **data uji yang sama dengan `main.py`** (split beku `split_ids.csv`), rata-rata **macro**.
   * Tabel keyakinan & kalibrasi (Brier, ECE), *reliability diagram*, distribusi keyakinan benar vs salah, dan hasil uji McNemar.
   * Grafik diagram batang perbandingan akurasi, presisi, recall, dan F1-score.
   * Heatmap *Confusion Matrix* gabungan kedua algoritma.
   * Galeri *WordCloud* resolusi tinggi untuk masing-masing kelas sentimen (Positif = hijau, Negatif = merah, Netral = biru).
3. **Tab 3 — 📋 Data Komentar**:
   * Tabel interaktif yang menampilkan transparansi data: **Komentar Mentah → Komentar Bersih → Label Sentimen → Likes → Tanggal**.
   * Fitur pencarian kata kunci (*live search filter*).
   * Fitur filter dropdown sentimen (Semua, Positif, Negatif, Netral).
   * Tombol *Download Dataset* (format CSV dengan encoding UTF-8 BOM).
4. **Tab 4 — 🔮 Prediksi Sentimen Interaktif**:
   * Area uji coba teks komentar baru secara *real-time*.
   * Disediakan tombol jalan pintas (*quick test preset*) dengan berbagai contoh komentar berita ekonomi.
   * Hasil prediksi dari 3 mesin: **InSet Lexicon** (aturan per klausa yang sama dengan `main.py`, beserta skornya), **Naïve Bayes**, dan **Random Forest**.
   * Transparansi rincian teks setelah melalui *preprocessing*.
   * Diagram batang probabilitas keyakinan model (*confidence score*).

---

## 💡 Kegunaan & Manfaat Sistem

1. **Bagi Akademisi & Peneliti NLP**:
   * Menjadi rujukan implementasi komparatif algoritma Naive Bayes dan Random Forest pada korpus opini media sosial Bahasa Indonesia.
   * Membuktikan secara empiris keunggulan pelabelan *pra-stemming* dan efektivitas *negation handling* leksikon.
2. **Bagi Pengambil Kebijakan (Pemerintah & Bank Indonesia)**:
   * Menyediakan instrumen *social listening* objektif untuk memantau keresahan masyarakat terhadap pelemahan nilai tukar Rupiah.
   * Membantu memetakan poin-poin krusial keluhan publik (daya beli, sembako, lapangan kerja) guna merumuskan komunikasi kebijakan yang tepat sasaran.
3. **Bagi Pelaku Usaha & Masyarakat**:
   * Memberikan gambaran iklim sentimen pasar dan respon masyarakat luas terhadap stabilitas moneter nasional.

---

## 📁 Struktur Direktori Proyek

```
Jurnal - Program/
├── data/
│   ├── raw/
│   │   ├── daftar_video.txt        # Daftar URL video sumber
│   │   └── komentar_raw.csv        # Dataset mentah hasil scraping YouTube (5.733 baris)
│   ├── processed/
│   │   ├── komentar_clean.csv      # Dataset bersih hasil praproses & pelabelan (5.530 baris)
│   │   └── split_ids.csv           # Split latih/uji yang dibekukan
│   ├── annotation/
│   │   └── anotasi_sampel_blind.csv# 400 sampel acak blind untuk uji Cohen's Kappa
│   └── lexicon/
│       ├── koreksi_domain.csv      # Koreksi kamus domain (kata, bobot, kategori, alasan)
│       ├── inset_positif.txt       # Kamus InSet resmi kata positif + bobot
│       ├── inset_positif.tsv       # Format TSV kata positif
│       ├── inset_negatif.txt       # Kamus InSet resmi kata negatif + bobot
│       └── inset_negatif.tsv       # Format TSV kata negatif
├── output/
│   ├── model_nbc.pkl               # Model Naive Bayes tersimpan
│   ├── model_rfc.pkl               # Model Random Forest tersimpan
│   ├── tfidf_vectorizer.pkl        # Model pembobot TF-IDF tersimpan
│   ├── hasil_ringkasan.json        # Ringkasan terstruktur seluruh metrik sistem
│   ├── keyakinan_ringkas.csv       # Data kalibrasi & confidence model
│   ├── proba_data_uji.csv          # Prediksi probabilitas detail data uji
│   ├── distribusi_label.png        # Grafik pie chart distribusi sentimen
│   ├── perbandingan_performa.png   # Grafik bar perbandingan NBC vs RFC
│   ├── confusion_matrix.png        # Grafik heatmap confusion matrix gabungan
│   ├── confusion_matrix_nbc.png    # Confusion matrix Naive Bayes
│   ├── confusion_matrix_rfc.png    # Confusion matrix Random Forest
│   ├── wordcloud_positif.png       # Awan kata sentimen positif
│   ├── wordcloud_negatif.png       # Awan kata sentimen negatif
│   └── wordcloud_netral.png        # Awan kata sentimen netral
├── proposal/
│   ├── Sempro-final.docx           # Naskah proposal penelitian lengkap
│   └── jurnal-sitasi/              # Kumpulan PDF referensi jurnal rujukan
├── scraping_komentar.py            # Pengumpulan komentar (youtube-comment-downloader)
├── ambil_metadata_video.py         # Metadata video (kanal, tanggal unggah) via yt-dlp
├── sentimen_inset.py               # Modul bersama: praproses & pelabelan InSet
├── main.py                         # Skrip pipeline utama (Preprocessing s.d. Evaluasi & Export)
├── dashboard.py                    # Aplikasi web dashboard interaktif Streamlit
├── requirements.txt                # Daftar dependensi pustaka Python
└── README.md                       # Dokumentasi lengkap sistem
```

---

## 🚀 Panduan Instalasi & Penggunaan

### 1. Prasyarat Sistem
* Komputer dengan OS Windows, macOS, atau Linux.
* Python versi 3.10 atau yang lebih baru.

### 2. Kloning & Pengaturan Virtual Environment
Buka terminal (PowerShell / Command Prompt) di direktori proyek:
```bash
# Buat virtual environment (jika belum ada)
python -m venv .venv

# Aktivasi virtual environment
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Windows CMD:
.\.venv\Scripts\activate.bat
# Linux / macOS:
source .venv/bin/activate
```

### 3. Instalasi Dependensi
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. (Opsional) Mengumpulkan Ulang Komentar
```bash
python scraping_komentar.py          # membaca data/raw/daftar_video.txt
```
*Mengubah data mentah berarti split dan sampel anotasi dibuat ulang oleh `main.py`.*

### 5. Menjalankan Pipeline Utama (`main.py`)
Untuk mengeksekusi pembersihan teks, pelabelan InSet resmi, ekstraksi fitur TF-IDF, pelatihan model NBC & RFC, validasi 5-Fold CV, uji McNemar, serta pembuatan seluruh grafik visualisasi:
```bash
python main.py
```
*Waktu eksekusi sekitar 1–2 menit (sebagian besar untuk validasi silang dan bootstrap).*

### 6. Menjalankan Dashboard Streamlit (`dashboard.py`)
Untuk membuka antarmuka visual interaktif:
```bash
streamlit run dashboard.py
```
Aplikasi akan otomatis terbuka pada peramban web: `http://localhost:8501`.

---

## 📚 Referensi Ilmiah

1. **Koto, F., & Rahmaningtyas, G. Y. (2017).** *InSet Lexicon: Evaluation of a Word List for Indonesian Sentiment Analysis in Microblogs*. International Conference on Asian Language Processing (IALP), 391–394.
2. **Okta, A. K. A., Sanjaya, A., & Setiawan, A. B. (2025).** *Penerapan Inset Lexicon untuk Analisis Sentimen Penonton Video JKT48 di YouTube*. Prosiding Seminar Nasional Mahasiswa Bidang Ilmu Komputer dan Aplikasinya (SENAMIKA), 6(2), 1276–1283.
3. **Juliandewi, K. A., dkk. (2025).** *Comparison of Naïve Bayes and Random Forest in Sentiment Analysis of State-Owned Banks Management by Danantara on X and YouTube*. Jurnal Nasional Komputasi dan Teknologi Informasi, 8(3), 527–537.
4. **Latifah, K., dkk. (2026).** *Evaluasi Efektivitas Pelabelan Otomatis InSet Lexicon dan TF-IDF pada Pemodelan Teks Bahasa Indonesia*. Jurnal Teknologi Sistem Informasi, 15(1), 105–113.
5. **Aisy, S. R., & Prasetiyo, B. (2023).** *Sentiment Analysis of the TPKS Law on Twitter Using InSet Lexicon with Multinomial Naïve Bayes and Support Vector Machine*. Scientific Journal of Informatics, 10(4), 415–424.

---

> **Dokumen Penelitian**: Tugas Proposal Skripsi — Program Studi Teknik Informatika S1  
> **Topik**: Analisis Sentimen Komentar YouTube Mengenai Pelemahan Nilai Tukar Rupiah
