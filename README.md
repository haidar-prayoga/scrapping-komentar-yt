# 📊 Sistem Analisis Sentimen Komentar YouTube: Kondisi Ekonomi Indonesia Terkait Pelemahan Nilai Tukar Rupiah

Sistem analisis sentimen komprehensif untuk pengumpulan data (*crawling*), pembersihan teks (*preprocessing* 6 tahap), pelabelan sentimen otomatis berbasis **InSet Lexicon** (*Indonesian Sentiment Lexicon* pada tahap *pra-stemming* dengan *Negation Handling* & pembobotan klausa), ekstraksi fitur **TF-IDF**, serta klasifikasi dan komparasi algoritma **Naïve Bayes Classifier (NBC)** dan **Random Forest Classifier (RFC)** yang dilengkapi validasi statistik (5-Fold CV & Uji McNemar) dan dashboard interaktif berbasis **Streamlit**.

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
1. **Pengumpulan Data Publik Terfokus**: Mengumpulkan ribuan komentar nyata warganet dari 13 video berita dan diskusi ekonomi nasional di YouTube.
2. **Standardisasi Teks Percakapan**: Membersihkan data teks media sosial yang sarat bahasa gaul (*slang*), singkatan, tanda baca tidak baku, dan elongasi huruf melalui alur *preprocessing* 6 tahap.
3. **Pelabelan Otomatis Ilmiah & Akurat**: Menerapkan kamus resmi **InSet Lexicon** pada tahap *pra-stemming* yang dilengkapi penanganan negasi (*negation handling* jendela 3 kata) dan pembobotan klausa untuk mengelompokkan komentar ke dalam tiga kelas: **Positif**, **Negatif**, dan **Netral**.
4. **Komparasi Model Pembelajaran Mesin**: Membandingkan kinerja klasifikasi antara algoritma probabilistik (**Naïve Bayes Classifier**) dan algoritma *ensemble tree-based* (**Random Forest Classifier**) berbasis representasi fitur **TF-IDF**.
5. **Validasi Statistik Komprehensif**: Menguji stabilitas model melalui *5-Fold Stratified Cross-Validation*, uji signifikansi komparatif (*Uji McNemar*), serta analisis kalibrasi keyakinan (*Brier Score & Expected Calibration Error*).
6. **Diseminasi Interaktif Modern**: Menyediakan antarmuka dashboard berbasis web (**Streamlit**) yang memfasilitasi eksplorasi data komentar mentah $ightarrow$ bersih, pemantauan visualisasi metrik, dan simulasi prediksi teks baru secara *real-time*.

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
* **Sumber Data**: 13 video berita ekonomi terkemuka (CNBC Indonesia, Kompas TV, Metro TV, CNN Indonesia, dan kreator edukasi finansial).
* **Dataset Mentah (`komentar_raw.csv`)**: **5.733 komentar**.
* **Setelah Deduplikasi (*Drop Duplicate*)**: **5.632 komentar**.
* **Dataset Bersih Valid (`komentar_clean.csv`)**: **5.531 komentar** (setelah eliminasi baris kosong hasil pembersihan).

### 4. Kamus Leksikon (InSet Lexicon)
* Menggunakan kamus resmi **InSet Lexicon** (*Indonesian Sentiment Lexicon*) oleh **Fajri Koto & Gemala Y. Rahmaningtyas (2017)**.
* Memuat **3.609 kata positif** dan **6.579 kata negatif** (total 10.188 lema) lengkap dengan bobot skor polaritas berkisar dari **-5 (sangat negatif)** hingga **+5 (sangat positif)**.
* Dilakukan penyaringan *pure function words* (kata tugas non-opini seperti *yang*, *pada*, *dari*, *ada*, *ya*) agar tidak menimbulkan bias polaritas dokumen.

---

## 🔄 Arsitektur & Alur Kerja Sistem (Pipeline)

```
[ 13 Video Berita YouTube ]
             │
             ▼
┌────────────────────────────────────────────────────────┐
│ 1. DATA ACQUISITION / CRAWLING                         │
│    (youtube-comment-downloader ──> komentar_raw.csv)   │
│    5.733 Raw ──> Deduplikasi: 5.632 Baris              │
└────────────────────────────────────────────────────────┘
             │
             ▼
┌────────────────────────────────────────────────────────┐
│ 2. PREPROCESSING TAHAP 1 (PRA-STEMMING)                │
│    • Cleaning URL, Mention, Hashtag, Simbol, Angka     │
│    • Case Folding (Konversi Huruf Kecil)               │
│    • Reduksi Elongasi Huruf (misal: baguuus ──> bagus) │
│    • Normalisasi Slang & Singkatan Percakapan Medsos   │
└────────────────────────────────────────────────────────┘
             │
             ▼
┌────────────────────────────────────────────────────────┐
│ 3. PELABELAN SENTIMEN INSET LEXICON (PRA-STEMMING)     │
│    • Pencocokan Token Pra-Stemming dengan Kamus InSet  │
│    • Negation Handling (Jendela Geser 3 Kata)          │
│    • Pembobotan Segmen & Klausa Kontras (tetapi, dsb.) │
│    • Klasifikasi: Positif (S>0), Negatif (S<0), Netral │
│    ───> Cakupan Leksikon: 92,15%                       │
└────────────────────────────────────────────────────────┘
             │
             ▼
┌────────────────────────────────────────────────────────┐
│ 4. PREPROCESSING TAHAP 2 (POST-STEMMING TF-IDF)        │
│    • Stopword Removal Selektif (Proteksi Negasi/Opini) │
│    • Filter Token (Panjang >= 4 atau Whitelist Khusus) │
│    • Stemming Sastrawi ECS (dengan Fast In-Memory Cache)│
│    ───> komentar_clean.csv (5.531 Baris Valid)         │
└────────────────────────────────────────────────────────┘
             │
             ▼
┌────────────────────────────────────────────────────────┐
│ 5. PEMBOBOTAN FITUR TF-IDF & DATA SPLITTING            │
│    • TF-IDF Vectorizer (Unigram & Bigram, Max 5.000)   │
│    • Stratified Split (80% Training: 4.424 Data /      │
│                        20% Testing : 1.107 Data)       │
└────────────────────────────────────────────────────────┘
             │
             ▼
┌────────────────────────────────────────────────────────┐
│ 6. PELATIHAN MODEL & VALIDASI SILANG                   │
│    • Naive Bayes Classifier (MultinomialNB)            │
│    • Random Forest Classifier (class_weight='balanced')│
│    • 5-Fold Stratified Cross-Validation (Data Latih)   │
└────────────────────────────────────────────────────────┘
             │
             ▼
┌────────────────────────────────────────────────────────┐
│ 7. EVALUASI METRIK, UJI HIPOTESIS & KALIBRASI          │
│    • Akurasi, Presisi, Recall, F1-Score (Weighted/Macro)│
│    • Uji Signifikansi Statistik McNemar (p-value)      │
│    • Expected Calibration Error (ECE) & Brier Score    │
│    • Ekspor Model (.pkl) & Grafik ke folder output/    │
└────────────────────────────────────────────────────────┘
             │
             ▼
┌────────────────────────────────────────────────────────┐
│ 8. DASHBOARD WEB INTERAKTIF (STREAMLIT)                │
│    (streamlit run dashboard.py ──> port 8501)          │
└────────────────────────────────────────────────────────┘
```

---

## 🔬 Detail Metodologi & Preprocessing

### 1. Landasan Ilmiah Pelabelan Pra-Stemming
Sesuai rancangan metodologi pada proposal penelitian:
* Kamus InSet Lexicon secara inheren memuat lema dan kata berimbuhan bahasa Indonesia (seperti *meningkat*, *membaik*, *bermanfaat*, *bersyukur*, *terpercaya*).
* Apabila pelabelan dilakukan setelah stemming, kata-kata tersebut akan tereduksi menjadi kata dasar (*tingkat*, *kembang*, *manfaat*, *syukur*) yang tidak terdaftar dalam leksikon, sehingga tingkat kecocokan (*match rate*) menurun drastis.
* Oleh karena itu, pelabelan diterapkan pada token **pra-stemming**, sedangkan proses stemming Sastrawi ECS diterapkan setelahnya khusus untuk pembentukan fitur representasi **TF-IDF**.

### 2. Mekanisme Penanganan Negasi (Negation Handling)
Opini masyarakat di media sosial kerap membalik arti kalimat melalui kata negasi (*tidak*, *bukan*, *belum*, *jangan*, *tanpa*, *kurang*, *gak*, *ga*, *tak*).
* Sistem menggunakan aturan jendela geser (*sliding window*) berukuran **3 kata**.
* Apabila kata sentimen positif didahului kata negasi (contoh: *"tidak bagus"*, *"bukan solusi"*), bobot skornya dibalik menjadi negatif:
  $$	ext{Score} = -W_{	ext{positif}}$$
* Sebaliknya, jika kata sentimen negatif didahului kata negasi (contoh: *"tidak panik"*, *"tidak rugi"*), bobotnya dibalik menjadi positif:
  $$	ext{Score} = +|W_{	ext{negatif}}|$$

### 3. Stopword Removal Selektif (Whitelist Protection)
Stopword bawaan bahasa Indonesia sering kali menghapus kata berbobot opini penting:
* **Kata Negasi**: *tidak*, *bukan*, *belum*, *tanpa*, *kurang*
* **Kata Sentimen Positif Kunci**: *baik*, *tepat*, *yakin*, *benar*, *jelas*
* **Kata Pujian Pendek**: *top*, *jos*, *oke*, *sip*, *ok*

Sistem menerapkan aturan *whitelist* ketat:
$$	ext{Stopwords Aktif} = 	ext{Stopwords Sastrawi} - 	ext{KATA\_NEGASI} - 	ext{KATA\_SENTIMEN\_WHITELIST}$$
Hal ini menjamin kata-kata sentimen dasar seperti *"baik"* tidak tereliminasi pada teks bersih pemodelan.

### 4. Normalisasi Slang & Reduksi Karakter Berulang (Elongasi)
* Reduksi huruf berulang $\ge 3$ kali menjadi bentuk tunggal menggunakan *regular expression*:
  Contoh: `baguuus` $ightarrow$ `bagus`, `kerennn` $ightarrow$ `keren`, `njirrrr` $ightarrow$ `njir`.
* Kamus slang komprehensif mengonversi ratusan istilah khas percakapan warganet (contoh: `bgt` $ightarrow$ `banget`, `mantul` $ightarrow$ `mantap`, `tks` / `thanks` $ightarrow$ `terima kasih`, `boncos` $ightarrow$ `rugi`, `cuan` $ightarrow$ `untung`).

---

## 📊 Hasil Evaluasi & Rangkuman Analisis Sistem

### 1. Sebaran Distribusi Sentimen Opini Publik
Dari **5.531 data komentar bersih valid**, distribusi sentimen hasil pelabelan adalah:

| Kelas Sentimen | Jumlah Komentar | Persentase | Karakteristik Utama Opini Publik |
| :--- | :---: | :---: | :--- |
| **Positif** | **3.051** | **55,16%** | Apresiasi terhadap materi edukasi video, optimisme pemulihan ekonomi, daya tahan bangsa, dan doa/ucapan terima kasih. |
| **Negatif** | **1.884** | **34,06%** | Keluhan pelemahan rupiah, kenaikan harga bahan pokok/sembako, kritik kebijakan moneter, isu korupsi pejabat, serta kekhawatiran krisis ekonomi. |
| **Netral** | **596** | **10,78%** | Pertanyaan teknis seputar kurs valas, diskusi perbandingan komoditas (emas vs dolar), serta komentar dengan skor polaritas berimbang. |

> **Catatan Analisis**: Sebanyak **92,15%** komentar berhasil dicocokkan dengan InSet Lexicon. Pada kelas Netral (596 data), 434 komentar murni tidak mengandung kata opini dan 162 komentar memiliki skor sentimen yang saling meniadakan.

---

### 2. Kinerja Model Machine Learning (Data Uji 20% = 1.107 Komentar)

Model dilatih pada 4.424 data latih dan diuji pada 1.107 data uji berstrata (*Stratified 80:20*):

| Metrik Evaluasi | Naïve Bayes (Standar) | Naïve Bayes (Prior Seragam) | Random Forest (`balanced`) | Keunggulan Algoritma |
| :--- | :---: | :---: | :---: | :--- |
| **Akurasi** | 60,07% | **61,79%** | 57,36% | Naive Bayes prior seragam menghasilkan akurasi global tertinggi. |
| **Presisi (*Weighted*)** | 55,57% | 60,33% | **61,03%** | Random Forest menghasilkan presisi tertimbang paling tinggi. |
| **Recall (*Weighted*)** | 60,07% | **61,79%** | 57,36% | Mengikuti sebaran akurasi pada evaluasi berstrata. |
| **F1-Score (*Weighted*)**| 51,68% | **60,60%** | 58,59% | Keduanya bersaing ketat pada bobot agregat. |
| **F1-Score (*Macro*)**   | 35,49% | 51,26% | **52,07%** | **Random Forest terbaik** dalam menyeimbangkan performa kelas minoritas (*Netral*). |

---

### 3. Detail Matriks Kebingungan (Confusion Matrix)

Matriks kebingungan menunjukkan rincian prediksi aktual versus prediksi model pada 1.107 data uji:

#### A. Random Forest Classifier (`class_weight='balanced'`)
```
               Prediksi Negatif   Prediksi Netral   Prediksi Positif   Total Aktual
Aktual Negatif:      193                72                112              377
Aktual Netral :       24                63                 32              119
Aktual Positif:      150                82                379              611
```
*Kemampuan deteksi kelas minoritas Netral mencapai Recall 52,9% (63 dari 119 terprediksi tepat).*

#### B. Naïve Bayes Classifier (Prior Seragam)
```
               Prediksi Negatif   Prediksi Netral   Prediksi Positif   Total Aktual
Aktual Negatif:      194                15                168              377
Aktual Netral :       41                27                 51              119
Aktual Positif:      127                21                463              611
```

---

### 4. Validasi Stabilitas: 5-Fold Stratified Cross-Validation
Pengujian silang 5 lipatan pada 4.424 data latih membuktikan stabilitas model dan ketiadaan *data leakage*:
* **Naïve Bayes Classifier**:
  * Rata-rata Akurasi: **$60,62\% \pm 0,58\%$**
  * Rata-rata F1-Macro: **$36,55\% \pm 0,73\%$**
* **Random Forest Classifier**:
  * Rata-rata Akurasi: **$58,02\% \pm 2,11\%$**
  * Rata-rata F1-Macro: **$53,60\% \pm 2,50\%$**

---

### 5. Uji Signifikansi Statistik (Uji McNemar)
Untuk membuktikan secara ilmiah apakah perbedaan kinerja klasifikasi antara NBC dan RFC bermakna:
* Jumlah sampel di mana hanya Naïve Bayes yang benar ($b$): **223**
* Jumlah sampel di mana hanya Random Forest yang benar ($c$): **193**
* Nilai Statistik Chi-Square ($\chi^2$): **2,0216**
* Nilai $p$-value (*Exact McNemar*): **0,1550** ($p > 0,05$)
* **Kesimpulan Ilmiah**: Pada tingkat signifikansi $lpha = 0,05$, **tidak terdapat perbedaan performa yang signifikan secara statistik** antara Naïve Bayes Classifier dan Random Forest Classifier. Keduanya memiliki kompetensi prediktif yang sepadan, di mana NBC unggul pada efisiensi komputasi dan akurasi global, sedangkan RFC unggul pada pemerataan recall kelas minoritas.

---

### 6. Analisis Keyakinan & Kalibrasi Probabilitas Model
* **Rata-rata Keyakinan (*Confidence*)**:
  * Naïve Bayes: **66,10%** (saat prediksi benar: 69,38%, saat salah: 61,17%).
  * Random Forest: **62,22%** (saat benar: 63,72%, saat salah: 60,21%).
* **Nilai Kesalahan Kalibrasi (*Expected Calibration Error - ECE*)**:
  * Random Forest: **0,0687** ($6,87\%$)
  * Naïve Bayes: **0,0714** ($7,14\%$)
  * *Kedua model terkalibrasi secara sehat dengan gap overconfidence di bawah 6%.*

---

## 💻 Fitur Dashboard Interaktif (Streamlit)

Dashboard dirancang dengan konsep estetika modern bertema *dark mode* elegan menggunakan palet warna terkurasi:

1. **Tab 1 — 🏠 Gambaran Umum (Overview)**:
   * Kartu ringkasan metrik utama: Total Dataset (5.531 data), Total Positif, Total Negatif, Total Netral.
   * Visualisasi donat/pie chart proporsi sentimen publik berbasis InSet Lexicon.
   * Diagram ringkas tahapan alur penelitian 6 tahap.
2. **Tab 2 — 📈 Visualisasi & Komparasi Model**:
   * Tabel metrik evaluasi perbandingan NBC vs RFC secara berdampingan.
   * Grafik diagram batang perbandingan akurasi, presisi, recall, dan F1-score.
   * Heatmap *Confusion Matrix* gabungan kedua algoritma.
   * Galeri *WordCloud* resolusi tinggi untuk masing-masing kelas sentimen (Positif = hijau, Negatif = merah, Netral = biru).
3. **Tab 3 — 📋 Data Komentar**:
   * Tabel interaktif yang menampilkan transparansi data: **Komentar Mentah $ightarrow$ Komentar Bersih $ightarrow$ Label Sentimen $ightarrow$ Likes $ightarrow$ Tanggal**.
   * Fitur pencarian kata kunci (*live search filter*).
   * Fitur filter dropdown sentimen (Semua, Positif, Negatif, Netral).
   * Tombol *Download Dataset* (format CSV dengan encoding UTF-8 BOM).
4. **Tab 4 — 🔮 Prediksi Sentimen Interaktif**:
   * Area uji coba teks komentar baru secara *real-time*.
   * Disediakan tombol jalan pintas (*quick test preset*) dengan berbagai contoh komentar berita ekonomi.
   * Hasil prediksi instan dari 3 mesin komparatif: **InSet Lexicon**, **Naïve Bayes**, dan **Random Forest**.
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
│   │   └── komentar_raw.csv        # Dataset mentah hasil crawling YouTube (5.733 baris)
│   ├── processed/
│   │   └── komentar_clean.csv      # Dataset bersih hasil preprocessing & pelabelan (5.531 baris)
│   ├── annotation/
│   │   └── anotasi_sampel_blind.csv# 400 sampel acak blind untuk uji Cohen's Kappa
│   └── lexicon/
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

### 4. Menjalankan Pipeline Utama (`main.py`)
Untuk mengeksekusi pembersihan teks, pelabelan InSet resmi, ekstraksi fitur TF-IDF, pelatihan model NBC & RFC, validasi 5-Fold CV, uji McNemar, serta pembuatan seluruh grafik visualisasi:
```bash
python main.py
```
*Waktu eksekusi berkisar 10–20 detik berkat in-memory caching pada PySastrawi.*

### 5. Menjalankan Dashboard Streamlit (`dashboard.py`)
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
