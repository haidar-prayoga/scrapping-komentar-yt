"""
Script: gabung_anotasi.py
Deskripsi:
  1. Mengambil label dari data/annotation/anotasi_anotator2.xlsx dan memasukkannya
     ke kolom anotator_2 pada data/annotation/anotasi_manual.csv.
  2. Menjaga kolom anotator_1 di anotasi_manual.csv apa adanya.
  3. Menghitung ulang kesepakatan (Cohen's Kappa & akurasi kesepakatan langsung).
  4. Menyaring komentar yang labelnya berbeda (anotator_1 != anotator_2) ke dalam
     data/annotation/perlu_diskusi.csv untuk diisi label_konsensus hasil diskusi.
  5. Opsi '--cek':
     - Memvalidasi bahwa seluruh label_konsensus di perlu_diskusi.csv telah terisi.
     - Menyinkronkan kembali label_konsensus tersebut ke anotasi_manual.csv.
     - Memastikan dataset anotasi manual 100% lengkap dan valid sebelum main.py dijalankan.
"""

import os
import sys
import argparse
import pandas as pd
import openpyxl
from sklearn.metrics import cohen_kappa_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DIR_ANOTASI = os.path.join(BASE_DIR, 'data', 'annotation')
XLSX_A2 = os.path.join(DIR_ANOTASI, 'anotasi_anotator2.xlsx')
CSV_MANUAL = os.path.join(DIR_ANOTASI, 'anotasi_manual.csv')
CSV_DISKUSI = os.path.join(DIR_ANOTASI, 'perlu_diskusi.csv')
ROOT_DISKUSI = os.path.join(BASE_DIR, 'perlu_diskusi.csv')

VALID_LABELS = {'positif', 'negatif', 'netral'}


def muat_anotator2_xlsx():
    if not os.path.exists(XLSX_A2):
        sys.exit(f"[ERROR] File {XLSX_A2} tidak ditemukan.")
    
    wb = openpyxl.load_workbook(XLSX_A2)
    sheet_name = 'Anotasi' if 'Anotasi' in wb.sheetnames else wb.sheetnames[0]
    ws = wb[sheet_name]
    
    data = []
    for row in ws.iter_rows(min_row=2, max_col=5, values_only=True):
        if row[1] is not None: # id_komentar
            id_k = int(row[1])
            lbl = str(row[3]).strip().lower() if row[3] is not None else None
            catatan = str(row[4]).strip() if row[4] is not None else ''
            data.append({'id_komentar': id_k, 'label_a2': lbl, 'catatan_a2': catatan})
            
    df_a2 = pd.DataFrame(data).set_index('id_komentar')
    return df_a2


def gabung_dan_ekspor_diskusi():
    print("=" * 70)
    print("GABUNG ANOTASI: Update Anotator 2 & Ekspor Perlu Diskusi")
    print("=" * 70)
    
    if not os.path.exists(CSV_MANUAL):
        sys.exit(f"[ERROR] File {CSV_MANUAL} tidak ditemukan.")
        
    df_manual = pd.read_csv(CSV_MANUAL)
    df_a2 = muat_anotator2_xlsx()
    
    # Update kolom anotator_2 berdasarkan anotasi_anotator2.xlsx
    cocok = 0
    for idx, row in df_manual.iterrows():
        id_k = row['id_komentar']
        if id_k in df_a2.index:
            val_a2 = df_a2.loc[id_k, 'label_a2']
            if val_a2 in VALID_LABELS:
                df_manual.at[idx, 'anotator_2'] = val_a2
                cocok += 1
                
    print(f"[OK] Berhasil memperbarui {cocok} label anotator_2 dari {XLSX_A2}")
    
    # Pastikan format teks dan label bersih
    for col in ('anotator_1', 'anotator_2'):
        df_manual[col] = df_manual[col].astype(str).str.strip().str.lower()
        
    # Hitung Cohen's Kappa
    a1 = df_manual['anotator_1']
    a2 = df_manual['anotator_2']
    kappa = cohen_kappa_score(a1, a2)
    sepakat = (a1 == a2).mean() * 100
    beda_mask = (a1 != a2)
    n_beda = int(beda_mask.sum())
    
    print(f"\n--- Evaluasi Inter-Annotator Agreement ---")
    print(f"Total Sampel Komentar       : {len(df_manual)}")
    print(f"Jumlah Komentar Sepakat     : {len(df_manual) - n_beda} ({sepakat:.2f}%)")
    print(f"Jumlah Perlu Diskusi (Beda) : {n_beda} ({100 - sepakat:.2f}%)")
    print(f"Cohen's Kappa Antar-Anotator: {kappa:.4f}")
    
    # Ekspor komentar yang berbeda
    df_beda = df_manual[beda_mask].copy()
    
    # Jika anotator_1 == anotator_2, konsensus otomatis sama
    # Jika beda, biarkan label_konsensus yang sudah ada (atau kosongkan jika belum ada konsensus)
    for idx in df_manual[~beda_mask].index:
        df_manual.at[idx, 'label_konsensus'] = df_manual.at[idx, 'anotator_1']
        
    cols_export = ['id_komentar', 'komentar', 'anotator_1', 'anotator_2', 'label_konsensus', 'target_opini']
    df_beda[cols_export].to_csv(CSV_DISKUSI, index=False, encoding='utf-8-sig')
    df_beda[cols_export].to_csv(ROOT_DISKUSI, index=False, encoding='utf-8-sig')
    
    # Simpan kembali anotasi_manual.csv
    df_manual.to_csv(CSV_MANUAL, index=False, encoding='utf-8-sig')
    print(f"\n[OK] File beda anotasi disimpan ke:")
    print(f"     -> {CSV_DISKUSI}")
    print(f"     -> {ROOT_DISKUSI}")
    print(f"[OK] File utama diperbarui: {CSV_MANUAL}")


def cek_dan_terapkan_konsensus():
    print("=" * 70)
    print("GABUNG ANOTASI --CEK: Validasi & Penerapan Konsensus")
    print("=" * 70)
    
    file_diskusi = CSV_DISKUSI if os.path.exists(CSV_DISKUSI) else ROOT_DISKUSI
    if not os.path.exists(file_diskusi):
        sys.exit(f"[ERROR] File {file_diskusi} tidak ditemukan. Jalankan tanpa --cek dulu.")
        
    df_diskusi = pd.read_csv(file_diskusi)
    df_manual = pd.read_csv(CSV_MANUAL).set_index('id_komentar')
    
    # Cek kelengkapan label_konsensus
    missing = df_diskusi['label_konsensus'].isna() | (df_diskusi['label_konsensus'].astype(str).str.strip() == '')
    if missing.any():
        print(f"[PERINGATAN] Ada {missing.sum()} baris di perlu_diskusi.csv yang belum diisi label_konsensus-nya!")
        print(df_diskusi[missing][['id_komentar', 'komentar', 'anotator_1', 'anotator_2']].head(5))
        sys.exit(1)
        
    df_diskusi['label_konsensus'] = df_diskusi['label_konsensus'].astype(str).str.strip().str.lower()
    invalids = set(df_diskusi['label_konsensus']) - VALID_LABELS
    if invalids:
        print(f"[ERROR] Ditemukan label konsensus tidak valid: {invalids}")
        print("Label harus salah satu dari: positif, negatif, atau netral.")
        sys.exit(1)
        
    # Terapkan konsensus ke df_manual
    for _, r in df_diskusi.iterrows():
        id_k = r['id_komentar']
        if id_k in df_manual.index:
            df_manual.loc[id_k, 'label_konsensus'] = r['label_konsensus']
            if 'target_opini' in r and pd.notna(r['target_opini']):
                df_manual.loc[id_k, 'target_opini'] = str(r['target_opini']).strip().lower()
                
    # Pastikan baris yang sepakat tetap memakai nilai kesepakatan
    sepakat_mask = df_manual['anotator_1'] == df_manual['anotator_2']
    df_manual.loc[sepakat_mask, 'label_konsensus'] = df_manual.loc[sepakat_mask, 'anotator_1']
    
    df_out = df_manual.reset_index()
    df_out.to_csv(CSV_MANUAL, index=False, encoding='utf-8-sig')
    
    # Hitung metrik final
    kappa = cohen_kappa_score(df_out['anotator_1'], df_out['anotator_2'])
    dist_k = df_out['label_konsensus'].value_counts()
    dist_tg = df_out['target_opini'].value_counts()
    
    print("[SUKSES] Seluruh label konsensus valid dan telah disinkronkan ke anotasi_manual.csv!")
    print(f"Cohen's Kappa Antar-Anotator: {kappa:.4f}")
    print("\nDistribusi Label Konsensus Final:")
    for k, v in dist_k.items():
        print(f"  {k:<8}: {v:>4} ({v/len(df_out)*100:.2f}%)")
    print("\nDistribusi Target Opini:")
    for k, v in dist_tg.items():
        print(f"  {k:<8}: {v:>4} ({v/len(df_out)*100:.2f}%)")
    print("\nSekarang silakan jalankan: python main.py")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Penggabungan & Evaluasi Anotasi Multi-Anotator")
    parser.add_argument('--cek', action='store_true', help="Validasi dan sinkronisasi hasil diskusi konsensus")
    args = parser.parse_args()
    
    if args.cek:
        cek_dan_terapkan_konsensus()
    else:
        gabung_dan_ekspor_diskusi()

