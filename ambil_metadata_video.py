"""
Mengambil metadata video YouTube (kanal, judul, tanggal unggah, penonton, jumlah komentar)
untuk membuktikan kriteria seleksi video (mis. "video terbaru") dan menghitung jumlah kanal.

Cara pakai:
  1. pip install yt-dlp
  2. Buat data/raw/daftar_video.txt : satu URL atau ID video per baris
  3. python ambil_metadata_video.py
  -> data/raw/metadata_video.csv  (dibaca otomatis oleh main_v4.py)

CATATAN: skrip ini belum diuji pada jaringan YouTube di lingkungan penyusun.
Jika yt-dlp gagal, isi metadata_video.csv secara manual dengan kolom:
video_id,judul,kanal,tanggal_unggah,jumlah_penonton,jumlah_komentar
"""
import os
import sys
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
IN_TXT = os.path.join(BASE, 'data', 'raw', 'daftar_video.txt')
OUT_CSV = os.path.join(BASE, 'data', 'raw', 'metadata_video.csv')

try:
    from yt_dlp import YoutubeDL
except ImportError:
    sys.exit("Pasang dulu: pip install yt-dlp")

if not os.path.exists(IN_TXT):
    sys.exit(f"Buat dulu {IN_TXT} (satu URL/ID per baris)")

with open(IN_TXT, encoding='utf-8') as f:
    daftar = [x.strip() for x in f if x.strip()]

rows = []
with YoutubeDL({'quiet': True, 'skip_download': True, 'no_warnings': True}) as ydl:
    for item in daftar:
        url = item if item.startswith('http') else f'https://www.youtube.com/watch?v={item}'
        try:
            info = ydl.extract_info(url, download=False)
        except Exception as e:                      # video privat / dihapus / jaringan
            print(f"[GAGAL] {item}: {e}")
            continue
        tgl = info.get('upload_date')               # format YYYYMMDD
        rows.append({
            'video_id': info.get('id'),
            'judul': info.get('title'),
            'kanal': info.get('channel') or info.get('uploader'),
            'tanggal_unggah': f"{tgl[:4]}-{tgl[4:6]}-{tgl[6:]}" if tgl else None,
            'jumlah_penonton': info.get('view_count'),
            'jumlah_komentar': info.get('comment_count'),
            'url': url,
        })
        print(f"[OK] {info.get('id')} | {info.get('channel')} | {tgl}")

pd.DataFrame(rows).to_csv(OUT_CSV, index=False, encoding='utf-8-sig')
print(f"\nTersimpan: {OUT_CSV}  ({len(rows)}/{len(daftar)} video)")
