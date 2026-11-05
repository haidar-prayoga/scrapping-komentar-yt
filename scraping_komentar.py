"""
=============================================================================
  PENGUMPULAN DATA: SCRAPING KOMENTAR YOUTUBE
  Pustaka : youtube-comment-downloader (tanpa API key)
  Masukan : data/raw/daftar_video.txt  (satu URL atau ID video per baris;
            baris diawali '#' diabaikan)
  Keluaran: data/raw/komentar_raw.csv  (dibaca oleh main.py)
-----------------------------------------------------------------------------
  Kolom keluaran:
    komentar           teks komentar asli
    likes              jumlah suka (angka; "1,2 rb"/"1.2K" dikonversi)
    tanggal            waktu relatif dari YouTube, mis. "4 bulan yang lalu"
    tanggal_perkiraan  tanggal absolut perkiraan (dari time_parsed) -> YYYY-MM-DD
    video              URL video
    cid                ID komentar YouTube
    balasan            True jika komentar merupakan balasan
-----------------------------------------------------------------------------
  Cara pakai:
    pip install youtube-comment-downloader
    python scraping_komentar.py              # menimpa komentar_raw.csv
    python scraping_komentar.py --tambah     # menambahkan ke file yang ada

  PERINGATAN: menjalankan ulang scraping akan mengubah data mentah, sehingga
  split_ids.csv dan sampel anotasi otomatis dibuat ulang oleh main.py.
=============================================================================
"""
import os
import re
import sys
import time
from datetime import datetime, timezone

import pandas as pd

try:
    from youtube_comment_downloader import YoutubeCommentDownloader, SORT_BY_RECENT
except ImportError:
    sys.exit("Pasang dulu: pip install youtube-comment-downloader")

BASE = os.path.dirname(os.path.abspath(__file__))
IN_TXT = os.path.join(BASE, 'data', 'raw', 'daftar_video.txt')
OUT_CSV = os.path.join(BASE, 'data', 'raw', 'komentar_raw.csv')
JEDA_ANTAR_VIDEO = 2.0   # detik, agar tidak membebani server


def ke_url(item: str) -> str:
    item = item.strip()
    return item if item.startswith('http') else f'https://www.youtube.com/watch?v={item}'


def parse_likes(v) -> int:
    """'1,2 rb' / '1.2K' / '3' -> int."""
    if v is None:
        return 0
    s = str(v).strip().lower().replace(' ', '')
    m = re.match(r'([\d.,]+)(rb|k|jt|m)?', s)
    if not m:
        return 0
    angka = float(m.group(1).replace(',', '.'))
    kali = {'rb': 1e3, 'k': 1e3, 'jt': 1e6, 'm': 1e6}.get(m.group(2), 1)
    return int(round(angka * kali))


def main():
    if not os.path.exists(IN_TXT):
        sys.exit(f"Buat dulu {IN_TXT} (satu URL/ID video per baris)")
    with open(IN_TXT, encoding='utf-8') as f:
        daftar = [ke_url(x) for x in f if x.strip() and not x.strip().startswith('#')]

    dl = YoutubeCommentDownloader()
    rows = []
    for url in daftar:
        n0 = len(rows)
        try:
            for c in dl.get_comments_from_url(url, sort_by=SORT_BY_RECENT, language='id'):
                tp = c.get('time_parsed')
                rows.append({
                    'komentar': c.get('text', ''),
                    'likes': parse_likes(c.get('votes')),
                    'tanggal': c.get('time', ''),
                    'tanggal_perkiraan': (datetime.fromtimestamp(tp, tz=timezone.utc).date().isoformat()
                                          if tp else None),
                    'video': url,
                    'cid': c.get('cid'),
                    'balasan': bool(c.get('reply', False)),
                })
        except Exception as e:                      # video privat/dihapus/komentar dimatikan
            print(f"[GAGAL] {url}: {e}")
        print(f"[OK] {url}: {len(rows) - n0:,} komentar")
        time.sleep(JEDA_ANTAR_VIDEO)

    baru = pd.DataFrame(rows)
    if '--tambah' in sys.argv and os.path.exists(OUT_CSV):
        lama = pd.read_csv(OUT_CSV)
        baru = pd.concat([lama, baru], ignore_index=True)
        if 'cid' in baru.columns:                  # dedup hanya baris yang punya cid
            ada = baru['cid'].notna()
            baru = pd.concat([baru[~ada], baru[ada].drop_duplicates(subset='cid', keep='first')])
    baru.to_csv(OUT_CSV, index=False, encoding='utf-8-sig')
    print(f"\nTersimpan: {OUT_CSV}  ({len(baru):,} komentar dari {baru['video'].nunique()} video)")


if __name__ == '__main__':
    main()
