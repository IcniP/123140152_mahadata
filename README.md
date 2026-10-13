# Replikasi skala kecil: segmentasi seed-based paralel di Apache Spark

Tugas Eksplorasi Mandiri Mahadata (IF25-40402), Institut Teknologi Sumatera.
Artikel acuan: Chen, F., Wang, N., Yu, B., Qin, Y., & Wang, L. (2021). *A strategy of parallel seed-based image
segmentation algorithms for handling massive image tiles over the Spark platform.* Remote Sensing, 13(10), 1969.
https://doi.org/10.3390/rs13101969

Tingkat replikasi: **parsial**. Seluruh alur enam langkah (tile -> segmentasi -> buffer otomatis dengan accumulator ->
auxiliary band tunggal -> shuffle -> border area processing -> integrasi) dan dua algoritma (region growing, watershed)
dijalankan pada citra 1024x1024 dengan Spark mode lokal, sehingga gratis dan cukup untuk laptop atau Google Colab.

## Struktur

| Berkas | Isi |
|---|---|
| `segpipe.py` | algoritma, F-measure, pipeline Spark, `LocalContext` (pengganti Spark tanpa Java untuk uji logika) |
| `run_experiments.py` | eksperimen akurasi, waktu vs core, waktu/volume shuffle 1 vs 3 band dan buffer otomatis vs manual |
| `get_data.py` | menyiapkan 3 citra RGB di `data/` (Sentinel-2 tanpa akun, atau sintetis) |
| `make_report.py` | membuat laporan PDF dari `results/` |
| `tests/test_segpipe.py` | uji logika (assert) |
| `report_config.json` | kelas, URL repositori, kuartil jurnal untuk sampul/laporan |

## Menjalankan (Google Colab gratis atau laptop dengan Java 17+)

```bash
pip install -r requirements.txt
python get_data.py --source sentinel        # atau: --source synthetic (tanpa internet)
python tests/test_segpipe.py                # uji logika, beberapa detik
python run_experiments.py --backend spark --cores 2
python make_report.py                       # -> 123140152_Muhammad-Farisi-Suyitno_Eksplorasi-Mahadata.pdf
```

Opsi: `--quick` (satu citra, hasil cepat), `--backend local` (tanpa Spark; akurasi dan volume shuffle valid,
waktu bukan waktu Spark), `--images "data/*.png"` untuk citra sendiri (RGB, minimal 1024x1024).
Waktu jalan penuh dengan region growing Python murni: sekitar 30-60 menit pada 2 core.

## Catatan reprodusibilitas

* Seed dan algoritma deterministik (tidak ada proses acak, kecuali generator citra sintetis dengan seed tetap).
* Versi yang dipakai saat pengembangan: Python 3.13, NumPy 2.5, SciPy, scikit-image 0.26, pandas 3.0, Java 21.
  Versi PySpark dan semua komponen lain dicatat otomatis di `results/meta.json` dan dimuat pada laporan.
* `get_data.py --source sentinel` belum diuji dari lingkungan pengembangan (jaringan dibatasi). Jika gagal, letakkan
  citra RGB hasil unduhan Copernicus Browser sendiri di `data/` (file PNG) dan jalankan langkah berikutnya.
* Sumber yang diadaptasi: Adams & Bischof (1994) untuk region growing, Vincent & Soille (1991) untuk watershed
  (`skimage.segmentation.watershed`).
