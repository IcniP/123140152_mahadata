"""Membuat laporan PDF dari results/*.csv dan results/meta.json.

  python make_report.py            # -> NIM_Nama_Eksplorasi-Mahadata.pdf
Edit report_config.json (kelas, URL repositori, kuartil jurnal) sebelum membuat versi akhir.
"""
import json
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

R = os.environ.get("RESULTS", "results")
CFG = {"nama": "Muhammad Farisi Suyitno", "nim": "123140152", "kelas": "[isi kelas]",
       "repo_url": "[isi URL repositori GitHub]", "akses_repo": "publik",
       "kuartil": "[isi kuartil SJR dari scimagojr.com]"}
if os.path.exists("report_config.json"):
    CFG.update(json.load(open("report_config.json")))
meta = json.load(open(f"{R}/meta.json"))
acc = pd.read_csv(f"{R}/accuracy.csv")
tim = pd.read_csv(f"{R}/timing.csv")
cores = pd.read_csv(f"{R}/cores.csv") if os.path.exists(f"{R}/cores.csv") else None
source = open("data/SOURCE.txt").read().strip() if os.path.exists("data/SOURCE.txt") else "tidak diketahui"
SPARK = meta["backend"] == "spark"
NAME = {"rg": "region growing", "ws": "watershed"}
BLUE, ORANGE, INK, INK2 = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e"


def pct(x):
    return f"{100 * x:.1f}%"


def pp(x):
    return f"{100 * x:+.1f}".replace("+-", "-")


# ------------------------------------------------------------------ gambar
plt.rcParams.update({"font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False})


def fig_acc():
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.8))
    for ax, al in zip(axs, ["rg", "ws"]):
        g = acc[acc.algo == al].groupby("tile").mean(numeric_only=True)
        x = np.arange(len(g))
        for col, c, ls, lab in [("f_border_with", BLUE, "-", "area border, dengan processing"),
                                ("f_border_without", ORANGE, "-", "area border, tanpa processing"),
                                ("f_entire_with", BLUE, "--", "seluruh citra, dengan processing"),
                                ("f_entire_without", ORANGE, "--", "seluruh citra, tanpa processing")]:
            ax.plot(x, g[col], color=c, ls=ls, lw=2, marker="o", ms=5, label=lab)
        ax.set_xticks(x, g.index)
        ax.set_xlabel("Ukuran tile (piksel)")
        ax.set_ylabel("F-measure")
        ax.set_title(NAME[al], fontsize=9, color=INK)
        ax.grid(axis="y", color="#e3e2de", lw=0.6)
    h, l = axs[0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.legend(h, l, fontsize=7, frameon=False, loc="lower center", ncol=2)
    fig.savefig(f"{R}/fig_akurasi.png", dpi=200)
    plt.close(fig)


def fig_time():
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.8))
    for ax, al in zip(axs, ["rg", "ws"]):
        g = tim[tim.algo == al]
        man = sorted(g[g["mode"] == "manual"].buffer.unique())
        cats = ["otomatis"] + [str(b) for b in man]
        x = np.arange(len(cats))
        for k, (na, c) in enumerate([(1, BLUE), (3, ORANGE)]):
            h = g[g.n_aux == na]
            v = [h[h["mode"] == "auto"].time.mean()] + [h[(h["mode"] == "manual") & (h.buffer == b)].time.mean() for b in man]
            ax.bar(x + (k - 0.5) * 0.4, v, 0.36, color=c, label=f"{na} auxiliary band")
        ax.set_xticks(x, cats)
        ax.set_xlabel("Ukuran buffer (piksel)")
        ax.set_ylabel("Waktu eksekusi (detik)")
        ax.set_title(NAME[al], fontsize=9, color=INK)
        ax.grid(axis="y", color="#e3e2de", lw=0.6)
    h, l = axs[0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.legend(h, l, fontsize=7, frameon=False, loc="lower center", ncol=2)
    fig.savefig(f"{R}/fig_waktu.png", dpi=200)
    plt.close(fig)


def fig_cores():
    g = cores.groupby("cores").time.mean()
    fig, ax = plt.subplots(figsize=(3.6, 2.4))
    ax.bar(g.index.astype(str), g.values, 0.5, color=BLUE)
    ax.set_xlabel("Jumlah core (local[n])")
    ax.set_ylabel("Waktu eksekusi (detik)")
    ax.grid(axis="y", color="#e3e2de", lw=0.6)
    fig.tight_layout()
    fig.savefig(f"{R}/fig_core.png", dpi=200)
    plt.close(fig)


fig_acc()
fig_time()
if cores is not None:
    fig_cores()

# ------------------------------------------------------------------- gaya
S = {"b": ParagraphStyle("b", fontName="Helvetica", fontSize=10, leading=14.5, alignment=TA_JUSTIFY, spaceAfter=6),
     "h1": ParagraphStyle("h1", fontName="Helvetica-Bold", fontSize=13, leading=17, spaceBefore=10, spaceAfter=6),
     "lead": ParagraphStyle("lead", fontName="Helvetica", fontSize=10, leading=14.5, spaceAfter=6, keepWithNext=1),
     "h2": ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=10.5, leading=14, spaceBefore=6, spaceAfter=4),
     "c": ParagraphStyle("c", fontName="Helvetica", fontSize=8.5, leading=11, alignment=TA_CENTER, textColor=colors.HexColor(INK2)),
     "cell": ParagraphStyle("cell", fontName="Helvetica", fontSize=8, leading=10),
     "box": ParagraphStyle("box", fontName="Helvetica", fontSize=9, leading=12.5, backColor=colors.HexColor("#fff4d6"),
                           borderColor=colors.HexColor("#eda100"), borderWidth=0.8, borderPadding=6, spaceBefore=4, spaceAfter=10),
     "ref": ParagraphStyle("ref", fontName="Helvetica", fontSize=9.5, leading=13, leftIndent=24, firstLineIndent=-24, spaceAfter=5)}
story = []


def P(t, s="b"):
    story.append(Paragraph(t, S[s]))


def bullets(items):
    for t in items:
        story.append(Paragraph("&bull;&nbsp;&nbsp;" + t, ParagraphStyle("li", parent=S["b"], leftIndent=14, firstLineIndent=-10, spaceAfter=3)))


def table(rows, widths, head=1):
    t = Table([[Paragraph(str(c), S["cell"]) for c in r] for r in rows], colWidths=[w * cm for w in widths], repeatRows=head)
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, head - 1), colors.HexColor("#ecebe7")),
                           ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#b9b8b2")), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                           ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5)]))
    story.append(t)
    story.append(Spacer(1, 6))


def fig(path, w, cap):
    story.append(KeepTogether([Image(path, width=w * cm, height=w * cm * _ratio(path)), Paragraph(cap, S["c"]), Spacer(1, 8)]))


def _ratio(p):
    from PIL import Image as I
    w, h = I.open(p).size
    return h / w


# ------------------------------------------------------------------ sampul
story += [Spacer(1, 2.5 * cm), Paragraph("LAPORAN TUGAS EKSPLORASI MANDIRI", ParagraphStyle("t", parent=S["h1"], alignment=TA_CENTER, fontSize=15)),
          Paragraph("Apache Spark dan NoSQL &mdash; Mahadata / Big Data (IF25-40402)", ParagraphStyle("t2", parent=S["b"], alignment=TA_CENTER)),
          Spacer(1, 1.2 * cm),
          Paragraph("Replikasi Parsial Strategi Paralel Segmentasi Citra Seed-Based pada Apache Spark<br/>(Chen et al., 2021)",
                    ParagraphStyle("t3", parent=S["h1"], alignment=TA_CENTER, fontSize=14, leading=19)), Spacer(1, 1.5 * cm)]
table([["Nama", CFG["nama"]], ["NIM", CFG["nim"]], ["Kelas", CFG["kelas"]],
       ["Dosen pengampu", "Meida Cahyo Untoro, S.Kom., M.Kom."], ["Tautan repositori", CFG["repo_url"]],
       ["Tanggal", "13 Oktober 2026"]], [4, 11], head=0)
story += [Spacer(1, 2 * cm), Paragraph("Program Studi Teknik Informatika, Fakultas Teknologi Industri<br/>Institut Teknologi Sumatera", S["c"]), PageBreak()]

# --------------------------------------------------------------------- A
P("A. Ringkasan Jurnal", "h1")
P("<b>Sitasi.</b> Chen, F., Wang, N., Yu, B., Qin, Y., &amp; Wang, L. (2021). A strategy of parallel seed-based image segmentation "
  "algorithms for handling massive image tiles over the Spark platform. <i>Remote Sensing, 13</i>(10), 1969. "
  "https://doi.org/10.3390/rs13101969")
P(f"<b>Reputasi.</b> Artikel terbit di <i>Remote Sensing</i> (MDPI), akses terbuka (CC BY), memiliki DOI, dan mencantumkan tanggal diterima "
  f"20 April 2021, disetujui 14 Mei 2021, serta terbit 18 Mei 2021 (memenuhi syarat 2019 ke atas). Kuartil SJR: {CFG['kuartil']}.")
P("<b>Masalah.</b> Citra penginderaan jauh berukuran sangat besar sehingga perlu diproses pada klaster. Platform big data seperti Spark hanya "
  "menyediakan sedikit algoritma pengolahan citra, sedangkan algoritma sekuensial yang dijalankan langsung pada tile menghasilkan objek "
  "tidak lengkap di tepi tile dan volume komunikasi besar pada tahap shuffle.")
P("<b>Dataset dan lingkungan artikel.</b> Sepuluh citra Sentinel-2 RGB tahun 2020 dari Copernicus Open Access Hub, masing-masing dipotong "
  "4096 x 4096 piksel, dengan tile 256 sampai 2048 piksel. Klaster memiliki tujuh node fisik. Eksperimen waktu memakai empat executor dengan empat core "
  "dan memori 2 GB per executor, dengan driver 15 GB.")
P("<b>Metode.</b> Dua algoritma berbasis seed (region growing dan watershed) dijalankan dalam enam langkah: pretreatment citra, segmentasi tile, "
  "deteksi ukuran buffer, sintesis auxiliary band, pemrosesan area border, dan integrasi tile. Dua gagasan utama menekan volume komunikasi: "
  "(1) ukuran buffer dideteksi otomatis memakai <i>accumulator</i> Spark, bukan ditebak besar; dan (2) tiga auxiliary band (garis border, kategori, arah) "
  "digabung menjadi satu band yang menggantikan band seed.")
P("<b>Hasil utama.</b> Pada tile 256 dan kepadatan seed level 1, rata-rata F-measure dengan dan tanpa pemrosesan border adalah 95,2% dan 92,8% "
  "(region growing, seluruh citra), 90,2% dan 84,4% (region growing, area border), 99,8% dan 99,1% (watershed, seluruh citra), serta 98,1% dan 91,1% "
  "(watershed, area border). Waktu eksekusi umumnya bertambah seiring ukuran buffer dan jumlah auxiliary band, dan strategi yang diusulkan berjalan "
  "sekitar dua kali lebih cepat daripada strategi tanpa pengurangan volume komunikasi.")
P("<b>Alasan memilih.</b> Artikel memuat metode yang dapat dijalankan, datanya terbuka (Sentinel-2), parameternya cukup rinci, dan konsep Spark yang dipakai "
  "(RDD, <i>lazy evaluation</i>, action, persist, accumulator, shuffle) sesuai materi kuliah. Skalanya dapat diperkecil menjadi satu mesin dengan mode lokal.")

# --------------------------------------------------------------------- B
P("B. Tahapan Implementasi dan Eksperimen", "h1")
P("B.1 Tingkat replikasi", "h2")
P("Replikasi <b>parsial</b>: seluruh alur enam langkah dan kedua algoritma direplikasi, tetapi pada citra "
  f"{meta['size']} x {meta['size']} piksel (bukan 4096 x 4096), {len(meta['images'])} citra, dan Spark mode lokal (bukan klaster tujuh node). Rasio jumlah tile per sisi dipertahankan: tile {', '.join(map(str, meta['tiles']))} pada citra {meta['size']} "
  f"setara dengan tile {', '.join(str(t * 4096 // meta['size']) for t in meta['tiles'])} pada citra 4096.")
P("B.2 Rancangan eksperimen", "h2")
table([["Unsur", "Isi"],
       ["Tujuan", "Menguji apakah pemrosesan area border memperbaiki akurasi, dan apakah satu auxiliary band serta buffer otomatis menurunkan waktu dan volume shuffle."],
       ["Data", f"{len(meta['images'])} citra RGB {meta['size']} x {meta['size']} ({source})."],
       ["Lingkungan", "Spark mode lokal pada satu mesin gratis (laptop atau Google Colab)."],
       ["Langkah", "Pretreatment, referensi sekuensial, pipeline Spark, evaluasi F-measure, pengukuran waktu."],
       ["Keluaran dibandingkan", "F-measure dengan dan tanpa border processing (seluruh citra dan area border), waktu 1 vs 3 auxiliary band, buffer otomatis vs manual, volume shuffle."]],
      [3.6, 12.4])
P("B.3 Lingkungan dan versi perangkat lunak", "h2")
env = [["Komponen", "Versi"], ["Python", meta["python"]], ["NumPy / SciPy / scikit-image", f"{meta['numpy']} / {meta['scipy']} / {meta['skimage']}"],
       ["PySpark", meta.get("pyspark", "tidak dipakai (backend LocalContext)")],
       ["Mode eksekusi", f"local[{meta['cores']}]" if SPARK else "LocalContext (tanpa Spark)"],
       ["Jumlah CPU mesin", meta["cpu_count"]], ["Tanggal eksperimen", meta["tanggal"]]]
table(env, [6, 10])
P("B.4 Langkah demi langkah", "h2")
bullets([
    "<b>Praproses.</b> RGB diubah ke abu-abu dengan Persamaan (1), lalu median filter berradius 2 piksel. Seed region growing adalah puncak lokal maksimum dan minimum pada jendela "
    "33, 27, 21, 15, dan 9 (level 1 sampai 5). Pita spektral watershed adalah gradien (<font face='Courier'>rank.gradient</font> dengan disk radius 1) dan seed-nya piksel dengan gradien "
    "&lt; 2, 3, 4, 5, dan 6. Seed diberi id unik, dan id itu menjadi label objek.",
    "<b>Referensi.</b> Segmentasi sekuensial seluruh citra dengan algoritma yang sama. Region growing memakai implementasi sendiri (antrean prioritas, mengikuti Adams &amp; Bischof, 1994); "
    "watershed memakai <font face='Courier'>skimage.segmentation.watershed</font> (Vincent &amp; Soille, 1991).",
    "<b>Tile menjadi RDD.</b> Citra dipotong menjadi tile T x T dan di-<font face='Courier'>parallelize</font> dengan satu partisi per tile.",
    "<b>Segmentasi tile dan deteksi buffer.</b> Tiap tile disegmentasi sendiri. Objek yang menyentuh sisi dalam tile dicatat sebagai objek border, dan kedalaman terbesarnya ke arah dalam tile "
    "dikirim ke <i>accumulator</i> bertipe maksimum. Karena evaluasi Spark malas, <font face='Courier'>persist()</font> dan <font face='Courier'>count()</font> dipanggil sebelum nilai accumulator dibaca.",
    "<b>Auxiliary band tunggal.</b> Satu band int32 per tile: 0 untuk piksel inner (dikunci), id seed untuk seed di area border, dan -1 untuk piksel area border tanpa seed. "
    "Varian pembanding mengirim seed, garis border, kategori, dan arah sebagai band terpisah (11 byte per piksel dibanding 5 byte, termasuk pita spektral).",
    "<b>Shuffle.</b> Tiap tile mengirim potongan yang beririsan dengan jendela ber-buffer ke delapan tetangganya lewat <font face='Courier'>flatMap</font> dan <font face='Courier'>groupByKey</font>. "
    "Jumlah byte yang dikirim ke tetangga dihitung dengan accumulator penjumlah sebagai ukuran volume komunikasi.",
    "<b>Border area processing.</b> Jendela ber-buffer dirakit, lalu algoritma dijalankan ulang hanya pada area border dengan seed di dalamnya. Hasil dipotong ke tile dan hanya piksel area border yang diganti.",
    "<b>Evaluasi.</b> F-measure berbasis objek (Persamaan 2 sampai 6, &alpha; = 0,5) terhadap referensi sekuensial, untuk seluruh citra dan untuk area border, yaitu piksel objek yang menyentuh sisi tile pada hasil tanpa processing.",
    f"<b>Waktu.</b> Waktu dinding seluruh pipeline dengan <font face='Courier'>time.perf_counter</font>, satu kali pemanasan, dan {meta['repeats']} ulangan. "
    "Buffer manual dipilih 20%, 40%, 60%, 80%, dan 100% ukuran tile (padanan 100 sampai 500 piksel pada tile 512 di artikel).",
])
P("B.5 Perbedaan terhadap artikel", "h2")
table([["Aspek", "Artikel", "Replikasi ini", "Alasan"],
       ["Ukuran citra", "4096 x 4096, 10 citra", f"{meta['size']} x {meta['size']}, {len(meta['images'])} citra", "Sumber daya terbatas dan batas tenggat"],
       ["Data", "Sentinel-2 Copernicus", source, "Akun Copernicus dan unduhan tidak tersedia pada lingkungan pengembangan"],
       ["Klaster", "7 node, 4 executor x 4 core", f"{'local[' + str(meta['cores']) + ']' if SPARK else 'LocalContext'}, satu mesin", "Gratis dan ringan"],
       ["Kode", "Tidak dilampirkan", "Dibangun ulang dari deskripsi", "Rincian border processing hanya garis besar"],
       ["Auxiliary band", "Garis border, kategori, arah", "Mask + seed dalam satu band", "Disederhanakan dengan hasil setara"],
       ["Penentuan seed watershed", "Ambang gradien", "rank.gradient disk radius 1 (asumsi)", "Fungsi gradien tidak dirinci"]],
      [2.8, 3.8, 4.4, 5])

# --------------------------------------------------------------------- C
P("C. Hasil Pengujian dan Analisis", "h1")
if not SPARK or "sintetis" in source.lower() or "synthetic" in source.lower():
    msg = []
    if not SPARK:
        msg.append("Hasil ini dihasilkan dengan <b>LocalContext (tanpa Spark)</b>. Akurasi dan volume shuffle tidak bergantung pada backend karena algoritmanya deterministik, "
                   "tetapi <b>waktu eksekusi bukan waktu Spark</b>.")
    if "synthetic" in source.lower() or "sintetis" in source.lower():
        msg.append("Citra yang dipakai <b>sintetis</b>, bukan Sentinel-2.")
    story.append(Paragraph("<b>Catatan sumber data:</b> " + " ".join(msg), S["box"]))
g = acc.groupby(["algo", "tile"]).mean(numeric_only=True).reset_index()
P("C.1 Akurasi", "h2")
rows = [["Algoritma", "Tile", "Seluruh citra: dengan", "Seluruh citra: tanpa", "Area border: dengan", "Area border: tanpa", "Buffer otomatis rata-rata"]]
for _, r in g.iterrows():
    rows.append([NAME[r.algo], int(r.tile), pct(r.f_entire_with), pct(r.f_entire_without), pct(r.f_border_with), pct(r.f_border_without), f"{r.buffer:.0f}"])
P(f"Tabel 1 memuat rata-rata F-measure atas {acc.image.nunique()} citra dan lima level seed.")
table(rows, [2.6, 1.3, 2.6, 2.6, 2.6, 2.6, 2.2])
fig(f"{R}/fig_akurasi.png", 15.5, "Gambar 1. F-measure rata-rata menurut ukuran tile. Garis biru: dengan border area processing; oranye: tanpa.")

t0 = meta["size"] // 16 if meta["size"] // 16 in meta["tiles"] else min(meta["tiles"])
n0 = meta["size"] // t0
cmp_rows = [["Kasus", "Artikel dengan", "Artikel tanpa", f"Replikasi dengan", "Replikasi tanpa"]]
paper = {("rg", "entire"): (95.2, 92.8), ("rg", "border"): (90.2, 84.4), ("ws", "entire"): (99.8, 99.1), ("ws", "border"): (98.1, 91.1)}
a1 = acc[(acc.tile == t0) & (acc.level == 1)]
for (al, kind), (pw, po) in paper.items():
    h = a1[a1.algo == al]
    cmp_rows.append([f"{NAME[al]}, {'seluruh citra' if kind == 'entire' else 'area border'}", f"{pw}%", f"{po}%",
                     pct(h[f"f_{kind}_with"].mean()), pct(h[f"f_{kind}_without"].mean())])
P(f"Tabel 2 membandingkan dengan angka artikel (tile 256 pada citra 4096, 16 x 16 tile, level seed 1) terhadap kasus replikasi terdekat: tile {t0} pada citra {meta['size']} ({n0} x {n0} tile), level seed 1.")
table(cmp_rows, [5, 2.7, 2.7, 2.9, 2.9])

P("<b>Analisis akurasi.</b>", "lead")
items = []
for al in ["rg", "ws"]:
    d = acc[acc.algo == al]
    fbw, fbo, few, feo = d.f_border_with.mean(), d.f_border_without.mean(), d.f_entire_with.mean(), d.f_entire_without.mean()
    gap = d.groupby("tile").apply(lambda x: (x.f_border_with - x.f_border_without).mean(), include_groups=False)
    lvl = d.groupby("level").f_entire_with.mean()
    frac = ((d.f_border_with >= d.f_border_without - 1e-12).mean())
    items.append(f"<b>{NAME[al].capitalize()}.</b> F-measure area border {'naik' if fbw > fbo else 'turun'} dari {pct(fbo)} menjadi {pct(fbw)} ({pp(fbw - fbo)} poin) setelah border area processing, "
                 f"sedangkan untuk seluruh citra {pct(feo)} menjadi {pct(few)} ({pp(few - feo)} poin). Hasil dengan processing tidak lebih rendah pada {100 * frac:.0f}% kombinasi (citra, level, tile). "
                 f"Selisih area border {'mengecil' if gap.iloc[0] > gap.iloc[-1] else 'tidak mengecil'} dari {pp(gap.iloc[0])} poin pada tile {gap.index[0]} menjadi {pp(gap.iloc[-1])} poin pada tile {gap.index[-1]}. "
                 f"F-measure seluruh citra dengan processing {'naik' if lvl.iloc[-1] > lvl.iloc[0] else 'turun'} dari {pct(lvl.iloc[0])} (level 1) ke {pct(lvl.iloc[-1])} (level 5).")
bullets(items)
P("Selisih pada seluruh citra jauh lebih kecil daripada pada area border karena objek border hanya sebagian kecil piksel; pada tile kecil porsinya lebih besar sehingga manfaat "
  "pemrosesan border lebih terlihat. Angka replikasi tidak identik dengan artikel karena citra, implementasi region growing, dan definisi area border berbeda, sehingga yang dibandingkan adalah arah dan besar perubahannya.")

P("C.2 Efisiensi: auxiliary band, buffer, dan volume shuffle", "h2")
rows = [["Algoritma", "Band", "Mode buffer", "Buffer rata-rata (piksel)", "Waktu rata-rata (detik)", "Volume shuffle (MB)"]]
for al in ["rg", "ws"]:
    for na in (1, 3):
        for mode in ("auto", "manual"):
            h = tim[(tim.algo == al) & (tim.n_aux == na) & (tim["mode"] == mode)]
            rows.append([NAME[al], na, "otomatis" if mode == "auto" else "manual (rata-rata 5 ukuran)", f"{h.buffer.mean():.0f}", f"{h.time.mean():.2f}", f"{h.shuffle_bytes.mean() / 1e6:.1f}"])
P(f"Tabel 3 dihitung pada tile {meta['tile_waktu']} piksel dan level seed 3.")
table(rows, [2.6, 1.3, 4.3, 2.8, 2.7, 2.6])
fig(f"{R}/fig_waktu.png", 15.5, "Gambar 2. Waktu eksekusi menurut ukuran buffer untuk 1 dan 3 auxiliary band (rata-rata atas citra).")
items = []
for al in ["rg", "ws"]:
    h = tim[tim.algo == al]
    t1, t3 = h[h.n_aux == 1].time.mean(), h[h.n_aux == 3].time.mean()
    b1, b3 = h[h.n_aux == 1].shuffle_bytes.mean(), h[h.n_aux == 3].shuffle_bytes.mean()
    m1 = h[(h.n_aux == 1) & (h["mode"] == "manual")].groupby("buffer").time.mean()
    au = h[(h.n_aux == 1) & (h["mode"] == "auto")]
    near = m1.index[np.abs(m1.index.values - au.buffer.mean()).argmin()]
    items.append(f"<b>{NAME[al].capitalize()}.</b> Tiga auxiliary band membutuhkan waktu rata-rata {t3 / t1:.2f} kali waktu satu band ({t3:.2f} s dibanding {t1:.2f} s) dan "
                 f"mengirim {b3 / b1:.2f} kali volume shuffle. Dengan satu band, waktu buffer manual terkecil ({m1.index[0]} piksel) adalah {m1.iloc[0]:.2f} s dan terbesar ({m1.index[-1]} piksel) {m1.iloc[-1]:.2f} s. "
                 f"Buffer otomatis rata-rata {au.buffer.mean():.0f} piksel memakai {au.time.mean():.2f} s, dibanding {m1[near]:.2f} s pada buffer manual terdekat ({near} piksel).")
bullets(items)
if not SPARK:
    P("Karena backend lokal, waktu di atas mencerminkan komputasi Python pada satu proses dan tidak memuat biaya shuffle jaringan Spark; kolom volume shuffle tetap valid.")
if cores is not None:
    c = cores.groupby("cores").time.mean()
    P(f"Waktu pada citra {meta['images'][0]} berkurang dari {c.iloc[0]:.1f} s (1 core) menjadi {c.iloc[-1]:.1f} s ({c.index[-1]} core), percepatan {c.iloc[0] / c.iloc[-1]:.2f} kali (Gambar 3).")
    fig(f"{R}/fig_core.png", 8, "Gambar 3. Waktu eksekusi menurut jumlah core (padanan jumlah calculation node pada artikel).")
ratio_auto = tim[tim["mode"] == "auto"].buffer.mean() / meta["tile_waktu"]
lim3 = ("Buffer otomatis rata-rata " + f"{100 * ratio_auto:.0f}% ukuran tile, " + (
    "mendekati ukuran tile karena objek besar menyentuh sisi tile, sehingga pengurangan volume komunikasi lebih kecil daripada klaim artikel."
    if ratio_auto >= 0.8 else "sehingga ada penghematan komunikasi dibanding buffer sebesar tile."))
P("<b>Keterbatasan.</b> (1) Skala citra dan mesin jauh lebih kecil daripada artikel. (2) Waktu pada satu mesin dipengaruhi proses lain; pengulangan terbatas. "
  f"(3) {lim3} "
  "(4) Definisi area border dan aturan border processing dirancang sendiri karena artikel tidak merinci algoritmanya. (5) Waktu shuffle tidak diukur terpisah.")

# --------------------------------------------------------------------- D
P("D. Kendala dan Refleksi", "h1")
bullets([
    "<b>Artikel tanpa kode dan rincian parsial.</b> Aturan penentuan mask, seed, dan batas pemrosesan ulang di area border dirancang sendiri dari Gambar 2 dan 3 serta teks artikel; "
    "diuji dengan membandingkan satu tile (harus identik dengan sekuensial) dan memastikan border processing tidak menurunkan akurasi.",
    "<b>Region growing tidak tersedia di pustaka.</b> Implementasi Python murni memerlukan sekitar delapan detik per citra 1024 x 1024, sehingga ukuran citra dikecilkan.",
    "<b>Data.</b> Unduhan Copernicus memerlukan akun, sehingga disediakan skrip pengambil potongan Sentinel-2 dari AWS Earth Search (tanpa akun) dan generator citra sintetis untuk uji tanpa internet.",
    "<b>Evaluasi malas Spark.</b> Nilai accumulator kosong sebelum ada action, sehingga <font face='Courier'>count()</font> dan <font face='Courier'>persist()</font> perlu dipanggil sebelum membaca ukuran buffer. "
    "Accumulator di dalam transformasi dapat terhitung ganda jika tugas diulang; jenis maksimum aman, sedangkan penghitung byte hanya perkiraan.",
    "<b>Lingkungan pengembangan.</b> Kode dikembangkan dengan bantuan AI pada lingkungan yang tidak dapat memasang PySpark, sehingga logika diuji lewat <font face='Courier'>LocalContext</font> yang meniru API RDD. "
    f"{'Hasil dijalankan pada Spark mode lokal.' if SPARK else 'Pengukuran waktu Spark perlu dijalankan ulang pada mesin yang memiliki PySpark.'}",
])
P("<b>Jika diulang:</b> memakai citra Sentinel-2 asli untuk seluruh skenario, menentukan buffer per tile (bukan satu nilai global) seperti usulan artikel, mengukur waktu tahap shuffle dari Spark UI, "
  "dan mencoba klaster kecil di cloud untuk melihat biaya jaringan yang sebenarnya.")

# --------------------------------------------------------------------- E
P("E. Tautan Kode", "h1")
P(f"Repositori: {CFG['repo_url']} (akses: {CFG['akses_repo']}). Berkas utama: <font face='Courier'>segpipe.py</font> (algoritma dan pipeline Spark), "
  "<font face='Courier'>run_experiments.py</font>, <font face='Courier'>get_data.py</font>, <font face='Courier'>make_report.py</font>, dan "
  "<font face='Courier'>tests/test_segpipe.py</font>. Cara menjalankan: <font face='Courier'>pip install -r requirements.txt</font>, "
  "<font face='Courier'>python get_data.py --source sentinel</font>, <font face='Courier'>python run_experiments.py --backend spark</font>, "
  "<font face='Courier'>python make_report.py</font>. Rincian ada di README repositori.")
P("Pernyataan Penggunaan Generative AI", "h2")
P("Generative AI (Claude, Anthropic) digunakan untuk membantu menulis kode awal (<font face='Courier'>segpipe.py</font>, <font face='Courier'>run_experiments.py</font>, "
  "<font face='Courier'>get_data.py</font>, uji unit), menyusun skrip pembuat laporan, dan menyusun draf laporan ini. Penulis bertanggung jawab atas kebenaran isi dan "
  "menjalankan serta memahami kode tersebut.")

story.append(PageBreak())
P("Daftar Pustaka", "h1")
for r in [
    "Adams, R., &amp; Bischof, L. (1994). Seeded region growing. <i>IEEE Transactions on Pattern Analysis and Machine Intelligence, 16</i>, 641&ndash;647.",
    "Chen, F., Wang, N., Yu, B., Qin, Y., &amp; Wang, L. (2021). A strategy of parallel seed-based image segmentation algorithms for handling massive image tiles over the Spark platform. "
    "<i>Remote Sensing, 13</i>(10), 1969. https://doi.org/10.3390/rs13101969",
    "van der Walt, S., Sch&ouml;nberger, J. L., Nunez-Iglesias, J., Boulogne, F., Warner, J. D., Yager, N., Gouillart, E., &amp; Yu, T. (2014). scikit-image: Image processing in Python. "
    "<i>PeerJ, 2</i>, e453. https://doi.org/10.7717/peerj.453",
    "Vincent, L., &amp; Soille, P. (1991). Watersheds in digital spaces: An efficient algorithm based on immersion simulations. <i>IEEE Transactions on Pattern Analysis and Machine Intelligence, 13</i>, 583&ndash;598.",
    "Zaharia, M., Xin, R. S., Wendell, P., Das, T., Armbrust, M., Dave, A., Meng, X., Rosen, J., Venkataraman, S., Franklin, M. J., Ghodsi, A., Gonzalez, J., Shenker, S., &amp; Stoica, I. (2016). "
    "Apache Spark: A unified engine for big data processing. <i>Communications of the ACM, 59</i>, 56&ndash;65.",
    "Zhang, X., Feng, X., Xiao, P., He, G., &amp; Zhu, L. (2015). Segmentation quality evaluation using region-based precision and recall measures for remote sensing images. "
    "<i>ISPRS Journal of Photogrammetry and Remote Sensing, 102</i>, 73&ndash;84.",
]:
    P(r, "ref")


def footer(c, d):
    c.setFont("Helvetica", 8)
    c.setFillColor(colors.HexColor(INK2))
    c.drawString(2 * cm, 1.2 * cm, "Mahadata (IF25-40402) - Eksplorasi Mandiri")
    c.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Halaman {d.page}")


out = f"{CFG['nim']}_{CFG['nama'].replace(' ', '-')}_Eksplorasi-Mahadata.pdf"
SimpleDocTemplate(out, pagesize=A4, leftMargin=2.2 * cm, rightMargin=2.2 * cm, topMargin=2 * cm, bottomMargin=2 * cm,
                  title="Laporan Eksplorasi Mandiri Mahadata", author=CFG["nama"]).build(story, onFirstPage=lambda c, d: None, onLaterPages=footer)
print("tersimpan", out)
