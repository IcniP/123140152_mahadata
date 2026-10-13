"""Replikasi skala kecil: strategi paralel segmentasi seed-based di Spark.

Acuan: Chen, F., Wang, N., Yu, B., Qin, Y., & Wang, L. (2021). Remote Sensing, 13(10), 1969.
https://doi.org/10.3390/rs13101969

Alur (mengikuti Gambar 2 artikel):
  pretreatment -> segmentasi tile -> deteksi buffer otomatis (accumulator)
  -> sintesis auxiliary band -> shuffle tile ber-buffer -> border area processing -> integrasi tile
Label objek = id seed global, sehingga bagian objek yang sama di tile berbeda punya label sama.
"""
import heapq

import numpy as np
from scipy import ndimage as ndi
from skimage.filters import rank
from skimage.morphology import disk
from skimage.segmentation import watershed

try:
    from pyspark.accumulators import AccumulatorParam
except ImportError:  # tanpa pyspark: hanya LocalContext yang bisa dipakai
    AccumulatorParam = object

# Tabel 2 artikel: level kepadatan seed 1 (jarang) .. 5 (rapat)
RG_WINDOWS = {1: 33, 2: 27, 3: 21, 4: 15, 5: 9}
WS_THRESH = {1: 2, 2: 3, 3: 4, 4: 5, 5: 6}


class MaxParam(AccumulatorParam):
    def zero(self, v):
        return 0

    def addInPlace(self, a, b):
        return max(a, b)


class SumParam(AccumulatorParam):
    def zero(self, v):
        return 0

    def addInPlace(self, a, b):
        return a + b


# ---------------------------------------------------------------- pretreatment
def prepare(rgb, algo, level):
    """RGB uint8 -> (pita spektral G uint8, penanda seed int32). Persamaan (1) + median radius 2."""
    gray = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).round().astype(np.uint8)
    gray = ndi.median_filter(gray, footprint=disk(2))
    if algo == "rg":  # seed = puncak lokal maksimum/minimum pada jendela tertentu
        w = RG_WINDOWS[level]
        G = gray
        seeds = (G == ndi.maximum_filter(G, w)) | (G == ndi.minimum_filter(G, w))
    else:  # shortcut: gradien = rank.gradient(disk(1)); artikel hanya menyebut fungsi gradient scikit-image
        G = rank.gradient(gray, disk(1))
        seeds = G < WS_THRESH[level]
    return G, ndi.label(seeds)[0].astype(np.int32)


# ------------------------------------------------------------ algoritma seed-based
def region_grow(img, markers, mask=None):
    """Seeded region growing (Adams & Bischof, 1994), versi antrean prioritas."""
    h, w = img.shape
    lab = np.full((h + 2, w + 2), -1, np.int64)  # bingkai -1 = penghalang
    inner = markers.astype(np.int64)
    if mask is not None:
        inner = np.where(mask, inner, -1)
    lab[1:-1, 1:-1] = inner
    val = np.zeros((h + 2, w + 2), np.int64)
    val[1:-1, 1:-1] = img
    W = w + 2
    lab_f, val_f = lab.ravel().tolist(), val.ravel().tolist()
    top = int(max(markers.max(), 0))
    tot, cnt = [0] * (top + 1), [0] * (top + 1)
    seeds = np.flatnonzero(lab.ravel() > 0).tolist()
    for p in seeds:
        tot[lab_f[p]] += val_f[p]
        cnt[lab_f[p]] += 1
    heap, n, offs = [], 0, (1, -1, W, -W)
    for p in seeds:
        l = lab_f[p]
        m = tot[l] / cnt[l]
        for o in offs:
            q = p + o
            if lab_f[q] == 0:
                heap.append((abs(val_f[q] - m), n, q, l))
                n += 1
    heapq.heapify(heap)
    while heap:
        _, _, q, l = heapq.heappop(heap)
        if lab_f[q] != 0:
            continue
        lab_f[q] = l
        tot[l] += val_f[q]
        cnt[l] += 1
        m = tot[l] / cnt[l]
        for o in offs:
            r = q + o
            if lab_f[r] == 0:
                heapq.heappush(heap, (abs(val_f[r] - m), n, r, l))
                n += 1
    out = np.array(lab_f, np.int64).reshape(h + 2, w + 2)[1:-1, 1:-1]
    return np.where(out > 0, out, 0).astype(np.int32)


def segment(algo, G, markers, mask=None):
    if not (markers > 0).any():
        return np.zeros(G.shape, np.int32)
    if algo == "rg":
        return region_grow(G, markers, mask)
    if mask is not None:
        markers = np.where(mask, markers, 0)
    return watershed(G, markers, mask=mask).astype(np.int32)


# -------------------------------------------------------------------- evaluasi
def _best(a, b):
    """Sum_i max_k |a_i ∩ b_k| untuk label > 0 (pencocokan objek, Persamaan 2)."""
    m = (a > 0) & (b > 0)
    if not m.any():
        return 0
    k = int(b.max()) + 1
    u, c = np.unique(a[m].astype(np.int64) * k + b[m], return_counts=True)
    s = u // k
    first = np.r_[0, np.flatnonzero(np.diff(s)) + 1]
    return int(np.maximum.reduceat(c, first).sum())


def fmeasure(seg, ref, region=None, alpha=0.5):
    """F-measure berbasis objek, Persamaan (3)-(6). region = mask bool untuk evaluasi area border."""
    if region is not None:
        seg, ref = np.where(region, seg, 0), np.where(region, ref, 0)
    ns, nr = (seg > 0).sum(), (ref > 0).sum()
    if ns == 0 or nr == 0:
        return 0.0
    p, r = _best(seg, ref) / ns, _best(ref, seg) / nr
    return 0.0 if p * r == 0 else 1 / (alpha / p + (1 - alpha) / r)


# ------------------------------------------------------------- tiling dan Spark
def tile_rect(ti, tj, grid):
    H, W, s = grid
    return ti * s, min(H, (ti + 1) * s), tj * s, min(W, (tj + 1) * s)


def seg_tile(item, algo, nt, depth_acc):
    """Langkah 2-4: segmentasi tile, deteksi objek border + kedalamannya, sintesis 1 auxiliary band."""
    (ti, tj), (rect, G, M) = item
    L = segment(algo, G, M)
    on = {"top": ti > 0, "bottom": ti < nt[0] - 1, "left": tj > 0, "right": tj < nt[1] - 1}
    views = {"top": L, "bottom": L[::-1], "left": L.T, "right": L.T[::-1]}
    bit = {"top": 1, "right": 2, "bottom": 4, "left": 8}
    D = np.zeros(L.shape, np.uint8)  # arah: sisi tile yang disentuh objek border (bitmask)
    depth = 0
    for side, arr in views.items():
        if not on[side]:
            continue
        labs = np.unique(arr[0])
        labs = labs[labs > 0]
        if labs.size:
            D[np.isin(L, labs)] |= bit[side]
            depth = max(depth, int(np.flatnonzero(np.isin(arr, labs).any(axis=1)).max()) + 1)
    depth_acc.add(depth)
    mask = D != 0
    # auxiliary band tunggal: 0 = piksel inner (dikunci), id seed = seed di area border, -1 = area border tanpa seed
    A = np.where(mask, np.where(M > 0, M, -1), 0).astype(np.int32)
    return (ti, tj), (rect, G, L, A, D)


def emit_pieces(item, B, grid, nt, n_aux, bytes_acc):
    """Langkah 'tile buffering': kirim potongan tile ke tetangga (8 arah) yang jendela ber-buffernya beririsan."""
    (ti, tj), (rect, G, L, A, D) = item
    r0, r1, c0, c1 = rect
    H, W, _ = grid
    out = []
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            ui, uj = ti + di, tj + dj
            if not (0 <= ui < nt[0] and 0 <= uj < nt[1]):
                continue
            t0, t1, u0, u1 = tile_rect(ui, uj, grid)
            a0, a1 = max(r0, t0 - B), min(r1, t1 + B)
            b0, b1 = max(c0, u0 - B), min(c1, u1 + B)
            if a0 >= a1 or b0 >= b1:
                continue
            sl = (slice(a0 - r0, a1 - r0), slice(b0 - c0, b1 - c0))
            if n_aux == 1:
                data = {"G": G[sl], "A": A[sl]}
            else:  # pembanding artikel: seed + 3 auxiliary band (garis border, kategori, arah) dikirim terpisah
                data = {"G": G[sl], "seeds": np.maximum(A[sl], 0), "lines": (A[sl] != 0).astype(np.uint8),
                        "cats": L[sl], "dirs": D[sl]}
            is_self = di == 0 and dj == 0
            if is_self:
                data["L"] = L
            else:
                bytes_acc.add(sum(v.nbytes for v in data.values()))
            out.append(((ui, uj), (is_self, (a0, a1, b0, b1), data)))
    return out


def border_process(item, grid, B, algo):
    """Langkah 5: jalankan ulang algoritma pada mask area border di jendela ber-buffer, lalu potong ke tile."""
    (ti, tj), pieces = item
    H, W, _ = grid
    r0, r1, c0, c1 = tile_rect(ti, tj, grid)
    R0, R1, C0, C1 = max(0, r0 - B), min(H, r1 + B), max(0, c0 - B), min(W, c1 + B)
    Gw = np.zeros((R1 - R0, C1 - C0), np.uint8)
    Aw = np.zeros(Gw.shape, np.int32)
    for is_self, (a0, a1, b0, b1), d in pieces:
        sl = (slice(a0 - R0, a1 - R0), slice(b0 - C0, b1 - C0))
        Gw[sl] = d["G"]
        Aw[sl] = d["A"] if "A" in d else np.where(d["lines"] > 0, np.where(d["seeds"] > 0, d["seeds"], -1), 0)
        if is_self:
            L = d["L"]
    mask = Aw != 0
    if not mask.any():
        return (ti, tj), (r0, r1, c0, c1), L
    new = segment(algo, Gw, np.where(Aw > 0, Aw, 0), mask)
    own = (slice(r0 - R0, r1 - R0), slice(c0 - C0, c1 - C0))
    take = mask[own] & (new[own] > 0)
    out = L.copy()
    out[take] = new[own][take]
    return (ti, tj), (r0, r1, c0, c1), out


def run_pipeline(sc, G, M, algo, tile, n_aux=1, buffer=None, want_before=False):
    """Jalankan seluruh strategi. buffer=None -> ukuran buffer otomatis (accumulator + action + persist)."""
    H, W = G.shape
    grid = (H, W, tile)
    nt = (-(-H // tile), -(-W // tile))
    items = []
    for ti in range(nt[0]):
        for tj in range(nt[1]):
            r0, r1, c0, c1 = tile_rect(ti, tj, grid)
            items.append(((ti, tj), ((r0, r1, c0, c1), G[r0:r1, c0:c1].copy(), M[r0:r1, c0:c1].copy())))
    depth_acc, bytes_acc = sc.accumulator(0, MaxParam()), sc.accumulator(0, SumParam())
    seg = sc.parallelize(items, len(items)).map(lambda it: seg_tile(it, algo, nt, depth_acc))
    if buffer is None or want_before:
        seg = seg.persist()
    if buffer is None:
        seg.count()  # accumulator di Spark baru terisi setelah action
        B = depth_acc.value
    else:
        B = buffer
    B = max(1, min(int(B), tile))
    final = (seg.flatMap(lambda it: emit_pieces(it, B, grid, nt, n_aux, bytes_acc))
             .groupByKey().map(lambda it: border_process(it, grid, B, algo)).collect())
    res = {"labels": _mosaic(final, (H, W), lambda t: t[2]), "buffer": B, "shuffle_bytes": bytes_acc.value}
    if want_before:
        before = seg.collect()
        res["labels_before"] = _mosaic(before, (H, W), lambda t: t[1][2], lambda t: t[1][0])
        res["border_mask"] = _mosaic(before, (H, W), lambda t: t[1][3] != 0, lambda t: t[1][0], bool)
    return res


def _mosaic(rows, shape, get, rect=lambda t: t[1], dtype=np.int32):
    out = np.zeros(shape, dtype)
    for t in rows:
        r0, r1, c0, c1 = rect(t)
        out[r0:r1, c0:c1] = get(t)
    return out


# ------------------------------------------------------------------- konteks
class LocalContext:
    """Pengganti SparkContext tanpa Java (evaluasi malas seperti RDD). Untuk uji logika dan demo, bukan benchmark."""

    class _Acc:
        def __init__(self, v, p):
            self.value, self.p = v, p

        def add(self, x):
            self.value = self.p.addInPlace(self.value, x)

    class _RDD:
        def __init__(self, f):
            self.f = f

        def map(self, g):
            return LocalContext._RDD(lambda: [g(x) for x in self.f()])

        def flatMap(self, g):
            return LocalContext._RDD(lambda: [y for x in self.f() for y in g(x)])

        def groupByKey(self):
            def f():
                d = {}
                for k, v in self.f():
                    d.setdefault(k, []).append(v)
                return list(d.items())
            return LocalContext._RDD(f)

        def persist(self):
            f, cache = self.f, []

            def g():
                if not cache:
                    cache.append(f())
                return cache[0]
            self.f = g
            return self

        def count(self):
            return len(self.f())

        def collect(self):
            return self.f()

    def parallelize(self, data, n=None):
        return LocalContext._RDD(lambda: list(data))

    def accumulator(self, v, p):
        return LocalContext._Acc(v, p)


def get_spark_context(cores):
    """SparkContext mode lokal dengan `cores` core (gratis, tanpa klaster)."""
    from pyspark.sql import SparkSession
    spark = (SparkSession.builder.master(f"local[{cores}]").appName("seg-seedbased")
             .config("spark.driver.memory", "4g").config("spark.ui.enabled", "false")
             .config("spark.sql.shuffle.partitions", "8").getOrCreate())
    sc = spark.sparkContext
    sc.setLogLevel("ERROR")
    sc.addPyFile(__file__)  # worker perlu meng-import segpipe
    return sc


def stop_spark():
    from pyspark.sql import SparkSession
    s = SparkSession.getActiveSession()
    if s:
        s.stop()
