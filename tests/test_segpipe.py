"""Uji logika dengan LocalContext (tanpa Spark). Jalankan: python -m pytest -q  atau  python tests/test_segpipe.py"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import segpipe as sp
from get_data import synthetic

RGB = synthetic(0, 192)


def _run(algo, level, tile, **kw):
    G, M = sp.prepare(RGB, algo, level)
    ref = sp.segment(algo, G, M)
    return G, M, ref, sp.run_pipeline(sp.LocalContext(), G, M, algo, tile, want_before=True, **kw)


def test_fmeasure_identik_dan_acak():
    a = np.random.default_rng(0).integers(1, 20, (50, 50))
    assert sp.fmeasure(a, a) == 1.0
    assert sp.fmeasure(np.random.default_rng(1).integers(1, 20, (50, 50)), a) < 0.5


def test_satu_tile_sama_dengan_sekuensial():
    for algo in ("rg", "ws"):
        _, _, ref, r = _run(algo, 3, 192)
        assert (r["labels"] == ref).all(), algo


def test_border_processing_tidak_menurunkan_akurasi():
    for algo in ("rg", "ws"):
        _, _, ref, r = _run(algo, 3, 48)
        bm = r["border_mask"]
        assert bm.any()
        for reg in (None, bm):
            assert sp.fmeasure(r["labels"], ref, reg) >= sp.fmeasure(r["labels_before"], ref, reg) - 1e-9, algo
        assert sp.fmeasure(r["labels"], ref, bm) > sp.fmeasure(r["labels_before"], ref, bm), algo


def test_buffer_otomatis_dan_aux_band():
    _, _, _, a1 = _run("ws", 3, 48, n_aux=1)
    _, _, _, a3 = _run("ws", 3, 48, n_aux=3)
    assert 1 <= a1["buffer"] <= 48
    assert (a1["labels"] == a3["labels"]).all()  # jumlah band tidak mengubah hasil
    assert a3["shuffle_bytes"] > 2 * a1["shuffle_bytes"]  # 11 vs 5 byte/piksel
    _, _, _, m = _run("ws", 3, 48, buffer=10)
    assert m["buffer"] == 10 and m["shuffle_bytes"] < a1["shuffle_bytes"] or a1["buffer"] <= 10


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            f()
            print("ok", n)
