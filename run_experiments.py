"""Menjalankan eksperimen: (1) akurasi dengan/tanpa border processing, (2) waktu vs jumlah core,
(3) waktu & volume shuffle untuk 1 vs 3 auxiliary band dan buffer otomatis vs manual.

  python run_experiments.py --backend spark --cores 2          # hasil utama (butuh Java + pyspark)
  python run_experiments.py --backend local --quick            # uji cepat tanpa Spark
"""
import argparse
import glob
import json
import os
import platform
import time

import numpy as np
import pandas as pd
from PIL import Image

import segpipe as sp


def timed(fn):
    t = time.perf_counter()
    r = fn()
    return r, time.perf_counter() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["spark", "local"], default="spark")
    ap.add_argument("--images", default="data/*.png")
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--tiles", type=int, nargs="+", default=[64, 128, 256, 512])
    ap.add_argument("--levels", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--algos", nargs="+", default=["rg", "ws"])
    ap.add_argument("--cores", type=int, default=os.cpu_count())
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", default="results")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.quick:
        a.tiles, a.levels, a.repeats = [128, 256], [1, 3, 5], 1
    os.makedirs(a.out, exist_ok=True)
    paths = sorted(glob.glob(a.images))[: 1 if a.quick else None]
    assert paths, f"tidak ada citra di {a.images}; jalankan get_data.py"
    imgs = {os.path.splitext(os.path.basename(p))[0]: np.array(Image.open(p).convert("RGB"))[: a.size, : a.size]
            for p in paths}
    spark = a.backend == "spark"
    sc = sp.get_spark_context(a.cores) if spark else sp.LocalContext()
    meta = {"backend": a.backend, "cores": a.cores if spark else 1, "cpu_count": os.cpu_count(),
            "python": platform.python_version(), "size": a.size, "images": list(imgs), "tiles": a.tiles,
            "levels": a.levels, "repeats": a.repeats, "tanggal": time.strftime("%Y-%m-%d %H:%M")}
    if spark:
        import pyspark
        meta["pyspark"] = pyspark.__version__
    import scipy
    import skimage
    meta.update(numpy=np.__version__, scipy=scipy.__version__, skimage=skimage.__version__)

    # ---- 1. akurasi (F-measure terhadap hasil sekuensial)
    rows = []
    for name, rgb in imgs.items():
        for algo in a.algos:
            for lvl in a.levels:
                G, M = sp.prepare(rgb, algo, lvl)
                ref, t_seq = timed(lambda: sp.segment(algo, G, M))
                for tile in a.tiles:
                    r = sp.run_pipeline(sc, G, M, algo, tile, want_before=True)
                    bm = r["border_mask"]
                    rows.append({"image": name, "algo": algo, "level": lvl, "tile": tile, "buffer": r["buffer"],
                                 "border_frac": bm.mean(), "t_seq": t_seq,
                                 "f_entire_with": sp.fmeasure(r["labels"], ref),
                                 "f_entire_without": sp.fmeasure(r["labels_before"], ref),
                                 "f_border_with": sp.fmeasure(r["labels"], ref, bm),
                                 "f_border_without": sp.fmeasure(r["labels_before"], ref, bm)})
                print("akurasi", name, algo, lvl, flush=True)
    pd.DataFrame(rows).to_csv(f"{a.out}/accuracy.csv", index=False)

    # ---- 2. waktu vs jumlah core (padanan "jumlah calculation node" Gambar 8)
    name0 = next(iter(imgs))
    G, M = sp.prepare(imgs[name0], "rg", 3)
    tile0 = a.size // 8
    if spark and not a.quick:
        rows = []
        for n in range(1, a.cores + 1):
            if n != a.cores:
                sp.stop_spark()
                sc = sp.get_spark_context(n)
            sp.run_pipeline(sc, G, M, "rg", tile0)  # pemanasan JVM/worker
            for _ in range(a.repeats):
                _, t = timed(lambda: sp.run_pipeline(sc, G, M, "rg", tile0))
                rows.append({"cores": n, "time": t})
            print("core", n, flush=True)
        pd.DataFrame(rows).to_csv(f"{a.out}/cores.csv", index=False)
        sp.stop_spark()
        sc = sp.get_spark_context(a.cores)

    # ---- 3. waktu & volume shuffle: 1 vs 3 auxiliary band, buffer otomatis vs manual
    manual = [max(1, round(tile0 * f)) for f in (0.2, 0.4, 0.6, 0.8, 1.0)]
    rows = []
    for name, rgb in imgs.items():
        for algo in a.algos:
            G, M = sp.prepare(rgb, algo, 3)
            sp.run_pipeline(sc, G, M, algo, tile0)  # pemanasan
            for n_aux in (1, 3):
                for buf in [None] + manual:
                    for rep in range(a.repeats):
                        r, t = timed(lambda: sp.run_pipeline(sc, G, M, algo, tile0, n_aux=n_aux, buffer=buf))
                        rows.append({"image": name, "algo": algo, "n_aux": n_aux,
                                     "mode": "auto" if buf is None else "manual", "buffer": r["buffer"],
                                     "rep": rep, "time": t, "shuffle_bytes": r["shuffle_bytes"]})
            print("waktu", name, algo, flush=True)
    pd.DataFrame(rows).to_csv(f"{a.out}/timing.csv", index=False)
    meta["tile_waktu"] = tile0
    json.dump(meta, open(f"{a.out}/meta.json", "w"), indent=1)
    print("selesai ->", a.out)


if __name__ == "__main__":
    main()
