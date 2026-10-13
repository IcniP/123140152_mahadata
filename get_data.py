"""Menyiapkan 3 citra RGB 1024x1024 di folder data/.

  python get_data.py --source synthetic   # tanpa internet, citra sintetis bergaya lanskap
  python get_data.py --source sentinel    # potongan Sentinel-2 (tutupan awan < 15%) dari AWS Earth Search, tanpa akun
Atau letakkan sendiri PNG/JPG RGB (mis. hasil unduhan Copernicus) di data/ dan lewati skrip ini.
"""
import argparse
import os

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

# shortcut: koordinat perkiraan area Lampung (kota, hutan/rawa, lahan pertanian); ganti sesuai minat
SITES = {"T1": (105.26, -5.43), "T2": (105.78, -4.95), "T3": (104.97, -5.36)}


def synthetic(seed, size):
    rng = np.random.default_rng(seed)

    def field(sigma):
        f = ndi.gaussian_filter(rng.standard_normal((size, size)), sigma)
        return (f - f.mean()) / f.std()

    land = field(60) + 0.5 * field(20)
    cls = np.digitize(land, np.quantile(land, [0.2, 0.45, 0.7, 0.9]))
    palette = np.array([[34, 70, 40], [60, 100, 50], [120, 130, 70], [150, 125, 90], [170, 160, 150]], float)
    rgb = palette[cls] + field(2)[..., None] * 8 + field(6)[..., None] * 6
    x = np.arange(size)
    river = (size * (0.3 + 0.1 * seed) + 80 * np.sin(x / 140.0 + seed)).astype(int)
    for dx in range(-6, 7):
        rgb[x, np.clip(river + dx, 0, size - 1)] = [60, 90, 120]
    for _ in range(40):  # blok permukiman
        r, c, h, w = rng.integers(0, size - 60, 2).tolist() + rng.integers(10, 50, 2).tolist()
        rgb[r:r + h, c:c + w] = [175, 165, 160] + rng.normal(0, 6, 3)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def sentinel(lon, lat, size):
    import requests
    import rasterio
    from rasterio.warp import transform
    from rasterio.windows import Window
    q = {"collections": ["sentinel-2-l2a"], "intersects": {"type": "Point", "coordinates": [lon, lat]},
         "datetime": "2020-01-01T00:00:00Z/2020-12-31T23:59:59Z", "query": {"eo:cloud_cover": {"lt": 15}},
         "sortby": [{"field": "properties.eo:cloud_cover", "direction": "asc"}], "limit": 1}
    r = requests.post("https://earth-search.aws.element84.com/v1/search", json=q, timeout=60)
    r.raise_for_status()
    href = r.json()["features"][0]["assets"]["visual"]["href"]  # TCI 10 m, 3 band RGB uint8
    with rasterio.open(href) as ds:
        xs, ys = transform("EPSG:4326", ds.crs, [lon], [lat])
        row, col = ds.index(xs[0], ys[0])
        return ds.read([1, 2, 3], window=Window(col - size // 2, row - size // 2, size, size)).transpose(1, 2, 0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["synthetic", "sentinel"], default="synthetic")
    ap.add_argument("--size", type=int, default=1024)
    a = ap.parse_args()
    os.makedirs("data", exist_ok=True)
    for i, (name, (lon, lat)) in enumerate(SITES.items()):
        img = synthetic(i, a.size) if a.source == "synthetic" else sentinel(lon, lat, a.size)
        Image.fromarray(img).save(f"data/{name}.png")
        print("tersimpan", f"data/{name}.png", img.shape)
    open("data/SOURCE.txt", "w").write("synthetic (citra sintetis bergaya lanskap)" if a.source == "synthetic"
                                       else "Sentinel-2 L2A (AWS Earth Search), potongan tutupan awan rendah")
