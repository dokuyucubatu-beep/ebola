"""Eğitilmiş model paketlerini dist/ altında zipler ve models_manifest.json'a SHA256 + boyut yazar (URL'yi yayıncı doldurur).
Paketler: marburg_model_v1.zip (Marburg RF + AD referans örneği), tox21_models_v1.zip (12 RF + AD referansı).
Çalıştırma (depo kökünden): python scripts/build_release.py"""
import os, json, zipfile, hashlib, sys, platform
import sklearn, rdkit, numpy, pandas, joblib
BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
PACKS = {
    "marburg_model_v1.zip": ["results/marburg_model.joblib", "results/marburg_sample_features.npz"],
    "tox21_models_v1.zip": ["tox21/tox21_train_fp.npz"] + [f"tox21/models/rf_{t}.joblib" for t in
        ["NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"]],
}
def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

if __name__ == "__main__":
    dist = os.path.join(BASE, "dist"); os.makedirs(dist, exist_ok=True)
    man = {"surum": "v1", "ortam": {"python": platform.python_version(), "scikit-learn": sklearn.__version__, "rdkit": rdkit.__version__,
           "numpy": numpy.__version__, "pandas": pandas.__version__, "joblib": joblib.__version__},
           "uyari": "joblib/pickle dosyalari kod calistirabilir; yalnizca bu manifestteki SHA256 ile eslesen paketleri yukleyin.", "paketler": {}}
    for name, files in PACKS.items():
        out = os.path.join(dist, name)
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            for f in files:
                p = os.path.join(BASE, f)
                if not os.path.exists(p): sys.exit(f"eksik dosya: {f}")
                z.write(p, f)
        man["paketler"][name] = {"url": "", "sha256": sha(out), "bayt": os.path.getsize(out), "icerik": files}
        print(f"{name}: {os.path.getsize(out)/1e6:.1f} MB  sha256 {man['paketler'][name]['sha256'][:16]}...")
    eski = os.path.join(BASE, "models_manifest.json")
    if os.path.exists(eski):  # daha once doldurulmus URL'leri koru
        for n, v in json.load(open(eski, encoding="utf-8")).get("paketler", {}).items():
            if n in man["paketler"] and v.get("url"): man["paketler"][n]["url"] = v["url"]
    json.dump(man, open(eski, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print("yazildi: models_manifest.json (url alanlarini yayindan sonra doldurun, depoya commit edin)")
