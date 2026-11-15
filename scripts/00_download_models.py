"""Eğitilmiş modelleri indirir, SHA256'yı models_manifest.json ile doğrular ve depo içindeki beklenen klasörlere açar.
Kullanım (depo kökünden):
  python scripts/00_download_models.py                 # tüm paketler
  python scripts/00_download_models.py --yalniz marburg_model_v1.zip
  python scripts/00_download_models.py --kaynak yerel_klasor   # url yerine yerel dist/ benzeri klasörden (dosya adıyla)
GÜVENLİK: joblib dosyaları açılırken kod çalıştırabilir. Hash eşleşmezse hiçbir şey açılmaz. Manifesti yalnızca güvendiğiniz depodan alın."""
import os, sys, json, hashlib, zipfile, shutil, argparse, urllib.request, tempfile
BASE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

def safe_extract(zpath, dest):
    with zipfile.ZipFile(zpath) as z:
        for n in z.namelist():  # zip-slip koruması
            t = os.path.abspath(os.path.join(dest, n))
            if not t.startswith(dest + os.sep): sys.exit(f"guvensiz zip girdisi: {n}")
        z.extractall(dest)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--yalniz", action="append"); ap.add_argument("--kaynak")
    a = ap.parse_args()
    man = json.load(open(os.path.join(BASE, "models_manifest.json"), encoding="utf-8"))
    for name, info in man["paketler"].items():
        if a.yalniz and name not in a.yalniz: continue
        tmp = tempfile.mkdtemp(); zp = os.path.join(tmp, name)
        try:
            if a.kaynak: shutil.copy(os.path.join(a.kaynak, name), zp)
            else:
                if not info.get("url"): sys.exit(f"{name}: manifestte url bos (model paketi henuz yayimlanmamis). Bkz. README.")
                print(f"indiriliyor: {name} ({info['bayt']/1e6:.0f} MB)"); urllib.request.urlretrieve(info["url"], zp)
            if sha(zp) != info["sha256"]: sys.exit(f"{name}: SHA256 eslesmiyor, acilmadi.")
            safe_extract(zp, BASE); print(f"tamam: {name} dogrulandi ve acildi")
        finally: shutil.rmtree(tmp, ignore_errors=True)
    e = man["ortam"]; print(f"Not: modeller scikit-learn {e['scikit-learn']}, rdkit {e['rdkit']} ile uretildi; farkli surum uyari/hata verebilir.")
