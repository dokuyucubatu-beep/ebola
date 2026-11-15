# Filovirüs giriş testi QSAR (Ebola/Marburg): araştırma prototipi

> **Uyarı.** Bu bir araştırma prototipidir. **Potens tahmin etmez, hedefe özgü aktivite iddiası taşımaz, tasarım aracı değildir.**
> Tahmin ettiği şey, hücre bazlı bir filovirüs giriş testinde "aktif" çıkma eğilimidir; bu etiket lizozomotropi (katyonik amfifilik ilaç etkisi), sitotoksisite ve test artefaktlarını da içerir.
> Ayrıntılar ve tüm sınırlar için önce [MODEL_CARD.md](MODEL_CARD.md) dosyasını okuyun.

## Özet bulgular
- Marburg giriş qHTS'inde (PubChem AID 540276) iskelet-gruplu CV AUC 0,85; aynı platformdaki NCGC Ebola testine aktarım AUC 0,81. Model, "bazik amin × logP" kuralını geçiyor ve bazik amin taşımayan bileşiklerde de çalışıyor. y-randomizasyonda AUC ≈ 0,50.
- ChEMBL'deki Ebola verisine (farklı laboratuvarlar, potens) aktarım yok (AUC 0,57, pX Spearman ≈ 0). ChEMBL'de doküman gruplu CV, rastgele CV'nin ne kadar iyimser olduğunu gösteriyor.
- Analog önerici (`13_analogs.py`) eşleşmiş çift testinde baz çizgiden ayırt edilemedi (%62 vs logP %60); yalnızca yapısal çeşitlendirme ve bayrak kontrolü içindir.
- Bu depo aynı zamanda ChEMBL Ebola SAR verisinin sağlamlığı üzerine bir **negatif sonuç / kıyaslama** çalışmasıdır.

## Dizin
```
scripts/   01–14 numaralı betikler (aşağıya bakın), qsar_common.py, test_molecules.csv
results/   küçük özet tablolar ve günlükler (büyük dosyalar depoda yok)
tox21/     Tox21 metrik tabloları (modeller ve ham veri depoda yok)
MODEL_CARD.md
```

## Kurulum
```bash
conda env create -f environment.yml
conda activate qsar-filovirus
```
Betikler depo kökünden çalıştırılır (`python scripts/<ad>.py`). Windows'ta Türkçe çıktı için `PYTHONIOENCODING=utf-8` ayarlayın.

## Hazır modelleri indirmek (yalnızca skorlamak isteyenler için)
Eğitilmiş modeller ve uygulanabilirlik referansları depoda değil, ayrı bir yayın paketinde (Release/Zenodo). `models_manifest.json` her paketin adresini ve SHA256'sını taşır:
```bash
python scripts/00_download_models.py
python scripts/12_predict.py girdi.csv cikti.csv
```
Betik hash eşleşmezse hiçbir şey açmaz (joblib dosyaları kod çalıştırabilir). Modeller scikit-learn 1.9.0 / rdkit 2026.03.4 ile üretildi; farklı sürüm hata veya uyarı verebilir. Eğitim verisinin kendisi (Marburg örneği, Tox21 eğitim parmak izleri) paketlerde yalnızca AD referansı olarak bulunur.
Paketlerde yayımlanan `models_manifest.json` içindeki `url` alanları boşsa paket henüz yayımlanmamıştır; bu durumda aşağıdaki veriyi indirip `02`, `09`, `10` betikleriyle modelleri kendiniz üretin.

## Veri (depoda yok, siz indirin)
Ham veri ve eğitilmiş modeller depoya konmadı (boyut ve lisans). İndirmeler betiklenmedi; kaynaklar:

| Veri | Kaynak | Nereye |
|---|---|---|
| ChEMBL Ebola (organizma hedefi CHEMBL4434621; ayrıca CHEMBL4105829, CHEMBL4513181, CHEMBL5291667) | ChEMBL REST API (`/chembl/api/data/activity.json?target_chembl_id=...`), sayfalı JSON | `data/act_<hedef>_<offset>.json` |
| NCGC Ebola giriş qHTS (AID 1117312 ve ilişkili 1117303, 1117305, 1117308, 1117313, 1117318) | PubChem (AID veri tablosu CSV) | `data/pubchem/aid_<AID>.csv` |
| Marburg giriş qHTS (AID 540276) | PubChem FTP toplu AID paketi `0540001_0541000` | `data/marburg_dl/0540001_0541000/540276.csv.gz` |
| Tox21 (12 uç nokta) | MoleculeNet Tox21 CSV'si (tox21.csv; bu çalışmada başka bir projeden kopyalandı, kaynağını ve lisansını kendiniz doğrulayın) | `tox21/tox21.csv` |

Lisans/atıf: ChEMBL CC BY-SA 3.0; PubChem/NCGC verileri kaynak gösterilerek kullanılmalıdır; Tox21 için ilgili kullanım koşullarına bakın. Bu depo ham veriyi yeniden dağıtmaz.

## Betikler (sırayla)
| Betik | İş |
|---|---|
| `01_curate.py` | ChEMBL Ebola kürasyonu (regresyon 275, sınıflandırma 522; eşik 1 µM) |
| `02_train_tox21.py`, `03_inspect_tox21.py` | Tox21 RF modelleri ve uygulanabilirlik incelemesi |
| `04_features.py`, `05_cv_compare.py`, `06_fingerprint_only.py`, `06_why_fingerprints.py`, `07_doc_zscore.py` | ChEMBL üzerinde özellik/model karşılaştırması, doküman gruplu CV, doküman z-skoru |
| `08_ncgc_classification.py` | NCGC Ebola sınıflandırma ve ChEMBL dış testi |
| `09_marburg_curate.py`, `10_marburg_model.py` | Marburg kürasyonu, model, Ebola'ya aktarım |
| `11_validation.py` | Taban çizgileri (CAD kuralı), benzerlik aralığı AUC, y-randomizasyon |
| `12_predict.py` | Kendi moleküllerinizi skorlayın |
| `13_analogs.py` | Analog önerici (yapısal çeşitlendirme) |
| `14_mmp_validation.py` | Analog yönü için eşleşmiş çift doğrulaması |

Not: `12_predict.py` ve `13_analogs.py` için `results/marburg_model.joblib`, `results/marburg_sample_features.npz` ve `tox21/models/` gerekir: ya `00_download_models.py` ile indirin ya da `02`, `09`, `10` betiklerini çalıştırın. `00_download_models.py` ve `build_release.py` (paketleri yayına hazırlar) bu tablonun dışındaki yardımcı betiklerdir.

## Kendi moleküllerinizi skorlamak
```bash
python scripts/12_predict.py girdi.csv cikti.csv      # girdi: 'smiles' sütunlu CSV
python scripts/13_analogs.py "SMILES" analoglar.csv   # tek ana molekülden analoglar
```
Çıktıdaki `ad_bandi`, `egitimde_marburg`, `CAD_bayragi` ve `tox21_uyari` sütunlarını skorla birlikte okuyun. `egitimde_*=True` ise skor eğitimden hatırlanmıştır, tahmin sayılmaz.

## Bilinen sınırlar (kısa)
Sitotoksisite karşı taraması yok; Marburg ve NCGC Ebola aynı laboratuvar/platform; iskelet bölmesi yumuşak; PR-AUC ve EF örnek prevalansında (%6,9, gerçek %2,2); skor kalibre değil; analog skor değişimi SAR öngörmüyor.

## Lisans
Kod MIT lisanslıdır ([LICENSE](LICENSE)). Veri kaynaklarının lisansları ayrıdır (yukarı bakın).
