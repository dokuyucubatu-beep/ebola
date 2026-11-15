# Model kartı — Filovirüs (Marburg/Ebola) giriş testi aktivite tarama modeli

Durum: araştırma prototipi. Doğrulanmış bir potens veya tasarım aracı **değildir**.

## Ne tahmin ediyor
Bir bileşiğin hücre bazlı filovirüs **giriş (entry) testinde** aktif (inhibitör) çıkma eğilimini sıralar.
Ölçülen şey virüs girişi inhibisyonudur. Bağlanma afinitesi veya hedef (ör. NPC1, GP) özgüllüğü değildir.
Aktif etiketi, hücrelerde lizozomotropi (katyonik amfifilik ilaç etkisi), sitotoksisite ve test artefaktlarını da içerir.

## Eğitim verisi
- Kaynak: PubChem AID 540276 (NCGC, Marburg giriş qHTS). Etiket: Outcome=Active ve Phenotype=Inhibitor → 1; Inactive → 0. Inconclusive ve aktivatörler atıldı.
- 204 989 benzersiz bileşik (4 471 aktif, %2,2; aktif medyan potens 7,9 µM). Eğitim örneği: tüm aktifler + 60 000 rastgele inaktif (64 471; aktif %6,9).
- Standardizasyon: en büyük organik fragman, nötralizasyon, InChIKey tekilleştirme; çelişkili tekrarlar atıldı.

## Model
RandomForest (400 ağaç, class_weight=balanced_subsample), 217 RDKit 2D tanımlayıcı + 167 MACCS biti; temizleme + korelasyon filtresi (|r|>0,9, yalnız eğitimde).
Çıktı **kalibre değildir**; olasılık değil, sıralama skorudur. Örnekleme ve sınıf ağırlığı nedeniyle gerçek aktif oranını yansıtmaz.

## Doğrulama (results\marburg_cv.csv, marburg_auc_by_similarity.csv, validation_run.log)
| Test | Sonuç |
|---|---|
| Bemis-Murcko iskelet-gruplu 5-kat CV | AUC 0,847; PR-AUC 0,387 (örnek aktif oranı 0,069); EF1% 10,8 |
| Test→eğitim benzerlik aralığı AUC | 0,80–0,89 (benzerlik düştükçe belirgin düşüş yok) |
| CAD kuralı (bazik amin × logP) tek başına | AUC 0,59; küçük LogReg 0,74 |
| Bazik amin taşımayan bileşikler | RF 0,83; CAD 0,50 |
| y-randomizasyon (3 tekrar) | AUC ≈ 0,50 (sızıntı yok) |
| Aktarım: NCGC Ebola (AID 1117312; Marburg ile ortak 641 çıkarıldı, n=1883) | AUC 0,81; PR-AUC 0,61 (aktif oranı 0,25); CAD 0,73 |
| Aktarım: ChEMBL Ebola (1 µM sınıf, n=510) | AUC 0,57 (CAD 0,55) |
| Aktarım: ChEMBL pseudovirüs pX (n=274) | Spearman ρ ≈ 0,00 |

## Uygulanabilirlik alanı (AD)
Morgan2 sayım parmak izi, eğitim örneğine en yakın komşu Tanimoto: ≥0,5 "içinde", 0,3–0,5 "sınırda", <0,3 "dışında".
Bu eşikler sezgiseldir; CV'de benzerlik aralıkları arasında kesin bir AUC kırılması gözlenmedi. "Dışında" bileşiklerde skora güvenmeyin.
`egitimde_marburg=True` ise bileşik eğitim verisindedir; skoru eğitimden hatırlanmıştır, tahmin sayılmaz.

## Tox21 uyarı bayrağı (isteğe bağlı destek)
Bu proje için eğitilmiş 12 uç nokta RF (Morgan sayım r2, 2048). İskelet-bölmeli test AUC 0,75–0,95 (SR-MMP 0,87; SR-p53 0,83; SR-ARE 0,79; tox21\tox21_scaffold_split_metrics.csv).
`tox21_uyari` yalnızca SR-MMP, SR-p53, SR-ARE ≥0,5 ise dolar; **zayıf bir uyarıdır**, sitotoksisitenin kanıtı veya yokluğu değildir.
Tox21 kimyasal uzayı ilaç benzeri değildir; `tox21_ad_benzerlik` düşükse bayrağa güvenmeyin. `egitimde_tox21=True` ise skor eğitimden hatırlanmıştır.
Ebola kümelerinde bayrak oranı düşük (SR-MMP ≈ %6–8), yani ayırt edici güç sınırlıdır.

## CAD bayrağı
Bazik amin ve Crippen logP ≥ 3 ise `CAD_bayragi=True`: lizozomotropik/katyonik amfifilik aday. Bu bileşiklerin aktifliği çoğunlukla bu mekanizmadan gelebilir.

## Bilinen sınırlar
- Aktif etiketi sitotoksisiteyi ayırmaz (karşı tarama yok). Hedefe özgü aktivite iddiası yapılamaz.
- Gerçek tasarım verisine (ChEMBL, farklı laboratuvarlar ve testler) aktarım yok. Potens (IC50) tahmini yok.
- Marburg ve NCGC Ebola aynı laboratuvar/platform; aktarım başarısı platform ortaklığından kaynaklanabilir.
- İskelet bölmesi yumuşaktır (34 534 grup, çoğu tekil); yakın analoglar ayrılmamış olabilir.
- PR-AUC/EF örnek prevalansında (%6,9) hesaplandı; gerçek kütüphane prevalansı (%2,2) için kesinlik daha düşük olur.
- Veri 2D; 3D/konformer bilgisi yok.

## Önerilen kullanım
Kütüphane taramasında öncelik sıralaması: yüksek skor + AD "içinde" + CAD/Tox21 bayrakları birlikte okunur. Bayraklı aktifler için ortogonal doğrulama (sitotoksisite, lizozomotropi karşı tarama, hedef testi) şarttır.

## Kullanım
```
python scripts/12_predict.py girdi.csv [cikti.csv]
```
Ortam: `qsar` conda ortamı (scikit-learn, RDKit); `scripts\` klasöründeki `qsar_common.py` import edilebilir olmalı.

## Analog önerici (scripts/13_analogs.py)
Serbest üretici değildir. Ana moleküle 24 tanıdık medkim dönüşümü (aromatik H→F/Cl/CH3/OCH3/CF3/CN/OH, Cl↔F, piperidin→morfolin, asit→tetrazol vb.) en çok 2 adım uygulanır (ağır atom farkı ≤ 8) ve her analog `12_predict.py` ile skorlanır.
Çıktı: ana moleküle göre skor değişimi, gürültü bayrağı (`fark_gurultude`: |Δ| < 0,10 veya RF standart hatasının 2 katı), AD, CAD, Tox21, PAINS.

**Eşleşmiş çift testi (scripts/14_mmp_validation.py, results/mmp_validation_pairs.csv, results/mmp_run.log):** Marburg'da aktif A ve tek dönüşümle ulaşılan inaktif B (n=377 çift, 336 aktif). Model p(A)>p(B) yönünü %62,3 doğru buluyor (%95 GA 56,8–67,6; şans %50), ama "yüksek logP kazanır" tabanı %59,9 (54,6–65,0) ile ayırt edilemiyor. Kural bazında: "CH3 sil" %82 ve "halojen sil" %71 (substituent çıkarılınca skor düşüyor), buna karşılık "H→CH3" %29 ve "H→OCH3" %39 (şans altı: model eklenen metil/metoksi analoğunu daha aktif sayıyor, veride aktif olan ise ana molekül). |Δskor| ≥ 0,1 olan az sayıda çiftte (n=39) %72, ≥ 0,2'de (n=10) %60.
**Sonuç:** Skor değişimi büyüklük/lipofiliklik eğilimini izliyor, ikame bazında SAR öngörmüyor. Analog önericiyi yalnızca **yapısal çeşitlendirme ve bayrak kontrolü (AD, CAD, Tox21, PAINS) için** kullanın; skor farkıyla analog sıralamayın. Sınırlar: yalnız tek adım, 7 kuralın n≥15 olduğu çiftler; çiftlerin 108'inde B örneklemde, 262'sinde değil; seçim aktiften başladığı için yön yanlılığı olabilir.
CAD azaltan dönüşümler yalnızca piperidin→morfolin/N-asetilpiperazin ve N,N-dimetil→asetamid ile sınırlı; dietilamin gibi aminlerde CAD bayrağı kalkmaz.
Serbest bir optimizasyon döngüsü bilerek yok: skor CAD/lizozomotropi ile karışık olduğundan döngü bazik amin × yüksek logP molekülleri üretirdi.
