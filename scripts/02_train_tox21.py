"""Tox21 (12 uc nokta) icin bu proje kapsaminda yeni siniflandirici egitimi.
Morgan sayim parmak izi (r=2, 2048) + RandomForest (class_weight=balanced_subsample).
Dogrulama: Bemis-Murcko iskelet gruplari ile %80/%20 bolme (iskelet sizintisi yok).
Nihai model tum veriyle yeniden egitilir, joblib olarak kaydedilir.
"""
import os, json, numpy as np, pandas as pd, joblib
from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.MolStandardize import rdMolStandardize
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import GroupShuffleSplit
RDLogger.DisableLog("rdApp.*")

BASE = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(BASE, "tox21"); os.makedirs(os.path.join(OUT, "models"), exist_ok=True)
TASKS = ["NR-AR","NR-AR-LBD","NR-AhR","NR-Aromatase","NR-ER","NR-ER-LBD","NR-PPAR-gamma",
         "SR-ARE","SR-ATAD5","SR-HSE","SR-MMP","SR-p53"]
SEED = 42

df = pd.read_csv(os.path.join(OUT, "tox21.csv"))
chooser = rdMolStandardize.LargestFragmentChooser(); unch = rdMolStandardize.Uncharger()
gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

recs = []
for i, r in df.iterrows():
    m = Chem.MolFromSmiles(r.smiles)
    if m is None: continue
    m = unch.uncharge(chooser.choose(m))
    if m.GetNumHeavyAtoms() < 3 or not any(a.GetSymbol() == "C" for a in m.GetAtoms()): continue
    recs.append((Chem.MolToInchiKey(m), Chem.MolToSmiles(m), m, r[TASKS].values.astype(float)))
print("gecerli molekul:", len(recs), "/", len(df))

# InChIKey ile tekrarlari birlestir: etiketler ortalamaya, ortalama 0.5 ise celiski -> NaN
by = {}
for ik, smi, m, y in recs: by.setdefault(ik, []).append((smi, m, y))
keys, smis, mols, Y = [], [], [], []
for ik, lst in by.items():
    Ys = np.vstack([l[2] for l in lst])
    with np.errstate(all="ignore"):
        mean = np.nanmean(Ys, axis=0)
    y = np.where(np.isnan(mean), np.nan, np.where(mean > 0.5, 1.0, np.where(mean < 0.5, 0.0, np.nan)))
    keys.append(ik); smis.append(lst[0][0]); mols.append(lst[0][1]); Y.append(y)
Y = np.vstack(Y)
print("benzersiz bilesik:", len(keys))

X = np.zeros((len(mols), 2048), dtype=np.uint8)
for i, m in enumerate(mols):
    fp = gen.GetCountFingerprintAsNumPy(m); X[i] = np.minimum(fp, 255)
scaf = [MurckoScaffold.MurckoScaffoldSmiles(mol=m) or smis[i] for i, m in enumerate(mols)]  # iskeletsiz: kendi grubu
groups = pd.factorize(pd.Series(scaf))[0]
tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED).split(X, groups=groups))
print(f"iskelet bolme: egitim {len(tr)}, test {len(te)}")

res = []
for t_i, t in enumerate(TASKS):
    y = Y[:, t_i]; lab = ~np.isnan(y)
    tri, tei = tr[lab[tr]], te[lab[te]]
    rf = RandomForestClassifier(n_estimators=500, class_weight="balanced_subsample", min_samples_leaf=1,
                                n_jobs=-1, random_state=SEED)
    rf.fit(X[tri], y[tri])
    p = rf.predict_proba(X[tei])[:, 1]
    auc = roc_auc_score(y[tei], p); ap = average_precision_score(y[tei], p)
    base = y[tei].mean()
    res.append(dict(task=t, n_train=len(tri), n_test=len(tei), pos_rate_test=round(base, 3),
                    roc_auc=round(auc, 3), pr_auc=round(ap, 3), pr_auc_baseline=round(base, 3)))
    print(res[-1])
    # nihai model: tum etiketli veriyle
    full = RandomForestClassifier(n_estimators=500, class_weight="balanced_subsample", n_jobs=-1, random_state=SEED)
    full.fit(X[lab], y[lab]); joblib.dump(full, os.path.join(OUT, "models", f"rf_{t}.joblib"), compress=3)

R = pd.DataFrame(res); R.to_csv(os.path.join(OUT, "tox21_scaffold_split_metrics.csv"), index=False)
print("\nOrtalama ROC-AUC:", R.roc_auc.mean().round(3), " PR-AUC:", R.pr_auc.mean().round(3))
# egitim parmak izlerini (uygulanabilirlik alani icin) kaydet
np.savez_compressed(os.path.join(OUT, "tox21_train_fp.npz"), X=X, keys=np.array(keys), smiles=np.array(smis))
