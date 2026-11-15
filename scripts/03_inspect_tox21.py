"""Tox21 modellerinin incelenmesi:
 1) test->egitim en yakin komsu benzerligi ve benzerlige gore AUC (sizinti / gercek genelleme)
 2) Ebola bilesiklerinin Tox21 egitim uzayina uzakligi (uygulanabilirlik)
 3) Ebola bilesiklerinde uc nokta basina tahmin edilen bayrak orani
"""
import os, numpy as np, pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.MolStandardize import rdMolStandardize
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
RDLogger.DisableLog("rdApp.*")
BASE = os.path.join(os.path.dirname(__file__), "..")
T = os.path.join(BASE, "tox21"); R = os.path.join(BASE, "results")
TASKS = ["NR-AR","NR-AR-LBD","NR-AhR","NR-Aromatase","NR-ER","NR-ER-LBD","NR-PPAR-gamma",
         "SR-ARE","SR-ATAD5","SR-HSE","SR-MMP","SR-p53"]
gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
chooser = rdMolStandardize.LargestFragmentChooser(); unch = rdMolStandardize.Uncharger()

# --- egitim verisini 02 ile ayni sekilde yeniden kur (ayni tohum -> ayni bolme)
z = np.load(os.path.join(T, "tox21_train_fp.npz"), allow_pickle=True)
X, smis, keys = z["X"], list(z["smiles"]), list(z["keys"])
raw = pd.read_csv(os.path.join(T, "tox21.csv"))
ik2y = {}; ik2scaf = {}
for _, r in raw.iterrows():
    m = Chem.MolFromSmiles(r.smiles)
    if m is None: continue
    m = unch.uncharge(chooser.choose(m))
    if m.GetNumHeavyAtoms() < 3 or not any(a.GetSymbol() == "C" for a in m.GetAtoms()): continue
    k = Chem.MolToInchiKey(m); ik2y.setdefault(k, []).append(r[TASKS].values.astype(float))
    if k not in ik2scaf: ik2scaf[k] = MurckoScaffold.MurckoScaffoldSmiles(mol=m) or Chem.MolToSmiles(m)
Y = []
for k in keys:
    Ys = np.vstack(ik2y[k])
    with np.errstate(all="ignore"): mean = np.nanmean(Ys, axis=0)
    Y.append(np.where(np.isnan(mean), np.nan, np.where(mean > 0.5, 1.0, np.where(mean < 0.5, 0.0, np.nan))))
Y = np.vstack(Y)
scaf = [ik2scaf[k] for k in keys]
groups = pd.factorize(pd.Series(scaf))[0]
tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(X, groups=groups))

def tanimoto_max(A, B, chunk=500):
    """A satirlari icin B'ye en yuksek ikili Tanimoto."""
    A = (A > 0).astype(np.float32); B = (B > 0).astype(np.float32)
    nb = B.sum(1); out = np.zeros(len(A), dtype=np.float32)
    for i in range(0, len(A), chunk):
        a = A[i:i+chunk]; inter = a @ B.T
        out[i:i+chunk] = (inter / (a.sum(1)[:, None] + nb[None, :] - inter + 1e-9)).max(1)
    return out

# 1) test->egitim benzerligi
sim_te = tanimoto_max(X[te], X[tr])
print("1) test->egitim en yakin komsu Tanimoto: medyan %.2f; >=0.7: %.0f%%; <0.4: %.0f%%" %
      (np.median(sim_te), (sim_te >= .7).mean()*100, (sim_te < .4).mean()*100))
bins = [(0, .4), (.4, .6), (.6, .8), (.8, 1.01)]
rows = []
for ti, t in enumerate(TASKS):
    y = Y[:, ti]; lab = ~np.isnan(y); tri = tr[lab[tr]]
    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample", n_jobs=-1, random_state=42)
    rf.fit(X[tri], y[tri])
    tei = np.array([j for j, i in enumerate(te) if lab[i]])
    p = rf.predict_proba(X[te][tei])[:, 1]; yt = y[te][tei]; s = sim_te[tei]
    row = dict(task=t)
    for lo, hi in bins:
        mk = (s >= lo) & (s < hi)
        ok = mk.sum() >= 30 and 0 < yt[mk].sum() < mk.sum()
        row[f"auc_{lo}-{min(hi,1):.1f}"] = round(roc_auc_score(yt[mk], p[mk]), 3) if ok else np.nan
        row[f"n_{lo}-{min(hi,1):.1f}"] = int(mk.sum())
    rows.append((row, rf))
B = pd.DataFrame([r for r, _ in rows]); B.to_csv(os.path.join(R, "tox21_auc_by_similarity.csv"), index=False)
print("\nAUC, test bileşiğinin egitim setine benzerligine gore:")
print(B[["task"] + [c for c in B.columns if c.startswith("auc_")]].to_string(index=False))
print("ortalama AUC:", B[[c for c in B.columns if c.startswith("auc_")]].mean().round(3).to_dict())

# 2-3) Ebola bilesikleri
eb = pd.concat([pd.read_csv(os.path.join(R, "ebola_regression.csv"))[["smiles"]],
                pd.read_csv(os.path.join(R, "ebola_classification.csv"))[["smiles"]]]).drop_duplicates().reset_index(drop=True)
EX = np.zeros((len(eb), 2048), dtype=np.uint8)
for i, s in enumerate(eb.smiles):
    EX[i] = np.minimum(gen.GetCountFingerprintAsNumPy(Chem.MolFromSmiles(s)), 255)
sim_eb = tanimoto_max(EX, X)
print("\n2) Ebola bilesikleri (%d) -> Tox21 egitim seti: medyan Tanimoto %.2f; >=0.5: %.0f%%; >=0.7: %.0f%%; <0.3: %.0f%%" %
      (len(eb), np.median(sim_eb), (sim_eb >= .5).mean()*100, (sim_eb >= .7).mean()*100, (sim_eb < .3).mean()*100))
print("   (karsilastirma: Tox21 test->egitim medyan %.2f)" % np.median(sim_te))
inD = sim_eb >= 0.4
full = {}
print("\n3) Ebola bilesiklerinde tahmini bayrak orani (olasilik>=0.5), tum / benzerlik>=0.4:")
out = []
for ti, (t, (_, _)) in enumerate(zip(TASKS, rows)):
    y = Y[:, ti]; lab = ~np.isnan(y)
    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample", n_jobs=-1, random_state=42).fit(X[lab], y[lab])
    p = rf.predict_proba(EX)[:, 1]
    eb[f"p_{t}"] = p
    out.append(dict(task=t, tox21_pos_rate=round(np.nanmean(y), 3), ebola_flag_all=round((p >= .5).mean(), 3),
                    ebola_flag_inD=round((p[inD] >= .5).mean(), 3) if inD.any() else np.nan))
O = pd.DataFrame(out); print(O.to_string(index=False)); print("n (benzerlik>=0.4):", int(inD.sum()))
eb["max_sim_tox21"] = sim_eb
eb.to_csv(os.path.join(R, "ebola_tox21_predictions.csv"), index=False)
O.to_csv(os.path.join(R, "tox21_ebola_flag_rates.csv"), index=False)
