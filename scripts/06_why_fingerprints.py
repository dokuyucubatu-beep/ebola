"""1) Neden rastgele CV >> dokuman-gruplu CV? (dokumanlar-arasi varyans, test->egitim benzerligi)
2) Yalniz parmak izi modelleri: farkli FP turleri x (RF, Tanimoto/MinMax-kernel SVM) ; doc-CV ve rastgele CV;
   ek olarak dokuman-ici (within-document) Spearman."""
import os, sys, warnings, numpy as np, pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator as G
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.metrics import r2_score, roc_auc_score, average_precision_score
from scipy.stats import spearmanr
RDLogger.DisableLog("rdApp.*"); warnings.filterwarnings("ignore")
R = os.path.join(os.path.dirname(__file__), "..", "results"); SEED = 7

gens = {
 "ECFP4_bit":   (G.GetMorganGenerator(radius=2, fpSize=2048), False),
 "ECFP4_cnt":   (G.GetMorganGenerator(radius=2, fpSize=2048), True),
 "ECFP6_cnt":   (G.GetMorganGenerator(radius=3, fpSize=2048), True),
 "FCFP4_cnt":   (G.GetMorganGenerator(radius=2, fpSize=2048, atomInvariantsGenerator=G.GetMorganFeatureAtomInvGen()), True),
 "AtomPair_cnt":(G.GetAtomPairGenerator(fpSize=2048), True),
 "Torsion_cnt": (G.GetTopologicalTorsionGenerator(fpSize=2048), True),
 "RDKitFP_bit": (G.GetRDKitFPGenerator(maxPath=6, fpSize=2048), False),
}
def fp(smiles, name):
    g, cnt = gens[name]; out = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        out.append(g.GetCountFingerprintAsNumPy(m) if cnt else g.GetFingerprintAsNumPy(m))
    return np.array(out, dtype=np.float32)

def minmax(A, B):
    K = np.zeros((len(A), len(B)), dtype=np.float64)
    for i in range(len(A)):
        mn = np.minimum(A[i], B).sum(1); mx = np.maximum(A[i], B).sum(1); K[i] = mn / np.maximum(mx, 1e-9)
    return K

def cv_pred(X, y, splits, kind, clf):
    p = np.zeros(len(y))
    for tr, te in splits:
        if kind == "RF":
            m = (RandomForestClassifier(300, n_jobs=-1, random_state=SEED, class_weight="balanced_subsample") if clf
                 else RandomForestRegressor(300, n_jobs=-1, random_state=SEED, max_features=0.33))
            m.fit(X[tr], y[tr]); p[te] = m.predict_proba(X[te])[:, 1] if clf else m.predict(X[te])
        else:  # Tanimoto (MinMax) kernel SVM
            Ktr = minmax(X[tr], X[tr]); Kte = minmax(X[te], X[tr])
            m = (SVC(kernel="precomputed", C=1, class_weight="balanced") if clf else SVR(kernel="precomputed", C=3, epsilon=0.1))
            m.fit(Ktr, y[tr]); p[te] = m.decision_function(Kte) if clf else m.predict(Kte)
    return p

def within_doc_rho(y, p, g, minn=8):
    r, w = [], []
    for d in pd.unique(g):
        i = np.where(g == d)[0]
        if len(i) >= minn and np.std(y[i]) > 0:
            c = spearmanr(y[i], p[i])[0]
            if np.isfinite(c): r.append(c); w.append(len(i))
    return (np.average(r, weights=w), len(r)) if r else (np.nan, 0)

def within_doc_auc(y, p, g, minn=8):
    a, w = [], []
    for d in pd.unique(g):
        i = np.where(g == d)[0]
        if len(i) >= minn and 0 < y[i].sum() < len(i):
            a.append(roc_auc_score(y[i], p[i])); w.append(len(i))
    return (np.average(a, weights=w), len(a)) if a else (np.nan, 0)

for tag, clf in (("reg", False), ("clf", True)):
    z = np.load(os.path.join(R, f"features_{tag}.npz"), allow_pickle=True)
    y = z["y"].astype(float); g = z["group"]; smi = z["smiles"]
    print(f"\n===== {'SINIFLANDIRMA' if clf else 'REGRESYON'} n={len(y)} dokuman={len(set(g))} =====")
    # --- 1) neden: dokumanlar-arasi varyans
    dfm = pd.DataFrame(dict(y=y, g=g)); gm = dfm.groupby("g").y.transform("mean")
    ss_b = ((gm - y.mean()) ** 2).sum(); ss_t = ((y - y.mean()) ** 2).sum()
    print("dokumanlar-arasi varyans payi (eta^2): %.2f" % (ss_b / ss_t))
    if clf:
        print("dokuman basina aktif orani min/medyan/maks: %.2f / %.2f / %.2f; tamami tek sinif olan dokuman: %d" % (
              *np.percentile(dfm.groupby("g").y.mean(), [0, 50, 100]),
              int(((dfm.groupby("g").y.mean() == 0) | (dfm.groupby("g").y.mean() == 1)).sum())))
    gsp = list(GroupKFold(5).split(np.zeros(len(y)), y, g))
    rsp = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y) if clf
               else KFold(5, shuffle=True, random_state=SEED).split(np.zeros(len(y))))
    Xb = fp(smi, "ECFP4_bit") > 0
    def nn(sp):
        v = []
        for tr, te in sp:
            a = Xb[te].astype(np.float32); b = Xb[tr].astype(np.float32); inter = a @ b.T
            v.append((inter / (a.sum(1)[:, None] + b.sum(1)[None, :] - inter)).max(1))
        return np.concatenate(v)
    print("test->egitim en yakin komsu Tanimoto medyan: rastgele CV %.2f | dokuman CV %.2f" % (np.median(nn(rsp)), np.median(nn(gsp))))
    print("dokuman-CV katlari (test boyutlari):", [len(te) for _, te in gsp])
    # --- 2) yalniz parmak izi
    rows = []
    for name in gens:
        X = fp(smi, name)
        for kind in ("RF", "TanimotoSVM"):
            row = dict(fp=name, model=kind)
            for sch, sp in (("doc", gsp), ("rand", rsp)):
                p = cv_pred(X, y, sp, kind, clf)
                if clf:
                    row[f"auc_{sch}"] = roc_auc_score(y, p); row[f"prauc_{sch}"] = average_precision_score(y, p)
                else:
                    row[f"r2_{sch}"] = r2_score(y, p); row[f"rho_{sch}"] = spearmanr(y, p)[0]
                if sch == "doc":
                    wd, nd = (within_doc_auc if clf else within_doc_rho)(y, p, g)
                    row["within_doc_" + ("auc" if clf else "rho")] = wd; row["n_docs_eval"] = nd
            rows.append(row); print({k: (round(float(v), 3) if isinstance(v, (float, np.floating)) else v) for k, v in row.items()}, flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(R, f"fp_only_{tag}.csv"), index=False)
