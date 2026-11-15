"""Yalniz parmak izi modelleri + 'neden dustu' tanilari.
Tani: (1) pX varyansinin dokumanlar arasi payi, (2) doc-CV vs rastgele CV'de test->egitim benzerligi,
      (3) siniflandirmada dokuman basina aktif orani.
Modeller: RF (sayim), Tanimoto-cekirdekli SVR/SVC, Tanimoto kNN. Dogrulama: dokuman-gruplu 5-kat (birincil), rastgele (karsilastirma).
Dokuman-ici metrik: >=10 bilesikli dokumanlarda ortalama Spearman rho / AUC (laboratuvar kaymasindan bagimsiz)."""
import os, sys, warnings, numpy as np, pandas as pd
from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import rdFingerprintGenerator, MACCSkeys, rdMolDescriptors
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.metrics import r2_score, mean_squared_error, roc_auc_score, average_precision_score
from scipy.stats import spearmanr
warnings.filterwarnings("ignore"); RDLogger.DisableLog("rdApp.*")
R = os.path.join(os.path.dirname(__file__), "..", "results"); SEED = 7

def fpset(smiles):
    ms = [Chem.MolFromSmiles(s) for s in smiles]
    G = {"morgan1": rdFingerprintGenerator.GetMorganGenerator(radius=1, fpSize=2048),
         "morgan2": rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048),
         "morgan3": rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=2048),
         "fcfp4": rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048,
                    atomInvariantsGenerator=rdFingerprintGenerator.GetMorganFeatureAtomInvGen()),
         "atompair": rdFingerprintGenerator.GetAtomPairGenerator(fpSize=2048),
         "torsion": rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=2048),
         "rdkit": rdFingerprintGenerator.GetRDKitFPGenerator(maxPath=5, fpSize=2048)}
    out = {k: np.array([g.GetCountFingerprintAsNumPy(m) for m in ms], dtype=np.float32) for k, g in G.items()}
    out["maccs"] = np.array([list(MACCSkeys.GenMACCSKeys(m)) for m in ms], dtype=np.float32)
    return out

def tani(A, B):
    A = (A > 0).astype(np.float32); B = (B > 0).astype(np.float32)
    inter = A @ B.T
    return inter / (A.sum(1)[:, None] + B.sum(1)[None, :] - inter + 1e-9)

def fit_predict(kind, clf, Xtr, ytr, Xte):
    if kind == "RF":
        m = (RandomForestClassifier(300, n_jobs=-1, random_state=SEED, class_weight="balanced_subsample") if clf
             else RandomForestRegressor(300, n_jobs=-1, random_state=SEED, max_features=0.33)).fit(Xtr, ytr)
        return m.predict_proba(Xte)[:, 1] if clf else m.predict(Xte)
    Ktr, Kte = tani(Xtr, Xtr), tani(Xte, Xtr)
    if kind == "TanSVM":
        if clf: return SVC(C=1, kernel="precomputed", class_weight="balanced").fit(Ktr, ytr).decision_function(Kte)
        return SVR(C=3, epsilon=0.1, kernel="precomputed").fit(Ktr, ytr).predict(Kte)
    if kind == "TanKNN":  # benzerlik agirlikli 5-en yakin komsu
        idx = np.argsort(-Kte, 1)[:, :5]; w = np.take_along_axis(Kte, idx, 1) + 1e-6
        return (w * ytr[idx]).sum(1) / w.sum(1)

def oof(kind, clf, X, y, splits):
    p = np.zeros(len(y))
    for tr, te in splits: p[te] = fit_predict(kind, clf, X[tr], y[tr], X[te])
    return p

def within_doc(y, p, g, clf, minn=10):
    vals = []
    for d in pd.unique(g):
        i = np.where(g == d)[0]
        if len(i) < minn: continue
        if clf:
            if 0 < y[i].sum() < len(i): vals.append(roc_auc_score(y[i], p[i]))
        else:
            r = spearmanr(y[i], p[i])[0]
            if np.isfinite(r): vals.append(r)
    return (np.mean(vals) if vals else np.nan), len(vals)

def run(tag, clf):
    z = np.load(os.path.join(R, f"features_{tag}.npz"), allow_pickle=True)
    y = z["y"].astype(float); g = z["group"]; smi = z["smiles"]
    print(f"\n===== {'SINIFLANDIRMA' if clf else 'REGRESYON'}: n={len(y)}, dokuman={len(set(g))} =====")
    FP = fpset(smi)
    gsp = list(GroupKFold(5).split(np.zeros(len(y)), y, g))
    rsp = list((StratifiedKFold(5, shuffle=True, random_state=SEED) if clf else KFold(5, shuffle=True, random_state=SEED)).split(np.zeros(len(y)), y))
    # --- tanilar
    if not clf:
        df = pd.DataFrame(dict(y=y, g=g)); grand = y.mean()
        ssb = sum(len(s) * (s.y.mean() - grand) ** 2 for _, s in df.groupby("g")); sst = ((y - grand) ** 2).sum()
        print(f"[tani 1] pX varyansinin dokumanlar arasi payi: {ssb / sst:.2f} (dokuman ici: {1 - ssb / sst:.2f})")
    else:
        rate = pd.DataFrame(dict(y=y, g=g)).groupby("g").y.agg(["mean", "size"])
        big = rate[rate["size"] >= 10]
        print(f"[tani 3] >=10 bilesikli {len(big)} dokumanda aktif orani: hepsi aktif/inaktif (>=90% ya da <=10%) olan: "
              f"{int(((big['mean'] >= .9) | (big['mean'] <= .1)).sum())}; dokumanlar arasi aktif orani sd {big['mean'].std():.2f}")
    X2 = FP["morgan2"]
    for nm, sp in (("doc-CV", gsp), ("rastgele-CV", rsp)):
        ms = np.concatenate([tani(X2[te], X2[tr]).max(1) for tr, te in sp])
        print(f"[tani 2] {nm}: test->egitim en yakin komsu Tanimoto medyan {np.median(ms):.2f}, >=0.6 olan %{(ms >= .6).mean() * 100:.0f}")
    # --- modeller
    rows = []
    for fs, X in FP.items():
        for kind in ("RF", "TanSVM", "TanKNN"):
            row = dict(fp=fs, model=kind)
            for nm, sp in (("doc", gsp), ("rand", rsp)):
                p = oof(kind, clf, X, y, sp)
                if clf: row[f"auc_{nm}"] = roc_auc_score(y, p); row[f"prauc_{nm}"] = average_precision_score(y, p)
                else: row[f"r2_{nm}"] = r2_score(y, p); row[f"rho_{nm}"] = spearmanr(y, p)[0]
                if nm == "doc":
                    w, nd = within_doc(y, p, g, clf); row["icdoc_" + ("auc" if clf else "rho")] = w; row["n_doc_ic"] = nd
            rows.append(row); print({k: (round(float(v), 3) if isinstance(v, (float, np.floating)) else v) for k, v in row.items()}, flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(R, f"fp_only_{tag}.csv"), index=False)

if __name__ == "__main__":
    for t in sys.argv[1:] or ["reg", "clf"]: run(t, t == "clf")
