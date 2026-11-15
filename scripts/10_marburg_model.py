"""Marburg giris qHTS modeli + Ebola'ya aktarim olcumu.
Ornekleme: tum aktifler + rastgele INAKTIF alt-ornek (SEED) -> egitim/CV hizli. Metrikler prevalansa duyarsiz secildi
(ROC-AUC, zenginlestirme faktoru EF1%/EF5%); PR-AUC yalniz ornek prevalansinda raporlanir.
Dogrulama: Bemis-Murcko iskelet-gruplu 5-kat CV. Aktarim: NCGC Ebola (AID 1117312) ve ChEMBL Ebola; Marburg ile ortak bilesikler cikarilir."""
import os, sys, warnings, importlib.util, joblib, numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, MACCSkeys, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import spearmanr
warnings.filterwarnings("ignore")
HERE = os.path.dirname(__file__); BASE = os.path.join(HERE, ".."); R = os.path.join(BASE, "results")
sys.path.insert(0, HERE); from qsar_common import Clean, CorrFilter
SEED, N_INACT = 7, 60000

def feat_chunk(smis):
    RDLogger.DisableLog("rdApp.*")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    D, K, M, S = [], [], [], []
    for s in smis:
        m = Chem.MolFromSmiles(s)
        d = []
        for _, f in Descriptors.descList:
            try: d.append(float(f(m)))
            except Exception: d.append(np.nan)
        D.append(d); K.append(list(MACCSkeys.GenMACCSKeys(m))); M.append(np.minimum(gen.GetCountFingerprintAsNumPy(m), 255))
        try: sc = MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False)
        except Exception: sc = None  # nadir: kotu bag stereosu -> kendi grubu
        S.append(sc or "ACYC_" + s)
    return np.array(D, dtype=np.float64), np.array(K, dtype=np.uint8), np.array(M, dtype=np.uint8), S

def featurize(smiles):
    smiles = list(smiles); n = max(1, (os.cpu_count() or 4) - 1)
    parts = [smiles[i:i + 1000] for i in range(0, len(smiles), 1000)]
    with ProcessPoolExecutor(max_workers=n) as ex: res = list(ex.map(feat_chunk, parts))
    D = np.vstack([r[0] for r in res]); D[~np.isfinite(D)] = np.nan
    return D, np.vstack([r[1] for r in res]), np.vstack([r[2] for r in res]), sum([r[3] for r in res], [])

def maxsim(A, B, chunk=300):
    A = (A > 0).astype(np.float32); B = (B > 0).astype(np.float32); nb = B.sum(1); out = np.zeros(len(A), np.float32)
    for i in range(0, len(A), chunk):
        a = A[i:i + chunk]; inter = a @ B.T; out[i:i + chunk] = (inter / (a.sum(1)[:, None] + nb[None, :] - inter + 1e-9)).max(1)
    return out

def ef(y, p, frac):
    k = max(1, int(round(len(y) * frac))); top = np.argsort(-p)[:k]; return y[top].mean() / y.mean()

def model(kind):
    pre = [Clean(), CorrFilter(.9)]
    if kind == "RF": return make_pipeline(*pre, RandomForestClassifier(400, n_jobs=-1, random_state=SEED, class_weight="balanced_subsample", min_samples_leaf=2, max_features="sqrt"))
    return make_pipeline(*pre, HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_leaf_nodes=31, l2_regularization=1.0, class_weight="balanced", random_state=SEED))

if __name__ == "__main__":
    if __package__ is None and not os.path.exists(os.path.join(R, "marburg_clean.csv")): sys.exit("marburg_clean.csv yok")
    spec = importlib.util.spec_from_file_location("fpo", os.path.join(HERE, "06_fingerprint_only.py")); fpo = importlib.util.module_from_spec(spec); spec.loader.exec_module(fpo)
    full = pd.read_csv(os.path.join(R, "marburg_clean.csv")); allik = set(full.ik)
    rng = np.random.RandomState(SEED)
    ina = full[full.y == 0]; ina = ina.iloc[rng.permutation(len(ina))[:N_INACT]]
    S = pd.concat([full[full.y == 1], ina]).reset_index(drop=True); y = S.y.values
    print(f"ornek: {len(S)} bilesik, aktif {y.sum()} (%{y.mean()*100:.1f}); gercek prevalans %{full.y.mean()*100:.1f}", flush=True)
    D, K, M, scaf = featurize(S.smiles); grp = pd.factorize(pd.Series(scaf))[0]
    print(f"ozellikler hazir; iskelet grubu {len(set(grp))}", flush=True)
    np.savez_compressed(os.path.join(R, "marburg_sample_features.npz"), D=D, K=K, M=M, y=y, smiles=S.smiles.values)
    X = {"desc+maccs": np.hstack([D, K.astype(float)]), "morgan2": M.astype(float)}
    sp = list(GroupKFold(5).split(np.zeros(len(y)), y, grp))
    rows = []
    for fs, kinds in (("desc+maccs", ("RF", "HGB")), ("morgan2", ("RF",))):
        for kind in kinds:
            p = np.zeros(len(y))
            for tr, te in sp: p[te] = model(kind).fit(X[fs][tr], y[tr]).predict_proba(X[fs][te])[:, 1]
            row = dict(features=fs, model=kind, auc=roc_auc_score(y, p), prauc=average_precision_score(y, p), ef1=ef(y, p, .01), ef5=ef(y, p, .05))
            rows.append(row); print({k: (round(float(v), 3) if isinstance(v, (float, np.floating)) else v) for k, v in row.items()}, flush=True)
    res = pd.DataFrame(rows); res.to_csv(os.path.join(R, "marburg_cv.csv"), index=False)
    print("(taban: PR-AUC = ornek prevalansi %.3f; EF taban 1.0, azami EF1%% = %.1f)" % (y.mean(), 1 / y.mean() if y.mean() > 0.01 else 100))
    best = res.sort_values("auc", ascending=False).iloc[0]; bf, bk = best.features, best.model; print("en iyi:", bf, bk, round(best.auc, 3), flush=True)
    final = model(bk).fit(X[bf], y); joblib.dump(final, os.path.join(R, "marburg_model.joblib"), compress=3)

    def feats_for(smiles):
        Dx, Kx, Mx, _ = featurize(smiles); return {"desc+maccs": np.hstack([Dx, Kx.astype(float)]), "morgan2": Mx.astype(float)}, Mx
    # ---- aktarim 1: NCGC Ebola
    N = pd.read_csv(os.path.join(R, "ncgc_ebola_classification.csv")); ov = N.ik.isin(allik)
    print(f"\nNCGC Ebola {len(N)} bilesik; Marburg ile ortak {int(ov.sum())} (cikarildi)", flush=True)
    N = N[~ov].reset_index(drop=True); FN, MN = feats_for(N.smiles); pn = final.predict_proba(FN[bf])[:, 1]
    sim = maxsim(MN, M); print(f"  NCGC->Marburg-ornek en yakin komsu Tanimoto medyan {np.median(sim):.2f}; >=0.5 olan %{(sim>=.5).mean()*100:.0f}")
    print(f"  [aktarim] NCGC Ebola: AUC {roc_auc_score(N.y, pn):.3f}, PR-AUC {average_precision_score(N.y, pn):.3f} (prevalans {N.y.mean():.3f}), EF1% {ef(N.y.values, pn, .01):.2f}, EF5% {ef(N.y.values, pn, .05):.2f}", flush=True)
    N.assign(p_marburg=pn).to_csv(os.path.join(R, "marburg_model_on_ncgc_ebola.csv"), index=False)
    # ---- aktarim 2: ChEMBL Ebola
    reg = pd.read_csv(os.path.join(R, "ebola_regression.csv")); clf = pd.read_csv(os.path.join(R, "ebola_classification.csv"))
    ext = pd.concat([clf[["smiles", "ik", "label", "main_doc"]], reg[["smiles", "ik"]]]).drop_duplicates("smiles")
    ext = ext.merge(reg[["smiles", "pX"]], on="smiles", how="left"); ext = ext[~ext.ik.isin(allik)].reset_index(drop=True)
    FE, ME = feats_for(ext.smiles); pe = final.predict_proba(FE[bf])[:, 1]
    simc = maxsim(ME, M); print(f"\nChEMBL {len(ext)} bilesik (Marburg ortak cikarildi); en yakin komsu Tanimoto medyan {np.median(simc):.2f}; >=0.5 olan %{(simc>=.5).mean()*100:.0f}")
    m1 = ext.label.notna(); print(f"  [aktarim] ChEMBL 1uM sinif (n={m1.sum()}): AUC {roc_auc_score(ext.label[m1], pe[m1]):.3f}, PR-AUC {average_precision_score(ext.label[m1], pe[m1]):.3f} (prevalans {ext.label[m1].mean():.3f})")
    m2 = ext.pX.notna(); print(f"  [aktarim] ChEMBL pseudovirus pX (n={m2.sum()}): Spearman rho {spearmanr(ext.pX[m2], pe[m2])[0]:.3f}")
    v = []
    for dd in ext.main_doc.dropna().unique():
        i = np.where((ext.main_doc == dd) & m1)[0]
        if len(i) >= 10 and 0 < ext.label.values[i].sum() < len(i): v.append(roc_auc_score(ext.label.values[i], pe[i]))
    print(f"  [aktarim] dokuman-ici AUC ortalama {np.mean(v):.3f} ({len(v)} dokuman)" if v else "  dokuman-ici yetersiz")
    ext.assign(p_marburg=pe, sim=simc).to_csv(os.path.join(R, "marburg_model_on_chembl.csv"), index=False)
