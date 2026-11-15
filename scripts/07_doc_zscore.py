"""Dokuman z-skoru: pX, her dokumanin kendi olcumlerine gore z = (pX - dokuman ort.) / dokuman sd donusturulur.
Hedef = goreli potens (ayni dokuman/seri icinde). Bilesik basina olcum-duzeyi z'lerin medyani.
Onceden sabit kurallar (sonuca bakilmadan): dokuman >= MIN_N olcum ve sd >= MIN_SD; aksi halde bu dokumanin olcumleri atilir.
Dogrulama: dokuman-gruplu 5-kat CV; z hedefi dokumanlar arasi kaymayi icerdigi icin birlesik rho ~ dokuman-ici siralama."""
import os, sys, warnings, importlib.util, numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNetCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import spearmanr
warnings.filterwarnings("ignore")
HERE = os.path.dirname(__file__); R = os.path.join(HERE, "..", "results")
sys.path.insert(0, HERE); from qsar_common import Clean, CorrFilter
spec = importlib.util.spec_from_file_location("fpo", os.path.join(HERE, "06_fingerprint_only.py")); fpo = importlib.util.module_from_spec(spec); spec.loader.exec_module(fpo)
MIN_N, MIN_SD, SEED = 8, 0.15, 7

M = pd.read_csv(os.path.join(R, "ebola_regression_measurements.csv"))
st = M.groupby("doc").pX.agg(["size", "mean", "std"])
print("dokuman olcum sayisi / sd:\n", st.sort_values("size", ascending=False).round(2).to_string())
ok = st[(st["size"] >= MIN_N) & (st["std"] >= MIN_SD)].index
print(f"\nkural: n>={MIN_N}, sd>={MIN_SD} -> {len(ok)}/{len(st)} dokuman kaldi; atilan olcum {int((~M.doc.isin(ok)).sum())}/{len(M)}")
M = M[M.doc.isin(ok)].copy()
M["z"] = M.groupby("doc").pX.transform(lambda s: (s - s.mean()) / s.std())
agg = M.groupby("smiles").agg(z=("z", "median"), pX=("pX", "median"), main_doc=("doc", "first"), n=("z", "size")).reset_index()
print(f"nihai: {len(agg)} bilesik, {agg.main_doc.nunique()} dokuman; z sd {agg.z.std():.2f}")

z = np.load(os.path.join(R, "features_reg.npz"), allow_pickle=True)
pos = {s: i for i, s in enumerate(z["smiles"])}
idx = np.array([pos[s] for s in agg.smiles])
y = agg.z.values; g = agg.main_doc.values; smi = agg.smiles.values
desc = z["desc"][idx]
FP = fpo.fpset(smi)
gsp = list(GroupKFold(5).split(np.zeros(len(y)), y, g))
rsp = list(KFold(5, shuffle=True, random_state=SEED).split(np.zeros(len(y))))

def oof_desc(kind, sp):
    p = np.zeros(len(y))
    for tr, te in sp:
        pre = [Clean(), CorrFilter(0.9)] + ([StandardScaler()] if kind == "ElasticNet" else [])
        est = ElasticNetCV(l1_ratio=[.2, .5, .8], cv=3, max_iter=5000, random_state=SEED) if kind == "ElasticNet" else \
              RandomForestRegressor(300, n_jobs=-1, random_state=SEED, max_features=0.33)
        p[te] = make_pipeline(*pre, est).fit(desc[tr], y[tr]).predict(desc[te])
    return p

def within(p):
    v = [spearmanr(y[g == d], p[g == d])[0] for d in pd.unique(g) if (g == d).sum() >= 10]
    return np.nanmean(v), len(v)

rows = []
def rec(name, kind, fn):
    row = dict(features=name, model=kind)
    for nm, sp in (("doc", gsp), ("rand", rsp)):
        p = fn(sp)
        row[f"r2_{nm}"] = r2_score(y, p); row[f"rmse_{nm}"] = mean_squared_error(y, p) ** .5; row[f"rho_{nm}"] = spearmanr(y, p)[0]
        if nm == "doc": row["icdoc_rho"], row["n_doc_ic"] = within(p)
    rows.append(row); print({k: (round(float(v), 3) if isinstance(v, (float, np.floating)) else v) for k, v in row.items()}, flush=True)

for kind in ("ElasticNet", "RF"): rec("rdkit-desc", kind, lambda sp, k=kind: oof_desc(k, sp))
for fs in ("morgan2", "fcfp4", "atompair", "rdkit", "maccs"):
    for kind in ("RF", "TanSVM", "TanKNN"):
        rec(fs, kind, lambda sp, f=fs, k=kind: fpo.oof(k, False, FP[f], y, sp))
base = np.zeros(len(y))
for tr, te in gsp: base[te] = y[tr].mean()
print("\ntaban (egitim ortalamasi) doc-CV: R2 %.3f RMSE %.3f; sd(y)=%.3f" % (r2_score(y, base), mean_squared_error(y, base) ** .5, y.std()))
# y-randomizasyon (Z skoru) - en iyi doc-CV kombinasyonu icin SONRA; burada yalniz karsilastirma tablosu
pd.DataFrame(rows).to_csv(os.path.join(R, "docz_reg_cv.csv"), index=False)
agg.to_csv(os.path.join(R, "ebola_regression_docz.csv"), index=False)
