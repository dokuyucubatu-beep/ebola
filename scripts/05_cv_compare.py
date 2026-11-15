"""Ozellik kumesi x algoritma karsilastirmasi.
Birincil dogrulama: dokuman-gruplu 5-kat CV (yeni laboratuvar/seri senaryosu).
Karsilastirma icin: rastgele 5-kat CV (iyimser). Korelasyon filtresi her katta yalniz egitimle fit edilir.
Birlesik (out-of-fold) tahminler uzerinden metrik."""
import os, sys, warnings, numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNetCV, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.metrics import r2_score, mean_squared_error, roc_auc_score, average_precision_score, matthews_corrcoef
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(__file__)); from qsar_common import Clean, CorrFilter
warnings.filterwarnings("ignore")
R = os.path.join(os.path.dirname(__file__), "..", "results")
SEED = 7

def get_X(z, fs):
    parts = {"desc": z["desc"], "morgan": z["morgan"], "maccs": z["maccs"]}
    return np.hstack([parts[p].astype(float) for p in fs.split("+")])
FSETS = ["desc", "morgan", "maccs", "desc+maccs", "desc+morgan+maccs"]

def prep(): return [Clean(), CorrFilter(0.9), StandardScaler()]
REG = {
  "ElasticNet": lambda: make_pipeline(*prep(), ElasticNetCV(l1_ratio=[.2, .5, .8], cv=3, max_iter=5000, random_state=SEED)),
  "RF": lambda: make_pipeline(*prep()[:2], RandomForestRegressor(300, n_jobs=-1, random_state=SEED, max_features=0.33)),
  "SVR": lambda: make_pipeline(*prep(), SVR(C=3, epsilon=0.1, gamma="scale")),
}
CLF = {
  "LogReg": lambda: make_pipeline(*prep(), LogisticRegression(C=0.05, max_iter=2000, class_weight="balanced")),
  "RF": lambda: make_pipeline(*prep()[:2], RandomForestClassifier(300, n_jobs=-1, random_state=SEED, class_weight="balanced_subsample", max_features="sqrt")),
  "SVC": lambda: make_pipeline(*prep(), SVC(C=1, gamma="scale", class_weight="balanced")),
}

def oof(make, X, y, splits, clf):
    p = np.zeros(len(y))
    for tr, te in splits:
        m = make().fit(X[tr], y[tr])
        if clf:
            p[te] = m.predict_proba(X[te])[:, 1] if hasattr(m[-1], "predict_proba") else m.decision_function(X[te])
        else: p[te] = m.predict(X[te])
    return p

def run(tag, clf):
    z = np.load(os.path.join(R, f"features_{tag}.npz"), allow_pickle=True)
    y = z["y"].astype(float); g = z["group"]
    print(f"\n===== {'SINIFLANDIRMA' if clf else 'REGRESYON'}: n={len(y)}, dokuman={len(set(g))} =====")
    gsp = list(GroupKFold(5).split(np.zeros(len(y)), y, g))
    rsp = list((StratifiedKFold(5, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y) if clf
               else KFold(5, shuffle=True, random_state=SEED).split(np.zeros(len(y)))))
    rows = []
    models = CLF if clf else REG
    for fs in FSETS:
        X = get_X(z, fs)
        for name, mk in models.items():
            row = dict(features=fs, model=name)
            for sch, sp in (("doc", gsp), ("rand", rsp)):
                p = oof(mk, X, y, sp, clf)
                if clf:
                    row[f"auc_{sch}"] = roc_auc_score(y, p); row[f"prauc_{sch}"] = average_precision_score(y, p)
                else:
                    row[f"r2_{sch}"] = r2_score(y, p); row[f"rmse_{sch}"] = mean_squared_error(y, p) ** .5
                    row[f"rho_{sch}"] = spearmanr(y, p)[0]
            rows.append(row); print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    df = pd.DataFrame(rows)
    if clf: print("taban: PR-AUC = prevalans %.3f" % y.mean())
    else:
        base = np.zeros(len(y))   # taban: egitim-katlari ortalamasi
        for tr, te in gsp: base[te] = y[tr].mean()
        print("taban (egitim ortalamasi) doc-CV: R2 %.3f RMSE %.3f; sd(y)=%.3f" % (r2_score(y, base), mean_squared_error(y, base) ** .5, y.std()))
    df.to_csv(os.path.join(R, f"cv_compare_{tag}.csv"), index=False)

if __name__ == "__main__":
    for t in sys.argv[1:] or ["reg", "clf"]: run(t, t == "clf")
