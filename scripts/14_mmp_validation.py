"""Analog onericinin dogrulamasi: Marburg verisinde esleşmis cift (matched pair) yon testi.
Her aktif A'ya 13_analogs.py'deki TEK adimlik donusumler uygulanir; urun B veri kumesinde varsa (ayni kurala gore) ve inaktifse
(A aktif, B inaktif) bir 'ucurum cifti' olusur. Soru: model p(A) > p(B) diyor mu? (%50 = bilgi yok.)
Sizinti onlemi: A icin iskelet-gruplu CV tahmini (OOF); B ornekte ise OOF, degilse A'nin katinin modeli (B hicbir modelde egitimde degil)."""
import os, sys, warnings, importlib.util, numpy as np, pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import GroupKFold
warnings.filterwarnings("ignore"); RDLogger.DisableLog("rdApp.*")
HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(HERE, "..", "results"); sys.path.insert(0, HERE)
def load(n, f):
    s = importlib.util.spec_from_file_location(n, os.path.join(HERE, f)); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
mm = load("mm", "10_marburg_model.py"); an = load("an", "13_analogs.py")
BASIC = Chem.MolFromSmarts("[NX3;!$(NC=[O,S,N]);!$(N-a);!$(N-S(=O)=O);!$(N-[#7,#8]);!$(N#*);!$(N=*)]")
def flat(s):
    m = Chem.MolFromSmiles(s) if isinstance(s, str) else None
    return Chem.MolToSmiles(m, isomericSmiles=False) if m is not None else None

if __name__ == "__main__":
    full = pd.read_csv(os.path.join(R, "marburg_clean.csv")); full["flat"] = [flat(s) for s in full.smiles]; full = full[full.flat.notna()]
    g = full.groupby("flat").y.agg(["nunique", "first"]); g = g[g["nunique"] == 1]; lab = g["first"].to_dict()  # belirsiz (celiskili stereo) atilir
    smi_of = full.drop_duplicates("flat").set_index("flat").smiles.to_dict()
    z = np.load(os.path.join(R, "marburg_sample_features.npz"), allow_pickle=True); D, K, M, y, ssm = z["D"], z["K"], z["M"], z["y"], z["smiles"]
    X = np.hstack([D, K.astype(float)]); sflat = [flat(s) for s in ssm]; sidx = {f: i for i, f in enumerate(sflat)}
    # --- ciftler: aktif A -> tek adim urunler B (veride, inaktif)
    pairs = []
    for i in np.where(y == 1)[0]:
        for prod, rule in an.apply_all(Chem.MolFromSmiles(ssm[i])).items():
            f = flat(prod)
            if f != sflat[i] and lab.get(f) == 0: pairs.append((i, f, rule))
    P = pd.DataFrame(pairs, columns=["a", "bflat", "rule"]).drop_duplicates(["a", "bflat"])
    print(f"ucurum cifti (A aktif, B inaktif, tek donusum): {len(P)}; benzersiz aktif A: {P.a.nunique()}; B ornekte: {P.bflat.isin(sidx).sum()}", flush=True)
    # --- CV
    scaf = []
    for s in ssm:
        try: sc = MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(s), includeChirality=False)
        except Exception: sc = None
        scaf.append(sc or "ACYC_" + s)
    grp = pd.factorize(pd.Series(scaf))[0]; sp = list(GroupKFold(5).split(np.zeros(len(y)), y, grp)); fold = np.zeros(len(y), int)
    for k, (_, te) in enumerate(sp): fold[te] = k
    extra = sorted(set(P.bflat) - set(sidx)); print(f"ornek disi B: {len(extra)} -> ozellik hesaplaniyor", flush=True)
    De, Ke, _, _ = mm.feat_chunk([smi_of[f] for f in extra]); De[~np.isfinite(De)] = np.nan; Xe = np.hstack([De, Ke.astype(float)])
    oof = np.zeros(len(y)); pe = np.zeros((5, len(extra)))
    for k, (tr, te) in enumerate(sp):
        mod = mm.model("RF").fit(X[tr], y[tr]); oof[te] = mod.predict_proba(X[te])[:, 1]; pe[k] = mod.predict_proba(Xe)[:, 1]; print(f"  kat {k+1}/5", flush=True)
    eidx = {f: i for i, f in enumerate(extra)}
    P["pa"] = oof[P.a.values]; P["fold"] = fold[P.a.values]
    P["pb"] = [oof[sidx[f]] if f in sidx else pe[fo, eidx[f]] for f, fo in zip(P.bflat, P.fold)]
    # --- taban cizgileri: logP ve CAD farki
    def desc(f):
        m = Chem.MolFromSmiles(smi_of[f]); lp = Crippen.MolLogP(m); return lp, float(m.HasSubstructMatch(BASIC)) * max(lp, 0)
    lb = {f: desc(f) for f in set(P.bflat)}; la = {i: desc(flat(ssm[i])) for i in set(P.a)}
    P["dlogp"] = [la[a][0] - lb[b][0] for a, b in zip(P.a, P.bflat)]; P["dcad"] = [la[a][1] - lb[b][1] for a, b in zip(P.a, P.bflat)]
    P["dp"] = P.pa - P.pb
    score = lambda d: (np.where(d > 0, 1.0, np.where(d == 0, .5, 0.0)))
    P["win_model"] = score(P.dp); P["win_logp"] = score(P.dlogp); P["win_cad"] = score(P.dcad)
    ga = P.groupby("a"); rng = np.random.RandomState(0); keys = P.a.unique()
    def boot(col):
        m = ga[col].mean().reindex(keys).values; n = ga.size().reindex(keys).values; v = []
        for _ in range(2000):
            s = rng.randint(0, len(keys), len(keys)); v.append((m[s] * n[s]).sum() / n[s].sum())
        return np.percentile(v, [2.5, 97.5])
    print("\n[yon dogrulugu: p(aktif A) > p(inaktif B); %50 = bilgi yok; %95 GA aktif-A kumelerinde bootstrap]")
    for nm, col in (("model", "win_model"), ("taban: yuksek logP kazanir", "win_logp"), ("taban: yuksek CAD kazanir", "win_cad")):
        lo, hi = boot(col); print(f"  {nm}: {P[col].mean()*100:.1f}%  (GA {lo*100:.1f}-{hi*100:.1f}), n={len(P)}")
    for thr in (0.1, 0.2):
        s = P[P.dp.abs() >= thr]; print(f"  model, |dp|>={thr} (gurultu ustu): {s.win_model.mean()*100:.1f}% (n={len(s)})")
    print("\n[kural bazinda model dogrulugu, n>=15]"); t = P.groupby("rule").agg(n=("win_model", "size"), model=("win_model", "mean"), logp=("win_logp", "mean"), ort_dp=("dp", "mean"))
    print(t[t.n >= 15].sort_values("n", ascending=False).round(3).to_string())
    P.to_csv(os.path.join(R, "mmp_validation_pairs.csv"), index=False); print("bitti")
