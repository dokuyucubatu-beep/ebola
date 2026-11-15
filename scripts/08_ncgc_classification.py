"""NCGC Ebola giris qHTS (AID 1117312, Screen 2 ratio) -> aktif/inaktif siniflandirma.
Kurasyon: en buyuk organik fragman, InChIKey tekillestirme (celiskili tekrarlar atilir).
Etiket: PubChem ACTIVITY_OUTCOME (Active=1). Dogrulama: Bemis-Murcko iskelet-gruplu 5-kat CV (tek laboratuvar -> meşru), rastgele CV karsilastirma.
Dis test: ChEMBL Ebola seti (NCGC ile ortak bilesikler cikarilir).
Taban cizgileri: prevalans; 'katyonik amfifilik' kurali (bazik amin & Crippen logP)."""
import os, sys, warnings, importlib.util, joblib, numpy as np, pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, Crippen
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.MolStandardize import rdMolStandardize
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import spearmanr
warnings.filterwarnings("ignore"); RDLogger.DisableLog("rdApp.*")
HERE = os.path.dirname(__file__); BASE = os.path.join(HERE, ".."); R = os.path.join(BASE, "results"); P = os.path.join(BASE, "data", "pubchem")
sys.path.insert(0, HERE); from qsar_common import Clean, CorrFilter
spec = importlib.util.spec_from_file_location("fpo", os.path.join(HERE, "06_fingerprint_only.py")); fpo = importlib.util.module_from_spec(spec); spec.loader.exec_module(fpo)
SEED = 7
chooser = rdMolStandardize.LargestFragmentChooser(preferOrganic=True); unch = rdMolStandardize.Uncharger()
def std(smi):
    m = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
    if m is None: return None
    m = unch.uncharge(chooser.choose(m))
    if m.GetNumHeavyAtoms() < 6 or m.GetNumHeavyAtoms() > 120 or not any(a.GetSymbol() == "C" for a in m.GetAtoms()): return None
    return m

def load(aid):
    d = pd.read_csv(os.path.join(P, f"aid_{aid}.csv"), low_memory=False)
    d = d[pd.to_numeric(d.PUBCHEM_SID, errors="coerce").notna()].copy()
    d["y"] = (d.PUBCHEM_ACTIVITY_OUTCOME == "Active").astype(int)
    return d

def curate(d):
    rows = []
    for _, r in d.iterrows():
        m = std(r.PUBCHEM_EXT_DATASOURCE_SMILES)
        if m is None: continue
        rows.append((Chem.MolToInchiKey(m), Chem.MolToSmiles(m), r.y, r.get("Sample Name", "")))
    t = pd.DataFrame(rows, columns=["ik", "smiles", "y", "name"])
    g = t.groupby("ik").agg(smiles=("smiles", "first"), name=("name", "first"), y=("y", "mean"), n=("y", "size")).reset_index()
    conf = ((g.y > 0) & (g.y < 1)).sum(); tie = (g.y == 0.5).sum()
    g = g[g.y != 0.5]; g["y"] = (g.y > 0.5).astype(int)
    return g, conf, tie

d1 = load(1117312); D, conf, tie = curate(d1)
print(f"AID 1117312: {len(d1)} satir -> {len(D)} benzersiz bilesik (celiskili tekrar {conf}, esit oy atilan {tie}); aktif {D.y.sum()} (%{D.y.mean()*100:.0f})")
# kendi-icinde tekrar uyumu + ikinci ekran (1117305) ile uyum -> ulasilabilir ust sinir
D2, _, _ = curate(load(1117305)); mm = D.merge(D2, on="ik", suffixes=("", "_2"))
print(f"Screen2 ikinci surum (AID 1117305) ile ortak {len(mm)} bilesik; etiket uyumu %{(mm.y == mm.y_2).mean()*100:.0f}; Cohen-benzeri: aktif-aktif {int(((mm.y==1)&(mm.y_2==1)).sum())}, aktif-inaktif {int(((mm.y==1)&(mm.y_2==0)).sum())}, inaktif-aktif {int(((mm.y==0)&(mm.y_2==1)).sum())}")
D.to_csv(os.path.join(R, "ncgc_ebola_classification.csv"), index=False)

mols = [Chem.MolFromSmiles(s) for s in D.smiles]
FP = fpo.fpset(D.smiles)
desc = np.array([[f(m) if True else 0 for _, f in Descriptors.descList] for m in mols], dtype=float); desc[~np.isfinite(desc)] = np.nan
y = D.y.values
scaf = [MurckoScaffold.MurckoScaffoldSmiles(mol=m) or "ACYCLIC_" + D.ik.iloc[i] for i, m in enumerate(mols)]
grp = pd.factorize(pd.Series(scaf))[0]
print(f"iskelet grubu: {len(set(grp))}; en buyuk grup {np.bincount(grp).max()}")
ssp = list(GroupKFold(5).split(np.zeros(len(y)), y, grp))
rsp = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y))

# --- taban cizgisi: katyonik amfifilik kural
basic = Chem.MolFromSmarts("[NX3;!$(NC=[O,S,N]);!$(N-a);!$(N-S(=O)=O);!$(N-[#7,#8]);!$(N#*);!$(N=*)]")
logp = np.array([Crippen.MolLogP(m) for m in mols]); hasb = np.array([m.HasSubstructMatch(basic) for m in mols])
cad = hasb * np.clip(logp, 0, None)
print(f"[taban] bazik amin orani %{hasb.mean()*100:.0f}; aktiflik orani bazik aminlerde %{y[hasb].mean()*100:.0f} vs digerlerinde %{y[~hasb].mean()*100:.0f}")
print(f"[taban] 'bazik amin x logP' tek skor AUC {roc_auc_score(y, cad):.3f}; PR-AUC {average_precision_score(y, cad):.3f} (prevalans {y.mean():.3f})")

def get_X(fs):
    parts = {"desc": desc, **FP}
    return np.hstack([parts[p].astype(float) for p in fs.split("+")])
def pipe_clf(kind):
    if kind == "LogReg": return make_pipeline(Clean(), CorrFilter(.9), StandardScaler(), LogisticRegression(C=.05, max_iter=3000, class_weight="balanced"))
    return make_pipeline(Clean(), CorrFilter(.9), RandomForestClassifier(500, n_jobs=-1, random_state=SEED, class_weight="balanced_subsample", max_features="sqrt"))
def oof(kind, X, sp):
    p = np.zeros(len(y))
    for tr, te in sp:
        if kind == "TanSVM": p[te] = fpo.fit_predict("TanSVM", True, X[tr], y[tr], X[te])
        else: p[te] = pipe_clf(kind).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return p

rows = []
for fs in ["desc", "morgan2", "fcfp4", "atompair", "rdkit", "maccs", "desc+morgan2", "desc+maccs"]:
    X = get_X(fs)
    for kind in ["LogReg", "RF"] + (["TanSVM"] if "+" not in fs and fs != "desc" else []):
        row = dict(features=fs, model=kind)
        for nm, sp in (("scaf", ssp), ("rand", rsp)):
            p = oof(kind, X, sp); row[f"auc_{nm}"] = roc_auc_score(y, p); row[f"prauc_{nm}"] = average_precision_score(y, p)
        rows.append(row); print({k: (round(float(v), 3) if isinstance(v, (float, np.floating)) else v) for k, v in row.items()}, flush=True)
res = pd.DataFrame(rows); res.to_csv(os.path.join(R, "ncgc_cv_compare.csv"), index=False)
best = res.sort_values("auc_scaf", ascending=False).iloc[0]; print("\nen iyi (iskelet-CV AUC):", best.features, best.model, round(best.auc_scaf, 3))

# --- nihai model + ChEMBL dis testi
bf, bk = best.features, best.model
Xfull = get_X(bf)
if bk == "TanSVM": final = None
else: final = pipe_clf(bk).fit(Xfull, y); joblib.dump(final, os.path.join(R, "ncgc_ebola_clf.joblib"), compress=3)
reg = pd.read_csv(os.path.join(R, "ebola_regression.csv")); clf = pd.read_csv(os.path.join(R, "ebola_classification.csv"))
ext = pd.concat([clf[["smiles", "label", "main_doc"]].assign(src="clf"), reg[["smiles"]].assign(src="reg")]).drop_duplicates("smiles")
cm = [std(s) for s in ext.smiles]; ext["ik"] = [Chem.MolToInchiKey(m) if m is not None else None for m in cm]
ext = ext.merge(clf[["smiles", "label"]].rename(columns={"label": "lab1"}), on="smiles", how="left", suffixes=("_x", "")) if False else ext
ext = ext[ext.ik.notna()].copy()
ext = ext.merge(reg[["smiles", "pX"]], on="smiles", how="left")
ext = ext.merge(clf[["smiles", "label"]].rename(columns={"label": "lab"}), on="smiles", how="left")
overlap = ext.ik.isin(set(D.ik)); print(f"\nChEMBL bilesik {len(ext)}; NCGC ile ortak (cikarildi) {int(overlap.sum())}")
ext = ext[~overlap].reset_index(drop=True)
FPe = fpo.fpset(ext.smiles)
de = np.array([[f(Chem.MolFromSmiles(s)) for _, f in Descriptors.descList] for s in ext.smiles], dtype=float); de[~np.isfinite(de)] = np.nan
parts = {"desc": de, **FPe}; Xe = np.hstack([parts[p].astype(float) for p in bf.split("+")])
pe = fpo.fit_predict("TanSVM", True, Xfull, y, Xe) if bk == "TanSVM" else final.predict_proba(Xe)[:, 1]
sim = fpo.tani(FPe["morgan2"], FP["morgan2"]).max(1)
print(f"ChEMBL -> NCGC egitim en yakin komsu Tanimoto medyan {np.median(sim):.2f}; >=0.5 olan %{(sim>=.5).mean()*100:.0f}")
m1 = ext.lab.notna()
print(f"[dis] ChEMBL siniflandirma (1 uM etiketi, n={m1.sum()}, aktif %{ext.lab[m1].mean()*100:.0f}): AUC {roc_auc_score(ext.lab[m1], pe[m1]):.3f}, PR-AUC {average_precision_score(ext.lab[m1], pe[m1]):.3f}")
m2 = ext.pX.notna(); print(f"[dis] ChEMBL pseudovirus pX (n={m2.sum()}): Spearman rho {spearmanr(ext.pX[m2], pe[m2])[0]:.3f}")
# dokuman-ici dis test
vals = []
for dd in ext.main_doc.dropna().unique():
    i = np.where((ext.main_doc == dd) & m1)[0]
    if len(i) >= 10 and 0 < ext.lab.values[i].sum() < len(i): vals.append(roc_auc_score(ext.lab.values[i], pe[i]))
print(f"[dis] dokuman-ici AUC ortalama {np.mean(vals):.3f} ({len(vals)} dokuman)" if vals else "[dis] dokuman-ici AUC icin yeterli dokuman yok")
pd.DataFrame(dict(smiles=ext.smiles, p_ncgc=pe, sim_ncgc=sim, lab=ext.lab, pX=ext.pX)).to_csv(os.path.join(R, "ncgc_model_on_chembl.csv"), index=False)
