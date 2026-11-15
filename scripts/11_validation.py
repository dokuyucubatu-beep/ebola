"""Marburg modeli dogrulama: (1) taban cizgileri (CAD kurali, kucuk LogReg) ayni alt kumelerde, (2) benzerlik-araligi AUC (CV test->egitim),
(3) y-randomizasyon (Z skoru), (4) CAD'e kati stratifiye AUC (model bazik-amin/logP'nin otesinde bilgi veriyor mu?)."""
import os, sys, warnings, importlib.util, numpy as np, pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore"); RDLogger.DisableLog("rdApp.*")
HERE = os.path.dirname(__file__); R = os.path.join(HERE, "..", "results")
sys.path.insert(0, HERE); from qsar_common import Clean, CorrFilter
spec = importlib.util.spec_from_file_location("mm", os.path.join(HERE, "10_marburg_model.py")); mm = importlib.util.module_from_spec(spec); spec.loader.exec_module(mm)
SEED, NPERM = 7, 3
basic = Chem.MolFromSmarts("[NX3;!$(NC=[O,S,N]);!$(N-a);!$(N-S(=O)=O);!$(N-[#7,#8]);!$(N#*);!$(N=*)]")

def simple(smiles):
    out = []
    for s in smiles:
        m = Chem.MolFromSmiles(s); lp = Crippen.MolLogP(m); b = float(m.HasSubstructMatch(basic))
        out.append([b, lp, Descriptors.MolWt(m), Descriptors.TPSA(m), b * max(lp, 0)])
    return np.array(out)

def auc(y, p):
    y = np.asarray(y); return roc_auc_score(y, p) if 0 < y.sum() < len(y) else np.nan

def rf(n=300): return make_pipeline(Clean(), CorrFilter(.9), RandomForestClassifier(n, n_jobs=-1, random_state=SEED, class_weight="balanced_subsample", min_samples_leaf=2, max_features="sqrt"))

if __name__ == "__main__":
    z = np.load(os.path.join(R, "marburg_sample_features.npz"), allow_pickle=True)
    D, K, M, y, smi = z["D"], z["K"], z["M"], z["y"], z["smiles"]
    X = np.hstack([D, K.astype(float)])
    # iskelet gruplari (10 ile ayni yontem; feat_chunk tekrar hesaplamamak icin yeniden uretilir)
    from rdkit.Chem.Scaffolds import MurckoScaffold
    scaf = []
    for s in smi:
        m = Chem.MolFromSmiles(s)
        try: sc = MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False)
        except Exception: sc = None
        scaf.append(sc or "ACYC_" + s)
    grp = pd.factorize(pd.Series(scaf))[0]; sp = list(GroupKFold(5).split(np.zeros(len(y)), y, grp))
    S = simple(smi); print("basit ozellikler hazir", flush=True)
    # ---- (1) taban: CAD ve kucuk LogReg (iskelet-CV), model OOF + benzerlik
    pc = S[:, 4]; print(f"[Marburg orn.] CAD kurali AUC {auc(y, pc):.3f}; bazik amin orani %{S[:,0].mean()*100:.0f}; aktiflik bazikte %{y[S[:,0]==1].mean()*100:.1f} vs digerinde %{y[S[:,0]==0].mean()*100:.1f}", flush=True)
    pl = np.zeros(len(y)); po = np.zeros(len(y)); sim = np.zeros(len(y), np.float32)
    for k, (tr, te) in enumerate(sp):
        pl[te] = make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=2000, class_weight="balanced")).fit(S[tr], y[tr]).predict_proba(S[te])[:, 1]
        po[te] = rf(400).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]; sim[te] = mm.maxsim(M[te], M[tr]); print(f"  kat {k+1}/5", flush=True)
    print(f"[Marburg CV] kucuk LogReg (bazik, logP, MW, TPSA, bazik*logP) AUC {auc(y, pl):.3f}; RF desc+maccs AUC {auc(y, po):.3f}", flush=True)
    print(f"[Marburg CV] test->egitim en yakin komsu Tanimoto: medyan {np.median(sim):.2f}; >=0.5 %{(sim>=.5).mean()*100:.0f}; >=0.7 %{(sim>=.7).mean()*100:.0f}")
    bins = [(0, .3), (.3, .4), (.4, .5), (.5, .7), (.7, 1.01)]; rows = []
    for lo, hi in bins:
        i = (sim >= lo) & (sim < hi)
        rows.append(dict(bin=f"{lo}-{min(hi,1)}", n=int(i.sum()), aktif=int(y[i].sum()), auc_rf=auc(y[i], po[i]), auc_cad=auc(y[i], pc[i]), auc_lr=auc(y[i], pl[i])))
    bt = pd.DataFrame(rows); print(bt.round(3).to_string(index=False), flush=True); bt.to_csv(os.path.join(R, "marburg_auc_by_similarity.csv"), index=False)
    for nm, msk in (("bazik amin var", S[:, 0] == 1), ("bazik amin yok", S[:, 0] == 0)):
        print(f"[Marburg CV | {nm}, n={int(msk.sum())}] RF AUC {auc(y[msk], po[msk]):.3f}; CAD {auc(y[msk], pc[msk]):.3f}", flush=True)
    # ---- ayni aktarim alt kumelerinde taban
    N = pd.read_csv(os.path.join(R, "marburg_model_on_ncgc_ebola.csv")); SN = simple(N.smiles)
    print(f"\n[NCGC Ebola aktarim n={len(N)}] model AUC {auc(N.y, N.p_marburg):.3f}; CAD {auc(N.y, SN[:,4]):.3f}; CAD'e kati: bazik var n={int((SN[:,0]==1).sum())} model {auc(N.y[SN[:,0]==1], N.p_marburg[SN[:,0]==1]):.3f} / CAD {auc(N.y[SN[:,0]==1], SN[SN[:,0]==1,4]):.3f}; bazik yok n={int((SN[:,0]==0).sum())} model {auc(N.y[SN[:,0]==0], N.p_marburg[SN[:,0]==0]):.3f} / CAD {auc(N.y[SN[:,0]==0], SN[SN[:,0]==0,4]):.3f}", flush=True)
    C = pd.read_csv(os.path.join(R, "marburg_model_on_chembl.csv")); SC = simple(C.smiles); m1 = C.label.notna().values
    print(f"[ChEMBL aktarim 1uM n={int(m1.sum())}] model AUC {auc(C.label[m1], C.p_marburg[m1]):.3f}; CAD {auc(C.label[m1], SC[m1,4]):.3f}; bazik amin orani %{SC[:,0].mean()*100:.0f}", flush=True)
    # ---- (3) y-randomizasyon
    perm_auc = []
    for r in range(NPERM):
        yp = np.random.RandomState(100 + r).permutation(y); p = np.zeros(len(y))
        for tr, te in sp: p[te] = rf(150).fit(X[tr], yp[tr]).predict_proba(X[te])[:, 1]
        perm_auc.append(auc(yp, p)); print(f"  y-rastgele {r+1}/{NPERM}: AUC {perm_auc[-1]:.3f}", flush=True)
    a = auc(y, po); mu, sd = np.mean(perm_auc), np.std(perm_auc, ddof=1)
    print(f"[y-randomizasyon] gercek AUC {a:.3f}; karistirilmis ort {mu:.3f} sd {sd:.4f} (n={NPERM}); Z = {(a-mu)/sd:.0f}  (permutasyon sayisi az: Z kaba; alt sinir icin ortalama ~0.5 olmasi esas)")
    print("bitti")
