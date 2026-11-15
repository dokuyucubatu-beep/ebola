"""Kullanim:  python scripts/12_predict.py girdi.csv [cikti.csv]
girdi: 'smiles' sutunlu CSV (varsa 'name' sutunu korunur) ya da satir basina bir SMILES olan .smi/.txt.
Cikti sutunlari: skor, AD (uygulanabilirlik) bandi, CAD/lizozomotropi bayragi, Tox21 uyari bayraklari. MODEL_CARD.md'yi okuyun."""
import os, sys, warnings, joblib, numpy as np, pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize
warnings.filterwarnings("ignore"); RDLogger.DisableLog("rdApp.*")
HERE = os.path.dirname(os.path.abspath(__file__)); BASE = os.path.join(HERE, ".."); R = os.path.join(BASE, "results"); T = os.path.join(BASE, "tox21")
sys.path.insert(0, HERE)
import importlib.util
spec = importlib.util.spec_from_file_location("mm", os.path.join(HERE, "10_marburg_model.py")); mm = importlib.util.module_from_spec(spec); spec.loader.exec_module(mm)

TASKS = ["NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"]
WARN = ["SR-MMP", "SR-p53", "SR-ARE"]   # sitotoksisite/stres ile ilgili; yalnizca zayif uyari
BASIC = Chem.MolFromSmarts("[NX3;!$(NC=[O,S,N]);!$(N-a);!$(N-S(=O)=O);!$(N-[#7,#8]);!$(N#*);!$(N=*)]")
chooser = rdMolStandardize.LargestFragmentChooser(preferOrganic=True); unch = rdMolStandardize.Uncharger()

def prep(smi):
    m = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
    if m is None: return None
    m = unch.uncharge(chooser.choose(m)); n = m.GetNumHeavyAtoms()
    return m if 6 <= n <= 120 and any(a.GetSymbol() == "C" for a in m.GetAtoms()) else None

def ad_band(s):
    return "icinde" if s >= .5 else ("sinirda" if s >= .3 else "disinda")

def main(inp, out):
    d = pd.read_csv(inp) if inp.endswith(".csv") else pd.DataFrame({"smiles": [l.strip() for l in open(inp) if l.strip()]})
    out_df = predict_df(d); out_df.to_csv(out, index=False)
    print(f"{int(out_df.gecerli.sum())}/{len(d)} gecerli molekul -> {out}")

def predict_df(d):
    d = d.reset_index(drop=True); mols = [prep(s) for s in d.smiles]; ok = np.array([m is not None for m in mols])
    res = pd.DataFrame(index=d.index); res["gecerli"] = ok
    if ok.any():
        sm = [Chem.MolToSmiles(m) for m, o in zip(mols, ok) if o]
        D, K, M, _ = mm.feat_chunk(sm); D[~np.isfinite(D)] = np.nan
        model = joblib.load(os.path.join(R, "marburg_model.joblib")); Xm = np.hstack([D, K.astype(float)]); sc = model.predict_proba(Xm)[:, 1]
        Xp = model[:-1].transform(Xm); tp = np.array([t.predict_proba(Xp)[:, 1] for t in model[-1].estimators_]); sd = tp.std(0)
        ref = np.load(os.path.join(R, "marburg_sample_features.npz"), allow_pickle=True)["M"]; sim = mm.maxsim(M, ref)
        mol_ok = [m for m in mols if m is not None]
        hasb = np.array([m.HasSubstructMatch(BASIC) for m in mol_ok]); lp = np.array([Crippen.MolLogP(m) for m in mol_ok])
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        Xt = np.array([np.minimum(gen.GetCountFingerprintAsNumPy(m), 255) for m in mol_ok], dtype=np.uint8)
        simt = mm.maxsim(Xt, np.load(os.path.join(T, "tox21_train_fp.npz"), allow_pickle=True)["X"])
        cols = {"smiles_std": sm, "skor_giris_testi": sc.round(3), "skor_agac_sd": sd.round(3), "ad_benzerlik": sim.round(2), "ad_bandi": [ad_band(s) for s in sim],
                "egitimde_marburg": sim >= .99, "bazik_amin": hasb, "logP": lp.round(2), "CAD_bayragi": hasb & (lp >= 3)}
        for t in TASKS:
            cols["tox21_" + t] = joblib.load(os.path.join(T, "models", f"rf_{t}.joblib")).predict_proba(Xt)[:, 1].round(2)
        cols["tox21_ad_benzerlik"] = simt.round(2); cols["egitimde_tox21"] = simt >= .99
        cols["tox21_uyari"] = [",".join(t for t in WARN if cols["tox21_" + t][i] >= .5) for i in range(len(sm))]
        for k, v in cols.items(): res.loc[ok, k] = v
    return pd.concat([d, res], axis=1)

if __name__ == "__main__":
    if len(sys.argv) < 2: sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else os.path.splitext(sys.argv[1])[0] + "_tahmin.csv")
