"""Marburg giris qHTS (AID 540276) kurasyonu. Kural (kullanici karari): yalniz kesin etiketler.
 aktif = Outcome Active VE Phenotype Inhibitor; inaktif = Outcome Inactive; Inconclusive ve aktivator-aktifler atilir.
SMILES: en buyuk organik fragman, notralize, InChIKey tekillestirme (celiskili tekrarlar atilir)."""
import os, gzip, numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize
BASE = os.path.join(os.path.dirname(__file__), "..")
SRC = os.path.join(BASE, "data", "marburg_dl", "0540001_0541000", "540276.csv.gz")

def work(smi):
    RDLogger.DisableLog("rdApp.*")
    try:
        m = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
        if m is None: return None
        m = rdMolStandardize.Uncharger().uncharge(rdMolStandardize.LargestFragmentChooser(preferOrganic=True).choose(m))
        if not (6 <= m.GetNumHeavyAtoms() <= 120) or not any(a.GetSymbol() == "C" for a in m.GetAtoms()): return None
        return Chem.MolToInchiKey(m), Chem.MolToSmiles(m)
    except Exception:
        return None

if __name__ == "__main__":
    d = pd.read_csv(SRC, low_memory=False)
    d = d[pd.to_numeric(d.PUBCHEM_SID, errors="coerce").notna()].copy()
    print("satir", len(d))
    act = (d.PUBCHEM_ACTIVITY_OUTCOME == "Active") & (d.Phenotype == "Inhibitor")
    ina = d.PUBCHEM_ACTIVITY_OUTCOME == "Inactive"
    print(f"aktif+inhibitor {int(act.sum())} (Active toplam {(d.PUBCHEM_ACTIVITY_OUTCOME=='Active').sum()}); inaktif {int(ina.sum())}; atilan {int((~(act|ina)).sum())}")
    d = d[act | ina].copy(); d["y"] = act[d.index].astype(int)
    d["pot"] = pd.to_numeric(d.Potency, errors="coerce")
    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 4) - 1)) as ex:
        res = list(ex.map(work, d.PUBCHEM_EXT_DATASOURCE_SMILES.tolist(), chunksize=2000))
    ok = [r is not None for r in res]; print("standardize edilemeyen:", len(res) - sum(ok))
    d = d[ok].copy(); d["ik"] = [r[0] for r, o in zip(res, ok) if o]; d["smiles"] = [r[1] for r, o in zip(res, ok) if o]
    g = d.groupby("ik").agg(smiles=("smiles", "first"), y=("y", "mean"), n=("y", "size"), pot=("pot", "median")).reset_index()
    conf = ((g.y > 0) & (g.y < 1)).sum(); tie = (g.y == 0.5).sum()
    g = g[g.y != 0.5].copy(); g["y"] = (g.y > 0.5).astype(int)
    print(f"benzersiz bilesik {len(g)}; celiskili tekrar {conf} (esit oy atilan {tie}); aktif {int(g.y.sum())} (%{g.y.mean()*100:.1f}); aktif potens medyan {g[g.y==1].pot.median():.1f} uM")
    g.to_csv(os.path.join(BASE, "results", "marburg_clean.csv"), index=False)
