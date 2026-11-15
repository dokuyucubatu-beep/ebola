"""Analog onerici: bir ana molekulden kucuk, kimyasal olarak makul (biyoizosterik/ikame) degisikliklerle analog uretir,
her birini 12_predict.py ile skorlar. Serbest uretici DEGIL: skoru somurmesin diye degisiklikler sinirli, 2 adimla kisitli.
Kullanim: python scripts/13_analogs.py "SMILES" [cikti.csv] [--adim 2]
Cikti: ana molekule gore degisim (skor farki, agac sd), AD, CAD/Tox21 bayraklari, PAINS, uygulanan donusumler."""
import os, sys, importlib.util, numpy as np, pandas as pd
from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import AllChem, rdFingerprintGenerator
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
RDLogger.DisableLog("rdApp.*")
HERE = os.path.dirname(os.path.abspath(__file__)); BASE = os.path.join(HERE, ".."); R = os.path.join(BASE, "results")
spec = importlib.util.spec_from_file_location("pr", os.path.join(HERE, "12_predict.py")); pr = importlib.util.module_from_spec(spec); spec.loader.exec_module(pr)

# (ad, SMARTS reaksiyon). Hepsi tek merkezli, tanidik medkim donusumleri.
RULES = [
    ("aromatik CH -> N (piridin)", "[cH;R:1]>>[n:1]"),
    ("aromatik H -> F", "[cH:1]>>[c:1]F"),
    ("aromatik H -> Cl", "[cH:1]>>[c:1]Cl"),
    ("aromatik H -> CH3", "[cH:1]>>[c:1][CH3]"),
    ("aromatik H -> OCH3", "[cH:1]>>[c:1]O[CH3]"),
    ("aromatik H -> CF3", "[cH:1]>>[c:1]C(F)(F)F"),
    ("aromatik H -> CN", "[cH:1]>>[c:1]C#N"),
    ("aromatik H -> OH", "[cH:1]>>[c:1][OH]"),
    ("Cl -> F", "[c:1][Cl]>>[c:1]F"),
    ("F -> Cl", "[c:1][F]>>[c:1]Cl"),
    ("Cl -> CF3", "[c:1][Cl]>>[c:1]C(F)(F)F"),
    ("Cl -> CH3", "[c:1][Cl]>>[c:1][CH3]"),
    ("aromatik halojen sil", "[c:1][F,Cl,Br]>>[cH:1]"),
    ("OCH3 -> OH", "[c:1]O[CH3]>>[c:1][OH]"),
    ("OH -> OCH3", "[c:1][OH]>>[c:1]O[CH3]"),
    ("CH3 -> CF3", "[c:1][CH3]>>[c:1]C(F)(F)F"),
    ("CH3 sil", "[c:1][CH3]>>[cH:1]"),
    ("piperidin -> morfolin (bazikligi azaltir)", "[N:1]1[CH2:2][CH2:3][CH2:4][CH2:5][CH2:6]1>>[N:1]1[CH2:2][CH2:3][O:4][CH2:5][CH2:6]1"),
    ("piperidin -> piperazin-N-asil", "[N:1]1[CH2:2][CH2:3][CH2:4][CH2:5][CH2:6]1>>[N:1]1[CH2:2][CH2:3][N:4](C(=O)C)[CH2:5][CH2:6]1"),
    ("N,N-dimetil -> N-metil asetamid (amin bazikligi yok)", "[CH3:2][NX3;H0;!$(NC=O);!$(N-a):1][CH3:3]>>[CH3:2][N:1]([CH3:3])C(=O)C"),
    ("karboksilik asit -> tetrazol", "[c,C:1][CX3](=O)[OH]>>[c,C:1]c1nnn[nH]1"),
    ("ester -> amid", "[c,C:1][CX3](=O)O[CH3,CH2]>>[c,C:1][CX3](=O)[NH2]"),
    ("amid N-metil", "[CX3:1](=O)[NH:2]>>[CX3:1](=O)[N:2][CH3]"),
    ("fenil -> piridin (orto/meta/para)", "[cH:1]1[cH:2][cH:3][cH:4][cH:5][c:6]1>>[n:1]1[cH:2][cH:3][cH:4][cH:5][c:6]1"),
]
RXN = [(n, AllChem.ReactionFromSmarts(s)) for n, s in RULES]
params = FilterCatalogParams(); params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS); PAINS = FilterCatalog(params)

def apply_all(mol):
    out = {}
    for name, rxn in RXN:
        for ps in rxn.RunReactants((mol,)):
            try:
                p = ps[0]; Chem.SanitizeMol(p); smi = Chem.MolToSmiles(p)
                if Chem.MolFromSmiles(smi) is not None: out.setdefault(smi, name)
            except Exception: pass
    return out

def enumerate_analogs(parent_smi, steps=2, max_n=400, max_dheavy=8):
    parent = Chem.MolFromSmiles(parent_smi); ps = Chem.MolToSmiles(parent); n0 = parent.GetNumHeavyAtoms()
    seen = {ps: ""}; frontier = {ps: ""}
    for _ in range(steps):
        new = {}
        for smi, hist in frontier.items():
            for s2, name in apply_all(Chem.MolFromSmiles(smi)).items():
                if s2 in seen or s2 in new: continue
                m = Chem.MolFromSmiles(s2)
                if abs(m.GetNumHeavyAtoms() - n0) > max_dheavy: continue
                new[s2] = (hist + " ; " if hist else "") + name
                if len(seen) + len(new) > max_n: break
        seen.update(new); frontier = new
    seen.pop(ps); return ps, seen

if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args: sys.exit(__doc__)
    steps = int(sys.argv[sys.argv.index("--adim") + 1]) if "--adim" in sys.argv else 2
    if "--adim" in sys.argv: args = [a for a in args if a != str(steps)]
    ps, an = enumerate_analogs(args[0], steps)
    d = pd.DataFrame({"smiles": [ps] + list(an), "donusum": ["(ana molekul)"] + list(an.values())})
    r = pr.predict_df(d)
    if not r.gecerli.iloc[0]: sys.exit("ana molekul gecersiz ya da kapsam disi (6-120 agir atom)")
    fpg = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048); pf = fpg.GetFingerprint(Chem.MolFromSmiles(ps))
    r["ana_benzerlik"] = [DataStructs.TanimotoSimilarity(pf, fpg.GetFingerprint(Chem.MolFromSmiles(s))) for s in r.smiles]
    r["ana_benzerlik"] = r.ana_benzerlik.round(2)
    p0 = r.skor_giris_testi.iloc[0]; r["skor_degisim"] = (r.skor_giris_testi - p0).round(3)
    r["PAINS"] = [PAINS.HasMatch(Chem.MolFromSmiles(s)) if g else np.nan for s, g in zip(r.smiles, r.gecerli)]
    r["CAD_azaldi"] = (r.CAD_bayragi.iloc[0] is True or bool(r.CAD_bayragi.iloc[0])) and ~r.CAD_bayragi.astype(bool)
    # gurultu (sezgisel): RF standart hatasi (agac sd/sqrt(400)) + sabit 0.10 alt sinir; kucuk farklar yorumlanmaz
    se = r.skor_agac_sd / np.sqrt(400); r["fark_gurultude"] = r.skor_degisim.abs() < np.maximum(0.10, 2 * np.sqrt(se ** 2 + se.iloc[0] ** 2))
    out = args[1] if len(args) > 1 else os.path.join(R, "analog_oneriler.csv")
    rest = r.iloc[1:].sort_values("skor_giris_testi", ascending=False)
    pd.concat([r.iloc[:1], rest]).to_csv(out, index=False)
    a = r.iloc[0]
    print(f"ana: {ps} skor {a.skor_giris_testi:.2f} (sd {a.skor_agac_sd:.2f}), AD {a.ad_bandi}, CAD {bool(a.CAD_bayragi)}, Tox21 uyari: {a.tox21_uyari if isinstance(a.tox21_uyari, str) else '-'}")
    print(f"{len(rest)} analog uretildi -> {out}")
    cols = ["smiles", "donusum", "skor_giris_testi", "skor_degisim", "fark_gurultude", "ad_bandi", "CAD_bayragi", "tox21_uyari", "PAINS"]
    pd.set_option("display.width", 250, "display.max_colwidth", 60)
    print("\n-- skora gore ilk 10 --"); print(rest[cols].head(10).to_string(index=False))
    print("\n-- CAD bayragi kalkan ve AD disinda olmayan, skora gore ilk 10 --"); print(rest[rest.CAD_azaldi & (rest.ad_bandi != "disinda")][cols].head(10).to_string(index=False))
