"""Ebolavirus (CHEMBL4434621) verisini iki modele hazirlar.
 (a) regresyon: pseudovirus giris testleri, '=' IC50/EC50 nM -> pXC50 (bilesik basina medyan)
 (b) siniflandirma: tum IC50/EC50 nM (sinir degerler dahil) -> aktif/inaktif
Cikti: results/ebola_regression.csv, results/ebola_classification.csv, results/curation_log.txt
"""
import json, glob, math, statistics, collections, os
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
RDLogger.DisableLog("rdApp.*")

BASE = os.path.join(os.path.dirname(__file__), "..")
# Kullanici karari: 1 uM. Ilk taslak 10 uM (pX>=5) idi; sayilari gorduktan sonra (%81 aktif) degistirildi.
ACT_THRESHOLD = 6.0  # pXC50 >= 6 (<=1 uM) aktif
log = []
def L(s):
    print(s); log.append(s)

acts = []
for f in sorted(glob.glob(os.path.join(BASE, "data", "act_CHEMBL4434621_*.json"))):
    acts += json.load(open(f))["activities"]
L(f"ham kayit: {len(acts)}")

PSEUDO = ("pseudotyp", "pseudo-typ", "pseudovir", "vsv", "lentivir", "pseudo ")
def is_pseudo(desc):
    d = (desc or "").lower()
    return any(k in d for k in PSEUDO)

chooser = rdMolStandardize.LargestFragmentChooser()
unch = rdMolStandardize.Uncharger()
def std(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None: return None
    m = chooser.choose(m); m = unch.uncharge(m)
    if m.GetNumHeavyAtoms() < 8 or m.GetNumHeavyAtoms() > 120: return None
    if not any(a.GetSymbol() == "C" for a in m.GetAtoms()): return None
    return m

rows = []
for a in acts:
    if a["standard_type"] not in ("IC50", "EC50") or a["standard_units"] != "nM": continue
    if a["standard_value"] is None or not a["canonical_smiles"]: continue
    v = float(a["standard_value"])
    if v <= 0: continue
    m = std(a["canonical_smiles"])
    if m is None: continue
    rows.append(dict(
        mol=a["molecule_chembl_id"], ik=Chem.MolToInchiKey(m), smiles=Chem.MolToSmiles(m),
        type=a["standard_type"], rel=a["standard_relation"] or "=", value_nM=v,
        pX=-math.log10(v * 1e-9), doc=a["document_chembl_id"], assay=a["assay_chembl_id"],
        pseudo=is_pseudo(a["assay_description"]), desc=a["assay_description"]))
df = pd.DataFrame(rows)
L(f"IC50/EC50 nM, standardize edilebilen: {len(df)} olcum, {df.ik.nunique()} bilesik")
L(f"iliski dagilimi: {df.rel.value_counts().to_dict()}")

# ---- (a) regresyon: pseudovirus + '=' ----
ra = df[(df.pseudo) & (df.rel == "=")].copy()
L(f"\n(a) pseudovirus '=': {len(ra)} olcum, {ra.ik.nunique()} bilesik, {ra.doc.nunique()} dokuman")
# pX araligi 2-10 disi sil
bad = ra[(ra.pX < 3) | (ra.pX > 10)]
L(f"  pX araligi disi atilan: {len(bad)}")
ra = ra.drop(bad.index)
ra[["mol", "ik", "smiles", "type", "value_nM", "pX", "doc", "assay"]].to_csv(
    os.path.join(BASE, "results", "ebola_regression_measurements.csv"), index=False)  # olcum duzeyi (doc z-skoru icin)
# ayni bilesikte tekrarlar arasi yayilim
g = ra.groupby("ik").pX
spread = (g.max() - g.min())[g.size() > 1]
L(f"  tekrarli bilesik: {len(spread)}; medyan yayilim {spread.median():.2f} log; >1 log olan: {(spread > 1).sum()}")
agg = ra.groupby("ik").agg(smiles=("smiles", "first"), mol=("mol", "first"), pX=("pX", "median"),
                            n_meas=("pX", "size"), n_doc=("doc", "nunique"),
                            docs=("doc", lambda s: ";".join(sorted(set(s))))).reset_index()
agg["main_doc"] = agg.docs.str.split(";").str[0]
L(f"  nihai: {len(agg)} bilesik; pX ort {agg.pX.mean():.2f}, sd {agg.pX.std():.2f}, min {agg.pX.min():.2f}, max {agg.pX.max():.2f}")
L("  dokuman basina bilesik (ilk 8): " + str(agg.main_doc.value_counts().head(8).to_dict()))
agg.to_csv(os.path.join(BASE, "results", "ebola_regression.csv"), index=False)

# ---- (b) siniflandirma: tum testler ----
# '=' -> pX; '>' -> alt sinir (pX <= x), '<' -> ust sinir
def lab(r):
    if r.rel == "=": return int(r.pX >= ACT_THRESHOLD)
    if r.rel in (">", ">="): return 0 if r.pX <= ACT_THRESHOLD else None  # >10 uM => inaktif; >1 uM belirsiz
    if r.rel in ("<", "<="): return 1 if r.pX >= ACT_THRESHOLD else None
    return None
df["label"] = df.apply(lab, axis=1)
cb = df.dropna(subset=["label"]).copy()
L(f"\n(b) etiketlenebilen olcum: {len(cb)} / {len(df)}; bilesik {cb.ik.nunique()}")
# bilesik seviyesinde: olcumler cakisirsa cogunluk, esitlikte at (celiski)
def vote(s):
    m = s.mean()
    return 1 if m > 0.5 else (0 if m < 0.5 else None)
cc = cb.groupby("ik").agg(smiles=("smiles", "first"), mol=("mol", "first"), label=("label", vote),
                           n_meas=("label", "size"), mean_lab=("label", "mean"),
                           pX_med=("pX", "median"), docs=("doc", lambda s: ";".join(sorted(set(s))))).reset_index()
tie = cc.label.isna().sum()
conflict = ((cc.mean_lab > 0) & (cc.mean_lab < 1)).sum()
L(f"  bilesik: {len(cc)}; olcumleri celisen: {conflict}; esit oy nedeniyle atilan: {tie}")
cc = cc.dropna(subset=["label"]); cc["label"] = cc.label.astype(int)
cc["main_doc"] = cc.docs.str.split(";").str[0]
L(f"  nihai: {len(cc)} bilesik; aktif {cc.label.sum()} ({cc.label.mean()*100:.0f}%), inaktif {(cc.label==0).sum()}")
L(f"  esik duyarliligi (bilesik medyan pX): >=6 aktif sayisi {(cc.pX_med>=6).sum()}, >=5 {(cc.pX_med>=5).sum()}, >=4 {(cc.pX_med>=4).sum()}")
L("  dokuman basina bilesik (ilk 8): " + str(cc.main_doc.value_counts().head(8).to_dict()))
cc.to_csv(os.path.join(BASE, "results", "ebola_classification.csv"), index=False)
open(os.path.join(BASE, "results", "curation_log.txt"), "w", encoding="utf-8").write("\n".join(log))
