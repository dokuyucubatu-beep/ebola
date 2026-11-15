"""2D ozellikler: RDKit tanimlayicilari, Morgan sayim (r=2, 2048), MACCS.
Her iki veri seti (regresyon, siniflandirma) icin results/features_*.npz uretir."""
import os, numpy as np, pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, MACCSkeys, rdFingerprintGenerator
RDLogger.DisableLog("rdApp.*")
BASE = os.path.join(os.path.dirname(__file__), "..", "results")
gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
names = [n for n, _ in Descriptors.descList]

def feats(smiles):
    D, M, K = [], [], []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        d = []
        for n, f in Descriptors.descList:
            try: d.append(float(f(m)))
            except Exception: d.append(np.nan)
        D.append(d)
        M.append(np.minimum(gen.GetCountFingerprintAsNumPy(m), 255))
        K.append(np.array(list(MACCSkeys.GenMACCSKeys(m)), dtype=np.uint8))
    D = np.array(D, dtype=np.float64); D[~np.isfinite(D)] = np.nan
    return D, np.array(M, dtype=np.uint8), np.array(K, dtype=np.uint8)

for tag, f, ycol in [("reg", "ebola_regression.csv", "pX"), ("clf", "ebola_classification.csv", "label")]:
    df = pd.read_csv(os.path.join(BASE, f))
    D, M, K = feats(df.smiles)
    print(tag, len(df), "desc", D.shape, "nan-sutun:", int(np.isnan(D).any(0).sum()))
    np.savez_compressed(os.path.join(BASE, f"features_{tag}.npz"), desc=D, morgan=M, maccs=K,
                        y=df[ycol].values, group=df.main_doc.values, smiles=df.smiles.values,
                        mol=df["mol"].values, desc_names=np.array(names))
