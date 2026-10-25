"""Average scIB results over seeds and report supervised minus unsupervised differences.

Usage: python scib_summary.py <scib_dir>   (files named <dataset>_seed<s>.csv)
"""
import glob
import os
import sys
import pandas as pd

rows = []
for f in sorted(glob.glob(os.path.join(sys.argv[1], "*_seed*.csv"))):
    ds, seed = os.path.basename(f)[:-4].rsplit("_seed", 1)
    df = pd.read_csv(f, index_col=0)
    for m, r in df[df.index.isin(["scVI", "scANVI"])].iterrows():
        rows.append({"dataset": ds, "seed": int(seed), "method": m, **{k: float(v) for k, v in r.items()}})
T = pd.DataFrame(rows)
T.to_csv("scib_all.csv", index=False)
cols = [c for c in T.columns if c not in ("dataset", "seed", "method")]
g = T.groupby(["dataset", "method"])[cols].mean()
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
print(g.round(4).to_string())
print("\n== supervised - unsupervised, mean over seeds ==")
print((g.xs("scANVI", level="method") - g.xs("scVI", level="method")).round(4).to_string())
