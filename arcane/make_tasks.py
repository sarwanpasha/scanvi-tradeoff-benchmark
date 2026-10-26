"""Write task lists: tasks_train.txt (train.py args), tasks_eval.txt (tag seed), tasks_ct.txt."""
import sys

main = {"spleen_lymph": 5, "heart": 5, "lupus": 5, "pbmc_fine": 3, "hlca": 3}
holdout = {"pbmc_fine": ["pDC", "MAIT", "Plasmablast"], "spleen_lymph": ["pDCs", "Neutrophils"],
           "hlca": ["AT1", "Non-classical monocytes"]}
downsample = {"pbmc_fine": ["pDC:20", "MAIT:20"]}
only = sys.argv[1:]  # optional subset of datasets
train, ev, ct = [], [], []


def add(ds, seed, extra="", tag=None):
    for split in ["full", "cf0", "cf1"]:
        train.append(f"--ds {ds} --seed {seed} --split {split} {extra}".strip())
    ev.append(f"{tag or ds} {seed}")


for ds, n in main.items():
    if only and ds not in only:
        continue
    for s in range(n):
        add(ds, s)
        ct.append(f"{ds} {s}")
for ds, types in holdout.items():
    if only and ds not in only:
        continue
    for t in types:
        for s in range(3):
            tag = f"{ds}_ho-{t}".replace(" ", "~").replace("/", "-")
            add(ds, s, f"--holdout {t.replace(' ', '~')}", tag)
for ds, items in downsample.items():
    if only and ds not in only:
        continue
    for it in items:
        for s in range(3):
            tag = f"{ds}_ds-{it}".replace(":", "-")
            add(ds, s, f"--downsample {it}", tag)
open("tasks_train.txt", "w").write("\n".join(train) + "\n")
open("tasks_eval.txt", "w").write("\n".join(ev) + "\n")
open("tasks_ct.txt", "w").write("\n".join(ct) + "\n")
print(len(train), "train tasks,", len(ev), "eval tasks,", len(ct), "celltypist tasks")
