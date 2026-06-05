"""
k_sensitivity.py — readout robustness for the cross-dataset transfer benchmark.

Sweeps the kNN label-transfer readout over k ∈ {5, 15, 30, 50} on saved embeddings
and reports per-stratum (abundant / intermediate / rare) retention at each k.
Used to demonstrate that the abundant-type harm is not an artifact of a single
neighbourhood size; reported in the paper as Table 3.

CPU-only. Reuses embeddings produced by cross_dataset_arm.run_once
(saved as <emb_dir>/<dataset>_seed<seed>.npz with keys:
  ref_scvi, ref_scanvi, q_scvi, q_scanvi, rl (ref labels), ql (query labels)).
"""

import argparse, glob, os, json, re
import numpy as np


def load_embedding(npz_path):
    """Load a saved embedding bundle. Returns a dict with float embeddings and labels."""
    d = np.load(npz_path, allow_pickle=True)
    return {
        "ref_scvi":   d["ref_scvi"],
        "ref_scanvi": d["ref_scanvi"],
        "q_scvi":     d["q_scvi"],
        "q_scanvi":   d["q_scanvi"],
        "rl":         d["rl"],
        "ql":         d["ql"],
    }


def knn_predict(ref, ref_labels, query, k=15):
    """kNN majority-vote label transfer from ref to query in latent space."""
    from sklearn.neighbors import KNeighborsClassifier
    return KNeighborsClassifier(n_neighbors=k, n_jobs=-1).fit(ref, ref_labels).predict(query)


def per_stratum_retention(pred, ql):
    """
    Stratify query types into abundant (top tertile), intermediate, rare (bottom tertile)
    by query abundance, and return mean per-type retention within each stratum.
    Skips types with <10 cells.
    """
    types, counts = np.unique(ql, return_counts=True)
    if len(types) < 3:
        return {"abundant": float("nan"), "intermediate": float("nan"), "rare": float("nan")}
    size = {t: c for t, c in zip(types, counts)}
    thr_lo = np.percentile(list(size.values()), 33)
    thr_hi = np.percentile(list(size.values()), 66)
    grp = {"abundant": [], "intermediate": [], "rare": []}
    for t in types:
        m = ql == t
        if m.sum() < 10:
            continue
        if size[t] <= thr_lo:
            g = "rare"
        elif size[t] > thr_hi:
            g = "abundant"
        else:
            g = "intermediate"
        grp[g].append(float(np.mean(pred[m] == t)))
    return {g: float(np.mean(v)) if v else float("nan") for g, v in grp.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--emb_dir", default="embeddings",
                    help="Directory containing <dataset>_seed<seed>.npz files")
    ap.add_argument("--ks", nargs="+", type=int, default=[5, 15, 30, 50],
                    help="kNN k values to sweep")
    ap.add_argument("--datasets", nargs="*", default=None,
                    help="Restrict to these datasets (default: all found)")
    ap.add_argument("--out", default="k_sensitivity.json")
    args = ap.parse_args()

    # Discover embedding files. Filter out variant files (shuffled labels, capacity
    # sweeps, ablations) — we only sweep k on the main scVI/scANVI runs.
    excluded_tags = ["_shuffled", "_withISG", "_ablISG", "_ablCtl",
                     "_fullref", "_ablTarget", "_ablControl", "_dim"]
    candidates = sorted(glob.glob(os.path.join(args.emb_dir, "*_seed*.npz")))
    files = [f for f in candidates
             if not any(tag in os.path.basename(f) for tag in excluded_tags)]
    if args.datasets:
        files = [f for f in files
                 if any(os.path.basename(f).startswith(d + "_seed") for d in args.datasets)]
    if not files:
        raise SystemExit(f"No embedding files found in {args.emb_dir}")

    # Aggregate per (k, method, stratum) across datasets and seeds
    agg = {k: {"scVI":   {s: [] for s in ["abundant", "intermediate", "rare"]},
               "scANVI": {s: [] for s in ["abundant", "intermediate", "rare"]}}
           for k in args.ks}
    for f in files:
        d = load_embedding(f)
        for k in args.ks:
            psv = knn_predict(d["ref_scvi"], d["rl"], d["q_scvi"], k)
            psa = knn_predict(d["ref_scanvi"], d["rl"], d["q_scanvi"], k)
            for s, v in per_stratum_retention(psv, d["ql"]).items():
                agg[k]["scVI"][s].append(v)
            for s, v in per_stratum_retention(psa, d["ql"]).items():
                agg[k]["scANVI"][s].append(v)

    # Summary
    out = {}
    print(f"\n{'k':>4} {'method':>8} {'abund':>8} {'inter':>8} {'rare':>8} "
          f"{'harm_ab':>10} {'gain_ra':>10}")
    print("-" * 66)
    for k in args.ks:
        m = {meth: {s: float(np.nanmean(agg[k][meth][s])) for s in ["abundant", "intermediate", "rare"]}
             for meth in ["scVI", "scANVI"]}
        harm_ab = m["scVI"]["abundant"] - m["scANVI"]["abundant"]
        gain_ra = m["scANVI"]["rare"] - m["scVI"]["rare"]
        out[k] = dict(scVI=m["scVI"], scANVI=m["scANVI"],
                      harm_abundant=harm_ab, gain_rare=gain_ra)
        print(f"{k:>4} {'scVI':>8} {m['scVI']['abundant']:>8.3f} "
              f"{m['scVI']['intermediate']:>8.3f} {m['scVI']['rare']:>8.3f}")
        print(f"{k:>4} {'scANVI':>8} {m['scANVI']['abundant']:>8.3f} "
              f"{m['scANVI']['intermediate']:>8.3f} {m['scANVI']['rare']:>8.3f} "
              f"{harm_ab:>+10.3f} {gain_ra:>+10.3f}")

    json.dump(out, open(args.out, "w"), indent=2, default=float)
    print(f"\nsaved -> {args.out}")
    print("Trade-off is readout-robust if harm_abundant stays positive and similar")
    print("magnitude across all k.")


if __name__ == "__main__":
    main()
