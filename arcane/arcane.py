"""ARCANE: arbitration of complementary embeddings for annotation with error control.

Given kNN label evidence from a label-free (unsupervised) embedding U and a label-shaped
(supervised) embedding S of the same cells, ARCANE
  1. arbitrates between the two kNN labels with a rule that maximizes held-out balanced accuracy,
     estimated on reference batches that were held out during training (cross-fitting), and
  2. returns class-conditional (Mondrian) conformal prediction sets, and
  3. flags cells that lie far from the reference in both embeddings (scale-free kNN distance with a
     conformal threshold) as matching no reference type.
"""
import numpy as np
from sklearn.neighbors import NearestNeighbors


def knn_evidence(z_ref, y_ref, z_tgt, classes, k=15, exclude_self=False):
    """Return (n_tgt, n_classes) kNN label proportions and the mean neighbor distance."""
    nn = NearestNeighbors(n_neighbors=k + int(exclude_self)).fit(z_ref)
    dist, idx = nn.kneighbors(z_tgt)
    if exclude_self:
        dist, idx = dist[:, 1:], idx[:, 1:]
    cls_index = {c: i for i, c in enumerate(classes)}
    yi = np.array([cls_index[c] for c in y_ref])
    P = np.zeros((len(z_tgt), len(classes)))
    np.add.at(P, (np.repeat(np.arange(len(z_tgt)), k), yi[idx].ravel()), 1.0 / k)
    return P, dist.mean(1)


def centroid_scores(ref_U, ref_S, y_ref, tgt_U, tgt_S, classes):
    """Scaled distance to each reference type centroid, maximized over the two views.

    For view V and type t, d = ||z - mu_t|| / r_t, where r_t is the median distance of type-t
    reference cells to mu_t. A cell is typical of t only if it is typical in both views.
    """
    out = None
    for R, Z in ((ref_U, tgt_U), (ref_S, tgt_S)):
        mu = np.stack([R[y_ref == c].mean(0) for c in classes])
        r = np.array([np.median(np.linalg.norm(R[y_ref == c] - mu[i], axis=1)) + 1e-8 for i, c in enumerate(classes)])
        D = np.sqrt(((Z[:, None, :] - mu[None, :, :]) ** 2).sum(-1)) / r[None, :]
        out = D if out is None else np.maximum(out, D)
    return out


def hard(P):
    return P.argmax(1)


class Arbiter:
    """Balanced-accuracy-optimal arbitration between two label predictions.

    For an ordered disagreement (a, b), i.e. U predicts a and S predicts b, the rule outputs a if
    q_a(a, b) >= q_b(a, b), where q_t(a, b) = P(U = a, S = b | Y = t) is estimated on calibration
    cells. Pairs without calibration cells (an empirical tie) fall back to the label with the
    larger summed evidence. With objective="accuracy", class-conditional rates are replaced by
    joint counts, which maximizes overall accuracy instead of balanced accuracy (ablation).
    """

    def __init__(self, n_classes, min_count=1, objective="balanced"):
        self.T, self.min_count, self.objective = n_classes, min_count, objective

    def fit(self, yU, yS, y):
        T = self.T
        N = np.bincount(y, minlength=T).astype(float)
        cnt = np.zeros((T, T, T))  # cnt[a, b, t]
        np.add.at(cnt, (yU, yS, y), 1.0)
        self.cnt = cnt
        w = 1.0 / np.maximum(N, 1.0) if self.objective == "balanced" else np.ones(T)
        A = np.arange(T)
        qa = cnt[A[:, None], A[None, :], A[:, None]] * w[:, None]  # q_a(a, b)
        qb = cnt[A[:, None], A[None, :], A[None, :]] * w[None, :]  # q_b(a, b)
        support = cnt[A[:, None], A[None, :], A[:, None]] + cnt[A[:, None], A[None, :], A[None, :]]
        self.choose_U = qa >= qb
        self.known = support >= self.min_count
        return self

    def predict(self, PU, PS):
        yU, yS = hard(PU), hard(PS)
        out = yS.copy()
        agree = yU == yS
        d = ~agree
        cu = self.choose_U[yU[d], yS[d]]
        kn = self.known[yU[d], yS[d]]
        ev = PU + PS
        fallback_U = ev[np.where(d)[0], yU[d]] >= ev[np.where(d)[0], yS[d]]
        pick_U = np.where(kn, cu, fallback_U)
        out[d] = np.where(pick_U, yU[d], yS[d])
        return out


class MondrianConformal:
    """Class-conditional split conformal prediction on a score matrix (lower = more typical)."""

    def __init__(self, alpha=0.1):
        self.alpha = alpha

    def fit(self, Sc, y, n_classes):
        self.cal = [np.sort(Sc[y == t, t]) for t in range(n_classes)]
        pooled = np.sort(Sc[np.arange(len(y)), y])
        self.pooled = pooled
        self.tau = np.ones(n_classes)
        self.pooled_types = []
        for t, s in enumerate(self.cal):
            n = len(s)
            r = int(np.ceil((n + 1) * (1 - self.alpha)))
            if 1 <= r <= n:
                self.tau[t] = s[r - 1]
            else:  # too few calibration cells of this type: use the threshold pooled over all types
                rp = int(np.ceil((len(pooled) + 1) * (1 - self.alpha)))
                self.tau[t] = pooled[min(rp, len(pooled)) - 1]
                self.pooled_types.append(t)
        return self

    def sets(self, Sc):
        return Sc <= self.tau[None, :]

    def pvalues(self, Sc):
        """Conformal p-value of each class for each cell (type-conditional)."""
        S = Sc
        out = np.ones_like(S)
        for t, s in enumerate(self.cal):
            if t in self.pooled_types:
                s = self.pooled
            # number of calibration scores >= test score
            ge = len(s) - np.searchsorted(s, S[:, t], side="left")
            out[:, t] = (1.0 + ge) / (len(s) + 1.0)
        return out


def loo_scale(R, k=15, n=5000, seed=0):
    """Median leave-one-out kNN distance among (a sample of) reference cells; makes distances scale-free."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(R), min(n, len(R)), replace=False)
    d = NearestNeighbors(n_neighbors=k + 1).fit(R).kneighbors(R[idx])[0][:, 1:].mean(1)
    return float(np.median(d)) + 1e-12


def run_arcane(ref_U, ref_S, y_ref, tgt_U, tgt_S, cal, classes, k=15, alpha=0.1,
               objective="balanced", score_type="hybrid", alpha_unknown=0.05):
    """Fit ARCANE on calibration data and annotate target cells.

    cal: list of dicts with keys ref_U, ref_S, y_ref, tgt_U, tgt_S, y_tgt (one per cross-fit fold).
    Returns a dict with arbitrated labels, prediction sets, novelty scores, and evidence.
    """
    T = len(classes)
    ci = {c: i for i, c in enumerate(classes)}
    def score(rU, rS, yr, tU, tS, PU, PS):
        if score_type == "knn":
            return 1.0 - 0.5 * (PU + PS)
        D = centroid_scores(rU, rS, yr, tU, tS, classes)
        if score_type == "centroid":
            return D
        # hybrid: kNN evidence, with ties (including zero evidence) broken by centroid distance;
        # the tie-breaker is smaller than one evidence step 1/(2k), so it never reorders evidence levels
        return 1.0 - 0.5 * (PU + PS) + (0.999 / (2 * k)) * D / (1.0 + D)

    yU_c, yS_c, y_c, S_c, D_c = [], [], [], [], []
    for f in cal:
        keep = np.isin(f["y_tgt"], classes)
        PU, du = knn_evidence(f["ref_U"], f["y_ref"], f["tgt_U"][keep], classes, k)
        PS, ds = knn_evidence(f["ref_S"], f["y_ref"], f["tgt_S"][keep], classes, k)
        D_c.append(0.5 * (du / loo_scale(f["ref_U"], k) + ds / loo_scale(f["ref_S"], k)))
        yU_c.append(hard(PU)); yS_c.append(hard(PS))
        S_c.append(score(f["ref_U"], f["ref_S"], f["y_ref"], f["tgt_U"][keep], f["tgt_S"][keep], PU, PS))
        y_c.append(np.array([ci[c] for c in f["y_tgt"][keep]]))
    yU_c, yS_c, y_c, S_c, D_c = map(np.concatenate, (yU_c, yS_c, y_c, S_c, D_c))
    D_sorted = np.sort(D_c)
    r_u = int(np.ceil((len(D_sorted) + 1) * (1 - alpha_unknown)))
    tau_u = D_sorted[min(r_u, len(D_sorted)) - 1]
    arb = Arbiter(T, objective=objective).fit(yU_c, yS_c, y_c)
    conf = MondrianConformal(alpha).fit(S_c, y_c, T)
    PU, dU = knn_evidence(ref_U, y_ref, tgt_U, classes, k)
    PS, dS = knn_evidence(ref_S, y_ref, tgt_S, classes, k)
    Sq = score(ref_U, ref_S, y_ref, tgt_U, tgt_S, PU, PS)
    D = 0.5 * (dU / loo_scale(ref_U, k) + dS / loo_scale(ref_S, k))
    return dict(label=arb.predict(PU, PS), yU=hard(PU), yS=hard(PS), yavg=hard(PU + PS),
                sets=conf.sets(Sq), novelty=D, unknown=D > tau_u, tau_unknown=tau_u,
                novelty_p=(1.0 + len(D_sorted) - np.searchsorted(D_sorted, D, side="left")) / (len(D_sorted) + 1.0),
                PU=PU, PS=PS, dU=dU, dS=dS,
                score=Sq, arbiter=arb, conformal=conf, cal_size=np.bincount(y_c, minlength=T))
