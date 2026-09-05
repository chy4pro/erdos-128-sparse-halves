"""Core library for Erdos #128 (sparse halves in triangle-free graphs).

Conjecture (Erdos-Rousseau): every triangle-free graph G on n vertices has a set
S of floor(n/2) vertices with e(S) <= n^2/50.

Asymptotic / weighted form used here.  For a triangle-free graph H on vertex set
[m] and a weight vector w on the simplex (sum w = 1), the balanced blow-up of H
with part sizes w_i * n is triangle-free, and its min-half density is

    f(H,w) = min { sum_{ij in E(H)} y_i y_j : 0 <= y <= w, sum y = 1/2 }.

The conjecture implies  F(H) := max_w f(H,w) <= 1/50 = 0.02  for every
triangle-free H.  C_5 (uniform w) and Petersen (uniform w) both give exactly 0.02.

IMPORTANT (soundness):  every y we exhibit is FEASIBLE, hence  f(H,w) <= value(y).
So a heuristic inner minimiser can only ever *over*-estimate f; whenever it
returns <= 0.02 that is a rigorous certificate that (H,w) is not a counterexample.
Only values > 0.02 need further (exact) scrutiny.
"""
import itertools, math
import numpy as np
import networkx as nx


# ---------------------------------------------------------------- basic utils
def is_triangle_free(G):
    A = nx.to_numpy_array(G)
    return np.trace(A @ A @ A) < 0.5


def adj_matrix(G):
    return nx.to_numpy_array(G, nodelist=sorted(G.nodes()))


# --------------------------------------------------- exact min-half, brute force
def min_half_bruteforce(G, k=None):
    """Exact min over all k-subsets of e(S).  Returns (value, best_set).
    Uses bitmask adjacency; fine for n <= 24 or so."""
    nodes = sorted(G.nodes())
    n = len(nodes)
    idx = {v: i for i, v in enumerate(nodes)}
    if k is None:
        k = n // 2
    nbr = [0] * n
    for u, v in G.edges():
        nbr[idx[u]] |= 1 << idx[v]
        nbr[idx[v]] |= 1 << idx[u]
    best = None
    bestS = None
    for comb in itertools.combinations(range(n), k):
        mask = 0
        e = 0
        for v in comb:
            e += bin(nbr[v] & mask).count('1')
            mask |= 1 << v
        if best is None or e < best:
            best = e
            bestS = comb
            if best == 0:
                break
    return best, [nodes[i] for i in bestS]


# ------------------------------------------------------------ exact min-half MILP
def min_half_milp(G, k=None, time_limit=120.0, mip_rel_gap=0.0):
    """Exact min-half by MILP (HiGHS).  min sum_e y_e s.t. y_e >= x_u+x_v-1,
    sum x = k, x binary, y >= 0.  Exact because we minimise."""
    from scipy.optimize import milp, LinearConstraint, Bounds
    from scipy.sparse import lil_matrix
    nodes = sorted(G.nodes())
    n = len(nodes)
    idx = {v: i for i, v in enumerate(nodes)}
    E = [(idx[u], idx[v]) for u, v in G.edges()]
    m = len(E)
    if k is None:
        k = n // 2
    nv = n + m
    c = np.concatenate([np.zeros(n), np.ones(m)])
    rows = lil_matrix((m + 1, nv))
    lb = np.zeros(m + 1)
    ub = np.zeros(m + 1)
    for e, (u, v) in enumerate(E):
        rows[e, u] = -1.0
        rows[e, v] = -1.0
        rows[e, n + e] = 1.0
        lb[e] = -1.0
        ub[e] = np.inf
    rows[m, :n] = 1.0
    lb[m] = ub[m] = k
    integrality = np.concatenate([np.ones(n), np.zeros(m)])
    res = milp(c=c, constraints=LinearConstraint(rows.tocsr(), lb, ub),
               integrality=integrality,
               bounds=Bounds(np.zeros(nv), np.concatenate([np.ones(n), np.full(m, np.inf)])),
               options={'time_limit': time_limit, 'mip_rel_gap': mip_rel_gap,
                        'presolve': True})
    if res.x is None:
        return None, None, res
    x = res.x[:n]
    S = [nodes[i] for i in range(n) if x[i] > 0.5]
    # exact recount by brute force on the returned set
    val = sum(1 for u, v in G.edges() if u in set(S) and v in set(S))
    return val, S, res


# ------------------------------------------- inner minimisation for blow-ups
def _obj(A, y):
    return 0.5 * float(y @ (A @ y))


def _project_boxsimplex(y, w, total):
    """Euclidean projection onto {0 <= y <= w, sum y = total} (bisection on the
    Lagrange multiplier)."""
    lo, hi = -1e3, 1e3
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        s = np.clip(y - mid, 0.0, w).sum()
        if s > total:
            lo = mid
        else:
            hi = mid
    return np.clip(y - 0.5 * (lo + hi), 0.0, w)


def inner_min(A, w, total=0.5, starts=40, iters=400, rng=None):
    """Heuristic minimiser of 0.5 y'Ay over {0<=y<=w, sum y = total}.
    Returns (value, y).  Value is always an UPPER bound on the true minimum."""
    if rng is None:
        rng = np.random.default_rng(0)
    m = len(w)
    if w.sum() < total - 1e-12:
        return np.inf, None
    best = np.inf
    besty = None
    cands = []
    # deterministic starts: greedy independent-set-first fills
    order_sets = [np.argsort(-w), np.argsort(w)]
    for _ in range(6):
        order_sets.append(rng.permutation(m))
    for order in order_sets:
        y = np.zeros(m)
        rem = total
        for i in order:
            t = min(w[i], rem)
            y[i] = t
            rem -= t
            if rem <= 1e-15:
                break
        cands.append(y)
    for _ in range(starts):
        y = rng.random(m) * w
        if y.sum() <= 0:
            continue
        y = _project_boxsimplex(y, w, total)
        cands.append(y)
    for y0 in cands:
        y = y0.copy()
        for it in range(iters):
            g = A @ y
            # Frank-Wolfe: minimise linear g'z over the box-simplex -> fill
            # smallest gradient coords first
            z = np.zeros(m)
            rem = total
            for i in np.argsort(g):
                t = min(w[i], rem)
                z[i] = t
                rem -= t
                if rem <= 1e-15:
                    break
            d = z - y
            # exact line search on t in [0,1]:  q(t)=obj(y+td)
            a = _obj(A, d)
            b = float(y @ (A @ d))
            if a > 1e-14:
                t = -b / (2 * a)
                t = min(1.0, max(0.0, t))
                cand_ts = [t, 1.0, 0.0]
            else:
                cand_ts = [1.0, 0.0]
            vals = [(_obj(A, y + tt * d), tt) for tt in cand_ts]
            v, tt = min(vals)
            if tt <= 1e-12 or v >= _obj(A, y) - 1e-15:
                break
            y = y + tt * d
        v = _obj(A, y)
        if v < best:
            best = v
            besty = y
    return best, besty


def inner_min_exact_support(A, w, total=0.5, max_m=9):
    """Exact minimiser by enumerating the KKT face structure: each coordinate is
    at 0, at w_i, or interior.  Interior coords satisfy (Ay)_i = lambda.
    Only for very small m (3^m faces)."""
    m = len(w)
    assert m <= max_m
    best = np.inf
    besty = None
    for pat in itertools.product((0, 1, 2), repeat=m):
        Z = [i for i in range(m) if pat[i] == 0]
        U = [i for i in range(m) if pat[i] == 1]
        I = [i for i in range(m) if pat[i] == 2]
        fixed = sum(w[i] for i in U)
        rem = total - fixed
        if rem < -1e-12:
            continue
        if not I:
            if abs(rem) > 1e-12:
                continue
            y = np.zeros(m)
            y[U] = w[U]
            cand = [y]
        else:
            # solve  A_II yI + lam*1 = -A_IU w_U ... KKT: (Ay)_i = lam for i in I
            k = len(I)
            M = np.zeros((k + 1, k + 1))
            M[:k, :k] = A[np.ix_(I, I)]
            M[:k, k] = -1.0
            M[k, :k] = 1.0
            rhs = np.zeros(k + 1)
            base = np.zeros(m)
            base[U] = w[U]
            rhs[:k] = -(A[np.ix_(I, range(m))] @ base)
            rhs[k] = rem
            try:
                sol = np.linalg.solve(M, rhs)
            except np.linalg.LinAlgError:
                continue
            yI = sol[:k]
            if np.any(yI < -1e-9) or np.any(yI > w[I] + 1e-9):
                continue
            y = base.copy()
            y[I] = np.clip(yI, 0.0, w[I])
            cand = [y]
        for y in cand:
            if abs(y.sum() - total) > 1e-8:
                continue
            if np.any(y < -1e-9) or np.any(y > w + 1e-9):
                continue
            v = _obj(A, y)
            if v < best:
                best = v
                besty = y
    return best, besty


# -------------------------------------------- outer maximisation over weights
def outer_max(A, restarts=30, iters=250, rng=None, inner_starts=25, verbose=False):
    """Maximise f(H,w) = inner_min over w on the simplex.  Heuristic
    (multiplicative-weights style hill climbing + random restarts)."""
    if rng is None:
        rng = np.random.default_rng(1)
    m = A.shape[0]
    best = -np.inf
    bestw = None
    starts = [np.ones(m) / m]
    for _ in range(restarts):
        starts.append(rng.dirichlet(np.ones(m)))
    for w0 in starts:
        w = w0.copy()
        v, y = inner_min(A, w, starts=inner_starts, rng=rng)
        step = 0.08
        for it in range(iters):
            improved = False
            for _ in range(6):
                pert = rng.normal(size=m) * step
                w2 = w * np.exp(pert)
                w2 = w2 / w2.sum()
                v2, y2 = inner_min(A, w2, starts=inner_starts, rng=rng)
                if v2 > v + 1e-12:
                    w, v, y = w2, v2, y2
                    improved = True
                    break
            if not improved:
                step *= 0.6
                if step < 1e-4:
                    break
        if v > best:
            best, bestw = v, w
    return best, bestw
