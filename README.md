# Erdős #128 (sparse halves): independent re-verification of the exact β of the known triangle-free strongly regular graphs

**Prior work (please cite this instead):** these values (Clebsch 1/64, Hoffman–Singleton 1/100, Gewirtz 3/224, Higman–Sims 7/400, with explicit bisection certificates) were recorded on the erdosproblems.com #128 discussion thread by Cormundus (comment of 26 Jul 2026, with code at https://github.com/cormundus/erdos-128-census). This repository only re-derives them independently (different construction of the graphs, Hoffman-bound certificates) and adds nothing new.

Remark: in A. Razborov, *More about sparse halves in triangle-free graphs* (arXiv:2104.09406, §4.4), an ad hoc half of the
Higman–Sims graph with 199 edges is presented, "remarkably suggesting" that β(HS) = 1/50 − 10⁻⁴ might be tight. It is not:
the sparsest half of the Higman–Sims graph spans exactly **175** edges, so **β(HS) = 175/10⁴ = 7/400 = 0.0175**.

* Lower bound: for a d-regular graph on n vertices with least adjacency eigenvalue λ_min, every S with |S| = n/2 spans
  e(S) ≥ (n/8)(d + λ_min) (Hoffman ratio bound); for HS this is (100/8)(22 − 8) = 175.
* Upper bound: the vertex set of the Higman–Sims graph splits into two copies of the Hoffman–Singleton graph (50 vertices,
  7-regular, 175 edges); either copy is such a half.

The same argument gives β(Clebsch) = 1/64 (a perfect matching), β(Hoffman–Singleton) = 1/100 (five disjoint pentagons) and
β(Gewirtz) = 3/224 (the sparsest half induces the Coxeter graph). None of these graphs is a counterexample to the
conjecture (all values are below 1/50); the conjecture itself remains open.

`verify_all.py` (Python 3 with numpy and networkx) builds the graphs from the extended binary Golay code / S(3,6,22), checks the strongly-regular
parameters, exhibits the halves and the ±1 Hoffman eigenvectors, and prints `ALL CHECKS PASSED`:

    python3 verify_all.py

Produced during an AI-assisted search (Claude, 2026-09-04) before the maintainer noticed the earlier thread comment; kept only as an independent check. No claim about the conjecture is made.
