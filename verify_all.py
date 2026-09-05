"""One-shot verification of every headline numerical claim in report.md.
Prints PASS/FAIL for each.  Run time ~1 minute."""
import sys, itertools, math
import numpy as np, networkx as nx
sys.path.insert(0, '/Users/roychen/workspace/claudecode/automath/engine/out/claude_blitz_0905/O6_128/scripts')
from golay import higman_sims, m22_graph, gewirtz, steiner_3_6_22
from families import hoffman_singleton, clebsch
from lib128 import min_half_bruteforce

OUT = '/Users/roychen/workspace/claudecode/automath/engine/out/claude_blitz_0905/O6_128/out/'
ok = True
def chk(name, cond, detail=''):
    global ok
    ok &= bool(cond)
    print(f"[{'PASS' if cond else 'FAIL'}] {name}   {detail}")

# --- construction of the Steiner system and the three Golay graphs
hexads, dist = steiner_3_6_22()
cnt = {}
for B in hexads:
    for t in itertools.combinations(sorted(B), 3):
        cnt[t] = cnt.get(t, 0) + 1
chk("extended Golay weight distribution", dist == {0:1, 8:759, 12:2576, 16:759, 24:1}, str(dist))
chk("S(3,6,22): 77 hexads, every triple exactly once",
    len(hexads) == 77 and len(cnt) == math.comb(22,3) and set(cnt.values()) == {1})

graphs = {'Clebsch': clebsch(), 'HoffmanSingleton': hoffman_singleton(),
          'Gewirtz': gewirtz(), 'M22': m22_graph(), 'HigmanSims': higman_sims(),
          'Petersen': nx.petersen_graph(), 'C5': nx.cycle_graph(5)}
params = {'Clebsch': (16,5,0,2), 'HoffmanSingleton': (50,7,0,1), 'Gewirtz': (56,10,0,2),
          'M22': (77,16,0,4), 'HigmanSims': (100,22,0,6), 'Petersen': (10,3,0,1), 'C5': (5,2,0,1)}
for nm, G in graphs.items():
    n = G.number_of_nodes(); A = nx.to_numpy_array(G); d = int(A.sum(1)[0])
    tri = int(round(np.trace(A @ A @ A))) // 6
    lam, mu = set(), set()
    for u, v in itertools.combinations(range(n), 2):
        c = len(set(G[u]) & set(G[v]))
        (lam if G.has_edge(u, v) else mu).add(c)
    got = (n, d, (list(lam) or [0])[0], list(mu)[0])
    chk(f"{nm} is srg{params[nm]}, triangle-free",
        tri == 0 and got == params[nm] and A.sum(1).min() == A.sum(1).max(), f"got srg{got}")

# --- Prop 1 + Prop 3: exact beta for the four graphs
expected = {'Clebsch': (4, 1/64), 'HoffmanSingleton': (25, 1/100),
            'Gewirtz': (42, 3/224), 'HigmanSims': (175, 7/400)}
for nm, (emin, beta) in expected.items():
    G = graphs[nm]; n = G.number_of_nodes(); A = nx.to_numpy_array(G)
    d = int(A.sum(1)[0]); lm = np.linalg.eigvalsh(A).min()
    S = set(int(x) for x in np.load(OUT + nm + '_half.npy'))
    e = sum(1 for u, v in G.edges() if u in S and v in S)
    lb = n * (d + lm) / 8
    z = -np.ones(n); z[sorted(S)] = 1.0
    eig = np.abs(A @ z - lm * z).max() < 1e-8 and abs(z.sum()) < 1e-9
    chk(f"beta({nm}) = {emin}/{n}^2 = {beta:.6f} exactly",
        len(S) == n // 2 and e == emin and abs(lb - emin) < 1e-9 and eig
        and abs(e / n**2 - beta) < 1e-12,
        f"|S|={len(S)}, e(S)={e}, Hoffman LB={lb:.4f}, pm1-eigenvector={eig}")

# --- the induced structures
HoSi = hoffman_singleton()
S = sorted(int(x) for x in np.load(OUT + 'HigmanSims_half.npy'))
H = graphs['HigmanSims'].subgraph(S)
chk("Higman-Sims sparsest half induces the Hoffman-Singleton graph", nx.is_isomorphic(H, HoSi))
S = sorted(int(x) for x in np.load(OUT + 'Gewirtz_half.npy'))
H = nx.convert_node_labels_to_integers(graphs['Gewirtz'].subgraph(S).copy())
degs = set(dict(H.degree()).values())
chk("Gewirtz sparsest half induces the Coxeter graph (28, cubic, girth 7, diam 4, |Aut|=336)",
    H.number_of_nodes() == 28 and degs == {3} and nx.girth(H) == 7 and nx.diameter(H) == 4
    and sum(1 for _ in nx.algorithms.isomorphism.GraphMatcher(H, H).isomorphisms_iter()) == 336)
S = sorted(int(x) for x in np.load(OUT + 'HoffmanSingleton_half.npy'))
H = graphs['HoffmanSingleton'].subgraph(S)
comps = sorted(len(c) for c in nx.connected_components(H))
chk("Hoffman-Singleton sparsest half = 5 disjoint pentagons", comps == [5]*5 and
    set(dict(H.degree()).values()) == {2})

# --- Prop 4 closed forms
chk("max_x x(1-2x)/(1-x) = 3-2sqrt2 at x=1-1/sqrt2",
    abs(max((x*(1-2*x)/(1-x)) for x in np.linspace(1e-6, .4999, 2000001)) - (3-2*np.sqrt(2))) < 1e-9,
    f"3-2sqrt2 = {3-2*np.sqrt(2):.7f}")
r1, r2 = (29-np.sqrt(41))/100, (29+np.sqrt(41))/100
chk("window roots of 50x^2-29x+4", abs(50*r1*r1-29*r1+4) < 1e-9 and abs(50*r2*r2-29*r2+4) < 1e-9,
    f"({r1:.7f}, {r2:.7f});  (3-2sqrt2)/8 = {(3-2*np.sqrt(2))/8:.7f} > 0.02")
bad = []
for nm, G in graphs.items():
    n = G.number_of_nodes(); A = nx.to_numpy_array(G); d = int(A.sum(1)[0])
    lm = np.linalg.eigvalsh(A).min()
    if lm > -d*d/(n-d) + 1e-9: bad.append(nm)
    if (d+lm)/n > 0.16: bad.append(nm+'(TRIGGERS!)')
chk("Prop 4 (lam_min <= -d^2/(n-d)) holds and nothing triggers (d+lam)/n > 4/25",
    not bad, f"max (d+lam)/n = {max((int(nx.to_numpy_array(G).sum(1)[0])+np.linalg.eigvalsh(nx.to_numpy_array(G)).min())/G.number_of_nodes() for G in graphs.values()):.4f}")

# --- Petersen / C5 extremality
chk("Petersen: min 5-set spans exactly 2 = 10^2/50", min_half_bruteforce(nx.petersen_graph())[0] == 2)

print()
print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
