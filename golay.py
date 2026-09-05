"""Extended binary Golay code -> S(3,6,22) -> Higman-Sims, M22, Gewirtz graphs."""
import itertools
import numpy as np, networkx as nx

def golay24():
    # [23,12,7] cyclic Golay, generator g(x)=x^11+x^10+x^6+x^5+x^4+x^2+1
    g = [1,0,1,0,1,1,1,0,0,0,1,1]     # coefficients c0..c11 of x^0..x^11
    gpoly = np.array(g, dtype=np.int8)
    words = []
    for msg in range(1 << 12):
        m = np.array([(msg >> i) & 1 for i in range(12)], dtype=np.int8)
        c = np.zeros(23, dtype=np.int8)
        for i in range(12):
            if m[i]:
                c[i:i+12] ^= gpoly
        c24 = np.concatenate([c, [c.sum() % 2]])
        words.append(c24)
    return np.array(words, dtype=np.int8)

def steiner_3_6_22():
    W = golay24()
    wt = W.sum(axis=1)
    dist = {int(w): int((wt == w).sum()) for w in np.unique(wt)}
    octads = W[wt == 8]
    # fix two coordinates 22,23; octads containing both
    sel = octads[(octads[:, 22] == 1) & (octads[:, 23] == 1)]
    hexads = [tuple(np.nonzero(r[:22])[0].tolist()) for r in sel]
    return hexads, dist

def higman_sims():
    hexads, _ = steiner_3_6_22()
    assert len(hexads) == 77, len(hexads)
    G = nx.Graph()
    G.add_node(('inf',))
    for p in range(22):
        G.add_node(('p', p)); G.add_edge(('inf',), ('p', p))
    for i, B in enumerate(hexads):
        G.add_node(('B', i))
        for p in B:
            G.add_edge(('p', p), ('B', i))
    for i in range(77):
        for j in range(i+1, 77):
            if not (set(hexads[i]) & set(hexads[j])):
                G.add_edge(('B', i), ('B', j))
    return nx.convert_node_labels_to_integers(G)

def m22_graph():
    """77 hexads, adjacent iff disjoint: 16-regular triangle-free on 77 vertices."""
    hexads, _ = steiner_3_6_22()
    G = nx.Graph(); G.add_nodes_from(range(77))
    for i in range(77):
        for j in range(i+1, 77):
            if not (set(hexads[i]) & set(hexads[j])):
                G.add_edge(i, j)
    return G

def gewirtz():
    """56 hexads avoiding a fixed point, adjacent iff disjoint: 10-regular, 56 vertices."""
    hexads, _ = steiner_3_6_22()
    H = [B for B in hexads if 0 not in B]
    assert len(H) == 56, len(H)
    G = nx.Graph(); G.add_nodes_from(range(56))
    for i in range(56):
        for j in range(i+1, 56):
            if not (set(H[i]) & set(H[j])):
                G.add_edge(i, j)
    return G

if __name__ == '__main__':
    hexads, dist = steiner_3_6_22()
    print("Golay weight distribution:", dist)
    print("hexads:", len(hexads), "sizes:", set(len(b) for b in hexads))
    # verify Steiner property: every 3-subset of 22 points in exactly one hexad
    cnt = {}
    for B in hexads:
        for t in itertools.combinations(sorted(B), 3):
            cnt[t] = cnt.get(t, 0) + 1
    import math
    print("S(3,6,22) check: #triples covered", len(cnt), "of", math.comb(22,3),
          " multiplicities:", set(cnt.values()))
    for nm, f in [('HigmanSims', higman_sims), ('M22', m22_graph), ('Gewirtz', gewirtz)]:
        G = f()
        A = nx.to_numpy_array(G).astype(np.int64)
        degs = sorted(set(dict(G.degree()).values()))
        print(f"{nm}: n={G.number_of_nodes()} e={G.number_of_edges()} degs={degs} "
              f"triangles={int(np.trace(A@A@A))//6} "
              f"lam_min={np.linalg.eigvalsh(A.astype(float)).min():.4f}")
