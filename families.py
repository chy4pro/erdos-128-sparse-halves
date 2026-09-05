"""Library of structured triangle-free graphs."""
import itertools, math
import numpy as np, networkx as nx

def circulant(n, S):
    G = nx.Graph(); G.add_nodes_from(range(n))
    for i in range(n):
        for s in S:
            G.add_edge(i, (i+s) % n)
    return G

def cayley_Zm(m, S):
    return circulant(m, S)

def kneser(n, k):
    V = list(itertools.combinations(range(n), k))
    G = nx.Graph(); G.add_nodes_from(range(len(V)))
    for i in range(len(V)):
        for j in range(i+1, len(V)):
            if not (set(V[i]) & set(V[j])):
                G.add_edge(i, j)
    return G

def schrijver(n, k):
    """Schrijver graph SG(n,k): stable k-subsets of Z_n (no two cyclically adjacent)."""
    V = [c for c in itertools.combinations(range(n), k)
         if all((c[(i+1) % k] - c[i]) % n != 1 for i in range(k)) ]
    V = [c for c in V if not any(((b - a) % n) == 1 or ((a - b) % n) == 1 for a in c for b in c)]
    G = nx.Graph(); G.add_nodes_from(range(len(V)))
    for i in range(len(V)):
        for j in range(i+1, len(V)):
            if not (set(V[i]) & set(V[j])):
                G.add_edge(i, j)
    return G

def andrasfai(k):
    """And(k): circulant on 3k-1 vertices with connection set {1,4,7,...,3k-2} (+/-)."""
    n = 3*k - 1
    S = [1 + 3*i for i in range(k)]
    return circulant(n, S)

def mycielski(G):
    return nx.mycielskian(G)

def higman_sims():
    try:
        return nx.LCF_graph(0,0,0)
    except Exception:
        return None

def hoffman_singleton():
    """Robertson construction: 5 pentagons P_h, 5 pentagrams Q_i, join
    vertex j of P_h to vertex (h*i + j) mod 5 of Q_i."""
    G = nx.Graph()
    for h in range(5):
        for j in range(5):
            G.add_node(('P', h, j)); G.add_node(('Q', h, j))
    for h in range(5):
        for j in range(5):
            G.add_edge(('P', h, j), ('P', h, (j+1) % 5))       # pentagon
            G.add_edge(('Q', h, j), ('Q', h, (j+2) % 5))       # pentagram
    for h in range(5):
        for i in range(5):
            for j in range(5):
                G.add_edge(('P', h, j), ('Q', i, (h*i + j) % 5))
    return nx.convert_node_labels_to_integers(G)

def clebsch():
    """Folded 5-cube = Clebsch graph (triangle-free, 16 vertices, 5-regular)."""
    G = nx.Graph(); G.add_nodes_from(range(16))
    for v in range(16):
        for b in range(4):
            G.add_edge(v, v ^ (1 << b))
        G.add_edge(v, v ^ 0b1111)
    return G

def cyclotomic(q, k):
    """Cayley graph on F_q with connection set = k-th powers (needs q prime here)."""
    S = sorted({pow(x, k, q) for x in range(1, q)})
    S = sorted({s for s in S} | {(-s) % q for s in S})
    return circulant(q, [s for s in S if s != 0])

def grotzsch():
    return nx.mycielskian(nx.cycle_graph(5))

def cn_blowup_template(n):
    return nx.cycle_graph(n)

def named_library():
    lib = {}
    lib['C5'] = nx.cycle_graph(5)
    lib['C7'] = nx.cycle_graph(7)
    lib['C9'] = nx.cycle_graph(9)
    lib['C11'] = nx.cycle_graph(11)
    lib['Petersen'] = nx.petersen_graph()
    lib['Clebsch16'] = clebsch()
    lib['Grotzsch11'] = grotzsch()
    lib['Mycielski(C7)'] = nx.mycielskian(nx.cycle_graph(7))
    lib['Mycielski(Petersen)'] = nx.mycielskian(nx.petersen_graph())
    lib['Mycielski2(C5)'] = nx.mycielskian(nx.mycielskian(nx.cycle_graph(5)))
    lib['HoffmanSingleton50'] = hoffman_singleton()
    lib['Kneser(7,3)'] = kneser(7, 3)
    lib['Kneser(8,3)'] = kneser(8, 3)
    lib['Schrijver(7,3)'] = schrijver(7, 3)
    lib['Schrijver(8,3)'] = schrijver(8, 3)
    lib['Schrijver(9,4)'] = schrijver(9, 4)
    lib['Schrijver(10,4)'] = schrijver(10, 4)
    for k in range(2, 8):
        lib[f'Andrasfai({k})'] = andrasfai(k)
    lib['Heawood14'] = nx.heawood_graph()
    lib['Moebius-Kantor16'] = nx.moebius_kantor_graph()
    lib['Pappus18'] = nx.pappus_graph()
    lib['Desargues20'] = nx.desargues_graph()
    lib['Coxeter28'] = nx.LCF_graph(28, [-13,-9,7,-7,9,13], 4) if True else None
    lib['Dodecahedral20'] = nx.dodecahedral_graph()
    lib['Tutte-Coxeter30'] = nx.LCF_graph(30, [-13,-9,7,-7,9,13], 5)
    lib['Kneser(9,4)'] = kneser(9, 4)
    lib['Kneser(10,4)'] = kneser(10, 4)
    lib['Kneser(11,4)'] = kneser(11, 4)
    lib['C5xC5(cat)'] = nx.tensor_product(nx.cycle_graph(5), nx.cycle_graph(5))
    lib['C5[C5]?'] = None
    return {k: v for k, v in lib.items() if v is not None}
