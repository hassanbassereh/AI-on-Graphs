"""
Laplacian node embedding with and without D in the constraint.

  without D :  min tr(Y^T L Y)  s.t.  Y^T Y   = I   ->  L f = lambda f
  with D    :  min tr(Y^T L Y)  s.t.  Y^T D Y = I   ->  L f = lambda D f

Graph: three communities (two dense, one sparse) plus short dangling
chains (2 nodes each) on the dense communities.  The chain ends have degree 1.
Without D these low-degree "loose ends" are the cheapest way to fill the size
budget, so they take over the first dimensions; with D the communities appear.
"""
import numpy as np
import networkx as nx
import scipy.sparse as sp
import scipy.sparse.linalg as lg
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score


# ----------------------------------------------------------------------------
# 1. Graph
# ----------------------------------------------------------------------------
def build_graph(seed=10, sizes=(15, 15, 15), p_in=(0.8, 0.8, 0.3), p_out=0.02,
                n_chains=(3, 3, 0), chain_len=2):
    """Stochastic block model + dangling chains. Returns graph and node groups.
    group 0,1,2 = communities ; 3 = chains on community 0 ; 4 = chains on community 1"""
    rng = np.random.default_rng(seed)
    probs = [[p_in[i] if i == j else p_out for j in range(len(sizes))] for i in range(len(sizes))]
    G = nx.stochastic_block_model(list(sizes), probs, seed=seed)
    group = {v: G.nodes[v]["block"] for v in G}
    nxt = G.number_of_nodes()
    for c, k in enumerate(n_chains):
        members = [v for v in group if group[v] == c]
        for _ in range(k):
            prev = int(rng.choice(members))
            for _ in range(chain_len):
                G.add_edge(prev, nxt)
                group[nxt] = 3 + c          # chain node of community c
                prev, nxt = nxt, nxt + 1
    assert nx.is_connected(G), "choose another seed: graph is disconnected"
    nodes = sorted(G)
    return G, nodes, np.array([group[v] for v in nodes])


# ----------------------------------------------------------------------------
# 2. Laplacian embedding
# ----------------------------------------------------------------------------
def embedding_LAP(G, d, use_D=True):
    nodes = sorted(G)
    W = nx.to_scipy_sparse_array(G, nodelist=nodes, dtype=float, format="csr")
    deg = np.asarray(W.sum(axis=1)).ravel()
    D = sp.diags_array(deg).tocsc()
    L = (D - W).tocsc()
    # shift-invert just below 0  ->  the SMALLEST eigenvalues
    if use_D:
        w, v = lg.eigsh(L, k=d + 1, M=D, sigma=-1e-2, which="LM")   # L f = lambda D f
    else:
        w, v = lg.eigsh(L, k=d + 1, sigma=-1e-2, which="LM")        # L f = lambda f
    order = np.argsort(w)
    w, v = w[order], v[:, order]
    if w[1] < 1e-8:
        raise ValueError("graph is disconnected")
    return v[:, 1:], w[1:]          # drop constant eigenvector (rule 3)


# ----------------------------------------------------------------------------
# 3. Evaluation: link-reconstruction MAP (score = cosine similarity)
# ----------------------------------------------------------------------------
def cosine_similarity_matrix(X, eps=1e-12):
    """S_ij = <x_i, x_j> / (||x_i|| ||x_j||)   (rows of X = node embeddings)"""
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + eps)   # unit-length rows
    return Xn @ Xn.T                                            # n x n, values in [-1, 1]


def MAP(G, X):
    """Mean Average Precision of reconstructing the edges of G.
    For every node a: rank all other nodes b by cosine(x_a, x_b), walk down the
    list, and average the precision at each position where (a, b) is a true edge."""
    nodes = sorted(G)                       # rows of X follow this order
    A = nx.to_numpy_array(G, nodelist=nodes)
    S = cosine_similarity_matrix(X)
    aps = []
    for a in range(len(nodes)):
        hits, s = 0, 0.0
        ranked = [b for b in np.argsort(-S[a]) if b != a]
        for r, b in enumerate(ranked, 1):
            if A[a, b]:
                hits += 1
                s += hits / r
        aps.append(s / hits)
    return float(np.mean(aps))


# ----------------------------------------------------------------------------
# 4. Run
# ----------------------------------------------------------------------------
G, nodes, group = build_graph()
deg = np.array([G.degree(v) for v in nodes])
community = np.where(group >= 3, group - 3, group)       # chains belong to their community
palette = {0: "red", 1: "blue", 2: "green", 3: "orange", 4: "cyan"}
colors = [palette[g] for g in group]
labels_txt = "red/blue = dense communities, green = sparse community, orange/cyan = dangling chains"

print(f"{G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
for g, name in [(0, "red"), (1, "blue"), (2, "green"), (3, "orange chain"), (4, "cyan chain")]:
    print(f"  mean degree {name:13s}: {deg[group == g].mean():.1f}")

methods = [(False, r"without D:  $Lf=\lambda f$,  $Y^TY=I$"),
           (True,  r"with D:  $Lf=\lambda Df$,  $Y^TDY=I$")]
ds = list(range(1, 20))

# ---- Figure 1: MAP vs d, dimension 1, 2-D embedding -------------------------
fig1, ax = plt.subplots(2, 3, figsize=(18, 10))
for row, (use_D, title) in enumerate(methods):
    maps = [MAP(G, embedding_LAP(G, d, use_D)[0]) for d in ds]
    X2, w2 = embedding_LAP(G, 2, use_D)
    km = KMeans(3, n_init=10, random_state=0).fit_predict(X2)
    ari = adjusted_rand_score(community, km)
    chains = group >= 3
    budget = deg if use_D else np.ones_like(deg)
    share = (budget[chains] * X2[chains, 0] ** 2).sum() / (budget * X2[:, 0] ** 2).sum()
    print(f"\n{title}\n  eigenvalues {np.round(w2, 4)}"
          f"\n  share of dim-1 size budget used by chain nodes: {share:.0%}"
          f"\n  k-means (k=3) on 2-D embedding vs communities, ARI = {ari:.2f}"
          f"\n  MAP d=1..5: {np.round(maps[:5], 3)}")

    a = ax[row, 0]
    a.plot(ds, maps, ".-")
    a.set_xlabel("Embedding dimension size (d)", fontsize=13)
    a.set_ylabel("MAP score", fontsize=13)
    a.set_xticks(range(1, 20, 2))
    a.set_title(title, fontsize=13)

    a = ax[row, 1]
    a.scatter(range(len(nodes)), X2[:, 0], c=colors, s=35, edgecolors="k")
    a.set_xlabel("Node index", fontsize=13)
    a.set_ylabel("Dimension 1", fontsize=13)
    a.set_title(f"dim 1 — chain nodes use {share:.0%} of the size budget", fontsize=12)

    a = ax[row, 2]
    a.scatter(X2[:, 0], X2[:, 1], c=colors, s=20 + 8 * deg, edgecolors="k")
    a.set_xlabel("Dimension 1", fontsize=13)
    a.set_ylabel("Dimension 2", fontsize=13)
    a.set_title(f"2-D embedding (size = degree), k-means ARI = {ari:.2f}", fontsize=12)

fig1.suptitle(labels_txt, fontsize=12)
fig1.tight_layout()
fig1.savefig("Laplacian_D_vs_noD_MAP.png", dpi=110)

# ---- Figure 2: original graph + t-SNE of 5-D embeddings ----------------------
fig2, ax = plt.subplots(1, 3, figsize=(24, 8))
pos = nx.spring_layout(G, seed=42)
nx.draw(G, pos, ax=ax[0], node_color=colors, node_size=120 + 25 * deg,
        with_labels=True, font_size=7, edge_color="lightgray")
ax[0].set_title("Original graph", fontsize=14)
for k, (use_D, title) in enumerate(methods, 1):
    X5, _ = embedding_LAP(G, 5, use_D)
    Xt = TSNE(n_components=2, perplexity=15, random_state=42).fit_transform(X5)
    ax[k].scatter(Xt[:, 0], Xt[:, 1], c=colors, s=200, alpha=0.7, edgecolors="k")
    for i, (x, y) in enumerate(Xt):
        ax[k].annotate(str(nodes[i]), (x, y), fontsize=7, ha="center", va="center")
    ax[k].set_title(f"t-SNE of 5-D embedding, {title}", fontsize=13)
    ax[k].grid(alpha=0.3)
fig2.suptitle(labels_txt, fontsize=12)
fig2.tight_layout()
fig2.savefig("Laplacian_D_vs_noD_tsne.png", dpi=110)
plt.show()