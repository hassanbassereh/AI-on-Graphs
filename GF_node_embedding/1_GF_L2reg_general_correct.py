

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import scipy.sparse.linalg as lg
    


def embedding_GF_l2reg(graph, d, max_iter=50000, eta=0.001, regu=0.001):
    """
    Graph Factorization with L2 regularization using matrix operations.
    
    This version is closer to the original TensorFlow implementation.
    """
    
    # Convert to adjacency matrix
    if isinstance(graph, nx.Graph):
        n = graph.number_of_nodes()
        A = nx.to_numpy_array(graph)
    else:
        A = np.array(graph)
        n = A.shape[0]

    # Initialize embeddings
    X = 0.01 * np.random.randn(n, d)
    
    for iteration in range(max_iter):
        # Compute reconstruction
        reconstruction = X @ X.T
        
        # Compute gradient
        grad = -2 * ((A - reconstruction)) @ X + 2 * regu * X
        
        # Update embeddings
        X -= eta * grad

    p_d_p_t = np.dot(X, X.T)
    eig_err = np.linalg.norm(p_d_p_t - A)
    
    print(f'\nFinal SVD error: {eig_err:.4f}')  
    return X,eig_err




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
        ranked = [b for b in np.argsort(-S[a], kind="stable") if b != a]   # deterministic order for ties (d=1: cosine is only +1/-1)
        for r, b in enumerate(ranked, 1):
            if A[a, b]:
                hits += 1
                s += hits / r
        aps.append(s / hits)
    return float(np.mean(aps))



def my_graph():
    # Initialize an empty graph
    G = nx.Graph()

    # Create complete graph with 7 nodes (labeled 1 to 7)
    nodes_7 = range(1, 8)
    G.add_nodes_from(nodes_7)
    for i in nodes_7:
        for j in nodes_7:
            if i < j:
                G.add_edge(i, j)

    G.remove_edge(7,6)
    G.remove_edge(5,6)
    G.remove_edge(3,5)
    G.remove_edge(3,4)

    # Create complete graph with 8 nodes (labeled 8 to 15)
    nodes_8 = range(8, 16)
    G.add_nodes_from(nodes_8)
    for i in nodes_8:
        for j in nodes_8:
            if i < j:
                G.add_edge(i, j)

    G.remove_edge(8,9)
    G.remove_edge(8,10)
    G.remove_edge(8,15)
    G.remove_edge(9,14)
    G.remove_edge(9,15)
    G.remove_edge(9,11)

    # Create complete graph with 9 nodes (labeled 17 to 25)
    nodes_9 = range(17, 26)
    G.add_nodes_from(nodes_9)
    for i in nodes_9:
        for j in nodes_9:
            if i < j:
                G.add_edge(i, j)


    G.remove_edge(18,19)
    G.remove_edge(18,20)
    G.remove_edge(18,25)
    G.remove_edge(19,24)
    G.remove_edge(19,25)
    G.remove_edge(19,21)
    G.remove_edge(21,24)
    G.remove_edge(21,25)
    G.remove_edge(22,24)


    # Add 8 solitary nodes (labeled 27 to 34)
    solitary_nodes = range(27, 35)
    G.add_nodes_from(solitary_nodes)


    G.add_edge(1,27)
    G.add_edge(27,28)
    G.add_edge(28,29)
    G.add_edge(29,8)
    G.add_edge(30,34)
    G.add_edge(34,17)
    G.add_edge(31,32)
    G.add_edge(31,19)
    G.add_edge(30,10)
    G.add_edge(22,33)
    G.add_edge(7,33)
    return G 



from sklearn.manifold import TSNE

def plot_tsne_embedding(G,X, title="t-SNE Visualization", 
                        perplexity=5, n_iter=1000, random_state=42,
                        figsize=(18, 8)):
    """
    Plot t-SNE visualization of embeddings
    
    Parameters:
    -----------
    X : numpy.ndarray
        Embedding matrix of shape (num_nodes, d)
    node_list : list
        List of node IDs in the same order as X rows
    title : str
        Title for the plot
    perplexity : int, default=5
        t-SNE perplexity parameter (should be less than n_samples)
    n_iter : int, default=1000
        Number of iterations for t-SNE optimization
    random_state : int, default=42
        Random seed for reproducibility
    figsize : tuple, default=(10, 8)
        Figure size
    
    Returns:
    --------
    X_tsne : numpy.ndarray
        2D t-SNE embeddings
    """
    if isinstance(G, (nx.Graph, nx.DiGraph)):
    # Convert NetworkX graph to adjacency matrix
        node_list = sorted(G.nodes())
    else:
    # Assume user passed an adjacency matrix directly
        node_list = range(1,G.shape[0]+1)

    # 
    
    print(f"Running t-SNE on {X.shape[0]} nodes with {X.shape[1]} dimensions...")
    
    # Adjust perplexity if needed (must be less than n_samples)
    max_perplexity = X.shape[0] - 1
    if perplexity >= max_perplexity:
        perplexity = max(5, max_perplexity // 2)
        print(f"Adjusted perplexity to {perplexity} (must be < {max_perplexity})")
    
    # Run t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, 
                random_state=random_state, verbose=0)
    X_tsne = tsne.fit_transform(X)
    
    print(f"t-SNE completed. Final KL divergence: {tsne.kl_divergence_:.4f}")
    
    # Define colors based on node communities
    colors = []
    for node_id in node_list:
        if 1 <= node_id <= 7:
            colors.append('red')
        elif 8 <= node_id <= 15:
            colors.append('blue')
        elif 17 <= node_id <= 25:
            colors.append('green')
        elif 27 <= node_id <= 29:
            colors.append('purple')        
        elif 31 <= node_id <= 32:
            colors.append('lightgreen')
        elif node_id == 33:
            colors.append('yellow')
        else:
            colors.append('gray')  # bridge nodes
    
    # Create plot
    plt.figure(figsize=figsize)
    plt.subplot(1,2,1)
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_color=colors, node_size=500, font_size=10)
    plt.title('Original Graph')

    plt.subplot(1,2,2)
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=colors, s=200, alpha=0.6, edgecolors='black', linewidth=1.5)
    
    # Add node labels
    for i, (x, y) in enumerate(X_tsne):
        plt.annotate(str(node_list[i]), (x, y), fontsize=9, ha='center', 
                    va='center', fontweight='bold')
    
    # plt.title(f'{title}\n'
    #           f'Red: Comm1 (1-7), Blue: Comm2 (8-15), Green: Comm3 (17-25), Gray: Bridges\n'
    #           f'Explained variance: PC1={explained_var[0]*100:.1f}%, PC2={explained_var[1]*100:.1f}%')
    plt.title(f'{title}')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("1_GF_tsne.png")
    plt.show()
    
    return X_tsne




def plot_tsne_embedding_dequal2(G,X  ,figsize=(18, 8)):


    if isinstance(G, (nx.Graph, nx.DiGraph)):
    # Convert NetworkX graph to adjacency matrix
        node_list = sorted(G.nodes())
    else:
    # Assume user passed an adjacency matrix directly
        node_list = range(1,G.shape[0]+1)

    # 
        
    # Define colors based on node communities
    colors = []
    for node_id in node_list:
        if 1 <= node_id <= 7:
            colors.append('red')
        elif 8 <= node_id <= 15:
            colors.append('blue')
        elif 17 <= node_id <= 25:
            colors.append('green')
        elif 27 <= node_id <= 29:
            colors.append('purple')        
        elif 31 <= node_id <= 32:
            colors.append('lightgreen')
        elif node_id == 33:
            colors.append('yellow')
        else:
            colors.append('gray')  # bridge nodes
    
    # Create plot
    plt.figure(figsize=figsize)
    plt.subplot(1,2,1)
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_color=colors, node_size=500, font_size=10)
    plt.title('Original Graph')

    plt.subplot(1,2,2)
    plt.scatter(X[:, 0], X[:, 1], c=colors, s=200, alpha=0.6, edgecolors='black', linewidth=1.5)
    
    # Add node labels
    for i, (x, y) in enumerate(X):
        plt.annotate(str(node_list[i]), (x, y), fontsize=9, ha='center', 
                    va='center', fontweight='bold')
    
    # plt.title(f'{title}\n'
    #           f'Red: Comm1 (1-7), Blue: Comm2 (8-15), Green: Comm3 (17-25), Gray: Bridges\n'
    #           f'Explained variance: PC1={explained_var[0]*100:.1f}%, PC2={explained_var[1]*100:.1f}%')
    # plt.title(f'{title}')
    plt.xlabel(' Dimension 1')
    plt.ylabel(' Dimension 2')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("1_GF_dequal2.png")
    plt.show()
    
    return X



# Usage example:
G = my_graph()
# G = np.loadtxt("G_adj_mat.txt")
# G = np.loadtxt("1_B_knn_cpp.txt")

# G = nx.from_numpy_array(adj_matrix)

# Get SDNE embeddings
# X = sdne_embed(G, d=5, n_iter=500, alpha=0.09, beta=2, n_units=[32, 16], xeta=0.001)
# X =embedding_GF_l2reg(G,beta=0.5)

# X = embedding_GF_l2reg(G)


X,error =embedding_GF_l2reg(G,d=2)

# print(X)
# plt.scatter(X[:,0],X[:,1])
# plt.show()


plot_tsne_embedding_dequal2(G,X)


maps=[]
Error=[]
# for d in range(2,20):

#     X,error =embedding_GF_l2reg(G,d)
#     rec_adj = get_reconstructed_adj(X)
#     predicted_edge_list = get_edge_list_from_adj_mtrx(rec_adj)
#     map_score = computeMAP_onlyfor_undirected_graphs(predicted_edge_list,G)
#     maps.append(map_score)
#     Error.append(error)
#     print("d=",d,"map_score=",round(map_score,2),"error",error)


maps = [MAP(G, embedding_LAP(G, d)[0]) for d in range(1,20)]
Error = [MAP(G, embedding_LAP(G, d)[1]) for d in range(1,20)]


plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(range(2,20),maps,'.-')
plt.xlabel('Embedding dimention size (d)',fontsize=15)
plt.ylabel('MAP score',fontsize=15)
plt.xticks(range(2,20,2),[str(z) for z in range(2,20,2)])
plt.subplot(1,2,2)
plt.plot(range(2,20),Error,'.-')
plt.xlabel('Embedding dimention size (d)',fontsize=15)
plt.ylabel('Error',fontsize=15)
plt.xticks(range(2,20),[str(z) for z in range(2,20)])
plt.savefig('1_GF_MAP.png')
plt.show()

# X = embedding_GF_l2reg(G)
# print(f"Original embedding shape: {X.shape}")

# Visualize with t-SNE
X_tsne = plot_tsne_embedding(G,embedding_GF_l2reg(G,4)[0], 
                             title="t-SNE of Laplacian Embeddings (4 → 2D)",
                             perplexity=50)

print(f"t-SNE embedding shape: {X_tsne.shape}")