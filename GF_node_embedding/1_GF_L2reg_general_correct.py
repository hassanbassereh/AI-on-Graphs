

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




def get_reconstructed_adj(X):
    """Compute the adjacency matrix from the learned embedding

    Returns:
            A numpy array of size #nodes * #nodes containing the reconstructed adjacency matrix.
    """

    node_num = X.shape[0]


    adj_mtx_r = np.zeros((node_num, node_num))
    for v_i in range(node_num):
        for v_j in range(node_num):
            if v_i == v_j:
                continue
            adj_mtx_r[v_i, v_j] = get_edge_weight(X,v_i, v_j)
    return adj_mtx_r


def get_edge_weight(X, i, j):
    # return np.exp(-np.power(np.linalg.norm(X[i, :] - X[j, :]), 2))
    return np.dot(X[i], X[j]) / (np.linalg.norm(X[i]) * np.linalg.norm(X[j]) + 1e-8)


def get_edge_list_from_adj_mtrx(adj, threshold=0.0, is_undirected=True, edge_pairs=None):
    result = []
    node_num = adj.shape[0]
    if edge_pairs:
        for (st, ed) in edge_pairs:
            if adj[st, ed] >= threshold:
                result.append((st, ed, adj[st, ed]))
    else:
        for i in range(node_num):
            for j in range(node_num):
                if j == i:
                    continue
                if is_undirected and i >= j:
                    continue
                if adj[i, j] > threshold:
                    result.append((i, j, adj[i, j]))
    return result



def computePrecisionCurve_onlyfor_undirected_graph(predicted_edge_list, true_graph, max_k=-1):
    """
    Compute precision scores for undirected graph link prediction.
    
    Args:
        predicted_edge_list: List of predicted edges as tuples (node1, node2, score)
        true_graph: NetworkX undirected graph containing true edges
        max_k: Maximum number of predictions to consider (-1 for all predictions)
    
    Returns:
        precision_scores: List of precision@k values
        delta_factors: List indicating whether each prediction is correct (1) or not (0)
    """
    if max_k == -1:
        max_k = len(predicted_edge_list)
    else:
        max_k = min(max_k, len(predicted_edge_list))
    
    # Sort predicted edges by score in descending order
    sorted_edges = sorted(predicted_edge_list, key=lambda x: x[2], reverse=True)
    
    precision_scores = []
    delta_factors = []
    correct_edge = 0
    
    for i in range(max_k):
        node1, node2 = sorted_edges[i][0], sorted_edges[i][1]
        
        # Check if edge exists in undirected graph
        if true_graph.has_edge(node1, node2):
            correct_edge += 1
            delta_factors.append(1.0)
        else:
            delta_factors.append(0.0)
        
        # Precision@(i+1): correct predictions / total predictions
        precision_scores.append(1.0 * correct_edge / (i + 1))
    
    return precision_scores, delta_factors




def computeMAP_onlyfor_undirected_graphs(predicted_edge_list, true_graph, max_k=-1):
    """
    Compute Mean Average Precision (MAP) for undirected graph link prediction.
    """
    node_list = list(true_graph.nodes())  # Get actual nodes in graph
    node_num = len(node_list)
    
    # Create mapping from node ID to index
    node_to_idx = {node: idx for idx, node in enumerate(node_list)}
    
    # Initialize edge lists for each node
    node_edges = [[] for _ in range(node_num)]
    
    # Use a dictionary to track edges by their sorted node pair
    edge_dict = {}
    for (st, ed, w) in predicted_edge_list:
        edge_key = tuple(sorted([st, ed]))
        if edge_key not in edge_dict or w > edge_dict[edge_key]:
            edge_dict[edge_key] = w
    
    # Add each edge to both nodes' lists (only once per node pair)
    for (node1, node2), w in edge_dict.items():
        if node1 in node_to_idx and node2 in node_to_idx:
            idx1, idx2 = node_to_idx[node1], node_to_idx[node2]
            node_edges[idx1].append((node1, node2, w))
            node_edges[idx2].append((node2, node1, w))
    
    node_ap = [0.0] * node_num
    count = 0
    
    for idx, node in enumerate(node_list):
        if true_graph.degree(node) == 0:
            continue
        
        count += 1
        # Sort predictions for this node by score (descending)
        node_edges[idx].sort(key=lambda x: x[2], reverse=True)
        
        precision_scores, delta_factors = computePrecisionCurve_onlyfor_undirected_graph(
            node_edges[idx], true_graph, max_k)
        
        if sum(delta_factors) == 0:
            node_ap[idx] = 0
        else:
            precision_rectified = [p * d for p, d in zip(precision_scores, delta_factors)]
            node_ap[idx] = float(sum(precision_rectified) / sum(delta_factors))
    
    return sum(node_ap) / count if count > 0 else 0

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
for d in range(2,20):

    X,error =embedding_GF_l2reg(G,d)
    rec_adj = get_reconstructed_adj(X)
    predicted_edge_list = get_edge_list_from_adj_mtrx(rec_adj)
    map_score = computeMAP_onlyfor_undirected_graphs(predicted_edge_list,G)
    maps.append(map_score)
    Error.append(error)
    print("d=",d,"map_score=",round(map_score,2),"error",error)


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