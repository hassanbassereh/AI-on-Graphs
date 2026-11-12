import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from sklearn.manifold import TSNE


def embedding_GF_l2reg(graph, d, max_iter=10000, eta=0.1, regu=0.1):
    """
    Graph Factorization with L2 regularization using matrix operations.
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
    
    print(f'Final SVD error: {eig_err:.4f}')  
    return X, eig_err




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



def plot_tsne_embedding(G, X_d4, X_d7, title="t-SNE Visualization", 
                        perplexity=30, n_iter=1000, random_state=42,
                        figsize=(20, 6)):
    """
    Plot t-SNE visualization of embeddings with three subplots:
    1. Original graph
    2. t-SNE with d=4 embeddings
    3. t-SNE with d=7 embeddings
    """
    if isinstance(G, (nx.Graph, nx.DiGraph)):
        node_list = sorted(G.nodes())
    else:
        node_list = range(1, G.shape[0] + 1)
    
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
    
    # Create figure with 3 subplots
    fig = plt.figure(figsize=figsize)
    
    # Subplot 1: Original graph
    ax1 = plt.subplot(1, 3, 1)
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_color=colors, node_size=500, 
            font_size=10, ax=ax1)
    ax1.set_title('Original Graph Structure', fontsize=14)
    
    # Run t-SNE for d=4 embeddings
    print(f"Running t-SNE on {X_d4.shape[0]} nodes with {X_d4.shape[1]} dimensions...")
    max_perplexity = X_d4.shape[0] - 1
    if perplexity >= max_perplexity:
        perplexity_adjusted = max(5, max_perplexity // 2)
    else:
        perplexity_adjusted = perplexity
    
    tsne_d4 = TSNE(n_components=2, perplexity=perplexity_adjusted, n_iter=n_iter, 
                   random_state=random_state, verbose=0)
    X_tsne_d4 = tsne_d4.fit_transform(X_d4)
    print(f"t-SNE for d=4 completed. KL divergence: {tsne_d4.kl_divergence_:.4f}")
    
    # Subplot 2: t-SNE for d=4
    ax2 = plt.subplot(1, 3, 2)
    scatter2 = ax2.scatter(X_tsne_d4[:, 0], X_tsne_d4[:, 1], c=colors, 
                          s=200, alpha=0.7, edgecolors='black', linewidth=1.5)
    
    # Add node labels for d=4
    for i, (x, y) in enumerate(X_tsne_d4):
        ax2.annotate(str(node_list[i]), (x, y), fontsize=9, ha='center', 
                    va='center', fontweight='bold')
    
    ax2.set_title(f't-SNE: d=4 Embeddings', fontsize=14)
    ax2.set_xlabel('t-SNE Dimension 1')
    ax2.set_ylabel('t-SNE Dimension 2')
    ax2.grid(True, alpha=0.3)
    
    # Run t-SNE for d=7 embeddings
    print(f"\nRunning t-SNE on {X_d7.shape[0]} nodes with {X_d7.shape[1]} dimensions...")
    max_perplexity = X_d7.shape[0] - 1
    if perplexity >= max_perplexity:
        perplexity_adjusted = max(5, max_perplexity // 2)
    else:
        perplexity_adjusted = perplexity
    
    tsne_d7 = TSNE(n_components=2, perplexity=perplexity_adjusted, n_iter=n_iter, 
                   random_state=random_state, verbose=0)
    X_tsne_d7 = tsne_d7.fit_transform(X_d7)
    print(f"t-SNE for d=7 completed. KL divergence: {tsne_d7.kl_divergence_:.4f}")
    
    # Subplot 3: t-SNE for d=7
    ax3 = plt.subplot(1, 3, 3)
    scatter3 = ax3.scatter(X_tsne_d7[:, 0], X_tsne_d7[:, 1], c=colors, 
                          s=200, alpha=0.7, edgecolors='black', linewidth=1.5)
    
    # Add node labels for d=7
    for i, (x, y) in enumerate(X_tsne_d7):
        ax3.annotate(str(node_list[i]), (x, y), fontsize=9, ha='center', 
                    va='center', fontweight='bold')
    
    ax3.set_title(f't-SNE: d=7 Embeddings', fontsize=14)
    ax3.set_xlabel('t-SNE Dimension 1')
    ax3.set_ylabel('t-SNE Dimension 2')
    ax3.grid(True, alpha=0.3)
    


    
    plt.suptitle(f'{title}', fontsize=16, y=1.05)
    plt.tight_layout()
    plt.savefig("GF_tsne_comparison.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    return X_tsne_d4, X_tsne_d7


def run_multiple_trials(num_runs=6, max_d=20, max_iter=10000, eta=0.1, regu=0.1):
    """
    Run embedding_GF_l2reg multiple times and collect statistics
    """
    G = my_graph()
    
    # Initialize storage for results
    all_maps = []
    all_errors = []
    
    for run in range(num_runs):
        print(f"\n{'='*50}")
        print(f"Run {run+1}/{num_runs}")
        print('='*50)
        
        run_maps = []
        run_errors = []
        
        # for d in range(2, max_d):
        #     X, error = embedding_GF_l2reg(G, d, max_iter=max_iter, eta=eta, regu=regu)
        #     rec_adj = get_reconstructed_adj(X)
        #     predicted_edge_list = get_edge_list_from_adj_mtrx(rec_adj)
        #     map_score = computeMAP_onlyfor_undirected_graphs(predicted_edge_list, G)
        #     run_maps.append(map_score)
        #     run_errors.append(error)
        #     print(f"d={d}, map_score={round(map_score, 2)}, error={error:.4f}")
        

        run_maps = [MAP(G, embedding_LAP(G, d)[0]) for d in range(1,max_d)]
        run_errors = [MAP(G, embedding_LAP(G, d)[1]) for d in range(1,max_d)]


        all_maps.append(run_maps)
        all_errors.append(run_errors)
    
    return all_maps, all_errors


def plot_results_statistics(all_maps, all_errors, max_d=20, max_iter=10000, eta=0.1, regu=0.1):
    """
    Plot statistics from multiple runs - Only subplots 1 and 4
    """
    all_maps = np.array(all_maps)
    all_errors = np.array(all_errors)
    
    # Calculate statistics
    map_mean = np.mean(all_maps, axis=0)
    map_std = np.std(all_maps, axis=0)
    map_min = np.min(all_maps, axis=0)
    map_max = np.max(all_maps, axis=0)
    
    error_mean = np.mean(all_errors, axis=0)
    error_std = np.std(all_errors, axis=0)
    error_min = np.min(all_errors, axis=0)
    error_max = np.max(all_errors, axis=0)
    
    # Create figure with only 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: MAP scores with error bars
    ax1 = axes[0]
    x = range(2, max_d)
    ax1.errorbar(x, map_mean, yerr=map_std, fmt='o-', capsize=5, label='Mean ± 1 SD', color='blue')
    ax1.fill_between(x, map_min, map_max, alpha=0.2, label='Min-Max Range', color='blue')
    
    # Mark d=4 and d=7 with special markers
    ax1.plot(4, map_mean[2], 's', markersize=10, color='red', label='d=4', zorder=5)
    ax1.plot(7, map_mean[5], '^', markersize=10, color='green', label='d=7', zorder=5)
    
    ax1.set_xlabel('Embedding Dimension (d)', fontsize=12)
    ax1.set_ylabel('MAP Score', fontsize=12)
    ax1.set_title(f'GF: MAP Scores (η={eta}, λ={regu}, iter={max_iter})', fontsize=14)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(range(2, max_d, 2))
    
    # Plot 2: Reconstruction errors with error bars
    ax2 = axes[1]
    ax2.errorbar(x, error_mean, yerr=error_std, fmt='o-', capsize=5, 
                 color='orange', label='Mean ± 1 SD')
    ax2.fill_between(x, error_min, error_max, alpha=0.2, color='orange', 
                     label='Min-Max Range')
    
    # Mark d=4 and d=7 with special markers
    ax2.plot(4, error_mean[2], 's', markersize=10, color='red', label='d=4', zorder=5)
    ax2.plot(7, error_mean[5], '^', markersize=10, color='green', label='d=7', zorder=5)
    
    ax2.set_xlabel('Embedding Dimension (d)', fontsize=12)
    ax2.set_ylabel('Reconstruction Error', fontsize=12)
    ax2.set_title(f'GF: Reconstruction Errors (η={eta}, λ={regu}, iter={max_iter})', fontsize=14)
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(range(2, max_d, 2))
    
    plt.tight_layout()
    plt.savefig(f'GF_results_eta{eta}_regu{regu}_iter{max_iter}.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    # Find optimal dimension based on mean MAP
    optimal_idx = np.argmax(map_mean)
    optimal_d = optimal_idx + 2  # +2 because d starts from 2
    
    print(f"\nHyperparameters: η={eta}, λ={regu}, max_iter={max_iter}")
    print(f"\nOptimal dimension based on mean MAP: d = {optimal_d}")
    print(f"  Mean MAP at d={optimal_d}: {map_mean[optimal_idx]:.4f}")
    print(f"  Std MAP at d={optimal_d}: {map_std[optimal_idx]:.4f}")
    print(f"  Range at d={optimal_d}: [{map_min[optimal_idx]:.4f}, {map_max[optimal_idx]:.4f}]")
    
    # Print values for d=4 and d=7 specifically
    print(f"\nSpecific dimensions:")
    print(f"  d=4: MAP = {map_mean[2]:.4f} ± {map_std[2]:.4f}, Error = {error_mean[2]:.4f} ± {error_std[2]:.4f}")
    print(f"  d=7: MAP = {map_mean[5]:.4f} ± {map_std[5]:.4f}, Error = {error_mean[5]:.4f} ± {error_std[5]:.4f}")
    
    print(f"\nOverall statistics across all dimensions:")
    print(f"  Mean MAP: {np.mean(all_maps):.4f} ± {np.std(all_maps):.4f}")
    print(f"  Mean Error: {np.mean(all_errors):.4f} ± {np.std(all_errors):.4f}")
    print(f"  Num runs: {len(all_maps)}")
    
    return optimal_d, map_mean, error_mean


# Main execution
if __name__ == "__main__":
    # Define hyperparameters
    MAX_ITER = 10000
    ETA = 0.1
    REGU = 0.1
    NUM_RUNS = 10
    MAX_D = 20
    
    print("="*70)
    print(f"Running Graph Factorization (GF) with L2 Regularization")
    print(f"Hyperparameters: η={ETA}, λ={REGU}, max_iter={MAX_ITER}")
    print(f"Number of runs: {NUM_RUNS}, Dimensions: 2 to {MAX_D-1}")
    print("="*70)
    
    # Run multiple trials
    all_maps, all_errors = run_multiple_trials(
        num_runs=NUM_RUNS, 
        max_d=MAX_D,
        max_iter=MAX_ITER,
        eta=ETA,
        regu=REGU
    )
    
    # Plot results
    optimal_d, map_mean, error_mean = plot_results_statistics(
        all_maps, all_errors, 
        max_d=MAX_D,
        max_iter=MAX_ITER,
        eta=ETA,
        regu=REGU
    )
    
    # Generate embeddings for d=4 and d=7 for t-SNE visualization
    print(f"\nGenerating embeddings for t-SNE comparison...")
    G = my_graph()
    
    # Generate embeddings for d=4 and d=7
    print("\nGenerating d=4 embeddings...")
    X_d4, error_d4 = embedding_GF_l2reg(G, 4, max_iter=MAX_ITER, eta=ETA, regu=REGU)
    print(f"d=4 error: {error_d4:.4f}")
    
    print("\nGenerating d=7 embeddings...")
    X_d7, error_d7 = embedding_GF_l2reg(G, 7, max_iter=MAX_ITER, eta=ETA, regu=REGU)
    print(f"d=7 error: {error_d7:.4f}")
    
    # Plot t-SNE comparison
    X_tsne_d4, X_tsne_d7 = plot_tsne_embedding(
        G, X_d4, X_d7,
        title=f"Graph Factorization Embeddings Comparison (η={ETA}, λ={REGU})",
        perplexity=500,
        n_iter=1000,
        random_state=42,
        figsize=(20, 6)
    )