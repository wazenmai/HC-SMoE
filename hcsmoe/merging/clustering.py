import torch
from typing import Optional

@torch.no_grad()
def group_experts_by_clustering(
    model: str,
    num_groups: int,
    cluster: str,
    linkage: str,
    hierarchical_stopping_metric: str,
    num_experts: int,
    experts: torch.Tensor,
    experts2: Optional[torch.Tensor] = None,
    experts3: Optional[torch.Tensor] = None,
    init_center: Optional[torch.Tensor] = None,
    w1: float = 1.0,
    w2: float = 1.0,
    w3: float = 1.0,
):
    if model == "mixtral":
        # experts = experts.to(torch.float).to('cuda:7')
        # experts2 = experts2.to(torch.float).to('cuda:7') if experts2 is not None else None
        # experts3 = experts3.to(torch.float).to('cuda:7') if experts3 is not None else None
        experts = experts.to(torch.float)
        experts2 = experts2.to(torch.float) if experts2 is not None else None
        experts3 = experts3.to(torch.float) if experts3 is not None else None
    else:
        experts = experts.to(torch.float)
        experts2 = experts2.to(torch.float) if experts2 is not None else None
        experts3 = experts3.to(torch.float) if experts3 is not None else None


    if cluster == "hierarchical":
        labels, dom_experts = hierarchical_clustering(experts, num_groups, linkage)
        print(f"group: {labels}, dom: {dom_experts}")
        return dom_experts, labels
    elif cluster == 'hierarchical-dynamic':
        labels, dom_experts = hierarchical_clustering_dynamic(experts, linkage, hierarchical_stopping_metric, num_groups, 1)
        print(f"group: {labels}, dom: {dom_experts}")
        return dom_experts, labels

    def _standardize(x):
        x = (x - x.mean(dim=0)) / (x.std(dim=0) + 1e-6)
        min_value = x.min()
        return x - min_value  # Shift so the minimum value is 0

    def kmeans_plus_plus_init(experts, num_groups):
        """
        Implements K-means++ initialization.
        Selects the first center randomly, then each subsequent center is chosen
        with a probability proportional to the square of the distance from the closest center.
        """
        num_experts = experts.size(0)
        centers = []
        center_indices = []
        
        # Step 1: Randomly select the first center
        first_center_idx = torch.randint(0, num_experts, (1,)).item()
        centers.append(experts[first_center_idx])
        center_indices.append(first_center_idx)

        # Step 2: Select the remaining centers
        for _ in range(1, num_groups):
            # Compute distance of each expert from the nearest center
            dist_list = []
            for i, center in enumerate(centers):
                dist = torch.cdist(experts, center.unsqueeze(0)) # (num_experts, num_centers)
                dist_list.append(dist)
            distances = torch.min(torch.concat(dist_list, dim=-1), dim=-1).values ** 2

            # Compute selection probabilities
            probabilities = distances / distances.sum()

            # Randomly choose the next center with probabilities proportional to distance
            next_center_idx = torch.multinomial(probabilities, 1).item()
            centers.append(experts[next_center_idx])
            center_indices.append(next_center_idx)
        
        return torch.tensor(center_indices)

    if init_center is not None:
        indices = init_center
    else:
        indices = kmeans_plus_plus_init(experts, num_groups)
    # elif self.random_start_center:
    #     indices = torch.randperm(self.num_experts)[:num_groups]
    # else:
    #     indices = torch.arange(num_groups)
    centers = experts[indices]
    centers2 = experts2[indices] if experts2 is not None else None
    centers3 = experts3[indices] if experts3 is not None else None
    distances = None
    assignments = None
    print(f"initial center: {centers.shape} {indices}")

    s1 = experts.shape[1]
    s2 = experts2.shape[1] if experts2 is not None else 1.0
    s3 = experts3.shape[1] if experts3 is not None else 1.0

    for _ in range(100):
        distances1 = _standardize(torch.cdist(experts, centers) / s1)
        distances2 = _standardize(torch.cdist(experts2, centers2) / s2) if experts2 is not None else torch.zeros(1, device=experts.device)
        distances3 = _standardize(torch.cdist(experts3, centers3) / s3) if experts3 is not None else torch.zeros(1, device=experts.device)
        print(f"distances1: {distances1.shape} {distances1}")
        print(f"distances2: {distances2.shape} {distances2}")
        print(f"distances3: {distances3.shape} {distances3}")

        distances = (w1 * distances1 + w2 * distances2 + w3 * distances3) / (w1 + w2 + w3)
        assignments = torch.argmin(distances, dim=1)
        del distances, distances1, distances2, distances3

        new_centers = torch.stack([experts[assignments == k].mean(dim=0) for k in range(num_groups)])
        new_centers2 = torch.stack([experts2[assignments == k].mean(dim=0) for k in range(num_groups)]) if experts2 is not None else None
        new_centers3 = torch.stack([experts3[assignments == k].mean(dim=0) for k in range(num_groups)]) if experts3 is not None else None
        
        print(f"assignments: {assignments}")
        for k in range(num_groups):
            print(f"cluster {k} {experts[assignments==k]}")
        print(f"new_centers: {torch.sum(torch.isnan(new_centers))}, {new_centers[0]}")
        if experts2 is not None:
            print(f"new_centers2: {torch.sum(torch.isnan(new_centers2))}, {new_centers2[0]}")
        max_diff = 0
        for i in range(num_groups):
            diff = torch.max(torch.abs(new_centers[i] - centers[i]))
            diff2 = torch.max(torch.abs(new_centers2[i] - centers2[i])) if experts2 is not None else torch.zeros(1, device=experts.device)
            diff3 = torch.max(torch.abs(new_centers3[i] - centers3[i])) if experts3 is not None else torch.zeros(1, device=experts.device)
            max_diff = max(max_diff, diff.item(), diff2.item(), diff3.item())
            if max_diff > 0.1:
                print(f"diff: {diff.item()}, {diff2.item()}, {diff3.item()}")
        if max_diff < 1e-4:
            print("Converged!")
            break
        centers = new_centers
        centers2 = new_centers2 if experts2 is not None else None
        centers3 = new_centers3 if experts3 is not None else None
    
    center_indices = []
    for k in range(num_groups):
        cluster_members = experts[assignments == k]
        cluster_members2 = experts2[assignments == k] if experts2 is not None else None
        cluster_members3 = experts3[assignments == k] if experts3 is not None else None
        distances1 = torch.cdist(cluster_members, new_centers[k].unsqueeze(0))
        distances2 = torch.cdist(cluster_members2, new_centers2[k].unsqueeze(0)) if experts2 is not None else torch.zeros(1, device=experts.device)
        distances3 = torch.cdist(cluster_members3, new_centers3[k].unsqueeze(0)) if experts3 is not None else torch.zeros(1, device=experts.device)
        final_distances = (distances1 + distances2 + distances3) / 2

        closest_expert_idx = torch.argmin(final_distances, dim=0)
        center_indices.append(torch.where(assignments == k)[0][closest_expert_idx].item())
    # centers = experts[center_indices]
    del centers, centers2, centers3
    print(f"group: {assignments.cpu()}, dom: {center_indices}")
    return center_indices, assignments

@torch.no_grad()
def compute_silhouette_score(tensor_list, cluster_labels):
        """
        Compute the silhouette score based on a list of tensors, 
        the cluster assignments, and the dominant experts for each group.
        """
        
        def compute_pairwise_distances(tensor_list):
            """Compute pairwise distances between all tensors in the list."""
            num_tensors = tensor_list.shape[0]
            distances = torch.zeros((num_tensors, num_tensors))

            for i in range(num_tensors):
                for j in range(i, num_tensors):
                    dist = torch.norm(tensor_list[i] - tensor_list[j])
                    distances[i, j] = dist
                    distances[j, i] = dist  # Symmetric matrix

            return distances
        
        # Step 1: Compute pairwise distances
        pairwise_distances = compute_pairwise_distances(tensor_list)

        num_tensors = tensor_list.shape[0]
        unique_labels = torch.unique(cluster_labels)

        silhouette_scores = torch.zeros(num_tensors)

        # Step 2: For each sample, compute silhouette score
        for i in range(num_tensors):
            # a(i): Mean intra-cluster distance (within the same cluster)
            same_cluster = [j for j in range(num_tensors) if cluster_labels[j] == cluster_labels[i] and j != i]
            if len(same_cluster) > 0:
                a_i = torch.mean(pairwise_distances[i, same_cluster])
            else:
                a_i = 0  # If there are no other points in the cluster

            # b(i): Mean nearest-cluster distance (distance to points in the nearest cluster)
            b_i = float('inf')
            for label in unique_labels:
                if label == cluster_labels[i]:
                    continue
                other_cluster = [j for j in range(num_tensors) if cluster_labels[j] == label]
                if len(other_cluster) > 0:
                    mean_dist_to_other_cluster = torch.mean(pairwise_distances[i, other_cluster])
                    b_i = min(b_i, mean_dist_to_other_cluster)

            # Step 3: Compute silhouette score for the sample
            silhouette_scores[i] = (b_i - a_i) / max(a_i, b_i)

        # Step 4: Average silhouette score across all samples
        overall_silhouette_score = torch.mean(silhouette_scores)

        return overall_silhouette_score


def safe_average(tensor):
    """Compute average ignoring inf values."""
    non_inf_mask = ~torch.isinf(tensor)
    if non_inf_mask.sum() == 0:
        return float('inf')
    return tensor[non_inf_mask].mean()

@torch.no_grad()
def compute_distance(pair_distances, clusters, method='average', X=None):
    if method == 'average':
        # dist(cluster i, cluster j) = sum_{x in cluster i, y in cluster j} dist(x, y) / (|cluster i| * |cluster j|)
        cluster_labels = torch.unique(clusters)
        distances = torch.zeros((len(cluster_labels), len(cluster_labels)))
        # Iterate through all pairs of clusters (ci, cj)
        for i, ci in enumerate(cluster_labels):
            for j, cj in enumerate(cluster_labels):
                if i >= j:
                    continue
                dist = []
                # Iterate through all pairs of points (vi, vj) for vi in ci and vj in cj
                for vi in torch.where(clusters == ci)[0]:
                    for vj in torch.where(clusters == cj)[0]:
                        dist.append(pair_distances[vi, vj].item())
                new_dist = torch.sum(torch.tensor(dist)) / (torch.sum(clusters == ci) * torch.sum(clusters == cj))
                distances[i, j] = new_dist
                distances[j, i] = new_dist
        distances.fill_diagonal_(float('inf'))
        idx = torch.argmin(distances)
        final_i, final_j = cluster_labels[idx // distances.shape[0]], cluster_labels[idx % distances.shape[0]]
    elif method == 'ward':
        # 1. Compute the center of each cluster
        cluster_labels = torch.unique(clusters)
        cluster_centers = torch.zeros((len(cluster_labels), X.shape[1]))
        for i, cluster in enumerate(cluster_labels):
            cluster_centers[i] = X[clusters == cluster].mean(dim=0)
        
        # 2. Compute the distance between each pair of clusters
        distances = torch.zeros((len(cluster_labels), len(cluster_labels)))
        for i, ci in enumerate(cluster_labels):
            for j, cj in enumerate(cluster_labels):
                if i >= j:
                    continue
                ni = torch.sum(clusters == ci)
                nj = torch.sum(clusters == cj)
                new_dist = (ni * nj) / (ni + nj) * torch.cdist(cluster_centers[i].unsqueeze(0), cluster_centers[j].unsqueeze(0), p=2)
                distances[i, j] = new_dist
                distances[j, i] = new_dist
        distances.fill_diagonal_(float('inf'))
        idx = torch.argmin(distances)
        final_i, final_j = cluster_labels[idx // distances.shape[0]], cluster_labels[idx % distances.shape[0]]
    else:
        raise NotImplementedError("Unsupported linkage method: {}".format(method))
    
    return final_i, final_j

@torch.no_grad()
def pairwise_distances(X, method='single'):
    """Compute pairwise Euclidean distances between points."""
    dot_product = torch.mm(X, X.t())
    square_norm = dot_product.diag()
    distances = square_norm.unsqueeze(0) - 2.0 * dot_product + square_norm.unsqueeze(1)
    distances = torch.clamp(distances, min=0.0).sqrt()
    if method == 'single' or method == 'average':
        distances.fill_diagonal_(float('inf'))
    elif method == 'complete':
        distances.fill_diagonal_(0.0)
    return distances

@torch.no_grad()
def linkage_step(distances, pair_distances, clusters=None, method='single', X=None):
    """Perform a single step of hierarchical clustering using the specified linkage method."""
    """
    Single linkage: d(ci, cj) = min_{x in ci, y in cj} dist(x, y) -> the closest pair of points
    Complete linkage: d(ci, cj) = max_{x in ci, y in cj} dist(x, y) -> the farthest pair of points
    Average linkage: d(ci, cj) = sum_{x in ci, y in cj} dist(x, y) / (|ci| * |cj|) -> the average distance between all pairs
    Ward linkage: d(ci, cj) = (|ci| * |cj|) / (|ci| + |cj|) * dist(mu(ci), mu(cj)) -> the increase in variance when merging clusters
    """
    
    ### 1. Find the pair of clusters with the smallest distance
    if method == 'single':
        # d(ci, cj) = min_{x in ci, y in cj} dist(x, y)
        min_idx = torch.argmin(distances).item()
        i, j = min_idx // distances.shape[0], min_idx % distances.shape[0]
        # print(f"min_idx: {min_idx}, ({i}, {j})")
    elif method == 'complete':
        # d(ci, cj) = max_{x in ci, y in cj} dist(x, y)
        max_idx = torch.argmax(distances).item()
        i, j = max_idx // distances.shape[0], max_idx % distances.shape[0]
    else:
        i, j = compute_distance(pair_distances, clusters, method, X)
    
    if i > j:
        i, j = j, i
    
    if method == 'average' or method == 'ward':
        return i, j, distances
    
    ### 2. Update the distance matrix
    # We merge cluster j to cluster i, so other clusters to cluster j will be inf. (cluster j dissapears)
    # And the distance from cluster i to other clusters will be updated based on the linkage method.
    for k in range(distances.shape[0]):
        if k != i and k != j: # skip the merged cluster
            if method == 'single':
                new_dist = torch.min(distances[i, k], distances[j, k])
            elif method == 'complete':
                new_dist = torch.max(distances[i, k], distances[j, k])
            distances[i, k] = new_dist
            distances[k, i] = new_dist

    if method == 'single':
        distances[i, i] = float('inf')
        distances[j, :] = float('inf')
        distances[:, j] = float('inf')
    elif method == 'complete':
        distances[i, i] = 0.0
        distances[j, :] = 0.0
        distances[:, j] = 0.0
    
    return i, j, distances

@torch.no_grad()
def hierarchical_clustering(X, n_clusters, method='single'):
    """Perform hierarchical clustering using the specified linkage method."""
    print("hierarchical clustering - {} to {} clusters".format(method, n_clusters))
    device = X.device
    n_samples = X.shape[0]
    
    # Compute pairwise distances
    distances = pairwise_distances(X, method)
    pair_distances = distances.clone()
    
    # Initialize clusters
    clusters = torch.tensor([i for i in range(n_samples)])
    
    # Perform clustering
    while len(torch.unique(clusters)) > n_clusters:
        i, j, distances = linkage_step(distances, pair_distances, clusters, method, X)
        print(f"clusters: {len(torch.unique(clusters))}, merge ({i}, {j})")
        cj = clusters[j]
        # Merge cluster j to cluster i
        clusters[clusters == cj] = clusters[i]

    
    # Reassign cluster IDs to be contiguous
    d = {}
    element_id = 0
    for i, idx in enumerate(clusters):
        if idx.item() not in d:
            d[idx.item()] = element_id
            element_id += 1
        clusters[i] = d[idx.item()]
    
    center_indices = []
    for k in range(n_clusters):
        cluster_members = X[clusters == k]
        cluster_center = cluster_members.mean(dim=0)
        distances = torch.cdist(cluster_members, cluster_center.unsqueeze(0), p=2)
        closest_expert_idx = torch.argmin(distances, dim=0).item()
        center_indices.append(torch.where(clusters == k)[0][closest_expert_idx].item())
    
    del distances
    return clusters, center_indices


def hierarchical_clustering_dynamic(X, linkage='single', stopping_metric='silhouette', max_clusters=8, min_clusters=2):
    """
    Perform hierarchical clustering using PyTorch on GPU with dynamic stopping criterion.
    :param experts: Tensor of experts (n_samples, n_features)
    :param max_clusters: The maximum number of clusters allowed.
    :param linkage: Linkage method: 'single', 'complete', 'average', 'ward'
    :param stopping_metric: Stopping criterion: 'silhouette' or 'inertia'
    :param min_clusters: Minimum number of clusters.
    :return: Cluster assignments and the ID of the expert closest to the group center
    """
    n_samples = X.shape[0]
    
    # Compute pairwise distances
    distances = pairwise_distances(X, linkage)
    pair_distances = distances.clone()
    
    # Initialize clusters
    clusters = torch.tensor([i for i in range(n_samples)])
    best_score = -float('inf')  # Best silhouette score or lowest inertia
    best_clusters = None
    
    # Perform clustering
    while len(torch.unique(clusters)) > min_clusters:
        i, j, distances = linkage_step(distances, pair_distances, clusters, linkage, X)
        cj = clusters[j]
        # Merge cluster j to cluster i
        clusters[clusters == cj] = clusters[i]

        # Compute the stopping metric
        if len(torch.unique(clusters)) <= max_clusters:
            if stopping_metric == 'silhouette' and len(clusters) >= 2:
                score = compute_silhouette_score(X, clusters)
                if score > best_score:
                    best_score = score
                    del best_clusters
                    best_clusters = clusters.clone()
                    print(f"Update score to {score}, {best_clusters}")
            elif stopping_metric == 'inertia':
                inertia = 0.0
                for idx, cluster in enumerate(clusters):
                    cluster_experts = X[cluster]
                    centroid = cluster_experts.mean(dim=0)
                    inertia += torch.sum((cluster_experts - centroid) ** 2).item()
                if inertia < best_score:
                    best_score = inertia
                    del best_clusters
                    best_clusters = clusters.clone()
                    print(f"Update score to {score}, {best_clusters}")

    
    # Reassign cluster IDs to be contiguous
    d = {}
    element_id = 0
    for i, idx in enumerate(best_clusters):
        if idx.item() not in d:
            d[idx.item()] = element_id
            element_id += 1
        best_clusters[i] = d[idx.item()]
    
    center_indices = []
    for k in range(len(torch.unique(best_clusters))):
        cluster_members = X[best_clusters == k]
        cluster_center = cluster_members.mean(dim=0)
        distances = torch.cdist(cluster_members, cluster_center.unsqueeze(0), p=2)
        closest_expert_idx = torch.argmin(distances, dim=0).item()
        center_indices.append(torch.where(best_clusters == k)[0][closest_expert_idx].item())
    
    del distances
    return best_clusters, center_indices
