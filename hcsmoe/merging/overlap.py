import torch
import torch.nn.functional as F
from scipy.stats import wasserstein_distance

def overlap_rate(X: torch.Tensor, Y: torch.Tensor):
    # Calculate the min and max of both datasets along each dimension
    X_min, X_max = X.min(dim=0).values, X.max(dim=0).values
    Y_min, Y_max = Y.min(dim=0).values, Y.max(dim=0).values
    
    # Calculate the intersection range
    intersection_min = torch.max(X_min, Y_min)
    intersection_max = torch.min(X_max, Y_max)
    
    # Calculate the union range
    union_min = torch.min(X_min, Y_min)
    union_max = torch.max(X_max, Y_max)
    
    # Calculate the lengths of the intersection and union ranges
    intersection_length = (intersection_max - intersection_min).clamp(min=0)
    union_length = union_max - union_min
    
    # Calculate the overlap rate as the ratio of the intersection to the union
    overlap_rate = intersection_length.sum() / union_length.sum()
    
    return overlap_rate.item()

def bhattacharyya_distance(mean_X, cov_X, mean_Y, cov_Y, epsilon=1e-6):
    # Regularize the covariance matrices by adding a small value to the diagonal
    cov_X_reg = cov_X + epsilon * torch.eye(cov_X.size(0))
    cov_Y_reg = cov_Y + epsilon * torch.eye(cov_Y.size(0))
    
    # Mean difference
    mean_diff = mean_X - mean_Y
    
    # Average covariance
    cov_mean = (cov_X_reg + cov_Y_reg) / 2
    
    # First term: Mahalanobis distance
    inv_cov_mean = torch.inverse(cov_mean)
    term1 = 0.125 * torch.dot(mean_diff, torch.mv(inv_cov_mean, mean_diff))
    
    # Second term: Log-determinant of covariances
    try:
        term2 = 0.5 * torch.logdet(cov_mean) - 0.25 * (torch.logdet(cov_X_reg) + torch.logdet(cov_Y_reg))
    except RuntimeError:
        print("Log determinant calculation failed due to numerical issues.")
        return float('inf')  # Set distance to infinity if logdet fails
    
    # Bhattacharyya distance
    BC = term1 + term2
    return BC

def overlap_rate_bhattacharyya(X: torch.Tensor, Y: torch.Tensor):
    # Compute mean and covariance of X and Y
    mean_X = torch.mean(X, dim=0)
    mean_Y = torch.mean(Y, dim=0)
    
    cov_X = torch.cov(X.T)  # Covariance matrix for X
    cov_Y = torch.cov(Y.T)  # Covariance matrix for Y
    
    # Calculate Bhattacharyya distance
    BC = bhattacharyya_distance(mean_X, cov_X, mean_Y, cov_Y)
    print(f"Bhattacharyya distance: {BC:.4f}")
    # Overlap rate approximation
    overlap_rate = torch.exp(-BC)
    
    return overlap_rate.item()


def get_prob_distributions(expert_outputs):
    return F.softmax(expert_outputs, dim=-1)  # Normalize across the last dimension


def compute_kl_divergence(p, q):
    epsilon = 1e-10
    p = p + epsilon
    q = q + epsilon
    return - (F.kl_div(p.log(), q, reduction='batchmean') + 
            F.kl_div(q.log(), p, reduction='batchmean')) / 2



# def compute_wasserstein_overlapping_rate(p, q):
#     p = p.float().reshape(-1).cpu().detach().numpy()
#     q = q.float().reshape(-1).cpu().detach().numpy()
#     dist = -wasserstein_distance(p, q)
#     del p, q
#     return dist


def compute_wasserstein_distance(p, q):
    # Ensure the tensors are on the same device and sort them
    p_sorted, _ = torch.sort(p)
    q_sorted, _ = torch.sort(q)
    
    # Compute the Wasserstein distance as the mean absolute difference
    # between the sorted values (CDF difference in a way)
    wasserstein_dist = torch.mean(torch.abs(p_sorted - q_sorted))
    
    return -wasserstein_dist

# Example usage
# experts_outputs = [torch.randn(32, 2048, device='cuda') for _ in range(8)]  # Random outputs from 8 experts
# wasserstein_distances = compute_wasserstein_overlapping_rate_pytorch(experts_outputs)
# print(wasserstein_distances)



# Example usage
# d = 100  # Dimensionality of the data
# n_X = 100  # Number of data points in X
# n_Y = 100  # Number of data points in Y

# # Generate random tensors for X and Y
# X = torch.randn(n_X, d)
# Y = torch.randn(n_Y, d)
# print(f"X: {X.shape}")
# print(X)
# print(f"Y: {Y.shape}")
# print(Y)

# # Calculate overlap rate
# overlap = overlap_rate(X, Y)
# print(f"Overlap rate: {overlap:.4f}")

# overlap = overlap_rate_bhattacharyya(X, Y)
# print(f"Overlap rate (Bhattacharyya): {overlap}")
