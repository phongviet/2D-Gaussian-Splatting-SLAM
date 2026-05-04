import numpy as np
import torch

def umeyama_alignment(x, y, with_scale=True):
    """
    Computes the Sim(3) transformation that aligns x to y.
    x: (N, 3) numpy array (estimated)
    y: (N, 3) numpy array (ground truth)
    Returns: s (scalar), R (3x3), t (3,) such that y \approx s * R @ x + t
    """
    mu_x = x.mean(axis=0)
    mu_y = y.mean(axis=0)
    
    x_centered = x - mu_x
    y_centered = y - mu_y
    
    n = x.shape[0]
    
    sigma_x = (x_centered**2).sum() / n
    
    # Covariance matrix
    sigma = (y_centered.T @ x_centered) / n
    
    U, D, Vt = np.linalg.svd(sigma)
    
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
        
    R = U @ S @ Vt
    
    if with_scale:
        s = np.trace(np.diag(D) @ S) / (sigma_x + 1e-8)
    else:
        s = 1.0
        
    t = mu_y - s * R @ mu_x
    
    return s, R, t

def apply_sim3_to_gaussians(gaussians, s, R, t):
    """
    Applies Sim(3) transformation to a GaussianModel.
    y = s * R @ x + t
    """
    from scipy.spatial.transform import Rotation as ScipyRot
    
    with torch.no_grad():
        # 1. Transform means
        means = gaussians.get_xyz
        device = means.device
        
        R_torch = torch.from_numpy(R).to(device).float()
        t_torch = torch.from_numpy(t).to(device).float()
        
        means_new = s * (means @ R_torch.T) + t_torch
        
        # 2. Transform rotations
        # q_gs is [w, x, y, z]
        q_gs = gaussians._rotation.detach().cpu().numpy()
        # Scipy uses [x, y, z, w]
        q_scipy = q_gs[:, [1, 2, 3, 0]]
        
        r_old = ScipyRot.from_quat(q_scipy)
        r_sim3 = ScipyRot.from_matrix(R)
        r_new = r_sim3 * r_old
        
        q_new_scipy = r_new.as_quat()
        # Convert back to [w, x, y, z]
        q_new_gs = q_new_scipy[:, [3, 0, 1, 2]]
        q_new_gs_torch = torch.from_numpy(q_new_gs).to(device).float()
        
        # 3. Transform scales
        scales = gaussians.get_scaling
        scales_new = scales * s
        
        # Update Gaussian parameters
        gaussians._xyz.copy_(means_new)
        gaussians._scaling.copy_(torch.log(scales_new))
        gaussians._rotation.copy_(q_new_gs_torch)
        
    print(f"Applied Sim(3) alignment: scale={s:.4f}")
