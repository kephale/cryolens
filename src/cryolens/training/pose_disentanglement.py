"""
Pose disentanglement losses and utilities.

This module provides losses and helper functions for learning disentangled
representations where structural identity is separated from 3D orientation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


def compute_pose_kld(
    pose_mu: torch.Tensor,
    pose_log_var: torch.Tensor,
    reduction: str = 'mean'
) -> torch.Tensor:
    """Compute KL divergence loss for variational pose distribution.
    
    Regularizes the pose distribution toward a standard normal prior,
    preventing collapse while allowing the network to learn diverse
    orientations.
    
    Parameters
    ----------
    pose_mu : torch.Tensor
        Mean of the pose distribution, shape (batch_size, pose_dims)
    pose_log_var : torch.Tensor
        Log variance of the pose distribution, shape (batch_size, pose_dims)
    reduction : str
        How to reduce the loss: 'mean', 'sum', or 'none'
        
    Returns
    -------
    torch.Tensor
        KL divergence loss
        
    Notes
    -----
    The KL divergence between q(pose|x) and N(0,I) is:
    KL = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    """
    kld = -0.5 * torch.sum(1 + pose_log_var - pose_mu**2 - pose_log_var.exp(), dim=1)
    
    if reduction == 'mean':
        return kld.mean()
    elif reduction == 'sum':
        return kld.sum()
    else:
        return kld


def compute_cycle_consistency_loss(
    model: nn.Module,
    z: torch.Tensor,
    global_weight: torch.Tensor,
    num_random_poses: int = 1,
    pose_distance_metric: str = 'mse'
) -> Tuple[torch.Tensor, dict]:
    """Compute cycle consistency loss for pose disentanglement.
    
    This loss teaches the model that:
    1. Different poses should produce different reconstructions
    2. The pose can be recovered from the reconstruction
    3. Pose controls rotation in Gaussian splat space (not input volume space)
    
    The key insight is that we rotate in splat coordinate space where the
    missing wedge artifact is not present, avoiding contradictions.
    
    Parameters
    ----------
    model : nn.Module
        The VAE model with encode/decode/reparameterize methods
    z : torch.Tensor
        Latent structure representation (detached to prevent gradient flow)
    global_weight : torch.Tensor
        Global amplitude weights (detached)
    num_random_poses : int
        Number of random poses to sample per batch item (default: 1)
    pose_distance_metric : str
        Distance metric for pose comparison: 'mse', 'l1', or 'cosine'
        
    Returns
    -------
    loss : torch.Tensor
        Cycle consistency loss
    info : dict
        Dictionary with additional information:
        - 'pose_distance_mean': Average distance between random and recovered poses
        - 'pose_distance_std': Standard deviation of distances
        
    Notes
    -----
    The process is:
    1. Sample random poses from N(0,I)
    2. Decode z with random poses to get reconstructions
    3. Re-encode reconstructions to recover poses
    4. Compute distance between random and recovered poses
    
    This encourages the model to use pose as a true rotation parameter.
    """
    batch_size = z.shape[0]
    device = z.device
    
    # Get pose dimensions from model
    if hasattr(model, 'model'):
        # Handle wrapped model (e.g., LightningModule)
        pose_dims = model.model.pose_channels
    else:
        pose_dims = model.pose_channels
    
    # Sample random poses from standard normal distribution
    random_poses = torch.randn(
        batch_size * num_random_poses,
        pose_dims,
        device=device
    )
    
    # Expand z and global_weight to match number of random poses
    z_expanded = z.detach().repeat_interleave(num_random_poses, dim=0)
    global_weight_expanded = global_weight.detach().repeat_interleave(num_random_poses, dim=0)
    
    # Decode with random poses
    with torch.no_grad():
        # We don't want gradients through the decoder for the initial reconstruction
        recon_random_pose = model.decode(z_expanded, random_poses, global_weight_expanded)
    
    # Re-encode to recover pose
    # Now we DO want gradients through the encoder
    _, _, reencoded_pose_mu, reencoded_pose_log_var, _ = model.encode(recon_random_pose)
    reencoded_pose = model.reparameterize(reencoded_pose_mu, reencoded_pose_log_var)
    
    # Compute distance between random and recovered poses
    if pose_distance_metric == 'mse':
        distances = F.mse_loss(reencoded_pose, random_poses, reduction='none').sum(dim=1)
        loss = distances.mean()
    elif pose_distance_metric == 'l1':
        distances = F.l1_loss(reencoded_pose, random_poses, reduction='none').sum(dim=1)
        loss = distances.mean()
    elif pose_distance_metric == 'cosine':
        # Cosine distance: 1 - cosine_similarity
        cos_sim = F.cosine_similarity(reencoded_pose, random_poses, dim=1)
        distances = 1 - cos_sim
        loss = distances.mean()
    else:
        raise ValueError(f"Unknown pose distance metric: {pose_distance_metric}")
    
    # Collect diagnostic information
    info = {
        'pose_distance_mean': distances.mean().item(),
        'pose_distance_std': distances.std(unbiased=False).item() if len(distances) > 1 else 0.0,
        'random_pose_norm': random_poses.norm(dim=1).mean().item(),
        'recovered_pose_norm': reencoded_pose.norm(dim=1).mean().item()
    }
    
    return loss, info


class PoseScheduler:
    """Scheduler for gradually increasing pose loss weights during training.
    
    This implements the gradual weight scheduling strategy:
    - Start with small pose_beta for first warmup_epochs
    - Linearly ramp up to final_pose_beta over ramp_epochs
    - Keep constant after that
    
    Parameters
    ----------
    initial_pose_beta : float
        Initial weight for pose KLD loss (default: 0.0001)
    final_pose_beta : float
        Final weight for pose KLD loss (default: 0.001)
    warmup_epochs : int
        Number of epochs to keep at initial weight (default: 50)
    ramp_epochs : int
        Number of epochs to ramp from initial to final (default: 50)
    cycle_consistency_weight : float
        Fixed weight for cycle consistency loss (default: 0.1)
    """
    
    def __init__(
        self,
        initial_pose_beta: float = 0.0001,
        final_pose_beta: float = 0.001,
        warmup_epochs: int = 50,
        ramp_epochs: int = 50,
        cycle_consistency_weight: float = 0.1
    ):
        self.initial_pose_beta = initial_pose_beta
        self.final_pose_beta = final_pose_beta
        self.warmup_epochs = warmup_epochs
        self.ramp_epochs = ramp_epochs
        self.cycle_consistency_weight = cycle_consistency_weight
    
    def get_pose_beta(self, current_epoch: int) -> float:
        """Get current pose_beta weight based on epoch.
        
        Parameters
        ----------
        current_epoch : int
            Current training epoch
            
        Returns
        -------
        float
            Current pose_beta weight
        """
        if current_epoch < self.warmup_epochs:
            return self.initial_pose_beta
        elif current_epoch < self.warmup_epochs + self.ramp_epochs:
            # Linear ramp
            progress = (current_epoch - self.warmup_epochs) / self.ramp_epochs
            return self.initial_pose_beta + progress * (self.final_pose_beta - self.initial_pose_beta)
        else:
            return self.final_pose_beta
    
    def get_cycle_weight(self, current_epoch: int) -> float:
        """Get current cycle consistency weight (constant).
        
        Parameters
        ----------
        current_epoch : int
            Current training epoch
            
        Returns
        -------
        float
            Current cycle consistency weight
        """
        return self.cycle_consistency_weight
    
    def get_info(self, current_epoch: int) -> dict:
        """Get current scheduler state information.
        
        Parameters
        ----------
        current_epoch : int
            Current training epoch
            
        Returns
        -------
        dict
            Dictionary with current weights and phase information
        """
        pose_beta = self.get_pose_beta(current_epoch)
        
        if current_epoch < self.warmup_epochs:
            phase = "warmup"
        elif current_epoch < self.warmup_epochs + self.ramp_epochs:
            phase = "ramp"
        else:
            phase = "full"
        
        return {
            'pose_beta': pose_beta,
            'cycle_weight': self.cycle_consistency_weight,
            'phase': phase,
            'epoch': current_epoch
        }


def compute_pose_metrics(
    pose_mu: torch.Tensor,
    pose_log_var: torch.Tensor,
    mol_id: Optional[torch.Tensor] = None
) -> dict:
    """Compute diagnostic metrics for pose distribution.
    
    These metrics help validate that pose disentanglement is working:
    - High entropy: Pose is not collapsed to a single value
    - High variance: Pose varies across the batch
    - Low same-structure variance: Same structures have similar poses (optional)
    
    Parameters
    ----------
    pose_mu : torch.Tensor
        Mean of pose distribution, shape (batch_size, pose_dims)
    pose_log_var : torch.Tensor
        Log variance of pose distribution, shape (batch_size, pose_dims)
    mol_id : Optional[torch.Tensor]
        Molecular IDs for computing same-structure variance (optional)
        
    Returns
    -------
    dict
        Dictionary with diagnostic metrics:
        - 'pose_entropy': Average entropy of pose distribution
        - 'pose_mu_variance': Variance of pose means across batch
        - 'pose_sigma_mean': Average standard deviation
        - 'pose_mu_norm': Average L2 norm of pose means
    """
    # Compute standard deviations
    pose_sigma = torch.exp(0.5 * pose_log_var)
    
    # Entropy of each pose dimension (approximation for Gaussian)
    # H = 0.5 * log(2 * pi * e * sigma^2)
    entropy = 0.5 * (1 + torch.log(2 * torch.pi * pose_sigma**2))
    
    metrics = {
        'pose_entropy': entropy.mean().item(),
        'pose_mu_variance': pose_mu.var(dim=0).mean().item(),
        'pose_sigma_mean': pose_sigma.mean().item(),
        'pose_mu_norm': pose_mu.norm(dim=1).mean().item(),
        'pose_log_var_mean': pose_log_var.mean().item()
    }
    
    # If we have mol_id, compute same-structure variance
    if mol_id is not None:
        unique_ids = torch.unique(mol_id)
        if len(unique_ids) > 1:
            same_struct_vars = []
            for mol in unique_ids:
                mask = mol_id == mol
                if mask.sum() > 1:
                    same_struct_vars.append(pose_mu[mask].var(dim=0).mean().item())
            
            if same_struct_vars:
                metrics['pose_same_structure_variance'] = sum(same_struct_vars) / len(same_struct_vars)
    
    return metrics
