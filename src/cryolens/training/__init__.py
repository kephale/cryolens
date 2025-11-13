"""Training module for CryoLens."""

# Import losses with better error handling to prevent module caching issues
# This is particularly important when installing from zip files in environments
# like Google Colab where dependencies might not be fully initialized on first import
try:
    from .losses import (
        MissingWedgeLoss, 
        NormalizedMSELoss, 
        ContrastiveAffinityLoss, 
        AffinityCosineLoss,
    )
    
    __all__ = [
        'MissingWedgeLoss',
        'NormalizedMSELoss', 
        'ContrastiveAffinityLoss',
        'AffinityCosineLoss',
    ]
except ImportError as e:
    import warnings
    warnings.warn(f"Failed to import losses: {e}. Loss functions will not be available.")
    __all__ = []

# Import curriculum learning components
try:
    from .curriculum import CurriculumScheduler, FixedCurriculumScheduler
    from .curriculum_vae import CurriculumVAEMixin
    
    __all__.extend([
        'CurriculumScheduler',
        'FixedCurriculumScheduler',
        'CurriculumVAEMixin',
    ])
except ImportError as e:
    import warnings
    warnings.warn(f"Failed to import curriculum components: {e}. Curriculum learning will not be available.")

# Import pose disentanglement components
try:
    from .pose_disentanglement import (
        compute_pose_kld,
        compute_cycle_consistency_loss,
        compute_pose_metrics,
        PoseScheduler,
    )
    
    __all__.extend([
        'compute_pose_kld',
        'compute_cycle_consistency_loss',
        'compute_pose_metrics',
        'PoseScheduler',
    ])
except ImportError as e:
    import warnings
    warnings.warn(f"Failed to import pose disentanglement components: {e}. Pose disentanglement will not be available.")
