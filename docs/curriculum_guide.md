# Curriculum Learning and Pose Disentanglement Guide

This guide shows how to use the curriculum learning and pose disentanglement components that are now part of the core `cryolens` library.

## Overview

The refactoring moves reusable components into `cryolens`:

- **`cryolens.training.curriculum`**: CurriculumScheduler for alternating structure selection
- **`cryolens.training.curriculum_vae`**: Mixin for curriculum-aware VAE training
- **`cryolens.training.pose_disentanglement`**: Pose KLD and cycle consistency losses

This allows training scripts to focus on experiment-specific logic while reusing battle-tested components.

## Basic Usage

### 1. Create a Curriculum Scheduler

```python
from cryolens.training.curriculum import CurriculumScheduler

# Initialize scheduler
scheduler = CurriculumScheduler(
    all_items=["1abc", "2def", "3ghi", "4jkl", "background"],  # Your PDB codes
    items_per_phase=6,          # How many structures per phase
    epochs_per_phase=300,        # How long each phase lasts
    reserved_items=["background"]  # Items always included (e.g., background)
)

# Get structures for an epoch
structures = scheduler.get_items_for_epoch(current_epoch=0)
# Returns: ["background", "1abc", "2def", "3ghi", "4jkl", "5mno"]
```

### 2. Use the Curriculum VAE Mixin

```python
from pytorch_lightning import LightningModule
from cryolens.training.curriculum_vae import CurriculumVAEMixin
from cryolens.models.vae import AffinityVAE

class MyCurriculumVAE(LightningModule, CurriculumVAEMixin):
    def __init__(self, scheduler, all_pdb_codes, dataloader_params, ...):
        super().__init__()
        
        # Initialize your model
        self.model = AffinityVAE(...)
        
        # Setup curriculum learning
        self.setup_curriculum(
            scheduler=scheduler,
            dataloader_factory=lambda filtered_structure_ids: 
                create_my_dataloader(
                    structure_ids=filtered_structure_ids,
                    **dataloader_params
                ),
            valid_structure_ids=all_pdb_codes,
            checkpoint_dir=Path("checkpoints")
        )
    
    def training_step(self, batch, batch_idx):
        img, local_mol_id = batch
        
        # Map local indices to global for similarity loss
        global_mol_id = self.map_to_global_indices(local_mol_id)
        
        # Forward pass
        output, z, pose, global_weight, mu, log_var, pose_mu, pose_log_var = self.model(img)
        
        # Your loss computation using global_mol_id
        ...
```

The mixin automatically:
- Calls `on_train_epoch_start()` to check for curriculum changes
- Creates/updates the dataloader when the curriculum changes
- Handles index mapping between local and global structure IDs
- Logs curriculum changes to a JSONL file

### 3. Add Pose Disentanglement Losses

```python
from cryolens.training.pose_disentanglement import (
    compute_pose_kld,
    compute_cycle_consistency_loss,
    compute_pose_metrics,
    PoseScheduler
)

class MyCurriculumVAE(LightningModule, CurriculumVAEMixin):
    def __init__(self, ...):
        super().__init__()
        
        # Initialize pose scheduler
        self.pose_scheduler = PoseScheduler(
            initial_pose_beta=0.0001,
            final_pose_beta=0.001,
            warmup_epochs=50,
            ramp_epochs=50,
            cycle_consistency_weight=0.1
        )
    
    def training_step(self, batch, batch_idx):
        img, local_mol_id = batch
        global_mol_id = self.map_to_global_indices(local_mol_id)
        
        # Forward pass (now returns pose_mu and pose_log_var)
        output, z, pose, global_weight, mu, log_var, pose_mu, pose_log_var = self.model(img)
        
        # Standard losses
        r_loss = self.reconstruction_loss(output, img)
        kld = -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1).mean()
        s_loss = self.similarity_loss(global_mol_id, mu)
        
        # Pose KLD loss with gradual scheduling
        pose_beta = self.pose_scheduler.get_pose_beta(self.current_epoch)
        pose_kld = compute_pose_kld(pose_mu, pose_log_var)
        pose_kld_weighted = pose_beta * pose_kld
        
        # Cycle consistency loss
        cycle_weight = self.pose_scheduler.get_cycle_weight(self.current_epoch)
        cycle_loss, cycle_info = compute_cycle_consistency_loss(
            model=self.model,
            z=z.detach(),
            global_weight=global_weight.detach(),
            num_random_poses=1
        )
        cycle_loss_weighted = cycle_weight * cycle_loss
        
        # Total loss
        loss = r_loss + kld + s_loss + pose_kld_weighted + cycle_loss_weighted
        
        # Log everything
        self.log("reconstruction_loss", r_loss)
        self.log("kld_loss", kld)
        self.log("similarity_loss", s_loss)
        self.log("pose_kld", pose_kld_weighted)
        self.log("cycle_loss", cycle_loss_weighted)
        self.log("pose_beta", pose_beta)
        
        # Log pose metrics periodically
        if batch_idx % 100 == 0:
            metrics = compute_pose_metrics(pose_mu, pose_log_var, global_mol_id)
            for key, value in metrics.items():
                self.log(f"pose_metrics/{key}", value)
        
        return loss
```

### 4. Handle Resuming from Checkpoints

```python
# In your training script
if resume_checkpoint:
    epoch = extract_epoch_from_checkpoint(resume_checkpoint)
    
    # Tell the model to resume
    model.set_resume_epoch(epoch)
    
    # Restore scheduler state if you have a curriculum log
    curriculum_log = checkpoint_dir / "curriculum_log.jsonl"
    if curriculum_log.exists():
        scheduler.restore_from_log(str(curriculum_log), epoch)
    
    # Resume training
    trainer.fit(model, ckpt_path=resume_checkpoint)
```

## Complete Example

See `cryolens-scripts/src/cryolens/scripts/alternating_curriculum.py` for a complete working example that:

1. Creates a CurriculumScheduler
2. Uses CurriculumVAEMixin for automatic dataloader updates
3. Implements pose disentanglement with PoseScheduler
4. Handles checkpointing and resuming
5. Logs curriculum changes and pose metrics

## Key Design Principles

### Separation of Concerns

- **cryolens**: Reusable, well-tested components
- **cryolens-scripts**: Experiment-specific glue code

### Composability

All components work independently:
- Use CurriculumScheduler without the mixin
- Use pose losses without curriculum learning
- Mix and match as needed

### Flexibility

- `CurriculumScheduler` supports any items (not just PDB codes)
- `FixedCurriculumScheduler` for baseline experiments
- `PoseScheduler` weights are fully configurable
- All loss functions support different reduction modes

## Migration Guide

If you have existing training scripts using the old alternating curriculum code:

### Old Code (in training script)

```python
class AlternatingCurriculumVAE(DensityVAE):
    def __init__(self, scheduler, ...):
        # Lots of curriculum-specific code here
        self.scheduler = scheduler
        self.current_structures = None
        # ... 100+ lines of curriculum logic
    
    def on_train_epoch_start(self):
        # Manual curriculum handling
        # ... 50+ lines
    
    def train_dataloader(self):
        # Manual dataloader creation
        # ... 30+ lines
```

### New Code (using mixin)

```python
from cryolens.training.curriculum_vae import CurriculumVAEMixin

class AlternatingCurriculumVAE(DensityVAE, CurriculumVAEMixin):
    def __init__(self, scheduler, ...):
        super().__init__(...)
        
        # One-line setup!
        self.setup_curriculum(
            scheduler=scheduler,
            dataloader_factory=self._create_dataloader,
            valid_structure_ids=all_pdb_codes,
            checkpoint_dir=checkpoint_dir
        )
    
    def _create_dataloader(self, filtered_structure_ids):
        # Your dataloader creation logic
        return create_tomotwin_dataloader(
            structure_ids=filtered_structure_ids,
            ...
        )
    
    def training_step(self, batch, batch_idx):
        img, local_mol_id = batch
        global_mol_id = self.map_to_global_indices(local_mol_id)
        # Rest of your training logic
```

The mixin handles all the curriculum management automatically!

## Testing

All new components include docstrings with usage examples. To test:

```python
# Test curriculum scheduler
from cryolens.training.curriculum import CurriculumScheduler

scheduler = CurriculumScheduler(
    all_items=["a", "b", "c", "d", "e"],
    items_per_phase=3,
    epochs_per_phase=100,
    reserved_items=["a"]
)

# Test alternating behavior
for epoch in [0, 100, 200, 300]:
    items = scheduler.get_items_for_epoch(epoch)
    print(f"Epoch {epoch}: {items}")
# Should show sequential, then random, then sequential, ...

# Test pose losses
from cryolens.training.pose_disentanglement import compute_pose_kld

pose_mu = torch.randn(32, 4)
pose_log_var = torch.randn(32, 4)
kld = compute_pose_kld(pose_mu, pose_log_var)
print(f"Pose KLD: {kld.item()}")
```

## Performance Considerations

### Memory

- Curriculum changes don't leak memory (old dataloaders are properly cleaned up)
- Index mapping is small (one int64 per structure)
- Pose losses have minimal memory overhead

### Speed

- Curriculum changes happen at epoch boundaries (negligible overhead)
- Index mapping is a simple lookup (O(1))
- Cycle consistency uses detached z (no extra backprop through decoder)

## Troubleshooting

### "No local_to_global_mapping found"

Call `setup_curriculum()` in your `__init__` before using `map_to_global_indices()`.

### Curriculum not changing

Check that:
- `on_train_epoch_start()` is being called (it should be automatic with PyTorch Lightning)
- Your epochs_per_phase matches your training schedule
- You're not accidentally creating a FixedCurriculumScheduler

### Pose not disentangling

Monitor these metrics:
- `pose_entropy`: Should stay > 0 (not collapsed)
- `pose_mu_variance`: Should be high across batch
- `cycle_loss`: Should decrease over time

If pose collapses:
- Increase initial_pose_beta
- Increase cycle_consistency_weight
- Check that pose_log_var initialization isn't too negative

## Future Enhancements

Potential additions to the curriculum framework:

1. **Difficulty-based curriculum**: Order structures by complexity
2. **Performance-based curriculum**: Adapt based on loss curves
3. **Multi-level curriculum**: Nested curricula (e.g., SNR + structures)
4. **Curriculum metrics**: Track learning progress per structure

Contributions welcome!
