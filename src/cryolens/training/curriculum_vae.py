"""
Curriculum-aware VAE training mixin.

This module provides a mixin class that adds curriculum learning capabilities
to VAE training, handling dynamic dataloader updates and structure mapping.
"""

import json
import time
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable

import torch
from pytorch_lightning import LightningModule


class CurriculumVAEMixin:
    """Mixin to add curriculum learning capabilities to a VAE model.
    
    This mixin handles:
    1. Dynamic structure selection based on curriculum scheduler
    2. Dataloader recreation when curriculum changes
    3. Mapping between local dataloader indices and global similarity matrix indices
    4. Curriculum logging
    
    To use this mixin, your VAE class should:
    1. Inherit from both LightningModule and CurriculumVAEMixin
    2. Call setup_curriculum() in __init__
    3. Override train_dataloader() to use self.current_dataloader
    4. Use self.map_to_global_indices() in training_step before similarity loss
    
    Example
    -------
    >>> class MyCurriculumVAE(LightningModule, CurriculumVAEMixin):
    ...     def __init__(self, scheduler, ...):
    ...         super().__init__()
    ...         self.setup_curriculum(
    ...             scheduler=scheduler,
    ...             dataloader_factory=my_dataloader_factory,
    ...             valid_structure_ids=all_structures
    ...         )
    ...     
    ...     def training_step(self, batch, batch_idx):
    ...         img, local_mol_id = batch
    ...         global_mol_id = self.map_to_global_indices(local_mol_id)
    ...         # Use global_mol_id for similarity loss
    """
    
    def setup_curriculum(
        self,
        scheduler,
        dataloader_factory: Callable,
        valid_structure_ids: List[str],
        checkpoint_dir: Optional[Path] = None
    ):
        """Setup curriculum learning components.
        
        Parameters
        ----------
        scheduler : CurriculumScheduler
            Curriculum scheduler that controls structure selection
        dataloader_factory : Callable
            Factory function to create dataloaders. Should accept:
            - filtered_structure_ids: List[str]
            And return:
            - (dataloader, dataset) tuple
        valid_structure_ids : List[str]
            Global list of all valid structure IDs (for index mapping)
        checkpoint_dir : Optional[Path]
            Directory for saving curriculum logs
        """
        self.curriculum_scheduler = scheduler
        self.dataloader_factory = dataloader_factory
        self.valid_structure_ids = valid_structure_ids
        self.curriculum_checkpoint_dir = checkpoint_dir
        
        # State tracking
        self.current_structures: Optional[List[str]] = None
        self.current_dataloader = None
        self.current_dataset = None
        self._resuming_from_checkpoint = False
        self._resume_epoch: Optional[int] = None
        self.local_to_global_mapping: Optional[torch.Tensor] = None
    
    def on_train_epoch_start(self):
        """Called at the start of each training epoch.
        
        Checks if curriculum has changed and recreates dataloader if needed.
        """
        # Call parent's method if it exists
        if hasattr(super(), 'on_train_epoch_start'):
            super().on_train_epoch_start()
        
        current_epoch = self.current_epoch
        
        # Handle resume epoch override
        if current_epoch == 0 and self._resume_epoch is not None:
            epoch_to_use = self._resume_epoch
        else:
            epoch_to_use = current_epoch
        
        # Force scheduler update if resuming
        if self._resuming_from_checkpoint and epoch_to_use > 0:
            self.curriculum_scheduler.current_phase = -1
            self.curriculum_scheduler.current_items = None
        
        # Get structures for current epoch
        structures_for_epoch = self.curriculum_scheduler.get_items_for_epoch(epoch_to_use)
        
        # Check if we need to update the dataloader
        needs_update = (
            structures_for_epoch != self.current_structures or
            (self._resuming_from_checkpoint and epoch_to_use > 0)
        )
        
        if needs_update:
            if self._resuming_from_checkpoint:
                self._resuming_from_checkpoint = False
            
            self._update_dataloader(structures_for_epoch, epoch_to_use)
    
    def _update_dataloader(self, structures: List[str], epoch: int):
        """Update dataloader with new structure list.
        
        Parameters
        ----------
        structures : List[str]
            New list of structures to use
        epoch : int
            Current epoch number
        """
        self.current_structures = structures
        
        print(f"\nCurriculum change at epoch {epoch}")
        print(f"New structures: {structures}")
        
        # Create local-to-global index mapping
        self._create_index_mapping(structures)
        
        # Create new dataloader
        self.current_dataloader, self.current_dataset = self.dataloader_factory(
            filtered_structure_ids=structures
        )
        
        print(f"Dataloader updated with {len(structures)} structures")
        
        # Reload batch files if supported
        if hasattr(self.current_dataset, 'reload_batches'):
            self.current_dataset.reload_batches()
        
        # Log curriculum change
        if self.global_rank == 0 and self.curriculum_checkpoint_dir is not None:
            self._log_curriculum_change(epoch, structures)
    
    def _create_index_mapping(self, structures: List[str]):
        """Create mapping from local dataloader indices to global similarity matrix indices.
        
        Parameters
        ----------
        structures : List[str]
            Current structure list
        """
        device = self.device if hasattr(self, 'device') else torch.device('cpu')
        
        self.local_to_global_mapping = torch.zeros(
            len(structures),
            dtype=torch.long,
            device=device
        )
        
        for local_idx, structure_name in enumerate(structures):
            if structure_name in self.valid_structure_ids:
                global_idx = self.valid_structure_ids.index(structure_name)
                self.local_to_global_mapping[local_idx] = global_idx
            else:
                # Use -1 for items not in global list (e.g., "background")
                self.local_to_global_mapping[local_idx] = -1
        
        print("Local to global mapping:")
        for i, struct in enumerate(structures[:10]):  # Show first 10
            print(f"  Local {i} ({struct}) -> Global {self.local_to_global_mapping[i].item()}")
        if len(structures) > 10:
            print(f"  ... and {len(structures) - 10} more")
    
    def map_to_global_indices(self, local_indices: torch.Tensor) -> torch.Tensor:
        """Map local dataloader indices to global similarity matrix indices.
        
        Parameters
        ----------
        local_indices : torch.Tensor
            Local indices from dataloader
            
        Returns
        -------
        torch.Tensor
            Global indices for similarity matrix
        """
        if self.local_to_global_mapping is None:
            raise RuntimeError("Index mapping not initialized. Call setup_curriculum first.")
        
        return self.local_to_global_mapping[local_indices]
    
    def train_dataloader(self):
        """Return current dataloader.
        
        Initializes dataloader on first call or after curriculum change.
        """
        if self.current_dataloader is None:
            # Determine initial epoch
            if self._resume_epoch is not None:
                epoch_to_use = self._resume_epoch
            elif hasattr(self.curriculum_scheduler, 'resume_epoch') and \
                 self.curriculum_scheduler.resume_epoch is not None:
                epoch_to_use = self.curriculum_scheduler.resume_epoch
            else:
                epoch_to_use = 0
            
            # Initialize with first structure set
            structures = self.curriculum_scheduler.get_items_for_epoch(epoch_to_use)
            self._update_dataloader(structures, epoch_to_use)
            
            # Log initialization
            if self.global_rank == 0 and self.curriculum_checkpoint_dir is not None:
                self._log_curriculum_init(epoch_to_use, structures)
        
        return self.current_dataloader
    
    def _log_curriculum_change(self, epoch: int, structures: List[str]):
        """Log curriculum change to file.
        
        Parameters
        ----------
        epoch : int
            Current epoch
        structures : List[str]
            Structure list for this phase
        """
        phase = epoch // self.curriculum_scheduler.epochs_per_phase
        phase_type = "random" if (phase % 2) == 1 else "sequential"
        
        log_entry = {
            "epoch": epoch,
            "phase": phase,
            "phase_type": phase_type,
            "structures": structures,
            "num_structures": len(structures),
            "timestamp": time.time(),
            "event": "curriculum_change"
        }
        
        log_file = self.curriculum_checkpoint_dir / "curriculum_log.jsonl"
        with open(log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    
    def _log_curriculum_init(self, epoch: int, structures: List[str]):
        """Log curriculum initialization to file.
        
        Parameters
        ----------
        epoch : int
            Current epoch
        structures : List[str]
            Structure list for this phase
        """
        event = "dataloader_init" if epoch == 0 else "resume_dataloader_init"
        
        log_entry = {
            "epoch": epoch,
            "phase": epoch // self.curriculum_scheduler.epochs_per_phase,
            "structures": structures,
            "num_structures": len(structures),
            "timestamp": time.time(),
            "event": event
        }
        
        log_file = self.curriculum_checkpoint_dir / "curriculum_log.jsonl"
        with open(log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    
    def set_resume_epoch(self, epoch: int):
        """Set the epoch to resume from.
        
        Parameters
        ----------
        epoch : int
            Epoch number to resume from
        """
        self._resume_epoch = epoch
        self._resuming_from_checkpoint = True
        self.curriculum_scheduler.resume_epoch = epoch
