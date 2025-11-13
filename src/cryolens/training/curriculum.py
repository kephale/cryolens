"""
Curriculum learning strategies for training.

This module provides curriculum schedulers that control which data structures
are used during different phases of training.
"""

import json
import random
import time
from pathlib import Path
from typing import List, Optional, Set


class CurriculumScheduler:
    """Manages alternating curriculum for structure selection.
    
    This scheduler alternates between sequential and random selection phases,
    ensuring all structures are seen systematically while also providing
    variety through random sampling.
    
    Parameters
    ----------
    all_items : List[str]
        Full list of available items (e.g., PDB codes)
    items_per_phase : int
        Number of items to use per phase (default: 6)
    epochs_per_phase : int
        Number of epochs per phase (default: 300)
    reserved_items : Optional[List[str]]
        Items that should always be included (e.g., "background")
    """
    
    def __init__(
        self,
        all_items: List[str],
        items_per_phase: int = 6,
        epochs_per_phase: int = 300,
        reserved_items: Optional[List[str]] = None
    ):
        """Initialize curriculum scheduler."""
        self.all_items = all_items
        self.items_per_phase = items_per_phase
        self.epochs_per_phase = epochs_per_phase
        self.reserved_items = reserved_items or []
        
        self.current_phase = -1
        self.current_items = None
        self.used_sequential_items: Set[str] = set()
        self.curriculum_log = []
        self.resume_epoch = None
        
        # Separate reserved items from regular items
        self.regular_items = [item for item in all_items if item not in self.reserved_items]
        
        # Calculate phases
        total_items = len(self.regular_items)
        available_slots = max(1, items_per_phase - len(self.reserved_items))
        self.max_sequential_phases = (total_items + available_slots - 1) // available_slots
        
        print(f"Curriculum Scheduler initialized:")
        print(f"  Total items: {len(self.all_items)}")
        print(f"  Regular items: {len(self.regular_items)}")
        print(f"  Reserved items: {self.reserved_items}")
        print(f"  Items per phase: {items_per_phase}")
        print(f"  Epochs per phase: {epochs_per_phase}")
        print(f"  Max sequential phases: {self.max_sequential_phases}")
    
    def restore_from_log(self, log_file_path: str, current_epoch: int):
        """Restore scheduler state from curriculum log file.
        
        Parameters
        ----------
        log_file_path : str
            Path to curriculum log file
        current_epoch : int
            Current epoch to restore to
        """
        try:
            if not Path(log_file_path).exists():
                print(f"No existing curriculum log found at {log_file_path}")
                return
            
            print(f"Restoring curriculum state from {log_file_path}")
            with open(log_file_path, 'r') as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line)
                        self.curriculum_log.append(entry)
            
            # Set current phase based on epoch
            self.current_phase = current_epoch // self.epochs_per_phase
            
            # Restore used sequential items
            for entry in self.curriculum_log:
                if entry.get('phase_type') == 'sequential':
                    items = entry.get('items', [])
                    self.used_sequential_items.update(items)
            
            # Get current items for the epoch
            self.current_items = self.get_items_for_epoch(current_epoch)
            
            print(f"Restored curriculum state:")
            print(f"  Current epoch: {current_epoch}")
            print(f"  Current phase: {self.current_phase}")
            print(f"  Used sequential items: {len(self.used_sequential_items)}")
            print(f"  Current items: {self.current_items}")
            
        except Exception as e:
            print(f"Warning: Could not restore curriculum log: {e}")
            print("Starting with fresh curriculum state")
    
    def get_items_for_epoch(self, epoch: int) -> List[str]:
        """Get the items to use for a given epoch.
        
        Parameters
        ----------
        epoch : int
            Current training epoch
            
        Returns
        -------
        List[str]
            List of items to use for this epoch
        """
        # Handle resume epoch override
        if epoch == 0 and self.resume_epoch is not None:
            epoch = self.resume_epoch
        
        phase = epoch // self.epochs_per_phase
        
        # Return cached items if we're in the same phase
        if phase == self.current_phase and self.current_items is not None:
            return self.current_items
        
        # Determine if this is a sequential or random phase
        is_random_phase = (phase % 2) == 1
        
        # Calculate number of regular items to select
        num_regular_items = max(1, self.items_per_phase - len(self.reserved_items))
        
        if is_random_phase:
            # Random phase: select items with replacement
            random.seed(phase * 12345)  # Deterministic per phase
            selected_items = random.sample(self.regular_items, num_regular_items)
            phase_type = "random"
        else:
            # Sequential phase: select next unused items
            available_items = [
                item for item in self.regular_items 
                if item not in self.used_sequential_items
            ]
            
            if len(available_items) < num_regular_items:
                # Reset if not enough unused items
                self.used_sequential_items.clear()
                available_items = self.regular_items
            
            # Select next batch
            selected_items = available_items[:num_regular_items]
            
            # Mark as used
            self.used_sequential_items.update(selected_items)
            
            phase_type = "sequential"
        
        # Combine reserved and selected items
        items = list(self.reserved_items) + selected_items
        
        # Update state if phase changed
        if phase != self.current_phase:
            self.current_phase = phase
            self.current_items = items
            
            log_entry = {
                "epoch": epoch,
                "phase": phase,
                "phase_type": phase_type,
                "items": items,
                "regular_items": selected_items,
                "reserved_items": self.reserved_items,
                "timestamp": time.time()
            }
            self.curriculum_log.append(log_entry)
            
            print(f"\nCURRICULUM CHANGE - Epoch {epoch}:")
            print(f"  Phase: {phase} ({phase_type})")
            print(f"  Items: {items}")
            print(f"  Phase duration: epochs {epoch} to {epoch + self.epochs_per_phase - 1}")
            if phase_type == "sequential":
                print(f"  Used sequential items so far: {len(self.used_sequential_items)}/{len(self.regular_items)}")
        else:
            self.current_items = items
        
        return items
    
    def save_curriculum_log(self, filepath: str):
        """Save curriculum log to JSON file.
        
        Parameters
        ----------
        filepath : str
            Path to save the log file
        """
        with open(filepath, 'w') as f:
            json.dump(self.curriculum_log, f, indent=2)
        print(f"Curriculum log saved to: {filepath}")


class FixedCurriculumScheduler:
    """A simple scheduler that uses a fixed set of items throughout training.
    
    This is useful for baseline training or when you don't want curriculum learning.
    
    Parameters
    ----------
    items : List[str]
        Fixed list of items to use throughout training
    """
    
    def __init__(self, items: List[str]):
        """Initialize fixed curriculum scheduler."""
        self.items = items
        self.current_items = items
        self.curriculum_log = []
        
        print(f"Fixed Curriculum Scheduler initialized:")
        print(f"  Using fixed set of {len(items)} items")
    
    def get_items_for_epoch(self, epoch: int) -> List[str]:
        """Get the items to use for a given epoch.
        
        Parameters
        ----------
        epoch : int
            Current training epoch
            
        Returns
        -------
        List[str]
            List of items (always the same for fixed curriculum)
        """
        return self.items
    
    def restore_from_log(self, log_file_path: str, current_epoch: int):
        """Restore scheduler state (no-op for fixed curriculum)."""
        pass
    
    def save_curriculum_log(self, filepath: str):
        """Save curriculum log (no-op for fixed curriculum)."""
        pass
