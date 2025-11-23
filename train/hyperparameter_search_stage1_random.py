"""
Hyperparameter Search for Stage-1 Random Q-Former Training.

This script performs grid search or random search over hyperparameters
(excluding Q-Former architecture params like n_queries, num_layers, num_heads).

Search Space:
=============
- Learning rate: [1e-5, 5e-5, 1e-4]
- Batch size: [8, 16, 32]
- Weight decay: [0.0001, 0.001, 0.01]
- Task E focal gamma: [1.0, 1.5, 2.0]
- Task E focal alpha: [0.75, 0.85, 0.95]
- Task E positive weight: [1.0, 1.255, 2.0]
- Task S tau_head: [0.5, 1.0, 2.0]
- Task S tau_lq: [0.5, 1.0, 2.0]
- Task S teacher tau: [0.3, 0.5, 0.7]
- Task loss weights: w_task_e / w_task_s ratios
- Drop-LQ probability: [0.0, 0.05, 0.1]

Usage:
======
python train/hyperparameter_search_stage1_random.py --mode grid --max_trials 50
python train/hyperparameter_search_stage1_random.py --mode random --max_trials 100
"""

import sys
import argparse
import pickle
import random
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from itertools import product
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from train.stage1_train_random import (
    Stage1RandomConfig,
    Stage1RandomTrainer,
    Stage1RandomDataset,
    collate_stage1_random_batch,
    load_and_split_data,
)


class HyperparameterSearchConfig:
    """Configuration for hyperparameter search."""
    
    def __init__(self):
        # Search space definitions
        self.search_space = {
            # Optimization
            'lr': [1e-5, 5e-5, 1e-4],
            'batch_size': [8, 16, 32],
            'weight_decay': [0.0001, 0.001, 0.01],
            'warmup_ratio': [0.05, 0.1, 0.15],
            
            # Task E hyperparameters
            'task_e_tau': [0.3, 0.5, 0.7],
            'task_e_focal_gamma': [1.0, 1.5, 2.0],
            'task_e_focal_alpha': [0.75, 0.85, 0.95],
            'task_e_w_pos': [1.0, 1.255, 2.0],
            'task_e_w_longtail': [50.0, 100.0, 150.0],
            
            # Task S hyperparameters
            'task_s_tau_head': [0.5, 1.0, 2.0],
            'task_s_tau_lq': [0.5, 1.0, 2.0],
            'task_s_rho_top': [3.0, 5.0, 7.0],
            'task_s_l_prime': [5, 10, 15],
            'teacher_tau': [0.3, 0.5, 0.7],
            
            # Multi-task weights
            'w_task_e': [0.5, 1.0, 2.0],
            'w_task_s': [0.5, 1.0, 2.0],
            
            # Regularization
            'p_drop_lq_unified': [0.0, 0.05, 0.1, 0.15],
        }
        
        # Fixed parameters (Q-Former architecture - not searched)
        self.fixed_params = {
            'n_queries': 32,
            'hidden_dim': 768,
            'num_layers': 6,
            'num_heads': 8,
            'dropout': 0.1,
        }
        
        # Training settings
        self.num_epochs = 5  # Quick evaluation per trial
        self.max_steps = 10000  # Early stopping per trial
        self.data_path = r"D:\LLMs\DR-QFormer\DR-QFormer\smoking_64.pkl"
        self.val_split = 0.1
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.seed = 42


class HyperparameterSearch:
    """Hyperparameter search manager."""
    
    def __init__(self, config: HyperparameterSearchConfig, mode: str = 'random', max_trials: int = 50):
        """
        Args:
            config: Search configuration
            mode: 'grid' or 'random'
            max_trials: Maximum number of trials to run
        """
        self.config = config
        self.mode = mode
        self.max_trials = max_trials
        
        # Results tracking
        self.results = []
        self.best_params = None
        self.best_val_loss = float('inf')
        
        # Create results directory
        self.results_dir = Path("./hyperparameter_search_results")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data once
        print("="*80)
        print("Loading dataset for hyperparameter search...")
        print("="*80)
        self.train_data, self.val_data = load_and_split_data(
            config.data_path,
            val_split=config.val_split,
            shuffle=True,
            seed=config.seed,
        )
        
    def generate_trial_configs(self) -> List[Dict]:
        """Generate trial configurations based on search mode."""
        if self.mode == 'grid':
            return self._grid_search_configs()
        elif self.mode == 'random':
            return self._random_search_configs()
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
    
    def _grid_search_configs(self) -> List[Dict]:
        """Generate all possible combinations (grid search)."""
        keys = list(self.config.search_space.keys())
        values = [self.config.search_space[k] for k in keys]
        
        # Generate all combinations
        all_combinations = list(product(*values))
        
        # Shuffle and limit
        random.shuffle(all_combinations)
        all_combinations = all_combinations[:self.max_trials]
        
        # Convert to list of dicts
        configs = []
        for combo in all_combinations:
            config_dict = dict(zip(keys, combo))
            configs.append(config_dict)
        
        print(f"📊 Grid search: {len(configs)} configurations generated (from {len(list(product(*values)))} total)")
        return configs
    
    def _random_search_configs(self) -> List[Dict]:
        """Generate random combinations (random search)."""
        configs = []
        
        for _ in range(self.max_trials):
            config_dict = {}
            for key, values in self.config.search_space.items():
                config_dict[key] = random.choice(values)
            configs.append(config_dict)
        
        print(f"🎲 Random search: {len(configs)} configurations generated")
        return configs
    
    def create_config_from_params(self, params: Dict) -> Stage1RandomConfig:
        """Create Stage1RandomConfig from hyperparameter dict."""
        config = Stage1RandomConfig()
        
        # Update with searched hyperparameters
        for key, value in params.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        # Update with fixed parameters
        for key, value in self.config.fixed_params.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        # Override training settings for quick evaluation
        config.num_epochs = self.config.num_epochs
        config.max_steps = self.config.max_steps
        config.train_data_path = self.config.data_path
        config.val_split = self.config.val_split
        config.device = self.config.device
        config.seed = self.config.seed
        
        # Disable frequent checkpointing during search
        config.save_interval = 999999  # Don't save during search
        config.log_interval = 100
        config.eval_interval = 1000
        
        return config
    
    def evaluate_config(self, trial_id: int, params: Dict) -> Dict:
        """Evaluate a single hyperparameter configuration."""
        print(f"\n{'='*80}")
        print(f"Trial {trial_id + 1}/{self.max_trials}")
        print(f"{'='*80}")
        print("Hyperparameters:")
        for key, value in params.items():
            print(f"  {key}: {value}")
        print("="*80)
        
        try:
            # Create config
            config = self.create_config_from_params(params)
            
            # Create temporary save directory for this trial
            config.save_dir = str(self.results_dir / f"trial_{trial_id}")
            
            # Initialize trainer
            trainer = Stage1RandomTrainer(config)
            
            # Create datasets and dataloaders
            train_dataset = Stage1RandomDataset(self.train_data[0], self.train_data[1])
            val_dataset = Stage1RandomDataset(self.val_data[0], self.val_data[1])
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=config.batch_size,
                shuffle=True,
                collate_fn=collate_stage1_random_batch,
                num_workers=0,
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                collate_fn=collate_stage1_random_batch,
                num_workers=0,
            )
            
            # Training loop
            best_val_loss = float('inf')
            start_time = time.time()
            
            for epoch in range(config.num_epochs):
                # Train
                train_metrics = trainer.train_epoch(train_loader, epoch)
                
                # Validate
                val_metrics = trainer.evaluate(val_loader)
                
                print(f"\nEpoch {epoch + 1}/{config.num_epochs}:")
                print(f"  Train - Total: {train_metrics['total_loss']:.4f}, E: {train_metrics['task_e_loss']:.4f}, S: {train_metrics['task_s_loss']:.4f}")
                print(f"  Val   - Total: {val_metrics['val_loss']:.4f}, E: {val_metrics['val_task_e_loss']:.4f}, S: {val_metrics['val_task_s_loss']:.4f}")
                
                if val_metrics['val_loss'] < best_val_loss:
                    best_val_loss = val_metrics['val_loss']
                
                # Early stopping if reached max_steps
                if trainer.global_step >= config.max_steps:
                    break
            
            training_time = time.time() - start_time
            
            result = {
                'trial_id': trial_id,
                'params': params,
                'best_val_loss': best_val_loss,
                'final_train_loss': train_metrics['total_loss'],
                'final_val_loss': val_metrics['val_loss'],
                'final_val_task_e_loss': val_metrics['val_task_e_loss'],
                'final_val_task_s_loss': val_metrics['val_task_s_loss'],
                'training_time': training_time,
                'global_steps': trainer.global_step,
                'status': 'completed',
            }
            
            print(f"\n✅ Trial {trial_id + 1} completed:")
            print(f"   Best Val Loss: {best_val_loss:.4f}")
            print(f"   Training Time: {training_time:.1f}s")
            
            # Clean up GPU memory
            del trainer
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"\n❌ Trial {trial_id + 1} failed: {str(e)}")
            result = {
                'trial_id': trial_id,
                'params': params,
                'best_val_loss': float('inf'),
                'status': 'failed',
                'error': str(e),
            }
        
        return result
    
    def run_search(self):
        """Run the hyperparameter search."""
        print("\n" + "="*80)
        print("Starting Hyperparameter Search")
        print("="*80)
        print(f"Mode: {self.mode}")
        print(f"Max Trials: {self.max_trials}")
        print(f"Results Directory: {self.results_dir}")
        print("="*80)
        
        # Generate trial configurations
        trial_configs = self.generate_trial_configs()
        
        # Run trials
        for trial_id, params in enumerate(trial_configs):
            result = self.evaluate_config(trial_id, params)
            self.results.append(result)
            
            # Update best params
            if result['status'] == 'completed' and result['best_val_loss'] < self.best_val_loss:
                self.best_val_loss = result['best_val_loss']
                self.best_params = params
                print(f"\n🌟 New best configuration found! Val Loss: {self.best_val_loss:.4f}")
            
            # Save intermediate results
            self.save_results()
        
        # Final summary
        self.print_summary()
    
    def save_results(self):
        """Save search results to JSON."""
        results_file = self.results_dir / "search_results.json"
        
        save_data = {
            'mode': self.mode,
            'max_trials': self.max_trials,
            'completed_trials': len([r for r in self.results if r['status'] == 'completed']),
            'failed_trials': len([r for r in self.results if r['status'] == 'failed']),
            'best_val_loss': self.best_val_loss,
            'best_params': self.best_params,
            'all_results': self.results,
            'fixed_params': self.config.fixed_params,
        }
        
        with open(results_file, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        print(f"\n💾 Results saved to {results_file}")
    
    def print_summary(self):
        """Print search summary."""
        print("\n" + "="*80)
        print("Hyperparameter Search Summary")
        print("="*80)
        
        completed_results = [r for r in self.results if r['status'] == 'completed']
        failed_results = [r for r in self.results if r['status'] == 'failed']
        
        print(f"Total Trials: {len(self.results)}")
        print(f"Completed: {len(completed_results)}")
        print(f"Failed: {len(failed_results)}")
        
        if completed_results:
            print(f"\n🏆 Best Configuration:")
            print(f"   Val Loss: {self.best_val_loss:.4f}")
            print(f"\n   Hyperparameters:")
            for key, value in self.best_params.items():
                print(f"     {key}: {value}")
            
            # Top 5 configurations
            print(f"\n📊 Top 5 Configurations:")
            sorted_results = sorted(completed_results, key=lambda x: x['best_val_loss'])
            for i, result in enumerate(sorted_results[:5], 1):
                print(f"\n   {i}. Val Loss: {result['best_val_loss']:.4f}")
                print(f"      Trial ID: {result['trial_id']}")
                key_params = ['lr', 'batch_size', 'task_e_focal_gamma', 'task_s_tau_head', 'w_task_e', 'w_task_s']
                for key in key_params:
                    if key in result['params']:
                        print(f"      {key}: {result['params'][key]}")
        
        print("\n" + "="*80)
        print("Search completed!")
        print(f"Results saved in: {self.results_dir}")
        print("="*80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Hyperparameter Search for Stage-1 Random Q-Former')
    parser.add_argument('--mode', type=str, default='random', choices=['grid', 'random'],
                        help='Search mode: grid or random')
    parser.add_argument('--max_trials', type=int, default=50,
                        help='Maximum number of trials to run')
    parser.add_argument('--data_path', type=str, default=r"D:\LLMs\DR-QFormer\DR-QFormer\smoking_64.pkl",
                        help='Path to training data')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Create search config
    search_config = HyperparameterSearchConfig()
    search_config.data_path = args.data_path
    search_config.seed = args.seed
    
    # Run search
    search = HyperparameterSearch(
        config=search_config,
        mode=args.mode,
        max_trials=args.max_trials,
    )
    
    search.run_search()


if __name__ == "__main__":
    main()
