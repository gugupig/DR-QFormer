"""
Hyperparameter Search for TASK E Training

This script performs grid search or random search over key hyperparameters
to find the best configuration for TASK E (entailment classification).

Key hyperparameters to search:
- Learning rate
- Batch size
- Focal loss alpha (class balance)
- Focal loss gamma (hard example mining)
- Positive class weight
- Drop-LQ probability
"""

import sys
import pickle
import random
import itertools
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field, asdict
import json

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from train.task_e_only import (
    TaskEConfig, 
    TaskETrainer, 
    SmokingDataset, 
    collate_task_e_batch,
    load_and_split_data
)


@dataclass
class SearchConfig:
    """Configuration for hyperparameter search."""
    # Data
    train_data_path: str = r"D:\LLMs\DR-QFormer\DR-QFormer\ms_xlm_embeddings.pkl"
    val_split: float = 0.1
    
    # Search space (lists of values to try)
    lr_space: List[float] = field(default_factory=lambda: [1e-5, 3e-5, 5e-5, 1e-4])
    batch_size_space: List[int] = field(default_factory=lambda: [8, 16, 32])
    focal_alpha_space: List[float] = field(default_factory=lambda: [0.5, 0.7, 0.85, 0.95])
    focal_gamma_space: List[float] = field(default_factory=lambda: [1.5, 2.0, 3.0])
    w_pos_space: List[float] = field(default_factory=lambda: [1.0, 5.0, 10.0, 20.0])
    drop_lq_space: List[float] = field(default_factory=lambda: [0.0, 0.1, 0.2])
    warmup_ratio_space: List[float] = field(default_factory=lambda: [0.05, 0.1, 0.15])
    
    # Search strategy
    search_type: str = "random"  # "grid" or "random"
    n_random_trials: int = 20  # Number of random trials if search_type="random"
    
    # Training settings (keep short for search)
    num_epochs: int = 2
    max_steps: int = 2000
    early_stop_patience: int = 5  # Stop if no improvement for N eval intervals
    eval_interval: int = 200
    
    # Fixed hyperparameters (not searched)
    n_queries: int = 16  # Smaller for faster search
    use_ca_layers: Optional[List[int]] = field(default_factory=lambda: [0, 3, 6, 9])
    
    # Output
    results_dir: str = "./search_results"
    save_top_k: int = 5  # Save top K configurations
    
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42


def analyze_data_distribution(data_path: str) -> Dict[str, float]:
    """Analyze label distribution in the dataset."""
    print(f"\n{'='*80}")
    print("📊 Analyzing Label Distribution")
    print(f"{'='*80}")
    
    with open(data_path, 'rb') as f:
        data_dict = pickle.load(f)
    
    total_pos = 0
    total_neg = 0
    total_samples = len(data_dict)
    
    for sample in data_dict.values():
        labels = sample['evidence_labels']
        total_pos += (labels == 1).sum()
        total_neg += (labels == 0).sum()
    
    total = total_pos + total_neg
    pos_ratio = total_pos / total
    
    print(f"\n📈 Dataset Statistics:")
    print(f"   Total samples: {total_samples:,}")
    print(f"   Total fragments: {total:,}")
    print(f"   Positive fragments: {total_pos:,} ({pos_ratio*100:.2f}%)")
    print(f"   Negative fragments: {total_neg:,} ({(1-pos_ratio)*100:.2f}%)")
    print(f"   Imbalance ratio: 1:{(total_neg/total_pos):.2f}")
    
    print(f"\n💡 Recommended Hyperparameters (based on distribution):")
    recommended_alpha = min(0.95, 0.5 + pos_ratio * 2)  # Favor positive class
    recommended_w_pos = min(50.0, 1.0 / pos_ratio)
    print(f"   focal_alpha: {recommended_alpha:.3f}")
    print(f"   w_pos: {recommended_w_pos:.2f}")
    print(f"{'='*80}\n")
    
    return {
        'pos_ratio': pos_ratio,
        'total_samples': total_samples,
        'total_pos': int(total_pos),
        'total_neg': int(total_neg),
        'recommended_alpha': recommended_alpha,
        'recommended_w_pos': recommended_w_pos,
    }


def create_trial_config(base_params: Dict, search_config: SearchConfig) -> TaskEConfig:
    """Create a TaskEConfig from trial parameters."""
    return TaskEConfig(
        # Data
        train_data_path=search_config.train_data_path,
        val_split=search_config.val_split,
        use_precomputed_embeddings=True,
        
        # Model (fixed for search)
        n_queries=search_config.n_queries,
        use_ca_layers=search_config.use_ca_layers,
        
        # Searched hyperparameters
        lr=base_params['lr'],
        batch_size=base_params['batch_size'],
        task_e_focal_alpha=base_params['focal_alpha'],
        task_e_focal_gamma=base_params['focal_gamma'],
        task_e_w_pos=base_params['w_pos'],
        p_drop_lq_unified=base_params['drop_lq'],
        warmup_ratio=base_params['warmup_ratio'],
        
        # Training settings
        num_epochs=search_config.num_epochs,
        max_steps=search_config.max_steps,
        save_interval=search_config.max_steps + 1,  # Don't save during search
        
        # Device
        device=search_config.device,
        seed=search_config.seed,
    )


def evaluate_configuration(
    config: TaskEConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    trial_num: int,
    total_trials: int,
    search_config: SearchConfig,
) -> Dict:
    """Train and evaluate a single configuration."""
    print(f"\n{'='*80}")
    print(f"🔬 Trial {trial_num}/{total_trials}")
    print(f"{'='*80}")
    print(f"Configuration:")
    print(f"  lr={config.lr:.1e}, batch_size={config.batch_size}, "
          f"alpha={config.task_e_focal_alpha:.2f}, gamma={config.task_e_focal_gamma:.1f}")
    print(f"  w_pos={config.task_e_w_pos:.2f}, drop_lq={config.p_drop_lq_unified:.2f}, "
          f"warmup={config.warmup_ratio:.2f}")
    
    # Create trainer
    trainer = TaskETrainer(config)
    
    # Training loop with early stopping
    best_val_loss = float('inf')
    no_improve_count = 0
    train_losses = []
    val_losses = []
    
    for epoch in range(config.num_epochs):
        # Train epoch
        trainer.qformer.train()
        trainer.head_e.train()
        
        epoch_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False)
        for batch in pbar:
            if trainer.global_step >= config.max_steps:
                break
            
            metrics = trainer.train_step(batch)
            epoch_loss += metrics['loss']
            num_batches += 1
            
            pbar.set_postfix({'loss': f"{metrics['loss']:.4f}"})
            
            # Evaluate periodically
            if trainer.global_step % search_config.eval_interval == 0:
                val_metrics = trainer.evaluate(val_loader)
                val_loss = val_metrics['val_loss']
                val_losses.append(val_loss)
                
                print(f"  Step {trainer.global_step}: val_loss={val_loss:.4f}")
                
                # Check for improvement
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                
                # Early stopping
                if no_improve_count >= search_config.early_stop_patience:
                    print(f"  ⚠️ Early stopping: no improvement for {search_config.early_stop_patience} intervals")
                    break
        
        avg_train_loss = epoch_loss / max(num_batches, 1)
        train_losses.append(avg_train_loss)
        
        if trainer.global_step >= config.max_steps or no_improve_count >= search_config.early_stop_patience:
            break
    
    # Final evaluation
    final_val_metrics = trainer.evaluate(val_loader)
    final_val_loss = final_val_metrics['val_loss']
    
    print(f"✅ Trial complete: best_val_loss={best_val_loss:.4f}, final_val_loss={final_val_loss:.4f}")
    
    return {
        'best_val_loss': best_val_loss,
        'final_val_loss': final_val_loss,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'total_steps': trainer.global_step,
        'converged': no_improve_count < search_config.early_stop_patience,
    }


def grid_search(search_config: SearchConfig) -> List[Dict]:
    """Perform grid search over hyperparameter space."""
    # Generate all combinations
    param_combinations = list(itertools.product(
        search_config.lr_space,
        search_config.batch_size_space,
        search_config.focal_alpha_space,
        search_config.focal_gamma_space,
        search_config.w_pos_space,
        search_config.drop_lq_space,
        search_config.warmup_ratio_space,
    ))
    
    print(f"\n🔍 Grid Search: {len(param_combinations)} total combinations")
    
    results = []
    
    # Load data once
    train_data, val_data = load_and_split_data(
        search_config.train_data_path,
        val_split=search_config.val_split,
        shuffle=True,
        seed=search_config.seed,
    )
    
    for idx, params in enumerate(param_combinations, 1):
        lr, batch_size, focal_alpha, focal_gamma, w_pos, drop_lq, warmup_ratio = params
        
        # Create trial config
        base_params = {
            'lr': lr,
            'batch_size': batch_size,
            'focal_alpha': focal_alpha,
            'focal_gamma': focal_gamma,
            'w_pos': w_pos,
            'drop_lq': drop_lq,
            'warmup_ratio': warmup_ratio,
        }
        
        config = create_trial_config(base_params, search_config)
        
        # Create datasets and dataloaders
        train_dataset = SmokingDataset(
            train_data[0], train_data[1],
            tokenizer=None, xlm_model=None,
            device=config.device,
            use_precomputed_embeddings=True
        )
        val_dataset = SmokingDataset(
            val_data[0], val_data[1],
            tokenizer=None, xlm_model=None,
            device=config.device,
            use_precomputed_embeddings=True
        )
        
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size,
            shuffle=True, collate_fn=collate_task_e_batch, num_workers=0
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size,
            shuffle=False, collate_fn=collate_task_e_batch, num_workers=0
        )
        
        # Evaluate configuration
        try:
            eval_results = evaluate_configuration(
                config, train_loader, val_loader,
                idx, len(param_combinations), search_config
            )
            
            results.append({
                'trial': idx,
                'params': base_params,
                'metrics': eval_results,
            })
        except Exception as e:
            print(f"❌ Trial {idx} failed: {e}")
            results.append({
                'trial': idx,
                'params': base_params,
                'metrics': {'error': str(e)},
            })
    
    return results


def random_search(search_config: SearchConfig) -> List[Dict]:
    """Perform random search over hyperparameter space."""
    print(f"\n🎲 Random Search: {search_config.n_random_trials} trials")
    
    results = []
    
    # Load data once
    train_data, val_data = load_and_split_data(
        search_config.train_data_path,
        val_split=search_config.val_split,
        shuffle=True,
        seed=search_config.seed,
    )
    
    for trial_idx in range(search_config.n_random_trials):
        # Sample random hyperparameters
        base_params = {
            'lr': random.choice(search_config.lr_space),
            'batch_size': random.choice(search_config.batch_size_space),
            'focal_alpha': random.choice(search_config.focal_alpha_space),
            'focal_gamma': random.choice(search_config.focal_gamma_space),
            'w_pos': random.choice(search_config.w_pos_space),
            'drop_lq': random.choice(search_config.drop_lq_space),
            'warmup_ratio': random.choice(search_config.warmup_ratio_space),
        }
        
        config = create_trial_config(base_params, search_config)
        
        # Create datasets and dataloaders
        train_dataset = SmokingDataset(
            train_data[0], train_data[1],
            tokenizer=None, xlm_model=None,
            device=config.device,
            use_precomputed_embeddings=True
        )
        val_dataset = SmokingDataset(
            val_data[0], val_data[1],
            tokenizer=None, xlm_model=None,
            device=config.device,
            use_precomputed_embeddings=True
        )
        
        train_loader = DataLoader(
            train_dataset, batch_size=base_params['batch_size'],
            shuffle=True, collate_fn=collate_task_e_batch, num_workers=0
        )
        val_loader = DataLoader(
            val_dataset, batch_size=base_params['batch_size'],
            shuffle=False, collate_fn=collate_task_e_batch, num_workers=0
        )
        
        # Evaluate configuration
        try:
            eval_results = evaluate_configuration(
                config, train_loader, val_loader,
                trial_idx + 1, search_config.n_random_trials, search_config
            )
            
            results.append({
                'trial': trial_idx + 1,
                'params': base_params,
                'metrics': eval_results,
            })
        except Exception as e:
            print(f"❌ Trial {trial_idx + 1} failed: {e}")
            results.append({
                'trial': trial_idx + 1,
                'params': base_params,
                'metrics': {'error': str(e)},
            })
    
    return results


def save_and_analyze_results(results: List[Dict], search_config: SearchConfig, data_stats: Dict):
    """Save results and generate analysis."""
    results_dir = Path(search_config.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Filter out failed trials
    valid_results = [r for r in results if 'error' not in r['metrics']]
    
    if not valid_results:
        print("\n❌ No valid results to analyze!")
        return
    
    # Sort by best validation loss
    valid_results.sort(key=lambda x: x['metrics']['best_val_loss'])
    
    # Save all results
    with open(results_dir / 'all_results.json', 'w') as f:
        json.dump({
            'search_config': asdict(search_config),
            'data_stats': data_stats,
            'results': results,
        }, f, indent=2)
    
    # Save top K results
    top_k = valid_results[:search_config.save_top_k]
    with open(results_dir / 'top_k_results.json', 'w') as f:
        json.dump(top_k, f, indent=2)
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"📊 Search Results Summary")
    print(f"{'='*80}")
    print(f"Total trials: {len(results)}")
    print(f"Valid trials: {len(valid_results)}")
    print(f"Failed trials: {len(results) - len(valid_results)}")
    
    print(f"\n🏆 Top {len(top_k)} Configurations:")
    print(f"{'='*80}")
    
    for rank, result in enumerate(top_k, 1):
        params = result['params']
        metrics = result['metrics']
        
        print(f"\nRank {rank}:")
        print(f"  Trial: {result['trial']}")
        print(f"  Best val loss: {metrics['best_val_loss']:.4f}")
        print(f"  Converged: {'✅' if metrics['converged'] else '⚠️'}")
        print(f"  Parameters:")
        print(f"    lr={params['lr']:.1e}, batch_size={params['batch_size']}")
        print(f"    focal_alpha={params['focal_alpha']:.2f}, focal_gamma={params['focal_gamma']:.1f}")
        print(f"    w_pos={params['w_pos']:.2f}, drop_lq={params['drop_lq']:.2f}")
        print(f"    warmup_ratio={params['warmup_ratio']:.2f}")
    
    # Generate config for best model
    best_params = top_k[0]['params']
    best_config_code = f"""
# Best Configuration (from hyperparameter search)
config = TaskEConfig(
    # Learning
    lr={best_params['lr']},
    batch_size={best_params['batch_size']},
    warmup_ratio={best_params['warmup_ratio']},
    
    # TASK E Loss
    task_e_focal_alpha={best_params['focal_alpha']},
    task_e_focal_gamma={best_params['focal_gamma']},
    task_e_w_pos={best_params['w_pos']},
    
    # Regularization
    p_drop_lq_unified={best_params['drop_lq']},
    
    # Training
    num_epochs=10,  # Increase for final training
    max_steps=50000,
)
"""
    
    with open(results_dir / 'best_config.py', 'w') as f:
        f.write(best_config_code)
    
    print(f"\n💾 Results saved to: {results_dir}")
    print(f"   - all_results.json: All trial results")
    print(f"   - top_k_results.json: Top {search_config.save_top_k} configurations")
    print(f"   - best_config.py: Best configuration code")
    print(f"{'='*80}\n")


def main():
    """Main hyperparameter search."""
    # Create search configuration
    search_config = SearchConfig(
        # Search space (adjust as needed)
        lr_space=[1e-5, 3e-5, 5e-5],
        batch_size_space=[8, 16],
        focal_alpha_space=[0.7, 0.85, 0.95],
        focal_gamma_space=[2.0, 3.0],
        w_pos_space=[5.0, 10.0, 20.0],
        drop_lq_space=[0.0, 0.1],
        warmup_ratio_space=[0.05, 0.1],
        
        # Search strategy
        search_type="random",  # "grid" or "random"
        n_random_trials=15,  # Number of random trials
        
        # Training (short for search)
        num_epochs=2,
        max_steps=2000,
        eval_interval=200,
        early_stop_patience=5,
    )
    
    print(f"\n{'='*80}")
    print(f"🚀 TASK E Hyperparameter Search")
    print(f"{'='*80}")
    print(f"Search type: {search_config.search_type}")
    print(f"Device: {search_config.device}")
    
    # Analyze data distribution first
    data_stats = analyze_data_distribution(search_config.train_data_path)
    
    # Perform search
    if search_config.search_type == "grid":
        results = grid_search(search_config)
    else:
        results = random_search(search_config)
    
    # Save and analyze results
    save_and_analyze_results(results, search_config, data_stats)
    
    print("\n✅ Hyperparameter search complete!")


if __name__ == "__main__":
    main()
