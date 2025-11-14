"""
Training script for joint multi-task DR-QFormer (E+S+C).

Usage:
    python scripts/train_joint.py --config configs/joint_train.yaml

Features:
    - YAML configuration loading
    - Automatic checkpoint saving/resuming
    - Logging to tensorboard and weights & biases
    - Multi-GPU support (coming soon)
"""

import sys
import argparse
from pathlib import Path
from typing import Optional

# Add parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import torch
    import yaml
    from torch.utils.data import DataLoader
    
    from train.task_joint import JointTrainer, JointTrainingConfig
    from train.joint_data import JointTrainingDataset, create_joint_dataloaders
    from train.schedule import ScheduleConfig
    
    print("✅ Successfully imported all modules")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def create_trainer_config(yaml_config: dict) -> JointTrainingConfig:
    """Convert YAML config to JointTrainingConfig."""
    model_cfg = yaml_config['model']
    training_cfg = yaml_config['training']
    
    # Build ScheduleConfig
    schedule_config = ScheduleConfig(
        max_steps=training_cfg['max_steps'],
        warmup_phase_end=yaml_config['loss_weights'].get('warmup_phase_end', 0.1),
        bridge_phase_end=yaml_config['loss_weights'].get('bridge_phase_end', 0.7),
        
        # Task weights
        w_E_start=yaml_config['loss_weights']['w_E']['start'],
        w_E_end=yaml_config['loss_weights']['w_E']['end'],
        w_E_phase_end_ratio=yaml_config['loss_weights']['w_E'].get('phase_end_ratio', 0.3),
        
        w_S_start=yaml_config['loss_weights']['w_S']['start'],
        w_S_end=yaml_config['loss_weights']['w_S']['end'],
        
        w_C_start=yaml_config['loss_weights']['w_C']['start'],
        w_C_end=yaml_config['loss_weights']['w_C']['end'],
        w_C_phase_end_ratio=yaml_config['loss_weights']['w_C'].get('phase_end_ratio', 0.3),
        
        # Curriculum
        lambda_teach_start=yaml_config['curriculum_task_s']['lambda_teach']['start'],
        lambda_teach_end=yaml_config['curriculum_task_s']['lambda_teach']['end'],
        lambda_teach_phase_end_ratio=yaml_config['curriculum_task_s']['lambda_teach'].get('phase_end_ratio', 0.7),
        
        lambda_post_start=yaml_config['curriculum_task_s']['lambda_post']['start'],
        lambda_post_end=yaml_config['curriculum_task_s']['lambda_post']['end'],
        lambda_post_phase_end_ratio=yaml_config['curriculum_task_s']['lambda_post'].get('phase_end_ratio', 0.7),
        
        lambda_entropy_start=yaml_config['curriculum_task_s']['lambda_entropy']['start'],
        lambda_entropy_end=yaml_config['curriculum_task_s']['lambda_entropy']['end'],
        lambda_entropy_phase_end_ratio=yaml_config['curriculum_task_s']['lambda_entropy'].get('phase_end_ratio', 0.7),
        
        # LQ entropy reg
        lq_entropy_task_e_start=yaml_config.get('lq_entropy_regularization', {}).get('task_e', {}).get('start', 0.0),
        lq_entropy_task_e_end=yaml_config.get('lq_entropy_regularization', {}).get('task_e', {}).get('end', 0.0),
        lq_entropy_task_e_phase_end_ratio=yaml_config.get('lq_entropy_regularization', {}).get('task_e', {}).get('phase_end_ratio', 1.0),
        
        lq_entropy_task_s_start=yaml_config.get('lq_entropy_regularization', {}).get('task_s', {}).get('start', 0.0),
        lq_entropy_task_s_end=yaml_config.get('lq_entropy_regularization', {}).get('task_s', {}).get('end', 0.0),
        lq_entropy_task_s_phase_end_ratio=yaml_config.get('lq_entropy_regularization', {}).get('task_s', {}).get('phase_end_ratio', 1.0),
        
        lq_entropy_task_c_start=yaml_config.get('lq_entropy_regularization', {}).get('task_c', {}).get('start', 0.0),
        lq_entropy_task_c_end=yaml_config.get('lq_entropy_regularization', {}).get('task_c', {}).get('end', 0.0),
        lq_entropy_task_c_phase_end_ratio=yaml_config.get('lq_entropy_regularization', {}).get('task_c', {}).get('phase_end_ratio', 1.0),
    )
    
    # Build JointTrainingConfig
    trainer_config = JointTrainingConfig(
        # Model
        n_queries=model_cfg['n_queries'],
        hidden_dim=model_cfg['hidden_dim'],
        num_layers=model_cfg['num_layers'],
        num_heads=model_cfg['num_heads'],
        
        # Task E
        task_e_tau=model_cfg['task_e']['tau'],
        task_e_p_drop_lq=model_cfg['task_e']['p_drop_lq'],
        task_e_focal_gamma=model_cfg['task_e']['focal_gamma'],
        task_e_focal_alpha=model_cfg['task_e']['focal_alpha'],
        
        # Task S
        task_s_tau_head=model_cfg['task_s']['tau_head'],
        task_s_tau_lq=model_cfg['task_s']['tau_lq'],
        task_s_rho_top=model_cfg['task_s']['rho_top'],
        task_s_l_prime=model_cfg['task_s']['l_prime'],
        
        # Task C
        task_c_llm_hidden_dim=model_cfg['task_c']['llm_hidden_dim'],
        task_c_softplus_beta=model_cfg['task_c']['softplus_beta'],
        task_c_margin_mode=yaml_config['margin_adaptive']['mode'],
        task_c_margin_fixed=yaml_config['margin_adaptive']['fixed_margin'],
        task_c_margin_adaptive_ratio=yaml_config['margin_adaptive']['adaptive_ratio'],
        task_c_margin_min=yaml_config['margin_adaptive']['min_margin'],
        task_c_margin_max=yaml_config['margin_adaptive']['max_margin'],
        
        # Training
        lr=training_cfg['optimizer']['lr'],
        weight_decay=training_cfg['optimizer']['weight_decay'],
        max_steps=training_cfg['max_steps'],
        batch_size=training_cfg['batch_size'],
        gradient_accumulation_steps=training_cfg['gradient_accumulation_steps'],
        max_grad_norm=training_cfg['max_grad_norm'],
        
        # Schedule
        schedule_config=schedule_config,
        
        # Device
        device=yaml_config['hardware']['device'],
    )
    
    return trainer_config


def train(config_path: str, resume_from: Optional[str] = None):
    """Main training loop."""
    print("="*80)
    print("Joint Training (E+S+C) - Main Entry Point")
    print("="*80)
    
    # Load config
    yaml_config = load_config(config_path)
    trainer_config = create_trainer_config(yaml_config)
    
    print(f"\n✅ Loaded config from: {config_path}")
    print(f"   Max steps: {trainer_config.max_steps}")
    print(f"   Batch size: {trainer_config.batch_size}")
    print(f"   Device: {trainer_config.device}")
    
    # Create trainer
    trainer = JointTrainer(trainer_config)
    
    # TODO: Load checkpoint if resume_from is specified
    if resume_from is not None:
        print(f"\n⏳ TODO: Resuming from checkpoint: {resume_from}")
    
    # Create datasets
    data_cfg = yaml_config['data']
    print(f"\n✅ Creating datasets...")
    print(f"   Train: {data_cfg['train_json']}")
    print(f"   Dev: {data_cfg['dev_json']}")
    
    # TODO: Replace with actual dataset loading
    print("\n⚠️  Using placeholder datasets (not implemented yet)")
    print("   To implement: Load from JSON files specified in config")
    
    # Create dummy dataloaders for testing
    train_loader = create_joint_dataloaders(
        train_json=data_cfg['train_json'],
        dev_json=data_cfg['dev_json'],
        batch_size=trainer_config.batch_size,
        max_fragments=data_cfg['max_fragments'],
        split='train'
    )
    
    dev_loader = create_joint_dataloaders(
        train_json=data_cfg['train_json'],
        dev_json=data_cfg['dev_json'],
        batch_size=trainer_config.batch_size,
        max_fragments=data_cfg['max_fragments'],
        split='dev'
    )
    
    # Training loop
    print("\n" + "="*80)
    print("Starting Training")
    print("="*80)
    
    num_epochs = (trainer_config.max_steps // len(train_loader)) + 1
    
    for epoch in range(1, num_epochs + 1):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch}/{num_epochs}")
        print(f"{'='*80}")
        
        epoch_metrics = trainer.train_epoch(train_loader, epoch)
        
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Total Loss: {epoch_metrics['loss_total']:.4f}")
        print(f"  Task E Loss: {epoch_metrics['loss_e']:.4f}")
        print(f"  Task S Loss: {epoch_metrics['loss_s']:.4f}")
        print(f"  Task C Loss: {epoch_metrics['loss_c']:.4f}")
        print(f"  Phase: {epoch_metrics['phase']}")
        print(f"  w_E={epoch_metrics['w_E']:.3f}, w_S={epoch_metrics['w_S']:.3f}, w_C={epoch_metrics['w_C']:.3f}")
        print(f"  λ_teach={epoch_metrics['lambda_teach']:.3f}, λ_post={epoch_metrics['lambda_post']:.3f}")
        
        # TODO: Evaluation
        # print(f"\nRunning evaluation...")
        # eval_metrics = trainer.evaluate(dev_loader)
        
        # TODO: Checkpoint saving
        # if epoch % save_every == 0:
        #     save_checkpoint(trainer, epoch, checkpoint_dir)
        
        if trainer.global_step >= trainer_config.max_steps:
            print(f"\n✅ Reached max steps ({trainer_config.max_steps}), stopping training.")
            break
    
    print("\n" + "="*80)
    print("Training Complete")
    print("="*80)


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Train joint multi-task DR-QFormer (E+S+C)")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/joint_train.yaml',
        help='Path to YAML configuration file'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    
    args = parser.parse_args()
    
    # Validate config file exists
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        print(f"   Please create a config file or specify a valid path with --config")
        sys.exit(1)
    
    # Start training
    train(str(config_path), resume_from=args.resume)


if __name__ == "__main__":
    main()
