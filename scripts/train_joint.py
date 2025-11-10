"""
Joint Training Script for DR-QFormer

Tasks: E (Entailment), S (Ranking), C (Contrastive NLL)

Usage:
    python scripts/train_joint.py --config configs/joint_train.yaml

Based on BLIP-2 Stage-1 philosophy:
- Shared Q-Former forward for all tasks
- Multi-objective training with dynamic scheduling
- Gradual shift from prior (teacher) to posterior (LLM feedback)
"""

import argparse
import yaml
import torch
import random
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from train.joint_data import create_joint_dataloader
from train.schedule import TrainingScheduler, ScheduleConfig
from train.task_joint import JointTrainer
from src.models.qformer import DRQFormer
from src.models.heads import EntailmentHead, FragmentRankingHead


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> dict:
    """Load YAML config."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_model(config: dict) -> DRQFormer:
    """Create DR-QFormer model."""
    model_cfg = config['model']
    
    model = DRQFormer(
        d=model_cfg['d'],
        N=model_cfg['num_lqs'],
        num_heads=model_cfg['num_heads'],
        num_blocks=model_cfg['num_blocks'],
        mlp_ratio=model_cfg['mlp_ratio'],
        num_lm_tokens=model_cfg['num_lm_tokens'],
        d_lm=model_cfg['d_lm'],
    )
    
    print(f"✅ DR-QFormer created: {sum(p.numel() for p in model.parameters()):,} params")
    return model


def create_heads(config: dict) -> tuple:
    """Create Task E and Task S heads."""
    model_cfg = config['model']
    heads_cfg = config['heads']
    
    # Task E Head
    task_e_head = EntailmentHead(
        d=model_cfg['d'],
        num_heads=model_cfg['num_heads'],
        aggregation_mode=heads_cfg['entailment']['aggregation_mode'],
        lse_r=heads_cfg['entailment']['lse_r'],
    )
    
    # Task S Head
    task_s_head = FragmentRankingHead(
        d=model_cfg['d'],
        num_heads=model_cfg['num_heads'],
        aggregation_mode=heads_cfg['ranking']['aggregation_mode'],
        head_lse_r=heads_cfg['ranking']['head_lse_r'],
        lq_lse_r=heads_cfg['ranking']['lq_lse_r'],
    )
    
    e_params = sum(p.numel() for p in task_e_head.parameters())
    s_params = sum(p.numel() for p in task_s_head.parameters())
    
    print(f"✅ Task E Head created: {e_params:,} params")
    print(f"✅ Task S Head created: {s_params:,} params")
    
    return task_e_head, task_s_head


def create_llm_placeholder(config: dict):
    """
    PLACEHOLDER: Create LLM for Task C.
    
    TODO: Replace with actual LLM loading:
    - Load from HuggingFace
    - Apply quantization (8-bit/4-bit)
    - Freeze parameters
    - Set device_map
    """
    llm_cfg = config['llm']
    
    print("⚠️  PLACEHOLDER: LLM not loaded")
    print(f"   TODO: Load {llm_cfg['model_name']}")
    print(f"   TODO: Set load_in_8bit={llm_cfg['load_in_8bit']}")
    print(f"   TODO: Freeze={llm_cfg['freeze']}")
    
    # Return dummy model
    import torch.nn as nn
    return nn.Identity()


def create_tokenizer_placeholder(config: dict):
    """
    PLACEHOLDER: Create tokenizer for Task C.
    
    TODO: Replace with actual tokenizer:
    - Load from HuggingFace
    - Set padding/truncation
    """
    tokenizer_cfg = config['data']['tokenizer']
    
    print("⚠️  PLACEHOLDER: Tokenizer not loaded")
    print(f"   TODO: Load {tokenizer_cfg['model_name']}")
    
    return None


def create_optimizer(model, task_e_head, task_s_head, config: dict):
    """Create optimizer."""
    opt_cfg = config['optimizer']
    
    # Collect all trainable parameters
    params = (
        list(model.parameters()) +
        list(task_e_head.parameters()) +
        list(task_s_head.parameters())
    )
    
    if opt_cfg['type'] == 'adamw':
        optimizer = torch.optim.AdamW(
            params,
            lr=opt_cfg['lr'],
            weight_decay=opt_cfg['weight_decay'],
            betas=tuple(opt_cfg['betas']),
            eps=opt_cfg['eps'],
        )
    else:
        raise ValueError(f"Unknown optimizer: {opt_cfg['type']}")
    
    print(f"✅ Optimizer created: {opt_cfg['type'].upper()}, lr={opt_cfg['lr']}")
    
    return optimizer


def create_scheduler(config: dict) -> TrainingScheduler:
    """Create training schedule."""
    sched_cfg = config['schedule']
    
    schedule_config = ScheduleConfig(
        total_steps=sched_cfg['total_steps'],
        warmup_frac=sched_cfg['warmup_frac'],
        bridge_frac=sched_cfg['bridge_frac'],
        closedloop_frac=sched_cfg['closedloop_frac'],
        w_E_start=sched_cfg['w_E_start'],
        w_E_end=sched_cfg['w_E_end'],
        w_E_decay_frac=sched_cfg['w_E_decay_frac'],
        w_S=sched_cfg['w_S'],
        w_C_start=sched_cfg['w_C_start'],
        w_C_end=sched_cfg['w_C_end'],
        w_C_ramp_frac=sched_cfg['w_C_ramp_frac'],
        lambda_teach_start=config['losses']['ranking']['lambda_teach_start'],
        lambda_teach_end=config['losses']['ranking']['lambda_teach_end'],
        lambda_post_start=config['losses']['ranking']['lambda_post_start'],
        lambda_post_end=config['losses']['ranking']['lambda_post_end'],
        lambda_ent_start=config['losses']['ranking']['lambda_ent_start'],
        lambda_ent_end=config['losses']['ranking']['lambda_ent_end'],
        lambda_dual=sched_cfg['lambda_dual'],
        enable_dual=sched_cfg['enable_dual'],
        lambda_ac=sched_cfg['lambda_ac'],
        enable_ac=sched_cfg['enable_ac'],
    )
    
    scheduler = TrainingScheduler(schedule_config)
    
    print("✅ Training scheduler created")
    
    return scheduler


def main(args):
    """Main training function."""
    print("=" * 80)
    print("DR-QFormer Joint Training (E/S/C)")
    print("=" * 80)
    
    # Load config
    config = load_config(args.config)
    print(f"✅ Config loaded from: {args.config}")
    
    # Set seed
    set_seed(config['seed'])
    print(f"✅ Random seed set: {config['seed']}")
    
    # Device
    device = torch.device(config['hardware']['device'])
    print(f"✅ Device: {device}")
    
    # Create model components
    print("\n" + "=" * 80)
    print("Creating Model Components")
    print("=" * 80)
    
    model = create_model(config)
    task_e_head, task_s_head = create_heads(config)
    llm_model = create_llm_placeholder(config)
    tokenizer = create_tokenizer_placeholder(config)
    
    # Create optimizer
    optimizer = create_optimizer(model, task_e_head, task_s_head, config)
    
    # Create scheduler
    scheduler = create_scheduler(config)
    
    # Create dataloaders
    print("\n" + "=" * 80)
    print("Creating DataLoaders")
    print("=" * 80)
    
    train_loader = create_joint_dataloader(
        data_path=config['data']['train_path'],
        batch_size=config['data']['batch_size'],
        max_fragments=config['data']['max_fragments'],
        num_workers=config['data']['num_workers'],
        shuffle=True,
    )
    
    val_loader = create_joint_dataloader(
        data_path=config['data']['val_path'],
        batch_size=config['validation']['batch_size'],
        max_fragments=config['validation']['max_fragments'],
        num_workers=config['data']['num_workers'],
        shuffle=False,
    )
    
    print(f"✅ Train batches: {len(train_loader)}")
    print(f"✅ Val batches: {len(val_loader)}")
    
    # Create trainer
    print("\n" + "=" * 80)
    print("Creating Trainer")
    print("=" * 80)
    
    trainer = JointTrainer(
        model=model,
        task_e_head=task_e_head,
        task_s_head=task_s_head,
        llm_model=llm_model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        focal_gamma=config['losses']['focal']['gamma'],
        focal_alpha=config['losses']['focal']['alpha'],
        softplus_beta=config['losses']['contrastive']['softplus_beta'],
        adaptive_margin=config['losses']['contrastive']['adaptive_margin'],
        margin_kappa=config['losses']['contrastive']['margin_kappa'],
        drop_lq_rate=config['model']['drop_lq_rate'],
        lq_entropy_reg_lambda=config['model']['lq_entropy_lambda'],
        log_interval=config['training']['log_interval'],
        save_dir=config['training']['save_dir'],
    )
    
    # Training loop
    print("\n" + "=" * 80)
    print("Starting Training")
    print("=" * 80)
    
    num_epochs = config['training']['num_epochs']
    
    for epoch in range(1, num_epochs + 1):
        print(f"\n{'=' * 80}")
        print(f"Epoch {epoch}/{num_epochs}")
        print(f"{'=' * 80}")
        
        # Train
        train_metrics = trainer.train_epoch(train_loader)
        
        print(f"\n[Epoch {epoch}] Train Results:")
        print(f"  Loss: {train_metrics['loss_total']:.4f}")
        print(f"  Task E: {train_metrics['loss_e']:.4f} | Acc: {train_metrics['e_accuracy']:.3f}")
        print(f"  Task S: {train_metrics['loss_s']:.4f} | NDCG: {train_metrics['s_ndcg']:.3f}")
        print(f"  Task C: {train_metrics['loss_c']:.4f} | Gain: {train_metrics['c_gain']:.3f}")
        
        # Save checkpoint
        if epoch % (config['training']['save_every_steps'] // len(train_loader)) == 0:
            trainer.save_checkpoint()
        
        # TODO: Add validation loop
        # TODO: Add early stopping
        # TODO: Add learning rate scheduling
    
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"Final checkpoint saved in: {config['training']['save_dir']}")
    
    print("\n⚠️  Remember to implement:")
    print("   - Task C dual-path Teacher Forcing")
    print("   - LLM integration")
    print("   - Posterior extraction qψ_U")
    print("   - Validation loop")
    print("   - Early stopping")
    print("   - Learning rate scheduling")
    print("   - Experiment tracking (wandb/tensorboard)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Joint Training for DR-QFormer")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/joint_train.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    
    args = parser.parse_args()
    
    main(args)
