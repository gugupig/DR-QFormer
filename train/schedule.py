"""
Curriculum Learning Scheduler for Joint Training (E+S+C).

Implements dynamic weight scheduling for:
- Task weights (w_E, w_S, w_C)
- Task S curriculum (λ_teach → λ_post transition)
- Learning rate schedules (warmup + cosine annealing)
- LQ entropy regularization (optional, with decay)

Training Phases:
1. Warm-up (0-10%): Prior-only, E+S teacher, C disabled
2. Bridge (10-70%): Gradual posterior integration, λ_teach→λ_post
3. Closed-loop (70-100%): Posterior-dominated, λ_post high
"""

from typing import Dict, Optional
from dataclasses import dataclass
import math


@dataclass
class ScheduleConfig:
    """Configuration for curriculum schedule."""
    # Total training steps
    max_steps: int
    
    # Training phases (cumulative ratios)
    warmup_phase_end: float = 0.1  # 10%
    bridge_phase_end: float = 0.7  # 70%
    closed_loop_phase_end: float = 1.0  # 100%
    
    # Task weights
    w_E_start: float = 1.0
    w_E_end: float = 0.5
    w_E_phase_end: float = 0.3  # Decay in first 30%
    
    w_S_start: float = 1.0
    w_S_end: float = 1.0
    
    w_C_start: float = 0.5
    w_C_end: float = 1.0
    w_C_phase_end: float = 0.3  # Ramp up in first 30%
    
    # Task S curriculum
    lambda_teach_start: float = 1.0
    lambda_teach_end: float = 0.2
    lambda_post_start: float = 0.0
    lambda_post_end: float = 0.8
    lambda_entropy_start: float = 0.01
    lambda_entropy_end: float = 0.001
    
    # Optional LQ entropy reg per task
    enable_lq_entropy_task_e: bool = False
    lq_entropy_task_e_start: float = 0.005
    lq_entropy_task_e_end: float = 0.0005
    
    enable_lq_entropy_task_s: bool = True
    lq_entropy_task_s_start: float = 0.01
    lq_entropy_task_s_end: float = 0.001
    
    enable_lq_entropy_task_c: bool = False
    lq_entropy_task_c_start: float = 0.008
    lq_entropy_task_c_end: float = 0.0001


def linear_schedule(
    current_step: int,
    total_steps: int,
    start_value: float,
    end_value: float,
    phase_end_ratio: float = 1.0,
) -> float:
    """
    Linear interpolation schedule.
    
    Args:
        current_step: Current training step
        total_steps: Total training steps
        start_value: Initial value
        end_value: Final value
        phase_end_ratio: Ratio of total steps for this phase (default: 1.0 = full training)
    
    Returns:
        value: Interpolated value at current_step
    
    Example:
        >>> # Decay from 1.0 to 0.5 over first 30% of training
        >>> value = linear_schedule(1500, 10000, 1.0, 0.5, phase_end_ratio=0.3)
        >>> # At step 1500/10000 = 15%, progress within phase = 15%/30% = 0.5
        >>> # value = 1.0 + (0.5 - 1.0) * 0.5 = 0.75
    """
    # Compute phase end step
    phase_end_step = int(total_steps * phase_end_ratio)
    
    # Compute progress within phase (clamped to [0, 1])
    if current_step >= phase_end_step:
        progress = 1.0
    else:
        progress = current_step / max(phase_end_step, 1)
    
    # Linear interpolation
    value = start_value + (end_value - start_value) * progress
    
    return value


def cosine_schedule(
    current_step: int,
    total_steps: int,
    start_value: float,
    end_value: float,
    phase_end_ratio: float = 1.0,
) -> float:
    """
    Cosine annealing schedule (smooth decay).
    
    Args:
        current_step: Current training step
        total_steps: Total training steps
        start_value: Initial value
        end_value: Final value
        phase_end_ratio: Ratio of total steps for this phase
    
    Returns:
        value: Cosine-annealed value at current_step
    
    Formula:
        progress = current_step / phase_end_step
        value = end_value + 0.5 * (start_value - end_value) * (1 + cos(π * progress))
    """
    # Compute phase end step
    phase_end_step = int(total_steps * phase_end_ratio)
    
    # Compute progress within phase (clamped to [0, 1])
    if current_step >= phase_end_step:
        progress = 1.0
    else:
        progress = current_step / max(phase_end_step, 1)
    
    # Cosine annealing
    value = end_value + 0.5 * (start_value - end_value) * (1 + math.cos(math.pi * progress))
    
    return value


class JointTrainingScheduler:
    """
    Curriculum learning scheduler for joint training.
    
    Manages dynamic weight schedules for:
    - Task weights (w_E, w_S, w_C)
    - Task S curriculum (λ_teach, λ_post, λ_entropy)
    - LQ entropy regularization (per task, optional)
    
    Usage:
        >>> config = ScheduleConfig(max_steps=10000)
        >>> scheduler = JointTrainingScheduler(config)
        >>> 
        >>> for step in range(10000):
        >>>     weights = scheduler.get_weights(step)
        >>>     # Use weights['w_E'], weights['lambda_teach'], etc.
    """
    
    def __init__(self, config: ScheduleConfig):
        self.config = config
        self.current_step = 0
    
    def get_weights(self, step: Optional[int] = None) -> Dict[str, float]:
        """
        Get all weights for current step.
        
        Args:
            step: Current training step (if None, use self.current_step)
        
        Returns:
            dict with keys:
                # Task weights
                - w_E: Task E weight
                - w_S: Task S weight
                - w_C: Task C weight
                
                # Task S curriculum
                - lambda_teach: Teacher supervision weight
                - lambda_post: Posterior alignment weight
                - lambda_entropy: Tail entropy regularization weight
                
                # Training phase
                - phase: "warmup", "bridge", or "closed_loop"
                - progress: Overall training progress [0, 1]
                
                # Optional LQ entropy reg
                - lq_entropy_task_e: LQ entropy weight for Task E
                - lq_entropy_task_s: LQ entropy weight for Task S
                - lq_entropy_task_c: LQ entropy weight for Task C
        """
        if step is None:
            step = self.current_step
        else:
            self.current_step = step
        
        cfg = self.config
        total_steps = cfg.max_steps
        
        # Overall progress
        progress = step / max(total_steps, 1)
        
        # Determine training phase
        if progress < cfg.warmup_phase_end:
            phase = "warmup"
        elif progress < cfg.bridge_phase_end:
            phase = "bridge"
        else:
            phase = "closed_loop"
        
        # Task weights
        w_E = linear_schedule(
            step, total_steps,
            cfg.w_E_start, cfg.w_E_end,
            cfg.w_E_phase_end
        )
        
        w_S = cfg.w_S_start  # Constant
        
        w_C = linear_schedule(
            step, total_steps,
            cfg.w_C_start, cfg.w_C_end,
            cfg.w_C_phase_end
        )
        
        # Task S curriculum
        lambda_teach = linear_schedule(
            step, total_steps,
            cfg.lambda_teach_start, cfg.lambda_teach_end,
            phase_end_ratio=1.0  # Full training
        )
        
        lambda_post = linear_schedule(
            step, total_steps,
            cfg.lambda_post_start, cfg.lambda_post_end,
            phase_end_ratio=1.0
        )
        
        lambda_entropy = linear_schedule(
            step, total_steps,
            cfg.lambda_entropy_start, cfg.lambda_entropy_end,
            phase_end_ratio=1.0
        )
        
        # Optional LQ entropy reg per task
        lq_entropy_task_e = 0.0
        if cfg.enable_lq_entropy_task_e:
            lq_entropy_task_e = cosine_schedule(
                step, total_steps,
                cfg.lq_entropy_task_e_start, cfg.lq_entropy_task_e_end,
                phase_end_ratio=1.0
            )
        
        lq_entropy_task_s = 0.0
        if cfg.enable_lq_entropy_task_s:
            lq_entropy_task_s = cosine_schedule(
                step, total_steps,
                cfg.lq_entropy_task_s_start, cfg.lq_entropy_task_s_end,
                phase_end_ratio=1.0
            )
        
        lq_entropy_task_c = 0.0
        if cfg.enable_lq_entropy_task_c:
            # Fast decay for Task C (avoid posterior conflict)
            lq_entropy_task_c = cosine_schedule(
                step, total_steps,
                cfg.lq_entropy_task_c_start, cfg.lq_entropy_task_c_end,
                phase_end_ratio=0.3  # Decay quickly in first 30%
            )
        
        # Warm-up phase: Disable Task C posterior extraction
        if phase == "warmup":
            w_C = 0.0  # Record NLL but don't use loss
            lambda_post = 0.0  # No posterior feedback yet
        
        return {
            # Task weights
            'w_E': w_E,
            'w_S': w_S,
            'w_C': w_C,
            
            # Task S curriculum
            'lambda_teach': lambda_teach,
            'lambda_post': lambda_post,
            'lambda_entropy': lambda_entropy,
            
            # Training phase
            'phase': phase,
            'progress': progress,
            
            # Optional LQ entropy reg
            'lq_entropy_task_e': lq_entropy_task_e,
            'lq_entropy_task_s': lq_entropy_task_s,
            'lq_entropy_task_c': lq_entropy_task_c,
        }
    
    def step(self):
        """Increment step counter."""
        self.current_step += 1
        return self.get_weights()


def get_lr_schedule(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
):
    """
    Create learning rate scheduler with cosine annealing after warmup.
    
    Args:
        optimizer: PyTorch optimizer
        num_warmup_steps: Number of warmup steps (linear increase)
        num_training_steps: Total training steps
        num_cycles: Number of cosine cycles (default: 0.5 = half cycle)
        last_epoch: Last epoch index (for resuming)
    
    Returns:
        lr_scheduler: PyTorch LR scheduler
    
    Schedule:
        - 0 to warmup_steps: Linear increase from 0 to base_lr
        - warmup_steps to total: Cosine decay from base_lr to 0
    """
    try:
        from torch.optim.lr_scheduler import LambdaLR
    except ImportError:
        print("Warning: PyTorch not available, returning None")
        return None
    
    def lr_lambda(current_step: int):
        # Warmup phase: Linear increase
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        # Cosine annealing phase
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress)))
    
    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


# ===================================================================
# Example Usage
# ===================================================================
if __name__ == "__main__":
    print("="*80)
    print("Joint Training Curriculum Scheduler Demo")
    print("="*80)
    
    # Create config
    config = ScheduleConfig(
        max_steps=10000,
        warmup_phase_end=0.1,
        bridge_phase_end=0.7,
    )
    
    # Create scheduler
    scheduler = JointTrainingScheduler(config)
    
    # Test key milestones
    milestones = [0, 500, 1000, 3000, 7000, 9999]
    
    print("\nWeight Schedule at Key Milestones:")
    print("-"*80)
    print(f"{'Step':<8} {'Phase':<12} {'w_E':<8} {'w_S':<8} {'w_C':<8} "
          f"{'λ_teach':<10} {'λ_post':<10}")
    print("-"*80)
    
    for step in milestones:
        weights = scheduler.get_weights(step)
        print(f"{step:<8} {weights['phase']:<12} {weights['w_E']:<8.3f} "
              f"{weights['w_S']:<8.3f} {weights['w_C']:<8.3f} "
              f"{weights['lambda_teach']:<10.3f} {weights['lambda_post']:<10.3f}")
    
    print("-"*80)
    print("\nPhase Descriptions:")
    print(f"  Warm-up (0-{int(config.max_steps*config.warmup_phase_end)}): "
          f"Prior-only, C disabled")
    print(f"  Bridge ({int(config.max_steps*config.warmup_phase_end)}-"
          f"{int(config.max_steps*config.bridge_phase_end)}): "
          f"λ_teach→λ_post transition")
    print(f"  Closed-loop ({int(config.max_steps*config.bridge_phase_end)}-"
          f"{config.max_steps}): "
          f"Posterior-dominated")
    print("="*80)
