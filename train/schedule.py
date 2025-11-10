"""
Training Schedule Management for Joint E/S/C Training

Implements dynamic weight scheduling for:
- Task weights: w_E, w_S, w_C
- Ranking sub-losses: λ_teach, λ_post, λ_ent
- Optional: dual mode weights, attention consistency weights

Following BLIP-2 Stage-1 spirit: multi-objective with shared forward pass,
gradually shifting from prior (teacher) to posterior (LLM feedback).
"""

from typing import Dict, Optional
from dataclasses import dataclass
import math


@dataclass
class ScheduleConfig:
    """Configuration for training schedule."""
    # Total training steps
    total_steps: int
    
    # Phase boundaries (as fraction of total_steps)
    warmup_frac: float = 0.1      # Warm-up: prior-only, ~10%
    bridge_frac: float = 0.6      # Bridge: prior→posterior transition, ~60%
    closedloop_frac: float = 0.3  # Closed-loop: posterior-dominant, ~30%
    
    # Task weights (w_E, w_S, w_C)
    w_E_start: float = 1.0
    w_E_end: float = 0.5
    w_E_decay_frac: float = 0.3   # Decay over first 30% of training
    
    w_S: float = 1.0              # Constant
    
    w_C_start: float = 0.5
    w_C_end: float = 1.0
    w_C_ramp_frac: float = 0.3    # Ramp up over first 30% of training
    
    # Ranking sub-losses (Task S)
    lambda_teach_start: float = 1.0
    lambda_teach_end: float = 0.2
    
    lambda_post_start: float = 0.0
    lambda_post_end: float = 0.8
    
    lambda_ent_start: float = 0.01
    lambda_ent_end: float = 0.001
    
    # Optional: Dual mode (QG)
    lambda_dual: float = 0.2
    enable_dual: bool = False
    
    # Optional: Attention consistency (between QA and QG)
    lambda_ac: float = 0.05
    enable_ac: bool = False


class TrainingScheduler:
    """
    Manages dynamic weight scheduling throughout training.
    
    Phases:
    1. Warm-up (0-10%): Prior-only, stabilize E/S, C records NLL only
    2. Bridge (10-70%): Gradually shift from teacher to posterior
    3. Closed-loop (70-100%): Posterior-dominant, full feedback loop
    """
    
    def __init__(self, config: ScheduleConfig):
        self.config = config
        self.current_step = 0
        
        # Compute phase boundaries
        self.warmup_steps = int(config.total_steps * config.warmup_frac)
        self.bridge_steps = int(config.total_steps * config.bridge_frac)
        self.closedloop_steps = config.total_steps - self.warmup_steps - self.bridge_steps
        
        # Compute task weight transition boundaries
        self.w_E_decay_steps = int(config.total_steps * config.w_E_decay_frac)
        self.w_C_ramp_steps = int(config.total_steps * config.w_C_ramp_frac)
        
        print("=" * 80)
        print("Training Schedule Initialized")
        print("=" * 80)
        print(f"Total steps: {config.total_steps}")
        print(f"\nPhases:")
        print(f"  Warm-up:      [0, {self.warmup_steps})")
        print(f"  Bridge:       [{self.warmup_steps}, {self.warmup_steps + self.bridge_steps})")
        print(f"  Closed-loop:  [{self.warmup_steps + self.bridge_steps}, {config.total_steps})")
        print(f"\nTask weight transitions:")
        print(f"  w_E decay:    [0, {self.w_E_decay_steps})")
        print(f"  w_C ramp:     [0, {self.w_C_ramp_steps})")
        print(f"\nOptional features:")
        print(f"  Dual mode (QG):            {'Enabled' if config.enable_dual else 'Disabled'}")
        print(f"  Attention consistency:     {'Enabled' if config.enable_ac else 'Disabled'}")
        print("=" * 80)
    
    def step(self) -> None:
        """Advance to next training step."""
        self.current_step += 1
    
    def get_progress(self) -> float:
        """Get training progress as fraction [0, 1]."""
        return self.current_step / self.config.total_steps
    
    def get_phase(self) -> str:
        """Get current training phase."""
        if self.current_step < self.warmup_steps:
            return "warmup"
        elif self.current_step < self.warmup_steps + self.bridge_steps:
            return "bridge"
        else:
            return "closedloop"
    
    def get_task_weights(self) -> Dict[str, float]:
        """
        Get current task weights: w_E, w_S, w_C.
        
        Returns:
            Dict with keys: 'w_E', 'w_S', 'w_C'
        """
        # w_E: Linear decay over first w_E_decay_frac
        if self.current_step < self.w_E_decay_steps:
            t = self.current_step / self.w_E_decay_steps
            w_E = self.config.w_E_start + t * (self.config.w_E_end - self.config.w_E_start)
        else:
            w_E = self.config.w_E_end
        
        # w_S: Constant
        w_S = self.config.w_S
        
        # w_C: Linear ramp up over first w_C_ramp_frac
        if self.current_step < self.w_C_ramp_steps:
            t = self.current_step / self.w_C_ramp_steps
            w_C = self.config.w_C_start + t * (self.config.w_C_end - self.config.w_C_start)
        else:
            w_C = self.config.w_C_end
        
        return {
            'w_E': w_E,
            'w_S': w_S,
            'w_C': w_C,
        }
    
    def get_ranking_lambdas(self) -> Dict[str, float]:
        """
        Get current λ for Task S sub-losses: λ_teach, λ_post, λ_ent.
        
        Phase-specific behavior:
        - Warm-up: λ_teach=1.0, λ_post=0.0 (prior-only)
        - Bridge: Linear crossover from teacher to posterior
        - Closed-loop: λ_teach=0.2, λ_post=0.8 (posterior-dominant)
        
        Returns:
            Dict with keys: 'lambda_teach', 'lambda_post', 'lambda_ent'
        """
        phase = self.get_phase()
        
        if phase == "warmup":
            # Prior-only: teacher=1.0, posterior=0.0
            lambda_teach = self.config.lambda_teach_start
            lambda_post = self.config.lambda_post_start
        
        elif phase == "bridge":
            # Linear crossover
            t = (self.current_step - self.warmup_steps) / self.bridge_steps
            lambda_teach = self.config.lambda_teach_start + t * (self.config.lambda_teach_end - self.config.lambda_teach_start)
            lambda_post = self.config.lambda_post_start + t * (self.config.lambda_post_end - self.config.lambda_post_start)
        
        else:  # closedloop
            # Posterior-dominant
            lambda_teach = self.config.lambda_teach_end
            lambda_post = self.config.lambda_post_end
        
        # λ_ent: Exponential decay (annealing)
        t_global = self.get_progress()
        log_ratio = math.log(self.config.lambda_ent_end / self.config.lambda_ent_start)
        lambda_ent = self.config.lambda_ent_start * math.exp(log_ratio * t_global)
        
        return {
            'lambda_teach': lambda_teach,
            'lambda_post': lambda_post,
            'lambda_ent': lambda_ent,
        }
    
    def get_optional_weights(self) -> Dict[str, float]:
        """
        Get optional feature weights: λ_dual, λ_ac.
        
        Returns:
            Dict with keys: 'lambda_dual', 'lambda_ac', 'enable_dual', 'enable_ac'
        """
        return {
            'lambda_dual': self.config.lambda_dual if self.config.enable_dual else 0.0,
            'lambda_ac': self.config.lambda_ac if self.config.enable_ac else 0.0,
            'enable_dual': self.config.enable_dual,
            'enable_ac': self.config.enable_ac,
        }
    
    def should_enable_posterior(self) -> bool:
        """
        Whether to enable posterior extraction from Task C.
        
        Returns:
            True if in Bridge or Closed-loop phase
        """
        return self.get_phase() in ["bridge", "closedloop"]
    
    def get_all_weights(self) -> Dict[str, float]:
        """
        Get all current weights in one dict.
        
        Returns:
            Combined dict with task weights, ranking lambdas, and optional weights
        """
        weights = {}
        weights.update(self.get_task_weights())
        weights.update(self.get_ranking_lambdas())
        weights.update(self.get_optional_weights())
        weights['phase'] = self.get_phase()
        weights['progress'] = self.get_progress()
        weights['enable_posterior'] = self.should_enable_posterior()
        return weights
    
    def print_current_schedule(self) -> None:
        """Print current schedule state (for debugging/logging)."""
        weights = self.get_all_weights()
        
        print(f"\n{'=' * 80}")
        print(f"Training Schedule - Step {self.current_step}/{self.config.total_steps}")
        print(f"{'=' * 80}")
        phase_str = str(weights['phase']).upper() if isinstance(weights['phase'], str) else 'UNKNOWN'
        print(f"Phase: {phase_str} | Progress: {weights['progress']:.1%}")
        print()
        
        print("Task Weights:")
        print(f"  w_E (Entailment):  {weights['w_E']:.3f}")
        print(f"  w_S (Ranking):     {weights['w_S']:.3f}")
        print(f"  w_C (Contrastive): {weights['w_C']:.3f}")
        print()
        
        print("Ranking Sub-losses (Task S):")
        print(f"  λ_teach (Teacher):   {weights['lambda_teach']:.3f}")
        print(f"  λ_post (Posterior):  {weights['lambda_post']:.3f}")
        print(f"  λ_ent (Entropy):     {weights['lambda_ent']:.4f}")
        print()
        
        if weights['enable_dual'] or weights['enable_ac']:
            print("Optional Features:")
            if weights['enable_dual']:
                print(f"  λ_dual (Dual mode): {weights['lambda_dual']:.3f}")
            if weights['enable_ac']:
                print(f"  λ_ac (Attn consist): {weights['lambda_ac']:.3f}")
            print()
        
        print(f"Posterior enabled: {weights['enable_posterior']}")
        print("=" * 80)


# ============================================================================
# Preset schedules for common scenarios
# ============================================================================

def get_default_schedule(total_steps: int) -> ScheduleConfig:
    """
    Default schedule for joint E/S/C training.
    
    Recommended for most use cases.
    """
    return ScheduleConfig(
        total_steps=total_steps,
        warmup_frac=0.1,
        bridge_frac=0.6,
        closedloop_frac=0.3,
        w_E_start=1.0,
        w_E_end=0.5,
        w_E_decay_frac=0.3,
        w_S=1.0,
        w_C_start=0.5,
        w_C_end=1.0,
        w_C_ramp_frac=0.3,
        lambda_teach_start=1.0,
        lambda_teach_end=0.2,
        lambda_post_start=0.0,
        lambda_post_end=0.8,
        lambda_ent_start=0.01,
        lambda_ent_end=0.001,
        lambda_dual=0.2,
        enable_dual=False,
        lambda_ac=0.05,
        enable_ac=False,
    )


def get_fast_schedule(total_steps: int) -> ScheduleConfig:
    """
    Faster schedule: shorter warm-up, quicker posterior ramp.
    
    For smaller datasets or when teacher signal is strong.
    """
    return ScheduleConfig(
        total_steps=total_steps,
        warmup_frac=0.05,
        bridge_frac=0.5,
        closedloop_frac=0.45,
        w_E_start=1.0,
        w_E_end=0.3,
        w_E_decay_frac=0.2,
        w_S=1.0,
        w_C_start=0.5,
        w_C_end=1.0,
        w_C_ramp_frac=0.2,
        lambda_teach_start=1.0,
        lambda_teach_end=0.1,
        lambda_post_start=0.0,
        lambda_post_end=0.9,
        lambda_ent_start=0.02,
        lambda_ent_end=0.0005,
        lambda_dual=0.0,
        enable_dual=False,
        lambda_ac=0.0,
        enable_ac=False,
    )


def get_conservative_schedule(total_steps: int) -> ScheduleConfig:
    """
    Conservative schedule: longer warm-up, slower posterior ramp.
    
    For large datasets or when teacher signal is noisy.
    """
    return ScheduleConfig(
        total_steps=total_steps,
        warmup_frac=0.15,
        bridge_frac=0.7,
        closedloop_frac=0.15,
        w_E_start=1.0,
        w_E_end=0.7,
        w_E_decay_frac=0.4,
        w_S=1.0,
        w_C_start=0.3,
        w_C_end=1.0,
        w_C_ramp_frac=0.4,
        lambda_teach_start=1.0,
        lambda_teach_end=0.3,
        lambda_post_start=0.0,
        lambda_post_end=0.7,
        lambda_ent_start=0.005,
        lambda_ent_end=0.001,
        lambda_dual=0.2,
        enable_dual=False,
        lambda_ac=0.05,
        enable_ac=False,
    )


# ============================================================================
# Test schedule
# ============================================================================

if __name__ == "__main__":
    print("Testing TrainingScheduler...")
    
    # Create default schedule for 10000 steps
    config = get_default_schedule(total_steps=10000)
    scheduler = TrainingScheduler(config)
    
    # Test key checkpoints
    test_steps = [0, 500, 1000, 3000, 5000, 7000, 9000, 9999]
    
    for step in test_steps:
        scheduler.current_step = step
        scheduler.print_current_schedule()
    
    print("\n✅ Schedule test complete!")
    print("\nPreset schedules available:")
    print("  - get_default_schedule(total_steps)")
    print("  - get_fast_schedule(total_steps)")
    print("  - get_conservative_schedule(total_steps)")
