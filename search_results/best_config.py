
# Best Configuration (from hyperparameter search)
config = TaskEConfig(
    # Learning
    lr=1e-05,
    batch_size=16,
    warmup_ratio=0.05,
    
    # TASK E Loss
    task_e_focal_alpha=0.85,
    task_e_focal_gamma=3.0,
    task_e_w_pos=10.0,
    
    # Regularization
    p_drop_lq_unified=0.0,
    
    # Training
    num_epochs=10,  # Increase for final training
    max_steps=50000,
)
