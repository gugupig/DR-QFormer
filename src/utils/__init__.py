"""Utility modules for DR-QFormer."""

try:
    from .masks import (
        build_sa_mask,
        build_ca_mask,
        build_padding_mask,
        build_causal_mask,
        combine_masks,
    )
    from .checkpoint import save_checkpoint, load_checkpoint, get_trainable_state_dict
    from .macs import (
        compute_macs_to_lqs,
        extract_answer_lq_posterior,
        compute_evidence_posterior,
        compute_evidence_posterior_from_ca,
        compute_evidence_posterior_from_ca_macs,  # Default: MACS algorithm for CA
        extract_span_indices,
        extract_posterior_from_llm_outputs,
    )
except ImportError:
    pass

__all__ = [
    "build_sa_mask",
    "build_ca_mask",
    "build_padding_mask",
    "build_causal_mask",
    "combine_masks",
    "save_checkpoint",
    "load_checkpoint",
    "get_trainable_state_dict",
    "compute_macs_to_lqs",
    "extract_answer_lq_posterior",
    "compute_evidence_posterior",
    "compute_evidence_posterior_from_ca",
    "compute_evidence_posterior_from_ca_macs",  # Default: MACS algorithm for CA
    "extract_span_indices",
    "extract_posterior_from_llm_outputs",
]
