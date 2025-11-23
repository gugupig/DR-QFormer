"""
Unit tests for MACS posterior extraction functions.

Tests:
1. compute_macs_to_lqs - attention aggregation
2. extract_answer_lq_posterior - SA part
3. compute_evidence_posterior - CA part
4. extract_posterior_from_llm_outputs - end-to-end
"""

import torch
import pytest
from src.utils.macs import (
    compute_macs_to_lqs,
    extract_answer_lq_posterior,
    compute_evidence_posterior,
    extract_posterior_from_llm_outputs,
)


class TestMACSFunctions:
    """Test suite for MACS posterior extraction."""
    
    @pytest.fixture
    def dummy_attentions(self):
        """Create dummy attention tensors (3 layers, 4 heads, 64 seq_len)."""
        batch_size = 2
        num_layers = 3
        num_heads = 4
        seq_len = 64
        
        attentions = []
        for _ in range(num_layers):
            # Random attention weights (should sum to 1 in last dim)
            att = torch.rand(batch_size, num_heads, seq_len, seq_len)
            att = att / att.sum(dim=-1, keepdim=True)  # Normalize
            attentions.append(att)
        
        return tuple(attentions)
    
    def test_compute_macs_to_lqs_shape(self, dummy_attentions):
        """Test MACS aggregation returns correct shape."""
        num_lqs = 32
        batch_size = 2
        seq_len = 64
        
        macs_map = compute_macs_to_lqs(
            attentions=dummy_attentions,
            num_lqs=num_lqs,
            alpha=0.8,
            use_zscore=True,
        )
        
        # Should return [batch, seq_len, num_lqs]
        assert macs_map.shape == (batch_size, seq_len, num_lqs)
    
    def test_compute_macs_to_lqs_no_nan(self, dummy_attentions):
        """Test MACS doesn't produce NaN or Inf."""
        macs_map = compute_macs_to_lqs(
            attentions=dummy_attentions,
            num_lqs=32,
            alpha=0.8,
            use_zscore=True,
        )
        
        assert not torch.isnan(macs_map).any()
        assert not torch.isinf(macs_map).any()
    
    def test_compute_macs_alpha_effect(self, dummy_attentions):
        """Test alpha parameter affects results."""
        macs_low_alpha = compute_macs_to_lqs(
            attentions=dummy_attentions,
            num_lqs=32,
            alpha=0.2,  # More smoothing
            use_zscore=False,
        )
        
        macs_high_alpha = compute_macs_to_lqs(
            attentions=dummy_attentions,
            num_lqs=32,
            alpha=0.9,  # Less smoothing
            use_zscore=False,
        )
        
        # Results should differ
        assert not torch.allclose(macs_low_alpha, macs_high_alpha)
    
    def test_extract_answer_lq_posterior_shape(self, dummy_attentions):
        """Test answer→LQ posterior extraction."""
        batch_size = 2
        num_lqs = 32
        answer_start = 40
        answer_end = 60
        
        lq_posterior = extract_answer_lq_posterior(
            attentions=dummy_attentions,
            answer_start_idx=answer_start,
            answer_end_idx=answer_end,
            num_lqs=num_lqs,
            aggregation='mean',
        )
        
        # Should return [batch, num_lqs]
        assert lq_posterior.shape == (batch_size, num_lqs)
    
    def test_extract_answer_lq_posterior_aggregation_modes(self, dummy_attentions):
        """Test different aggregation modes produce different results."""
        answer_start = 40
        answer_end = 60
        num_lqs = 32
        
        lq_mean = extract_answer_lq_posterior(
            attentions=dummy_attentions,
            answer_start_idx=answer_start,
            answer_end_idx=answer_end,
            num_lqs=num_lqs,
            aggregation='mean',
        )
        
        lq_max = extract_answer_lq_posterior(
            attentions=dummy_attentions,
            answer_start_idx=answer_start,
            answer_end_idx=answer_end,
            num_lqs=num_lqs,
            aggregation='max',
        )
        
        lq_sum = extract_answer_lq_posterior(
            attentions=dummy_attentions,
            answer_start_idx=answer_start,
            answer_end_idx=answer_end,
            num_lqs=num_lqs,
            aggregation='sum',
        )
        
        # All three should differ
        assert not torch.allclose(lq_mean, lq_max)
        assert not torch.allclose(lq_mean, lq_sum)
        assert not torch.allclose(lq_max, lq_sum)
        
        # Max should be >= mean
        # (Not always true with z-score, but rough check)
        # assert (lq_max >= lq_mean).float().mean() > 0.4
    
    def test_compute_evidence_posterior_shape(self):
        """Test evidence posterior computation shape."""
        batch_size = 2
        num_lqs = 32
        num_evidence = 50
        
        lq_posterior = torch.randn(batch_size, num_lqs)
        ca_weights = torch.randn(batch_size, num_lqs, num_evidence)
        
        evidence_posterior = compute_evidence_posterior(
            lq_posterior=lq_posterior,
            ca_weights=ca_weights,
            temperature=1.0,
        )
        
        # Should return [batch, num_evidence]
        assert evidence_posterior.shape == (batch_size, num_evidence)
    
    def test_compute_evidence_posterior_is_distribution(self):
        """Test evidence posterior is valid probability distribution."""
        batch_size = 4
        num_lqs = 32
        num_evidence = 64
        
        lq_posterior = torch.randn(batch_size, num_lqs).softmax(dim=-1)
        ca_weights = torch.randn(batch_size, num_lqs, num_evidence)
        
        evidence_posterior = compute_evidence_posterior(
            lq_posterior=lq_posterior,
            ca_weights=ca_weights,
            temperature=1.0,
        )
        
        # Should be non-negative
        assert (evidence_posterior >= 0).all()
        
        # Should sum to 1 (probability distribution)
        sums = evidence_posterior.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(batch_size), atol=1e-5)
    
    def test_compute_evidence_posterior_temperature_effect(self):
        """Test temperature affects distribution sharpness."""
        batch_size = 2
        num_lqs = 32
        num_evidence = 64
        
        lq_posterior = torch.randn(batch_size, num_lqs).softmax(dim=-1)
        ca_weights = torch.randn(batch_size, num_lqs, num_evidence)
        
        # Low temperature = sharper (more confident)
        evidence_sharp = compute_evidence_posterior(
            lq_posterior=lq_posterior,
            ca_weights=ca_weights,
            temperature=0.5,
        )
        
        # High temperature = flatter (less confident)
        evidence_flat = compute_evidence_posterior(
            lq_posterior=lq_posterior,
            ca_weights=ca_weights,
            temperature=2.0,
        )
        
        # Entropy should be lower for sharper distribution
        entropy_sharp = -(evidence_sharp * (evidence_sharp + 1e-8).log()).sum(dim=-1)
        entropy_flat = -(evidence_flat * (evidence_flat + 1e-8).log()).sum(dim=-1)
        
        assert (entropy_sharp < entropy_flat).all()
    
    def test_extract_posterior_from_llm_outputs(self, dummy_attentions):
        """Test end-to-end posterior extraction."""
        batch_size = 2
        num_lqs = 32
        num_evidence = 50
        
        # Mock LLM outputs
        llm_outputs = {
            'attentions': dummy_attentions,
            'answer_start_idx': 40,
            'answer_end_idx': 60,
        }
        
        # Mock Q-Former CA weights
        ca_weights = torch.randn(batch_size, num_lqs, num_evidence)
        ca_weights = ca_weights.softmax(dim=-1)  # Normalize
        
        # Extract posterior
        evidence_posterior = extract_posterior_from_llm_outputs(
            llm_outputs=llm_outputs,
            qformer_ca_weights=ca_weights,
            subset_indices=None,
            num_lqs=num_lqs,
            alpha=0.8,
            temperature=1.0,
        )
        
        # Check shape
        assert evidence_posterior.shape == (batch_size, num_evidence)
        
        # Check is valid distribution
        assert (evidence_posterior >= 0).all()
        sums = evidence_posterior.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(batch_size), atol=1e-5)
    
    def test_extract_posterior_with_subset(self, dummy_attentions):
        """Test posterior extraction with subset indices."""
        batch_size = 2
        num_lqs = 32
        num_evidence = 100
        subset_size = 30
        
        # Mock LLM outputs
        llm_outputs = {
            'attentions': dummy_attentions,
            'answer_start_idx': 40,
        }
        
        # Mock Q-Former CA weights (full)
        ca_weights = torch.randn(batch_size, num_lqs, num_evidence)
        ca_weights = ca_weights.softmax(dim=-1)
        
        # Subset indices
        subset_indices = torch.randint(0, num_evidence, (subset_size,))
        
        # Extract posterior on subset
        evidence_posterior = extract_posterior_from_llm_outputs(
            llm_outputs=llm_outputs,
            qformer_ca_weights=ca_weights,
            subset_indices=subset_indices,
            num_lqs=num_lqs,
        )
        
        # Should return posterior only on subset
        assert evidence_posterior.shape == (batch_size, subset_size)
        
        # Still valid distribution
        sums = evidence_posterior.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(batch_size), atol=1e-5)


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v'])
