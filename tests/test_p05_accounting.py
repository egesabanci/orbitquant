"""Tests for P0.5 true byte accounting."""
from __future__ import annotations

import numpy as np
import pytest

from baseline import p05_accounting as p05


def test_accounting_payload_and_side_info():
    """Payload and side info add correctly at d=64."""
    rep = p05.Representation(name="t", payload_bits_per_coord=2,
                             per_token_side_info_bits=8.0)
    a = p05.accounting(rep, d=64, n_tokens=8192)
    assert a["payload_bits"] == 128
    assert a["side_info_bits"] == 8.0
    assert a["total_bits_per_token"] == 136.0
    assert a["true_bits_per_coord_equiv"] == pytest.approx(2.125)


def test_protected_pool_cost_is_per_token_not_proportional_to_n():
    """Protected-pool cost must NOT grow with cache length.

    Regression: pool_extra = frac * n_tokens * extra charged 3276.8 b/token at
    n=8192 instead of the expected frac*extra = 0.4 b/token.
    """
    rep = p05.Representation(name="pool", payload_bits_per_coord=2,
                             per_token_side_info_bits=8.0,
                             protected_pool=(0.05, 8.0))
    a_small = p05.accounting(rep, d=64, n_tokens=100)
    a_large = p05.accounting(rep, d=64, n_tokens=8192)
    # per-token total must be independent of n (only predictor amortizes)
    assert a_small["total_bits_per_token"] == a_large["total_bits_per_token"]
    # expected: payload 128 + side 8 + membership 1 + frac*extra 0.4 = 137.4
    assert a_large["total_bits_per_token"] == pytest.approx(137.4, abs=1e-9)


def test_predictor_amortizes_with_cache_length():
    """Predictor weights amortize over tokens (P2.10)."""
    rep = p05.Representation(name="pred", payload_bits_per_coord=2,
                             per_token_side_info_bits=8.0,
                             predictor_bits_total=64)
    a_short = p05.accounting(rep, d=64, n_tokens=100)
    a_long = p05.accounting(rep, d=64, n_tokens=8192)
    assert a_short["predictor_amortized_bits"] == pytest.approx(0.64)
    assert a_long["predictor_amortized_bits"] == pytest.approx(64 / 8192)
    assert a_long["total_bits_per_token"] < a_short["total_bits_per_token"]


def test_deployability_score():
    """Fully-deployable reps score 1.0; per-block baseline scores lower."""
    good = p05.Representation(name="good", payload_bits_per_coord=2)
    bad = p05.Representation(name="bad", payload_bits_per_coord=2,
                             register_dequant=False, fused_attention=False,
                             no_dequant_materialize=False)
    assert p05.deployability(good)["score"] == 1.0
    assert p05.deployability(bad)["score"] == pytest.approx(2 / 5)
