"""P0.5 — True byte accounting and deployability score.

Every experiment must report PAID bytes per token, not nominal bits per
coordinate. This module computes the true per-token footprint of each
quantized representation, counting everything P0.5 requires (proposals.md):

- quantized payload bits (per-coordinate code indices)
- packed-bit padding (how index bits pack into whole bytes)
- side information: norms / radii / residual norms (P1.3: 8-bit log format),
  clip scales, zero points, codebook IDs
- protected-token pools (P1.5: per-token bit-width metadata)
- predictor weights (P2.10: amortized over cached tokens)
- layout metadata

And assigns a deployability score: paged-cache fit, register dequantization,
fused-attention support, query batching, and no dequantized K/V materialization.

Pure NumPy / pure Python. No model, no GPU.
"""
from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["Representation", "accounting", "deployability", "run_accounting"]


@dataclass
class Representation:
    """A quantized KV representation and its metadata costs.

    ``payload_bits_per_coord``: code index bits per coordinate.
    ``per_token_side_info_bits``: scalar side info per token (norms, radii,
      residual norms, clip scales, zero points, layout) AFTER P1.3-style
      quantization where applicable.
    ``codebook_ids_per_token``: codebook-version metadata bits per token
      (0 for a single shared codebook).
    ``protected_pool``: (fraction, extra_bits_per_protected) for P1.5-style
      protected pools; 0/0 when unused.
    ``predictor_bits_total``: P2.10 predictor weights, amortized over tokens.
    ``pack_bits_per_block``: padding overhead when packing indices into bytes.
    ``block_size``: coordinates per packed block (for padding calc).
    ``zero_point_scale``: per-block zero point + scale bits (0 if none).
    """
    name: str
    payload_bits_per_coord: float
    per_token_side_info_bits: float = 0.0
    codebook_ids_per_token: int = 0
    protected_pool: tuple[float, float] = (0.0, 0.0)
    predictor_bits_total: int = 0
    pack_bits_per_block: int = 0
    block_size: int = 64
    zero_point_scale: tuple[int, int] = (0, 0)  # (zp_bits, scale_bits) per block
    # deployability properties
    paged_cache_fit: bool = True
    register_dequant: bool = True
    fused_attention: bool = True
    query_batching: bool = True
    no_dequant_materialize: bool = True
    notes: str = ""


def accounting(rep: Representation, d: int, n_tokens: int) -> dict:
    """Compute the true paid bits/bytes per token for a representation.

    Returns a dict with the breakdown and the true per-token cost at the given
    head dimension and cache length (the latter amortizes predictor weights).
    """
    payload = rep.payload_bits_per_coord * d

    # packing padding: index bits packed into bytes in blocks of block_size
    # coords; each block's index bits rounded up to whole bytes.
    idx_bits_per_block = rep.payload_bits_per_coord * rep.block_size
    bytes_per_block = -(-int(idx_bits_per_block) // 8)  # ceil div
    padding_per_block = bytes_per_block * 8 - idx_bits_per_block
    padding = padding_per_block * (d / rep.block_size)

    # per-block zero point + scale
    zp_scale = 0.0
    if rep.zero_point_scale[0] > 0:
        zp_scale = (rep.zero_point_scale[0] + rep.zero_point_scale[1]) * (d / rep.block_size)

    # protected pool: per-token metadata marking pool membership (1 bit flag
    # per token, not per coordinate) + extra bits per protected token
    frac, extra = rep.protected_pool
    pool_meta = 1.0 if frac > 0 else 0.0  # 1 bit/token membership flag
    pool_extra = frac * extra if frac > 0 else 0.0  # averaged over tokens

    # predictor weights amortized over cached tokens
    predictor = rep.predictor_bits_total / n_tokens if n_tokens > 0 else 0.0

    side = rep.per_token_side_info_bits + rep.codebook_ids_per_token
    total_bits = payload + padding + zp_scale + side + pool_meta + pool_extra + predictor

    return {
        "payload_bits": payload,
        "packing_padding_bits": padding,
        "zp_scale_bits": zp_scale,
        "side_info_bits": side,
        "protected_meta_bits": pool_meta,
        "protected_extra_bits": pool_extra,
        "predictor_amortized_bits": predictor,
        "total_bits_per_token": total_bits,
        "total_bytes_per_token": total_bits / 8,
        "nominal_bits_per_coord": rep.payload_bits_per_coord,
        "true_bits_per_coord_equiv": total_bits / d,
    }


def deployability(rep: Representation) -> dict:
    """Deployability score: fraction of the five properties that hold."""
    props = {
        "paged_cache_fit": rep.paged_cache_fit,
        "register_dequant": rep.register_dequant,
        "fused_attention": rep.fused_attention,
        "query_batching": rep.query_batching,
        "no_dequant_materialize": rep.no_dequant_materialize,
    }
    return {"properties": props, "score": sum(props.values()) / len(props)}


# --------------------------------------------------------------------------- #
# Standard representations at b = 2 nominal bits, d = 64
# --------------------------------------------------------------------------- #
def default_representations(d: int = 64, n_tokens: int = 8192) -> list[Representation]:
    """The main representations at ~2 nominal bits/coord, with honest metadata.

    Side info uses the P1.3 8-bit log format where applicable (norms, radii,
    residual norms). Predictor cost is the P2.10 scalar predictor (64 bits
    total at d=64, amortized over n_tokens).
    """
    # per-block overhead for kivi/kvquant-style: 8-bit zp + 8-bit scale per
    # 128-coordinate block
    zp_scale = (8, 8)
    block = 128

    return [
        Representation(
            name="TurboQuant_mse b=2",
            payload_bits_per_coord=2,
            # unit-norm assumption: no norm header for unit vectors; real KV
            # needs a norm header -> 8-bit log (P1.3)
            per_token_side_info_bits=8.0,
            notes="exact-Beta Lloyd-Max, no block constants",
        ),
        Representation(
            name="TurboQuant_prod b=2",
            payload_bits_per_coord=2,
            # 1-bit MSE base + 1-bit residual QJL: same 2 b/coord payload but
            # carries a residual norm per token (P1.3 8-bit log)
            per_token_side_info_bits=16.0,  # key norm + residual norm
            notes="1-bit MSE base + 1-bit QJL residual (Algorithm 2)",
        ),
        Representation(
            name="QJL m=64",
            payload_bits_per_coord=1.0,
            # m=d sign bits = 1 b/coord + key norm per token
            per_token_side_info_bits=8.0,
            notes="1-bit sign sketch, zero per-block constants",
        ),
        Representation(
            name="PolarQuant 3.875",
            payload_bits_per_coord=3.875,
            per_token_side_info_bits=0.0,  # radius already inside the 62-bit block
            pack_bits_per_block=0,
            block_size=16,
            notes="recursive polar angles; paper's 62 bits/16 coords includes "
                  "the 16-bit radius (P1.3 could cut radius to 8-bit log: "
                  "-> 3.375 b/coord)",
        ),
        Representation(
            name="TurboQuant + protected pool 5%",
            payload_bits_per_coord=2,
            per_token_side_info_bits=8.0,
            protected_pool=(0.05, 8.0),  # 5% of tokens at +8 bits (4-bit + extra)
            notes="P1.5: 5% protected tokens at higher precision",
        ),
        Representation(
            name="TurboQuant + predictive (P2.10)",
            payload_bits_per_coord=2,
            per_token_side_info_bits=8.0,
            predictor_bits_total=64,  # scalar predictor, d=64
            notes="P2.10 scalar predictor amortized over cache",
        ),
        Representation(
            name="KIVI-style per-block (baseline)",
            payload_bits_per_coord=2,
            zero_point_scale=zp_scale,
            block_size=block,
            register_dequant=False,  # per-block scale in memory, not registers
            fused_attention=False,  # needs dequant before attention
            no_dequant_materialize=False,
            notes="per-128-coord block: 8-bit zp + 8-bit scale",
        ),
    ]


def run_accounting(d: int = 64, n_tokens: int = 8192) -> list[dict]:
    """Full accounting table for the standard representations."""
    rows = []
    for rep in default_representations(d, n_tokens):
        acc = accounting(rep, d, n_tokens)
        dep = deployability(rep)
        rows.append({"rep": rep, "acc": acc, "dep": dep})
    return rows
