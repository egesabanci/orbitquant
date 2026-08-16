# EXPERIMENT: <short, descriptive title>

> Doc per experiment — ONE idea, one clean single-responsibility analysis.
> Copy this template; delete guidance comments before finishing.

## Goal
_What hypothesis does this experiment test? What metric(s) does it move, and in
which direction? Tie it to the session goal (better = tq_mse / estimator;
lighter = tq_bytes_b2 / fwht_us)._

## Method / Protocol
_Concrete description: what code changed (`baseline/...`), what representation /
estimator / transform is tested, and the exact protocol (grid, seeds,
parameters). Reference the benchmark harness. If it is an algebraic rewrite
(e.g. vectorized FWHT), state the equivalence proof (same outputs on random
inputs)._

## Anti-cheat check
_State explicitly how this result is NOT an artifact: unchanged seeds/inputs,
honest rates, realistic vectors, no benchmark weakening._

## Results (before → after)
| Metric | Before | After | Δ | Verdict |
|---|---|---|---|---|
| tq_mse | 0.295923 | … | … | ↓ good |
| tq_bytes_b2 | 136.0 | … | … | ↓ good |
| tq_bias_b1 | 0.019938 | … | … | ↓ good |
| tq_var_b2 | 0.988199 | … | … | ↓ good |
| fwht_us | ~389 | … | … | ↓ good |

_Give the raw experiment numbers (run_experiment + log_experiment values)._

## Findings
_Interpretation: why did it move (or not)? What does it teach about the regime?
Be quantitative and honest; a well-measured negative result is a valid finding._

## Tradeoffs / risks
_Any quality penalty, complexity, added metadata, or GPU-box caveats, real vs
assumed._

## Verdict
**keep / discard / crash** — with one-line justification referencing the
metric deltas and the anti-cheat contract.

## ASI (what to remember after a context reset)
- _Key numbers, the exact code change, the part that worked/didn't, and what
  the next experiment should try._
