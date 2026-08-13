"""Fixed-size, cached human-invariant features for oracle braid scientists.

These features are deliberately labelled as an oracle family.  The environment
computes them from the whole closed braid, while the ordinary scientists must
infer global structure from their observations.  A knot invariant is unchanged
by every semantic Markov/braid rewrite, so callers should carry the vector across
those moves and recompute it only after a crossing change.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

FEATURE_SIZES = {
    "": 0,
    "classical": 6,
    "alexander": 18,
    "jones": 18,
    "combined": 30,
}


def invariant_feature_size(name: str) -> int:
    try:
        return FEATURE_SIZES[name]
    except KeyError as error:
        raise ValueError(f"unknown invariant feature set {name!r}") from error


def _signed_log(value: float, scale: float = 8.0) -> float:
    return math.copysign(math.log1p(abs(value)) / scale, value)


def _derivative_value(poly: dict[int, int], x: float, order: int) -> float:
    total = 0.0
    for exponent, coefficient in poly.items():
        falling = 1
        for offset in range(order):
            falling *= exponent - offset
        if falling:
            total += coefficient * falling * x ** (exponent - order)
    return total


def _polynomial_features(poly: dict[int, int], max_len: int) -> list[float]:
    """The coefficient/degree/evaluation summary used by Applebaum et al.

    Direct integer coefficients have no stable scale.  Degrees and support are
    normalized by the configured word capacity; values and the first three
    derivatives near one use a signed logarithm.  The result is fixed-size even
    when the polynomial degree grows.
    """
    if not poly:
        return [0.0] * 12
    degrees = sorted(poly)
    capacity = float(max(max_len, 1))
    values = [
        degrees[0] / capacity,
        degrees[-1] / capacity,
        (degrees[-1] - degrees[0]) / capacity,
        len(poly) / capacity,
    ]
    for point in (0.9, 1.1):
        for order in range(4):
            values.append(_signed_log(_derivative_value(poly, point, order)))
    return values


@lru_cache(maxsize=100_000)
def _cached_features(
    word: tuple[int, ...], strands: int, name: str, max_len: int
) -> tuple[float, ...]:
    if not name:
        return ()
    from rf_knots.invariants import alexander_polynomial, determinant, jones_polynomial
    from rf_knots.seifert import branched_cover_homology, signature

    # rf-knots quite reasonably rejects an empty Seifert matrix. The terminal
    # one-strand state is nevertheless the most frequent observation in this
    # game, and its normalized Alexander/Jones polynomials are both one.
    if not word and strands == 1:
        alexander = {0: 1}
        det = 1
        sigma = 0
        cover = ()
    else:
        alexander = alexander_polynomial(word, strands)
        det = determinant(word, strands)
        sigma = signature(word, strands)
        cover = branched_cover_homology(word, strands)
    mod3_rank = sum(int(value) % 3 == 0 for value in cover)
    classical = [
        sigma / max(float(max_len), 1.0),
        _signed_log(float(det)),
        (max(alexander) - min(alexander)) / max(float(max_len), 1.0),
        len(alexander) / max(float(max_len), 1.0),
        _signed_log(float(sum(abs(value) for value in alexander.values()))),
        mod3_rank / max(float(strands), 1.0),
    ]
    if name == "classical":
        result = classical
    elif name == "alexander":
        result = classical + _polynomial_features(alexander, max_len)
    else:
        jones = {0: 1} if not word and strands == 1 else jones_polynomial(word, strands)
        if name == "jones":
            result = classical + _polynomial_features(jones, max_len)
        elif name == "combined":
            result = (
                classical
                + _polynomial_features(alexander, max_len)
                + _polynomial_features(jones, max_len)
            )
        else:  # guarded before the expensive imports, retained defensively
            raise ValueError(f"unknown invariant feature set {name!r}")
    expected = invariant_feature_size(name)
    if len(result) != expected:
        raise AssertionError(f"feature set {name} has {len(result)} values, expected {expected}")
    return tuple(float(np.clip(value, -4.0, 4.0)) for value in result)


def invariant_features(
    word: tuple[int, ...] | list[int] | np.ndarray,
    strands: int,
    name: str,
    max_len: int,
) -> np.ndarray:
    """Return a fresh float32 vector; the immutable cached copy stays private."""
    invariant_feature_size(name)
    compact = tuple(int(value) for value in word if int(value))
    return np.asarray(
        _cached_features(compact, int(strands), name, int(max_len)), dtype=np.float32
    ).copy()
