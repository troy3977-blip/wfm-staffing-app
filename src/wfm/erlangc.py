from __future__ import annotations

import math


def offered_load_erlangs(volume: float, aht_seconds: float, interval_seconds: float) -> float:
    """
    Offered load a (Erlangs) = arrival_rate * AHT.
    With arrivals measured as count per interval:
      arrival_rate = volume / interval_seconds
      => a = volume * aht_seconds / interval_seconds
    """
    if interval_seconds <= 0:
        raise ValueError("interval_seconds must be > 0")
    if volume < 0:
        raise ValueError("volume must be >= 0")
    if aht_seconds <= 0 and volume > 0:
        raise ValueError("aht_seconds must be > 0 when volume > 0")
    if volume == 0:
        return 0.0
    return float(volume) * float(aht_seconds) / float(interval_seconds)


def _erlang_c_wait_probability(a: float, n: int) -> float:
    """
    Erlang C probability of wait (Pw).

    Pw = [ (a^n / n!) * (n/(n-a)) ] / [ sum_{k=0..n-1} a^k/k! + (a^n/n!) * (n/(n-a)) ]

    Requires n > a for stability.
    """
    if n <= 0:
        return 1.0
    if a <= 0:
        return 0.0
    if n <= a:
        return 1.0

    # Compute sum_{k=0}^{n-1} a^k/k! and term a^n/n! via recurrence to avoid overflow.
    term = 1.0  # a^0/0!
    s = term
    for k in range(1, n):
        term *= a / k
        s += term

    # term currently = a^(n-1)/(n-1)!
    term_n = term * a / n  # a^n / n!
    extra = term_n * (n / (n - a))
    denom = s + extra
    if denom <= 0:
        return 1.0
    return float(extra / denom)


def asa_erlang_c(a: float, n: int, aht_seconds: float) -> float:
    """
    Average Speed of Answer (ASA) for M/M/n without abandonment.

    ASA = Pw * (AHT / (n-a))
    """
    if a <= 0:
        return 0.0
    if n <= a:
        return float("inf")
    pw = _erlang_c_wait_probability(a, n)
    return float(pw) * float(aht_seconds) / float(n - a)


def service_level_erlang_c(a: float, n: int, aht_seconds: float, target_answer_time_seconds: float) -> float:
    """
    Service level for threshold T (seconds):

    SL(T) = 1 - Pw * exp(-(n-a) * (T / AHT))
    """
    if a <= 0:
        return 1.0
    if n <= a:
        return 0.0

    T = max(float(target_answer_time_seconds), 0.0)
    pw = _erlang_c_wait_probability(a, n)
    expo = math.exp(-(n - a) * (T / float(aht_seconds)))
    sl = 1.0 - pw * expo
    # Clamp for safety
    return max(0.0, min(1.0, float(sl)))