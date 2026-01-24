from __future__ import annotations

import math


def offered_load_erlangs(volume: float, aht_seconds: float, interval_seconds: float) -> float:
    """
    Offered load (Erlangs) = (Volume * AHT) / IntervalSeconds
    """
    if interval_seconds <= 0:
        raise ValueError("interval_seconds must be > 0")
    if volume < 0 or aht_seconds < 0:
        raise ValueError("volume and aht_seconds must be >= 0")
    return (volume * aht_seconds) / interval_seconds


def erlang_c_probability_of_wait(a: float, n: int) -> float:
    """
    Erlang C: probability that an arrival has to wait (delay probability).
    Stable only when n > a.

    Uses a numerically stable recursion for the sum:
      sum_{k=0}^{n-1} a^k / k!
    and then:
      P(wait) = (a^n / n!)*(n/(n-a)) / [ sum_{k=0}^{n-1} a^k/k! + (a^n/n!)*(n/(n-a)) ]
    """
    if a < 0:
        raise ValueError("offered load a must be >= 0")
    if n <= 0:
        raise ValueError("n must be >= 1")
    if a == 0:
        return 0.0
    if n <= a:
        # Unstable system; in practice you'd never staff at/below load.
        return 1.0

    # Compute sum_{k=0}^{n-1} a^k/k! via recursion:
    # term_k = a^k/k!
    term = 1.0
    s = term
    for k in range(1, n):
        term *= a / k
        s += term

    # term now equals a^(n-1)/(n-1)!, so multiply one more time for a^n/n!
    term *= a / n  # now a^n / n!
    numerator = term * (n / (n - a))
    denom = s + numerator
    return float(numerator / denom)


def service_level_erlang_c(a: float, n: int, aht_seconds: float, sl_time_seconds: float) -> float:
    """
    Service level: P(wait <= T) for Erlang C.
      SL(T) = 1 - P(wait)*exp(-(n-a)*mu*T), where mu = 1/AHT
    """
    if aht_seconds <= 0:
        raise ValueError("aht_seconds must be > 0")
    if sl_time_seconds < 0:
        raise ValueError("sl_time_seconds must be >= 0")
    if a == 0:
        return 1.0
    if n <= a:
        return 0.0

    pw = erlang_c_probability_of_wait(a, n)
    mu = 1.0 / aht_seconds
    exponent = -(n - a) * mu * sl_time_seconds
    return float(1.0 - pw * math.exp(exponent))


def asa_erlang_c(a: float, n: int, aht_seconds: float) -> float:
    """
    Average Speed of Answer (ASA) under Erlang C:
      ASA = P(wait) * AHT / (n - a)
    """
    if aht_seconds <= 0:
        raise ValueError("aht_seconds must be > 0")
    if a == 0:
        return 0.0
    if n <= a:
        return float("inf")

    pw = erlang_c_probability_of_wait(a, n)
    return float(pw * aht_seconds / (n - a))