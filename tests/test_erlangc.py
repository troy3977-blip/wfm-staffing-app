from wfm.erlangc import offered_load_erlangs, service_level_erlang_c, asa_erlang_c


def test_offered_load_basic():
    a = offered_load_erlangs(volume=120, aht_seconds=300, interval_seconds=900)
    assert a > 0


def test_service_level_bounds():
    a = 10.0
    sl = service_level_erlang_c(a=a, n=20, aht_seconds=300, target_answer_time_seconds=60)
    assert 0.0 <= sl <= 1.0


def test_asa_finite_when_stable():
    a = 10.0
    asa = asa_erlang_c(a=a, n=20, aht_seconds=300)
    assert asa > 0