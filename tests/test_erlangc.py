from wfm.erlangc import offered_load_erlangs, erlang_c_probability_of_wait, service_level_erlang_c, asa_erlang_c

def test_offered_load():
    a = offered_load_erlangs(volume=120, aht_seconds=300, interval_seconds=900)  # 15-min
    assert a == 40.0

def test_erlang_c_wait_prob_bounds():
    a = 10.0
    pw = erlang_c_probability_of_wait(a, 15)
    assert 0.0 <= pw <= 1.0

def test_service_level_increases_with_agents():
    a = 10.0
    sl_12 = service_level_erlang_c(a, 12, aht_seconds=300, sl_time_seconds=20)
    sl_20 = service_level_erlang_c(a, 20, aht_seconds=300, sl_time_seconds=20)
    assert sl_20 >= sl_12

def test_asa_decreases_with_agents():
    a = 10.0
    asa_12 = asa_erlang_c(a, 12, aht_seconds=300)
    asa_20 = asa_erlang_c(a, 20, aht_seconds=300)
    assert asa_20 <= asa_12