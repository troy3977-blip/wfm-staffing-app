from wfm.staffing import StaffingInputs, compute_required_agents


def test_closed_interval_returns_zero():
    inputs = StaffingInputs(
        volume=100,
        aht_seconds=300,
        interval_minutes=15,
        is_open=False,
        target_type="service_level",
        service_level_target=0.80,
        service_level_time_seconds=60,
        asa_target_seconds=30,
        occupancy_target=0.90,
        shrinkage=0.30,
    )
    res = compute_required_agents(inputs)
    assert res.required_on_phone == 0
    assert res.required_scheduled == 0


def test_basic_staffing_returns_positive():
    inputs = StaffingInputs(
        volume=120,
        aht_seconds=300,
        interval_minutes=15,
        is_open=True,
        target_type="service_level",
        service_level_target=0.80,
        service_level_time_seconds=20,
        asa_target_seconds=30,
        occupancy_target=0.90,
        shrinkage=0.30,
    )
    res = compute_required_agents(inputs)
    assert res.required_on_phone > 0
    assert res.required_scheduled >= res.required_on_phone