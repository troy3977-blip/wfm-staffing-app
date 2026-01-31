import pandas as pd

from wfm.monte_carlo import MonteCarloConfig, run_interval_monte_carlo


def test_monte_carlo_runs_small():
    df = pd.DataFrame(
        {
            "interval_start": ["2024-01-01 09:00:00", "2024-01-01 09:15:00"],
            "interval_minutes": [15, 15],
            "volume": [50, 60],
            "aht_seconds": [300, 300],
            "is_open": [True, True],
        }
    )

    cfg = MonteCarloConfig(n_sims=200, seed=1, volume_dist="poisson", volume_cv=0.15, aht_dist="lognormal", aht_cv=0.10)

    out = run_interval_monte_carlo(
        interval_df=df,
        cfg=cfg,
        simulate_volume=True,
        simulate_aht=False,
        target_type="service_level",
        service_level_target=0.80,
        service_level_time_seconds=60,
        asa_target_seconds=30,
        shrinkage=0.30,
        occupancy_target=0.90,
    )

    assert "scheduled_p90" in out.columns
    assert len(out) == 2