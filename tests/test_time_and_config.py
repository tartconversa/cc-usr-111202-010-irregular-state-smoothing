import unittest

from src.config import parse_config
from src.errors import StateEstimationError
from src.estimator import run_forward_filter
from src.observations import parse_observations
from src.timeutils import parse_time, q_to_seconds, to_iso, unit_factor


class TimeUtilsTests(unittest.TestCase):
    def test_iso_offsets_normalize_to_epoch_seconds(self):
        beijing = parse_time("2026-11-01T08:00:00+08:00")
        utc = parse_time("2026-11-01T00:00:00+00:00")
        self.assertEqual(beijing, utc)
        self.assertEqual(to_iso(utc), "2026-11-01T00:00:00+00:00")

    def test_numeric_axes_scale_by_configured_unit(self):
        self.assertEqual(parse_time(1, "seconds"), 1.0)
        self.assertEqual(parse_time(1, "milliseconds"), 0.001)
        self.assertEqual(parse_time(1, "minutes"), 60.0)
        self.assertEqual(parse_time(1, "hours"), 3600.0)
        self.assertEqual(unit_factor("ms"), 1e-3)

    def test_naive_iso_rejected(self):
        with self.assertRaises(StateEstimationError):
            parse_time("2026-11-01T00:00:00")

    def test_q_scales_with_cube_root_of_second_factor(self):
        # 1 分钟 = 60 秒：同一物理过程 q_s = q_min / 60
        self.assertAlmostEqual(q_to_seconds(60.0, "minutes"), 1.0, places=12)
        self.assertAlmostEqual(q_to_seconds(1.0, "seconds"), 1.0, places=12)
        self.assertAlmostEqual(q_to_seconds(0.001, "milliseconds"), 1.0, places=12)


def _numeric_config(time_unit, q_in_input_unit):
    raw = {
        "time_unit": time_unit,
        "initial_time": 0,
        "initial_mean": [0, 0],
        "initial_covariance": [[1, 0], [0, 0]],
        "q": q_in_input_unit,
        "sensors": {"s1": {"bias": 0, "variance": 1}},
        "q_grid": [q_in_input_unit],
    }
    return parse_config(raw)


class UnitConversionEquivalenceTests(unittest.TestCase):
    """同一物理过程分别用秒和毫秒表达，滤波结果必须一致。"""

    @staticmethod
    def _run(time_unit, q_input, rows):
        cfg = _numeric_config(time_unit, q_input)
        observations = parse_observations(rows, cfg)
        return run_forward_filter(observations, cfg), cfg

    def test_seconds_and_milliseconds_expressions_are_equivalent(self):
        seconds_rows = [
            {"observation_id": "a", "sensor": "s1", "event_at": 0,
             "received_at": 0, "value": 1},
            {"observation_id": "b", "sensor": "s1", "event_at": 1,
             "received_at": 1, "value": 2},
        ]
        ms_rows = [
            {"observation_id": "a", "sensor": "s1", "event_at": 0,
             "received_at": 0, "value": 1},
            {"observation_id": "b", "sensor": "s1", "event_at": 1000,
             "received_at": 1000, "value": 2},
        ]
        fit_s, cfg_s = self._run("seconds", 0.01, seconds_rows)
        fit_ms, cfg_ms = self._run("milliseconds", 0.01 * 1e-3, ms_rows)
        # 换算到秒基准的 q 一致
        self.assertAlmostEqual(cfg_s.q, cfg_ms.q, places=15)
        self.assertEqual(len(fit_s.steps), len(fit_ms.steps))
        for step_s, step_ms in zip(fit_s.steps, fit_ms.steps):
            # 内部时间都以秒计
            self.assertAlmostEqual(step_s.event_at, step_ms.event_at, places=12)
            self.assertAlmostEqual(step_s.dt, step_ms.dt, places=12)
            for i in range(2):
                self.assertAlmostEqual(
                    step_s.posterior_mean[i], step_ms.posterior_mean[i], places=12
                )
                for j in range(2):
                    self.assertAlmostEqual(
                        step_s.posterior_cov[i][j],
                        step_ms.posterior_cov[i][j],
                        places=12,
                    )


class ConfigValidationTests(unittest.TestCase):
    def test_non_positive_observation_variance_rejected(self):
        raw = {
            "time_unit": "seconds",
            "initial_time": 0,
            "initial_mean": [0, 0],
            "initial_covariance": [[1, 0], [0, 0]],
            "q": 0,
            "sensors": {"s1": {"bias": 0, "variance": 0}},
            "q_grid": [0],
        }
        with self.assertRaises(StateEstimationError):
            parse_config(raw)

    def test_negative_q_rejected_but_singular_initial_covariance_allowed(self):
        raw = {
            "time_unit": "seconds",
            "initial_time": 0,
            "initial_mean": [0, 0],
            "initial_covariance": [[1, 0], [0, 0]],
            "q": -1,
            "sensors": {"s1": {"bias": 0, "variance": 1}},
            "q_grid": [-1],
        }
        with self.assertRaises(StateEstimationError):
            parse_config(raw)
        raw["q"] = 0
        raw["q_grid"] = [0]
        cfg = parse_config(raw)  # 半正定初值合法
        self.assertEqual(cfg.initial_covariance[1][1], 0.0)

    def test_unsorted_cutoffs_rejected(self):
        raw = {
            "time_unit": "seconds",
            "initial_time": 0,
            "initial_mean": [0, 0],
            "initial_covariance": [[1, 0], [0, 0]],
            "q": 0,
            "sensors": {"s1": {"bias": 0, "variance": 1}},
            "q_grid": [0],
            "known_at_cutoffs": [2, 1],
        }
        with self.assertRaises(StateEstimationError):
            parse_config(raw)


if __name__ == "__main__":
    unittest.main()
