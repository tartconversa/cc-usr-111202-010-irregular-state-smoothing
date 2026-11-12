import unittest

from src.config import parse_config
from src.errors import ObservationConflictError, StateEstimationError
from src.observations import filter_known_at, parse_observations

BASE_CONFIG = {
    "time_unit": "seconds",
    "initial_time": "2026-11-01T00:00:00+08:00",
    "initial_mean": [0, 0],
    "initial_covariance": [[1, 0], [0, 0]],
    "q": 0.01,
    "sensors": {
        "s1": {"bias": 0, "variance": 1},
        "s2": {"bias": 0.2, "variance": 0.5},
    },
    "q_grid": [0, 0.01],
}


def config():
    return parse_config(BASE_CONFIG)


def row(obs_id, sensor="s1", event="2026-11-01T00:00:01+08:00",
        received="2026-11-01T00:00:01+08:00", value="1"):
    return {
        "observation_id": obs_id,
        "sensor": sensor,
        "event_at": event,
        "received_at": received,
        "value": value,
    }


class ObservationTests(unittest.TestCase):
    def test_exact_resend_is_deduplicated(self):
        rows = [row("o1"), row("o1")]
        result = parse_observations(rows, config())
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].observation_id, "o1")

    def test_conflicting_resend_is_rejected_not_silently_kept(self):
        rows = [row("o1", value="1"), row("o1", value="1.5")]
        with self.assertRaises(ObservationConflictError):
            parse_observations(rows, config())

    def test_conflict_on_any_content_field_is_rejected(self):
        for mutated in (
            row("o1", sensor="s2"),
            row("o1", event="2026-11-01T00:00:00+08:00"),
            row("o1", received="2026-11-01T00:00:09+08:00"),
        ):
            with self.subTest(mutated=mutated):
                with self.assertRaises(ObservationConflictError):
                    parse_observations([row("o1"), mutated], config())

    def test_blank_value_is_an_error_not_zero(self):
        rows = [row("o1", value="")]
        with self.assertRaises(StateEstimationError):
            parse_observations(rows, config())

    def test_unknown_sensor_rejected(self):
        with self.assertRaises(StateEstimationError):
            parse_observations([row("o1", sensor="s9")], config())

    def test_received_before_event_rejected(self):
        rows = [row(
            "o1",
            event="2026-11-01T00:00:05+08:00",
            received="2026-11-01T00:00:01+08:00",
        )]
        with self.assertRaises(StateEstimationError):
            parse_observations(rows, config())

    def test_naive_timestamp_rejected(self):
        rows = [row("o1", event="2026-11-01T00:00:01")]
        with self.assertRaises(StateEstimationError):
            parse_observations(rows, config())

    def test_known_at_filter_uses_received_not_event_time(self):
        observations = parse_observations(
            [
                row("early", event="2026-11-01T00:00:01+08:00",
                    received="2026-11-01T00:00:01+08:00"),
                row("late", event="2026-11-01T00:00:02+08:00",
                    received="2026-11-01T00:00:10+08:00"),
            ],
            config(),
        )
        cutoff = parse_config(BASE_CONFIG).initial_time + 6
        known = filter_known_at(observations, cutoff)
        self.assertEqual([o.observation_id for o in known], ["early"])


if __name__ == "__main__":
    unittest.main()
