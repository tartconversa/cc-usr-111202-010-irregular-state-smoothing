import json
import tempfile
import unittest
from pathlib import Path

from src.config import parse_config
from src.observations import parse_observations
from src.replay import ReplayCache
from src.run_estimation import run, run_hand_check
from src.state_reader import load

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_OBS = ROOT / "fixtures" / "observations.csv"
FIXTURE_STATE = ROOT / "fixtures" / "state.json"


class FixtureEndToEndTests(unittest.TestCase):
    def setUp(self):
        self.config = parse_config(load(FIXTURE_STATE))
        self.observations = parse_observations(
            load(FIXTURE_OBS), self.config, source_label=str(FIXTURE_OBS)
        )

    def test_duplicate_row_in_fixture_is_deduplicated(self):
        # observations.csv 中 o3 出现两次且内容一致
        ids = [o.observation_id for o in self.observations]
        self.assertEqual(len(ids), 5)
        self.assertEqual(ids.count("o3"), 1)

    def test_hand_check_from_state_config(self):
        result = run_hand_check(self.config, self.observations)
        self.assertTrue(result["passed"], result)
        self.assertAlmostEqual(result["posterior_position"], 0.5, places=12)
        self.assertAlmostEqual(result["posterior_position_variance"], 0.5, places=12)

    def test_versions_keep_filter_and_smoothing_separated(self):
        cache = ReplayCache(self.observations, self.config)
        known_at = self.config.known_at_cutoffs[0]
        version = cache.run_version(known_at, self.config.q)
        # 平滑与滤波步数相同但对象不同，且平滑显式记录差异
        self.assertEqual(len(version.smooth_steps), len(version.filter_result.steps))
        for smooth_step in version.smooth_steps:
            self.assertIsNot(
                smooth_step.smoothed_mean, smooth_step.filtered_mean
            )
            self.assertEqual(len(smooth_step.smoothed_mean), 2)
            self.assertEqual(len(smooth_step.filtered_mean), 2)

    def test_late_o4_only_appears_in_second_version(self):
        cache = ReplayCache(self.observations, self.config)
        first_cutoff, second_cutoff = self.config.known_at_cutoffs
        early = cache.run_version(first_cutoff, self.config.q)
        late = cache.run_version(second_cutoff, self.config.q)
        self.assertNotIn("o4", early.observation_ids)
        self.assertIn("o4", late.observation_ids)
        # o5 的 event_at 在 120 秒，早版本完全不知道它
        self.assertNotIn("o5", early.observation_ids)

    def test_run_writes_outputs_without_touching_fixtures(self):
        fixture_bytes_before = FIXTURE_OBS.read_bytes()
        with tempfile.TemporaryDirectory() as directory:
            summary, _, q_used, _, _, _ = run(
                FIXTURE_OBS, FIXTURE_STATE, Path(directory) / "out"
            )
            out = Path(directory) / "out"
            self.assertTrue((out / "summary.json").exists())
            self.assertTrue((out / "report.md").exists())
            self.assertEqual(
                len(list(out.glob("version_*.json"))),
                len(self.config.known_at_cutoffs),
            )
            payload = json.loads((out / "summary.json").read_text(encoding="utf-8"))
            self.assertTrue(payload["immutability_check"]["passed"])
            self.assertEqual(q_used, payload["q_selection"]["q"])
        self.assertEqual(FIXTURE_OBS.read_bytes(), fixture_bytes_before)


if __name__ == "__main__":
    unittest.main()
