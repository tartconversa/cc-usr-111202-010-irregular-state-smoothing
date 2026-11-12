import unittest

from src.errors import StateEstimationError
from src.linalg import check_psd
from src.model import noise_covariance, observation_matrix, process_noise, transition_matrix


class ModelTests(unittest.TestCase):
    def test_transition_matrix_uses_real_dt(self):
        self.assertEqual(transition_matrix(2.5), [[1.0, 2.5], [0.0, 1.0]])
        self.assertEqual(transition_matrix(0.0), [[1.0, 0.0], [0.0, 1.0]])
        with self.assertRaises(StateEstimationError):
            transition_matrix(-1.0)

    def test_process_noise_matches_continuous_white_acceleration_form(self):
        q, dt = 0.01, 2.0
        q_matrix = process_noise(q, dt)
        expected = [
            [q * dt ** 3 / 3.0, q * dt ** 2 / 2.0],
            [q * dt ** 2 / 2.0, q * dt],
        ]
        for i in range(2):
            for j in range(2):
                self.assertAlmostEqual(q_matrix[i][j], expected[i][j], places=15)
        check_psd(q_matrix, label="测试 Q")

    def test_zero_q_gives_zero_covariance_not_fake_inflation(self):
        self.assertEqual(process_noise(0.0, 10.0), [[0.0, 0.0], [0.0, 0.0]])

    def test_long_gap_grows_position_variance_cubically(self):
        short = process_noise(1.0, 1.0)
        long_gap = process_noise(1.0, 10.0)
        # 位置方差按 dt^3 增长，速度方差按 dt 增长，协方差项按 dt^2 增长
        self.assertAlmostEqual(long_gap[0][0] / short[0][0], 1000.0, places=10)
        self.assertAlmostEqual(long_gap[1][1] / short[1][1], 10.0, places=10)
        self.assertAlmostEqual(long_gap[0][1] / short[0][1], 100.0, places=10)

    def test_observation_and_noise_matrices(self):
        h = observation_matrix(("s1", "s2"))
        self.assertEqual(h, [[1.0, 0.0], [1.0, 0.0]])
        sensors = {
            "s1": type("S", (), {"variance": 1.0})(),
            "s2": type("S", (), {"variance": 0.5})(),
        }
        r = noise_covariance(("s1", "s2"), sensors)
        self.assertEqual(r, [[1.0, 0.0], [0.0, 0.5]])


if __name__ == "__main__":
    unittest.main()
