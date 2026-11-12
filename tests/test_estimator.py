import math
import unittest

from src.config import EstimationConfig, SensorSpec, parse_config
from src.estimator import (
    forecast_to,
    run_forward_filter,
    rts_smooth,
    total_innovation_log_likelihood,
)
from src.linalg import check_psd, jacobi_eigen
from src.observations import Observation
from src.replay import ReplayCache, select_q_on_training

T0 = 0.0


def make_config(q=0.01, sensors=None, initial_cov=((1.0, 0.0), (0.0, 0.0)),
                cutoffs=(), training_end=None, q_grid=(0.0, 0.01, 0.1),
                time_unit="seconds", initial_time=T0, numeric=True):
    sensors = sensors or {
        "s1": SensorSpec("s1", bias=0.0, variance=1.0),
        "s2": SensorSpec("s2", bias=0.2, variance=0.5),
    }
    return EstimationConfig(
        time_unit=time_unit,
        initial_time=initial_time,
        initial_mean=(0.0, 0.0),
        initial_covariance=initial_cov,
        q=q,
        sensors=sensors,
        known_at_cutoffs=tuple(cutoffs),
        training_end=training_end,
        q_grid=tuple(q_grid),
        hand_check=None,
        numeric_time_axis=numeric,
    )


def obs(obs_id, sensor, event_at, received_at=None, value=1.0):
    return Observation(
        observation_id=obs_id,
        sensor=sensor,
        event_at=float(event_at),
        received_at=float(event_at if received_at is None else received_at),
        value=float(value),
    )


class HandCheckTests(unittest.TestCase):
    def test_spec_hand_check_position_half_variance_half(self):
        """题目指定校验：位置方差1、速度方差0、q=0、dt=0、观测1/方差1。"""
        cfg = make_config(q=0.0)
        result = run_forward_filter([obs("o1", "s1", 0.0, value=1.0)], cfg, q=0.0)
        self.assertEqual(len(result.steps), 1)
        step = result.steps[0]
        self.assertEqual(step.dt, 0.0)
        self.assertAlmostEqual(step.posterior_mean[0], 0.5, places=12)
        self.assertAlmostEqual(step.posterior_cov[0][0], 0.5, places=12)
        # 半正定即可：速度方向仍可为零方差
        self.assertAlmostEqual(step.posterior_cov[1][1], 0.0, places=12)
        self.assertAlmostEqual(step.normalized_residual[0], 1.0 / (2.0 ** 0.5), places=12)


class SameTimeSensorTests(unittest.TestCase):
    def test_two_independent_sensors_share_one_prior_and_one_dt(self):
        cfg = make_config(q=0.0)
        observations = [
            obs("o1", "s1", 0.0, value=1.0),
            obs("o2", "s2", 0.0, received_at=1.0, value=1.2),
        ]
        result = run_forward_filter(observations, cfg, q=0.0)
        self.assertEqual(len(result.steps), 1)
        step = result.steps[0]
        self.assertEqual(step.sensors, ("s1", "s2"))
        self.assertEqual(step.dt, 0.0)
        # 手工融合：先验位置 N(0,1)；s1: z=1,R=1；s2: 去偏后 z=1.0,R=0.5
        # 精度 1 + 1 + 2 = 4 -> 方差 0.25；均值 (1*1 + 2*1.0)/4 = 0.75
        self.assertAlmostEqual(step.posterior_mean[0], 0.75, places=12)
        self.assertAlmostEqual(step.posterior_cov[0][0], 0.25, places=12)
        self.assertEqual(len(step.innovation), 2)
        # 创新协方差：H P H^T 非对角元等于共享先验位置方差 1，
        # 对角元再叠加各自独立的观测噪声（对角 R）
        self.assertAlmostEqual(step.innovation_cov[0][0], 2.0, places=12)
        self.assertAlmostEqual(step.innovation_cov[1][1], 1.5, places=12)
        self.assertAlmostEqual(step.innovation_cov[0][1], 1.0, places=12)
        self.assertAlmostEqual(step.innovation_cov[1][0], 1.0, places=12)


class LongGapTests(unittest.TestCase):
    def test_missing_observations_do_not_update_and_gap_uses_real_dt(self):
        cfg = make_config(q=1.0)
        result = run_forward_filter(
            [obs("o1", "s1", 0.0, value=0.0), obs("o2", "s1", 10.0, value=0.0)],
            cfg,
        )
        self.assertEqual(len(result.steps), 2)
        gap_step = result.steps[1]
        self.assertAlmostEqual(gap_step.dt, 10.0, places=12)
        # 第一步在 t=0 把位置方差压到 0.5；10 秒空档的预测协方差
        # P_pred = F P F^T + Q(10)，没有任何中间步或补零更新
        q10 = 1.0
        expected_q00 = q10 * 10.0 ** 3 / 3.0
        expected_q01 = q10 * 10.0 ** 2 / 2.0
        expected_q11 = q10 * 10.0
        self.assertAlmostEqual(
            gap_step.predicted_cov[0][0], 0.5 + expected_q00, places=10
        )
        self.assertAlmostEqual(
            gap_step.predicted_cov[0][1], expected_q01, places=10
        )
        self.assertAlmostEqual(
            gap_step.predicted_cov[1][1], expected_q11, places=10
        )

    def test_no_observations_only_forecasts_with_covariance_growth(self):
        cfg = make_config(q=1.0)
        result = run_forward_filter([], cfg)
        mean, cov, dt = forecast_to(result, 5.0, cfg.q)
        self.assertEqual(dt, 5.0)
        self.assertEqual(mean, [0.0, 0.0])
        self.assertAlmostEqual(cov[0][0], 1.0 + 125.0 / 3.0, places=12)
        self.assertAlmostEqual(cov[1][1], 5.0, places=12)

    def test_smoothing_empty_track_is_empty_and_well_defined(self):
        cfg = make_config()
        result = run_forward_filter([], cfg)
        self.assertEqual(rts_smooth(result, cfg.q), [])
        self.assertEqual(total_innovation_log_likelihood(result), 0.0)
        self.assertEqual(result.final_mean, list(cfg.initial_mean))
        self.assertEqual(result.final_time, cfg.initial_time)


class LateObservationTests(unittest.TestCase):
    def _replay(self):
        cfg = make_config(
            q=0.01,
            cutoffs=(6.0, 122.0),
            training_end=6.0,
        )
        observations = [
            obs("o1", "s1", 0.0, received_at=0.0, value=1.0),
            obs("o2", "s1", 5.0, received_at=5.0, value=2.0),
            # 迟到：发生在 t=2，但 t=10 才获知
            obs("o3", "s2", 2.0, received_at=10.0, value=1.5),
        ]
        return cfg, ReplayCache(observations, cfg), observations

    def test_late_observation_creates_new_version_without_rewriting_old(self):
        cfg, cache, _ = self._replay()
        early = cache.run_version(6.0, cfg.q)
        late = cache.run_version(122.0, cfg.q)
        self.assertEqual(early.observation_ids, ("o1", "o2"))
        self.assertEqual(late.observation_ids, ("o1", "o3", "o2"))

        # 旧版本对象与其中的数值不被回写
        early_first = early.filter_result.steps[0]
        late_first = late.filter_result.steps[0]
        self.assertEqual(early_first.observation_ids, ("o1",))
        self.assertEqual(late_first.observation_ids, ("o1",))
        self.assertEqual(
            early_first.posterior_mean, late_first.posterior_mean
        )
        self.assertEqual(
            early_first.posterior_cov, late_first.posterior_cov
        )
        # 迟到点之后的同键步骤，晚版本因融合新信息而应当不同
        early_t5 = early.filter_result.steps[1]
        late_t5 = late.filter_result.steps[2]
        self.assertNotEqual(early_t5.posterior_mean, late_t5.posterior_mean)

    def test_cache_returns_same_object_on_replay(self):
        cfg, cache, _ = self._replay()
        first = cache.run_version(6.0, cfg.q)
        second = cache.run_version(6.0, cfg.q)
        self.assertIs(first, second)
        cache.run_version(122.0, cfg.q)
        self.assertEqual(cache.cached_keys(), ((6.0, cfg.q), (122.0, cfg.q)))

    def test_future_observations_do_not_change_past_filter_outputs(self):
        cfg = make_config(q=0.01)
        base = run_forward_filter(
            [obs("o1", "s1", 0.0, value=1.0), obs("o2", "s1", 5.0, value=2.0)],
            cfg,
        )
        with_future = run_forward_filter(
            [
                obs("o1", "s1", 0.0, value=1.0),
                obs("o2", "s1", 5.0, value=2.0),
                obs("o9", "s1", 100.0, value=50.0),
            ],
            cfg,
        )
        for old, new in zip(base.steps, with_future.steps):
            self.assertEqual(old.posterior_mean, new.posterior_mean)
            self.assertEqual(old.posterior_cov, new.posterior_cov)
            self.assertEqual(old.innovation, new.innovation)


class RTSUpdateTests(unittest.TestCase):
    def test_smoother_uses_future_observations_and_last_step_equals_filter(self):
        cfg = make_config(q=0.01)
        observations = [
            obs("o1", "s1", 0.0, value=0.0),
            obs("o2", "s1", 1.0, value=1.0),
            obs("o3", "s1", 2.0, value=2.0),
        ]
        fit = run_forward_filter(observations, cfg)
        smooth = rts_smooth(fit, cfg.q)
        self.assertEqual(len(smooth), len(fit.steps))
        last = smooth[-1]
        self.assertEqual(last.smoothed_mean, last.filtered_mean)
        self.assertEqual(last.smoothed_cov, last.filtered_cov)
        # 早期平滑估计吸收了未来信息，与滤波不同（且位置向直线轨迹靠拢）
        first = smooth[0]
        self.assertNotEqual(first.smoothed_mean, first.filtered_mean)
        # 平滑协方差不大于滤波协方差（半正定差）
        diff = [
            [first.filtered_cov[i][j] - first.smoothed_cov[i][j] for j in range(2)]
            for i in range(2)
        ]
        eigenvalues = sorted(jacobi_eigen(diff)[0])
        self.assertGreaterEqual(eigenvalues[0], -1e-10)

    def test_all_filter_and_smoother_covariances_are_psd(self):
        cfg = make_config(q=0.01)
        observations = [
            obs("o1", "s1", 0.0, value=0.0),
            obs("o2", "s2", 0.0, received_at=0.0, value=0.5),
            obs("o3", "s1", 7.0, value=3.0),
            obs("o4", "s1", 120.0, value=50.0),
        ]
        fit = run_forward_filter(observations, cfg)
        for step in fit.steps:
            check_psd(step.predicted_cov, label="测试预测协方差")
            check_psd(step.posterior_cov, label="测试后验协方差")
        for step in rts_smooth(fit, cfg.q):
            check_psd(step.smoothed_cov, label="测试平滑协方差")
            check_psd(step.filtered_cov, label="测试滤波协方差")

    def test_singular_q_zero_rts_uses_audited_pinv(self):
        cfg = make_config(q=0.0, initial_cov=((1.0, 0.0), (0.0, 0.0)))
        fit = run_forward_filter(
            [obs("o1", "s1", 0.0, value=1.0), obs("o2", "s1", 1.0, value=2.0)],
            cfg, q=0.0,
        )
        smooth = rts_smooth(fit, q=0.0)
        # 不抛异常即说明伪逆路径生效；记录了截断的特征值数量
        self.assertTrue(any(step.pinv_clipped >= 0 for step in smooth))
        # 无噪声速度模型下平滑位置仍在合理量纲内
        for step in smooth:
            for value in step.smoothed_mean:
                self.assertTrue(math.isfinite(value))


class QSelectionTests(unittest.TestCase):
    def test_q_chosen_only_from_training_innovations_and_frozen_afterward(self):
        cfg = make_config(
            q_grid=(0.0, 0.01, 0.1),
            training_end=5.0,
            cutoffs=(5.0, 100.0),
        )
        observations = [
            obs("o1", "s1", 0.0, received_at=0.0, value=0.0),
            obs("o2", "s1", 1.0, received_at=1.0, value=1.1),
            obs("o3", "s1", 2.0, received_at=2.0, value=1.9),
            # 验证段观测：训练截止后才获知，绝不参与 q 选择
            obs("o4", "s1", 3.0, received_at=50.0, value=99.0),
        ]
        selection = select_q_on_training(observations, cfg, 5.0)
        self.assertEqual(selection.training_observation_ids, ("o1", "o2", "o3"))
        training_only_scores = dict(selection.grid_scores)

        # 用全部观测重新做“训练”会改变得分，证明选择口径确实受截止控制
        all_selection = select_q_on_training(observations, cfg, 100.0)
        self.assertNotEqual(
            dict(all_selection.grid_scores), training_only_scores
        )

        # 验证段冻结 q：两个版本用同一个 q 构建
        cache = ReplayCache(observations, cfg)
        early = cache.run_version(5.0, selection.q)
        late = cache.run_version(100.0, selection.q)
        self.assertEqual(early.q_used, late.q_used)

    def test_ties_select_smaller_q_deterministically(self):
        cfg = make_config(q_grid=(0.5, 0.0, 0.1), training_end=1.0)
        observations = [obs("o1", "s1", 0.0, value=1.0)]
        selection = select_q_on_training(observations, cfg, 1.0)
        # 单观测在 dt=0 上，所有 q 的创新似然相同 -> 并列选最小 q
        self.assertEqual(selection.q, 0.0)

    def test_innovation_likelihood_sums_per_group(self):
        cfg = make_config()
        fit = run_forward_filter(
            [obs("o1", "s1", 0.0, value=1.0), obs("o2", "s1", 1.0, value=2.0)],
            cfg,
        )
        manual = sum(step.log_likelihood for step in fit.steps)
        self.assertAlmostEqual(
            total_innovation_log_likelihood(fit), manual, places=14
        )


if __name__ == "__main__":
    unittest.main()
