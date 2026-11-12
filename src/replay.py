"""按获知版本（known_at）重放的缓存与回退重算策略。

离线重放规则：
1. 每个 known_at 截止对应一个不可变的“已知版本”：先按 received_at
   筛出当时可得的观测，再按 event_at 重放滤波；
2. 迟到观测只会产生截止更晚的新版本，已经算出的旧版本结果对象
   不会被回写或原地修改（测试通过对象身份与数值快照双重检查）；
3. 缓存在一份不可变观测批次内按 (known_at, q) 记忆化，命中即返回；
   如果原始观测批次本身发生变化，正确做法是丢弃整个缓存、从原始
   文件重新构建（确定性重算），而不是增量修补旧版本。
"""

from dataclasses import dataclass

from .estimator import (
    FilterResult,
    SmoothStep,
    forecast_to,
    run_forward_filter,
    rts_smooth,
    total_innovation_log_likelihood,
)
from .observations import Observation, filter_known_at


@dataclass(frozen=True)
class QSelection:
    q: float
    grid_scores: tuple[tuple[float, float], ...]
    training_observation_ids: tuple[str, ...]
    training_end: float


@dataclass
class VersionResult:
    known_at: float
    q_used: float
    filter_result: FilterResult
    smooth_steps: list[SmoothStep]
    forecast_mean: list[float]
    forecast_cov: list[list[float]]
    forecast_dt: float
    observation_ids: tuple[str, ...]


def select_q_on_training(observations, config, training_end):
    """只用训练时段内“当时可得”的创新似然在 q 网格上选 q。

    训练口径：received_at <= training_end（迟到样本在训练段尚未获知，
    自然不会参与）。似然并列时取较小的 q，保证选择确定且偏向简单模型。
    验证段必须冻结这里返回的 q，不得重新拟合。
    """
    if not config.q_grid:
        raise ValueError("q_grid 为空，无法选择过程噪声")
    training_obs = filter_known_at(observations, training_end)
    training_ids = tuple(
        obs.observation_id for obs in sorted(training_obs, key=lambda o: o.event_at)
    )
    scores = []
    for candidate in config.q_grid:
        fit = run_forward_filter(training_obs, config, q=candidate)
        scores.append((candidate, total_innovation_log_likelihood(fit)))
    best_q = max(scores, key=lambda pair: (pair[1], -pair[0]))[0]
    return QSelection(
        q=best_q,
        grid_scores=tuple(scores),
        training_observation_ids=training_ids,
        training_end=training_end,
    )


class ReplayCache:
    """版本化离线重放器；每个版本独立计算、独立缓存、互不回写。"""

    def __init__(self, observations, config):
        # 防御性拷贝：缓存生命周期内观测批次视为不可变。
        self._observations: list[Observation] = list(observations)
        self._config = config
        self._cache: dict[tuple[float, float], VersionResult] = {}

    def known_observations(self, known_at):
        return filter_known_at(self._observations, known_at)

    def run_version(self, known_at, q):
        """构建（或取回缓存的）某个 known_at 截止下的完整滤波/平滑版本。"""
        key = (known_at, q)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        known_obs = self.known_observations(known_at)
        filter_result = run_forward_filter(known_obs, self._config, q=q)
        smooth_steps = rts_smooth(filter_result, q)
        forecast_mean, forecast_cov, forecast_dt = forecast_to(
            filter_result, known_at, q
        )
        version = VersionResult(
            known_at=known_at,
            q_used=q,
            filter_result=filter_result,
            smooth_steps=smooth_steps,
            forecast_mean=forecast_mean,
            forecast_cov=forecast_cov,
            forecast_dt=forecast_dt,
            observation_ids=tuple(
                obs.observation_id
                for obs in sorted(known_obs, key=lambda o: (o.event_at, o.observation_id))
            ),
        )
        self._cache[key] = version
        return version

    def cached_keys(self):
        return tuple(sorted(self._cache))
