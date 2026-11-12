"""变时间间隔的前向卡尔曼滤波与 RTS 平滑。

核心口径（与项目问题陈述一一对应）：
- 传播使用观测之间的真实 dt；缺测不产生任何更新，长空档只通过
  F(dt)、Q(dt) 把协方差向前传播，不存在固定步长或补零；
- 同一 event_at 的多台独立仪器在同一个先验上做联合更新，
  dt 只传播一次；
- 更新采用 Joseph 形式，每步检查协方差对称半正定，不裁剪特征值；
- 创新与标准化残差按观测维度逐一记录，供训练段似然选 q 使用；
- RTS 平滑只在一个已知版本内部使用该版本的全部未来观测，
  平滑结果与实时滤波结果分别存放，禁止互相冒充。
"""

import math
from dataclasses import dataclass, field

from .linalg import (
    add,
    joseph_update,
    matmul,
    matvec,
    subtract,
    symmetrize_covariance,
    symmetric_pinv,
    transpose,
    vec_add,
    vec_subtract,
)
from .model import noise_covariance, observation_matrix, process_noise, transition_matrix


@dataclass
class FilterStep:
    event_at: float
    observation_ids: tuple[str, ...]
    sensors: tuple[str, ...]
    dt: float
    predicted_mean: list[float]
    predicted_cov: list[list[float]]
    posterior_mean: list[float]
    posterior_cov: list[list[float]]
    innovation: list[float]                 # 每个观测维度一个分量
    innovation_cov: list[list[float]]       # S
    normalized_residual: list[float]        # S^{-1/2} 意义下的标准化残差分量
    log_likelihood: float                   # 该联合观测组的创新高斯对数似然
    pinv_clipped: int = 0                   # RTS 阶段诊断字段，滤波时为 0


@dataclass
class FilterResult:
    steps: list[FilterStep] = field(default_factory=list)
    final_mean: list[float] = field(default_factory=lambda: [0.0, 0.0])
    final_cov: list[list[float]] = field(
        default_factory=lambda: [[0.0, 0.0], [0.0, 0.0]]
    )
    final_time: float = 0.0


def _group_by_event_time(observations):
    """按 event_at 升序分组；同刻观测保持输入顺序作为确定的联合更新顺序。"""
    ordered = sorted(observations, key=lambda obs: (obs.event_at, obs.observation_id))
    groups = []
    for obs in ordered:
        if not groups or groups[-1][0].event_at != obs.event_at:
            groups.append([obs])
        else:
            groups[-1].append(obs)
    return groups


def _innovation_log_likelihood(innovation, s, s_inv):
    """创新的高斯对数似然：

    L = -0.5 * (m ln(2pi) + ln det S + nu^T S^{-1} nu)

    S = H P H^T + R 在本模型中对称正定（观测方差严格为正），
    但 H P H^T 带来非对角元（同刻仪器共享先验位置方差），
    因此行列式走通用的部分主元消去而不是对角元乘积。
    """
    m = len(innovation)
    determinant = _determinant(s)
    quadratic = sum(
        innovation[i] * sum(s_inv[i][j] * innovation[j] for j in range(m))
        for i in range(m)
    )
    return -0.5 * (
        m * math.log(2.0 * math.pi) + math.log(determinant) + quadratic
    )


def _determinant(a):
    n = len(a)
    m = [row[:] for row in a]
    determinant = 1.0
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) < 1e-300:
            return 0.0
        if pivot != col:
            m[col], m[pivot] = m[pivot], m[col]
            determinant = -determinant
        determinant *= m[col][col]
        pivot_value = m[col][col]
        for row in range(col + 1, n):
            factor = m[row][col] / pivot_value
            m[row] = [
                value - factor * pivot_value_row
                for value, pivot_value_row in zip(m[row], m[col])
            ]
    return determinant


def _normalized_residuals(innovation, s):
    """逐维标准化残差 nu_i / sqrt(S_ii)。

    S = H P H^T + R 的对角元是每个观测维度自身的边际创新方差，
    因此逐维分量具有单位边际标准差口径；跨维联合评估由对数似然
    （含完整 S^{-1} 二次型与行列式）承担。
    """
    return [nu / math.sqrt(s[i][i]) for i, nu in enumerate(innovation)]


def run_forward_filter(observations, config, q=None):
    """按真实时间间隔重放一组“当时可得”的观测。

    observations 必须已经按 known_at 截止过滤；函数内部再按 event_at
    排序分组，保证先筛选、后按发生时刻重放的口径。
    """
    q = config.q if q is None else q
    mean = list(config.initial_mean)
    cov = [list(row) for row in config.initial_covariance]
    current_time = config.initial_time
    # 初值本身也要满足对称半正定不变量（允许半正定，不要求严格正定）。
    cov = symmetrize_covariance(cov, label="初始协方差")
    result = FilterResult(final_mean=mean, final_cov=cov, final_time=current_time)

    for group in _group_by_event_time(observations):
        event_time = group[0].event_at
        dt = event_time - current_time
        if dt < -1e-12:
            raise ValueError(
                f"观测时刻 {event_time} 早于当前滤波时间 {current_time}，时间序被破坏"
            )
        dt = max(dt, 0.0)

        # 1) 真实 dt 下的预测；缺测期间没有别的步骤，空档完整落在这一步。
        F = transition_matrix(dt)
        predicted_mean = matvec(F, mean)
        predicted_cov = symmetrize_covariance(
            add(matmul(matmul(F, cov), transpose(F)), process_noise(q, dt)),
            label="预测协方差",
        )

        # 2) 同刻联合观测：同一个先验，一次更新，dt 不重复传播。
        sensor_names = tuple(obs.sensor for obs in group)
        H = observation_matrix(sensor_names)
        R = noise_covariance(sensor_names, config.sensors)
        innovation = [
            obs.value - config.sensors[obs.sensor].bias - predicted_mean[0]
            for obs in group
        ]
        post_mean, post_cov, _gain, S, S_inv = joseph_update(
            predicted_mean, predicted_cov, H, innovation, R
        )

        step = FilterStep(
            event_at=event_time,
            observation_ids=tuple(obs.observation_id for obs in group),
            sensors=sensor_names,
            dt=dt,
            predicted_mean=predicted_mean,
            predicted_cov=predicted_cov,
            posterior_mean=post_mean,
            posterior_cov=post_cov,
            innovation=innovation,
            innovation_cov=S,
            normalized_residual=_normalized_residuals(innovation, S),
            log_likelihood=_innovation_log_likelihood(innovation, S, S_inv),
        )
        result.steps.append(step)
        mean, cov, current_time = post_mean, post_cov, event_time

    result.final_mean, result.final_cov, result.final_time = mean, cov, current_time
    return result


def forecast_to(filter_result, target_time, q):
    """把最后一个后验按真实间隔传播到 target_time（无观测更新）。"""
    dt = target_time - filter_result.final_time
    if dt < -1e-12:
        raise ValueError("forecast 目标时刻早于滤波最后时刻")
    dt = max(dt, 0.0)
    F = transition_matrix(dt)
    mean = matvec(F, filter_result.final_mean)
    cov = symmetrize_covariance(
        add(
            matmul(matmul(F, filter_result.final_cov), transpose(F)),
            process_noise(q, dt),
        ),
        label="预报协方差",
    )
    return mean, cov, dt


@dataclass
class SmoothStep:
    event_at: float
    observation_ids: tuple[str, ...]
    filtered_mean: list[float]
    filtered_cov: list[list[float]]
    smoothed_mean: list[float]
    smoothed_cov: list[list[float]]
    pinv_clipped: int


def rts_smooth(filter_result, q):
    """RTS 后向平滑。

    对每个滤波步 k，用预测协方差 P_{k+1|k} 的伪逆构造平滑增益
    G_k = P_{k|k} F^T P_{k+1|k}^{-1}。q=0 且初速度方差为 0 时
    预测协方差可能奇异，伪逆会显式截断近零特征值并记录数量，
    而不是静默求逆失败或偷偷加对角项。
    """
    steps = filter_result.steps
    if not steps:
        return []

    smoothed_means = [None] * len(steps)
    smoothed_covs = [None] * len(steps)
    clipped_counts = [0] * len(steps)
    smoothed_means[-1] = list(steps[-1].posterior_mean)
    smoothed_covs[-1] = [row[:] for row in steps[-1].posterior_cov]

    for k in range(len(steps) - 2, -1, -1):
        next_step = steps[k + 1]
        F = transition_matrix(next_step.dt)
        p_pred_inv, _rank, clipped, _eig = symmetric_pinv(next_step.predicted_cov)
        clipped_counts[k] = clipped
        gain = matmul(
            matmul(steps[k].posterior_cov, transpose(F)), p_pred_inv
        )
        diff_mean = vec_subtract(smoothed_means[k + 1], next_step.predicted_mean)
        smooth_mean = vec_add(
            steps[k].posterior_mean, matvec(gain, diff_mean)
        )
        diff_cov = subtract(
            smoothed_covs[k + 1], next_step.predicted_cov
        )
        smooth_cov = add(
            steps[k].posterior_cov,
            matmul(matmul(gain, diff_cov), transpose(gain)),
        )
        smooth_cov = symmetrize_covariance(smooth_cov, label=f"RTS 平滑协方差（步 {k}）")
        smoothed_means[k] = smooth_mean
        smoothed_covs[k] = smooth_cov

    return [
        SmoothStep(
            event_at=step.event_at,
            observation_ids=step.observation_ids,
            filtered_mean=list(step.posterior_mean),
            filtered_cov=[row[:] for row in step.posterior_cov],
            smoothed_mean=smoothed_means[i],
            smoothed_cov=smoothed_covs[i],
            pinv_clipped=clipped_counts[i],
        )
        for i, step in enumerate(steps)
    ]


def total_innovation_log_likelihood(filter_result):
    """一个 q 下全部观测组的创新对数似然之和（训练段选 q 的目标函数）。"""
    return sum(step.log_likelihood for step in filter_result.steps)
