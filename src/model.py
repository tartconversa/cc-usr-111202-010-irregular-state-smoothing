"""恒速（白加速度噪声）运动模型的矩阵构造。

状态 x = [位置, 速度]^T，真实时间间隔 dt（秒）：

    F = [[1, dt], [0, 1]]
    Q = q * [[dt^3/3, dt^2/2], [dt^2/2, dt]]

每台仪器只观测位置：H = [1, 0]，观测减已知偏置后进入创新。
"""

from .errors import StateEstimationError
from .linalg import check_psd


def transition_matrix(dt):
    if dt < 0:
        raise StateEstimationError(f"时间间隔必须非负，收到 dt={dt}")
    return [[1.0, float(dt)], [0.0, 1.0]]


def process_noise(q, dt):
    """连续白加速度噪声在真实间隔 dt 下的离散协方差。"""
    if q < 0:
        raise StateEstimationError(f"过程噪声谱强度 q 必须非负，收到 {q}")
    if dt < 0:
        raise StateEstimationError(f"时间间隔必须非负，收到 dt={dt}")
    matrix = [
        [q * dt ** 3 / 3.0, q * dt ** 2 / 2.0],
        [q * dt ** 2 / 2.0, q * dt],
    ]
    check_psd(matrix, label="过程噪声协方差 Q")
    return matrix


def observation_matrix(sensor_names):
    """同一发生时刻的多仪器联合观测矩阵，每个传感器一行 [1, 0]。"""
    return [[1.0, 0.0] for _ in sensor_names]


def noise_covariance(sensor_names, sensors):
    """独立仪器的观测噪声协方差为对角阵，对角元取各自已知正方差。"""
    m = len(sensor_names)
    matrix = [[0.0] * m for _ in range(m)]
    for i, name in enumerate(sensor_names):
        matrix[i][i] = sensors[name].variance
    return matrix
