"""状态估计组件的自定义异常。"""


class StateEstimationError(ValueError):
    """输入数据或模型配置不满足契约。"""


class ObservationConflictError(StateEstimationError):
    """同一唯一键的重送内容与首次记录不一致。"""


class CovarianceError(StateEstimationError):
    """协方差矩阵不满足对称半正定不变量。"""
