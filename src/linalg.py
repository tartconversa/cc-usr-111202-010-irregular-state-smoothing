"""状态估计用到的小规模线性代数工具（仅标准库）。

状态维度固定为 2（位置、速度），观测按同一发生时刻分组后做联合更新，
矩阵规模都很小，手写实现即可，不引入第三方依赖。

设计约束：
- Joseph 更新后只做浮点对称化，绝不通过裁剪负特征值来掩盖错误；
- 对称半正定检查独立成函数，预测与更新步骤都会调用；
- 对称矩阵求逆采用基于 Jacobi 特征分解的伪逆，允许初始协方差半正定
  （速度方差为 0），并在结果中报告被截断的特征值个数。
"""

import math

from .errors import CovarianceError

# 浮点容差。绝对容差配合按矩阵范数缩放的相对容差一起使用。
_ATOL = 1e-12
_RTOL = 1e-10


def zeros(rows, cols):
    return [[0.0 for _ in range(cols)] for _ in range(rows)]


def identity(n):
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def copy_matrix(a):
    return [row[:] for row in a]


def matmul(a, b):
    rows, inner, cols = len(a), len(b), len(b[0])
    out = zeros(rows, cols)
    for i in range(rows):
        ai = a[i]
        oi = out[i]
        for k in range(inner):
            aik = ai[k]
            if aik == 0.0:
                continue
            bk = b[k]
            for j in range(cols):
                oi[j] += aik * bk[j]
    return out


def matvec(a, x):
    return [sum(aij * xj for aij, xj in zip(row, x)) for row in a]


def add(a, b):
    return [[a[i][j] + b[i][j] for j in range(len(a[0]))] for i in range(len(a))]


def subtract(a, b):
    return [[a[i][j] - b[i][j] for j in range(len(a[0]))] for i in range(len(a))]


def vec_add(a, b):
    return [x + y for x, y in zip(a, b)]


def vec_subtract(a, b):
    return [x - y for x, y in zip(a, b)]


def transpose(a):
    return [list(col) for col in zip(*a)]


def scale(a, factor):
    return [[a[i][j] * factor for j in range(len(a[0]))] for i in range(len(a))]


def symmetric_part(a):
    """只用于消除浮点级不对称（量级由容差检查约束），不改变代数结构。"""
    n = len(a)
    return [[0.5 * (a[i][j] + a[j][i]) for j in range(n)] for i in range(n)]


def symmetrize_covariance(a, label="协方差"):
    """对称化并检查不对称只处于浮点噪声水平，然后做 PSD 校验。"""
    tol = _ATOL + _RTOL * max(
        (abs(value) for row in a for value in row), default=0.0
    )
    n = len(a)
    for i in range(n):
        for j in range(i + 1, n):
            if abs(a[i][j] - a[j][i]) > tol:
                raise CovarianceError(
                    f"{label}不对称超过浮点容差：P[{i}][{j}]={a[i][j]!r}, "
                    f"P[{j}][{i}]={a[j][i]!r}"
                )
    p = symmetric_part(a)
    check_psd(p, label)
    return p


def check_symmetric(a):
    n = len(a)
    tol = _ATOL + _RTOL * max(
        (abs(value) for row in a for value in row), default=0.0
    )
    for i in range(n):
        if len(a[i]) != n:
            raise CovarianceError("矩阵不是方阵，无法按协方差处理")
        for j in range(i + 1, n):
            if abs(a[i][j] - a[j][i]) > tol:
                raise CovarianceError(f"矩阵不对称超过浮点容差：[{i}][{j}] 与 [{j}][{i}]")


def check_psd(a, label="协方差"):
    """通过对称 Jacobi 特征分解验证半正定。

    负特征值只允许出现在浮点容差内；不在这里做任何裁剪——
    裁剪由调用方在明确的伪逆场景中显式完成。
    """
    check_symmetric(a)
    eigenvalues = _eigenvalues_symmetric(a)
    tol = _ATOL + _RTOL * max((abs(v) for v in eigenvalues), default=0.0)
    negatives = [v for v in eigenvalues if v < -tol]
    if negatives:
        raise CovarianceError(
            f"{label}出现显著负特征值 {negatives}（容差 {tol:.3g}），"
            "拒绝通过裁剪掩盖数值或建模错误"
        )


def jacobi_eigen(a, max_sweeps=100):
    """对称矩阵的循环 Jacobi 特征分解，返回 (特征值, 正交特征向量列)。

    用于 2x2 协方差；规模稍大时同样正确，只是效率一般。
    收敛判据用相对 Frobenius 范数（off^2 / frob^2），对元素整体
    很小或很大的矩阵都尺度无关，避免绝对阈值造成提前退出。
    """
    check_symmetric(a)
    n = len(a)
    m = [row[:] for row in a]
    v = identity(n)
    for _ in range(max_sweeps):
        off_sq = sum(
            m[i][j] * m[i][j] for i in range(n) for j in range(i + 1, n)
        )
        frob_sq = sum(value * value for row in m for value in row)
        if off_sq <= (_RTOL * _RTOL) * max(frob_sq, _ATOL * _ATOL):
            break
        for p in range(n - 1):
            for q in range(p + 1, n):
                apq = m[p][q]
                if abs(apq) <= _ATOL:
                    continue
                app, aqq = m[p][p], m[q][q]
                phi = 0.5 * math.atan2(2.0 * apq, aqq - app)
                c, s = math.cos(phi), math.sin(phi)
                for k in range(n):
                    mkp, mkq = m[k][p], m[k][q]
                    m[k][p] = c * mkp - s * mkq
                    m[k][q] = s * mkp + c * mkq
                for k in range(n):
                    mpk, mqk = m[p][k], m[q][k]
                    m[p][k] = c * mpk - s * mqk
                    m[q][k] = s * mpk + c * mqk
                for k in range(n):
                    vkp, vkq = v[k][p], v[k][q]
                    v[k][p] = c * vkp - s * vkq
                    v[k][q] = s * vkp + c * vkq
    eigenvalues = [m[i][i] for i in range(n)]
    return eigenvalues, v


def _eigenvalues_symmetric(a):
    values, _ = jacobi_eigen(a)
    return values


def symmetric_pinv(a, rank_tol=None):
    """对称矩阵的特征分解伪逆。

    RTS 平滑在 q=0、初始速度方差为 0 时会遇到奇异预测协方差，
    普通求逆会失败。这里显式截断接近 0 的特征值并返回
    (伪逆, 秩, 截断特征值个数, 特征值列表)，把奇异情况变成
    可审计的行为而不是静默处理。
    """
    check_symmetric(a)
    eigenvalues, vectors = jacobi_eigen(a)
    largest = max((abs(v) for v in eigenvalues), default=0.0)
    tol = rank_tol if rank_tol is not None else max(_ATOL, _RTOL * largest)
    inv_diag = [0.0] * len(eigenvalues)
    clipped = 0
    for i, value in enumerate(eigenvalues):
        if abs(value) <= tol:
            inv_diag[i] = 0.0
            clipped += 1
        else:
            inv_diag[i] = 1.0 / value
    n = len(eigenvalues)
    out = zeros(n, n)
    for i in range(n):
        for j in range(n):
            total = 0.0
            for k in range(n):
                total += vectors[i][k] * inv_diag[k] * vectors[j][k]
            out[i][j] = total
    rank = n - clipped
    return symmetric_part(out), rank, clipped, eigenvalues


def inverse(a):
    """一般方阵求逆（高斯-若尔当带部分主元），用于创新协方差等非对称情形。"""
    n = len(a)
    for row in a:
        if len(row) != n:
            raise CovarianceError("求逆目标不是方阵")
    m = [row[:] + [1.0 if i == j else 0.0 for j in range(n)] for i, row in enumerate(a)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) <= _ATOL:
            raise CovarianceError("矩阵奇异，无法求逆；对称半正定场景请使用 symmetric_pinv")
        if pivot != col:
            m[col], m[pivot] = m[pivot], m[col]
        pivot_value = m[col][col]
        m[col] = [value / pivot_value for value in m[col]]
        for row in range(n):
            if row == col:
                continue
            factor = m[row][col]
            if factor == 0.0:
                continue
            m[row] = [
                value - factor * pivot_row_value
                for value, pivot_row_value in zip(m[row], m[col])
            ]
    return [row[n:] for row in m]


def joseph_update(prior_mean, prior_cov, h, innovation, r):
    """Joseph 形式的卡尔曼更新，返回 (后验均值, 后验协方差, 增益, S, S_inv)。

    P_post = (I - K H) P (I - K H)^T + K R K^T
    该形式对舍入误差更稳健，且在半正定初值（初始速度方差为 0）下
    仍保持协方差半正定。h 为 m x 2 观测矩阵，r 为 m x m 观测噪声协方差。
    """
    n = len(prior_mean)
    s = add(matmul(matmul(h, prior_cov), transpose(h)), r)
    s_inv = inverse(s)
    # K = P H^T S^{-1}
    gain = matmul(matmul(prior_cov, transpose(h)), s_inv)
    identity_n = identity(n)
    ih = subtract(identity_n, matmul(gain, h))
    term1 = matmul(matmul(ih, prior_cov), transpose(ih))
    term2 = matmul(matmul(gain, r), transpose(gain))
    post_cov = symmetrize_covariance(add(term1, term2), label="更新后协方差")
    post_mean = [m + k for m, k in zip(prior_mean, matvec(gain, innovation))]
    return post_mean, post_cov, gain, s, s_inv
