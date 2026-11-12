import math
import unittest

from src.errors import CovarianceError
from src.linalg import (
    add,
    check_psd,
    inverse,
    jacobi_eigen,
    joseph_update,
    matmul,
    matvec,
    symmetric_part,
    symmetric_pinv,
    symmetrize_covariance,
    transpose,
)


class LinalgTests(unittest.TestCase):
    def test_matmul_matches_known_product(self):
        a = [[1, 2], [3, 4], [5, 6]]
        b = [[7, 8, 9], [10, 11, 12]]
        expected = [[27, 30, 33], [61, 68, 75], [95, 106, 117]]
        self.assertEqual(matmul(a, b), expected)
        self.assertEqual(matvec(a, [1, 1]), [3, 7, 11])

    def test_jacobi_recovers_diagonal_eigenvalues(self):
        matrix = [[2.0, 0.0], [0.0, 0.5]]
        values, vectors = jacobi_eigen(matrix)
        self.assertAlmostEqual(sorted(values)[0], 0.5, places=12)
        self.assertAlmostEqual(sorted(values)[1], 2.0, places=12)
        # 特征值顺序不固定，用 A = V diag(lambda) V^T 重建后比较
        diag = [[values[0], 0.0], [0.0, values[1]]]
        reconstructed = matmul(matmul(vectors, diag), transpose(vectors))
        for i in range(2):
            for j in range(2):
                self.assertAlmostEqual(reconstructed[i][j], matrix[i][j], places=12)

    def test_jacobi_converges_for_tiny_scale_matrices(self):
        # 元素整体极小时，相对范数判据仍需正确对角化
        scale = 1e-10
        matrix = [[scale, 0.3 * scale], [0.3 * scale, 0.5 * scale]]
        values, vectors = jacobi_eigen(matrix)
        diag = [[values[0], 0.0], [0.0, values[1]]]
        reconstructed = matmul(matmul(vectors, diag), transpose(vectors))
        for i in range(2):
            for j in range(2):
                self.assertAlmostEqual(
                    reconstructed[i][j] / scale, matrix[i][j] / scale, places=10
                )
        self.assertTrue(all(v > 0 for v in values))

    def test_psd_check_rejects_negative_eigenvalue_without_clipping(self):
        bad = [[1.0, 0.0], [0.0, -0.5]]
        with self.assertRaises(CovarianceError):
            check_psd(bad)
        with self.assertRaises(CovarianceError):
            symmetrize_covariance([row[:] for row in bad])
        # 输入没有被原地“修复”
        self.assertEqual(bad[1][1], -0.5)

    def test_singular_symmetric_pinv_explicitly_clips_zero_eigenvalue(self):
        singular = [[1.0, 0.0], [0.0, 0.0]]
        pinv, rank, clipped, eigenvalues = symmetric_pinv(singular)
        self.assertEqual(rank, 1)
        self.assertEqual(clipped, 1)
        self.assertAlmostEqual(pinv[0][0], 1.0, places=12)
        self.assertEqual(pinv[1][1], 0.0)
        self.assertIn(0.0, [round(v, 12) for v in eigenvalues])

    def test_inverse_round_trip(self):
        matrix = [[4.0, 1.0], [2.0, 3.0]]
        inv = inverse(matrix)
        product = matmul(matrix, inv)
        for i in range(2):
            for j in range(2):
                self.assertAlmostEqual(product[i][j], 1.0 if i == j else 0.0, places=12)

    def test_joseph_hand_check_posterior(self):
        """位置方差1、速度方差0、q=0、dt=0，一次位置观测1、方差1。"""
        prior_mean = [0.0, 0.0]
        prior_cov = [[1.0, 0.0], [0.0, 0.0]]
        h = [[1.0, 0.0]]
        r = [[1.0]]
        innovation = [1.0]  # 观测 1 减先验位置 0
        mean, cov, gain, s, s_inv = joseph_update(
            prior_mean, prior_cov, h, innovation, r
        )
        self.assertAlmostEqual(mean[0], 0.5, places=12)
        self.assertAlmostEqual(mean[1], 0.0, places=12)
        self.assertAlmostEqual(cov[0][0], 0.5, places=12)
        # 半正定：速度方向仍为零方差，不要求严格正定
        check_psd(cov)
        self.assertAlmostEqual(s[0][0], 2.0, places=12)
        self.assertAlmostEqual(gain[0][0], 0.5, places=12)
        self.assertAlmostEqual(gain[1][0], 0.0, places=12)

    def test_joseph_two_sensor_joint_matches_sequential_scalar_updates(self):
        """联合更新与“同先验确定顺序”的标量序列方案数值等价。

        序列方案在同一时刻不重复传播 dt；第二步的创新基于第一步
        更新后的条件均值重新构造（等价于信息滤波的逐行吸收）。
        """
        prior_mean = [0.0, 0.0]
        prior_cov = [[1.0, 0.0], [0.0, 0.0]]
        h = [[1.0, 0.0], [1.0, 0.0]]
        r = [[1.0, 0.0], [0.0, 0.5]]
        innovation = [1.0, 1.0]
        joint_mean, joint_cov, *_ = joseph_update(
            prior_mean, prior_cov, h, innovation, r
        )
        seq_mean, seq_cov, *_ = joseph_update(
            prior_mean, prior_cov, [[1.0, 0.0]], [innovation[0]], [[1.0]]
        )
        # 第二个创新相对第一步更新后的均值重新计算，不重复传播
        second_innovation = [innovation[1] - (seq_mean[0] - prior_mean[0])]
        seq_mean, seq_cov, *_ = joseph_update(
            seq_mean, seq_cov, [[1.0, 0.0]], second_innovation, [[0.5]]
        )
        for i in range(2):
            self.assertAlmostEqual(joint_mean[i], seq_mean[i], places=12)
            for j in range(2):
                self.assertAlmostEqual(joint_cov[i][j], seq_cov[i][j], places=12)

    def test_symmetrize_only_removes_floating_noise(self):
        noisy = [[1.0, 0.2 + 1e-15], [0.2, 0.5]]
        cleaned = symmetrize_covariance(noisy)
        self.assertTrue(all(math.isfinite(v) for row in cleaned for v in row))
        self.assertAlmostEqual(cleaned[0][1], symmetric_part(noisy)[0][1], places=15)

    def test_add_transpose_basics(self):
        a = [[1.0, 2.0], [3.0, 4.0]]
        self.assertEqual(add(a, a), [[2.0, 4.0], [6.0, 8.0]])
        self.assertEqual(transpose(a), [[1.0, 3.0], [2.0, 4.0]])


if __name__ == "__main__":
    unittest.main()
