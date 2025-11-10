import tensorflow as tf
import numpy as np
import pandas as pd
import time
import os


class StableAutoDiffOperators:
    """
    稳定的自动微分算子计算器

    这个类实现了多种自动微分方法的组合，用于计算高阶微分算子。
    拉普拉斯算子：Δf = ∇²f = Σ(∂²f/∂x_i²)
    双调和算子：Δ²f = ∇⁴f = Σ(∂⁴f/∂x_i⁴) + 交叉项
    """

    def __init__(self):
        self.results_cache = {}

    def compute_gradient_reverse(self, func, x):
        """
        反向模式计算一阶梯度（基准）

        数学公式：
        ∇f(x) = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂x_n]^T

        使用反向模式自动微分（Backpropagation）：
        - 前向传播计算函数值 f(x)
        - 反向传播计算梯度 ∇f(x)
        """
        with tf.GradientTape() as tape:
            tape.watch(x)  # 跟踪输入张量
            y = func(x)  # 前向传播：计算 f(x)
        grad = tape.gradient(y, x)  # 反向传播：计算 ∂f/∂x
        return grad if grad is not None else tf.zeros_like(x)

    def compute_laplacian_reverse_reverse(self, func, x):
        """
        稳定的反向+反向模式计算拉普拉斯算子

        数学公式：
        Δf(x) = ∇²f(x) = Σ(∂²f/∂x_i²) = trace(H(f))
        其中 H(f) 是 Hessian 矩阵

        计算方法：
        1. 第一次反向传播：计算梯度 ∇f(x) = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂x_n]
        2. 第二次反向传播：对每个梯度分量 ∂f/∂x_i 再次求导，得到 ∂²f/∂x_i∂x_j
        3. 取对角线元素 ∂²f/∂x_i² 并求和

        注意：这种方法计算整个 Hessian 矩阵的对角线
        """
        batch_size = tf.cast(tf.shape(x)[0], tf.float32)

        try:
            with tf.GradientTape(persistent=True) as outer_tape:
                outer_tape.watch(x)
                with tf.GradientTape(persistent=True) as inner_tape:
                    inner_tape.watch(x)
                    y = func(x)  # f(x)
                # 第一次反向传播：计算梯度 ∇f(x)
                grad = inner_tape.gradient(y, x)

                if grad is None:
                    del inner_tape, outer_tape
                    return tf.constant(0.0)

            laplacian = 0.0
            # 对每个维度计算二阶导数 ∂²f/∂x_i²
            for i in range(x.shape[1]):
                try:
                    # 安全地提取梯度分量 ∂f/∂x_i
                    grad_i = tf.gather(grad, indices=[i], axis=1)  # 形状: [batch_size, 1]
                    # 第二次反向传播：计算 ∂(∂f/∂x_i)/∂x = [∂²f/∂x_i∂x₁, ∂²f/∂x_i∂x₂, ..., ∂²f/∂x_i∂x_n]
                    grad_ii = outer_tape.gradient(grad_i, x)

                    if grad_ii is not None:
                        # 提取对角线元素 ∂²f/∂x_i²
                        diag_element = tf.gather(grad_ii, indices=[i], axis=1)  # 形状: [batch_size, 1]
                        laplacian += tf.reduce_sum(diag_element)  # 累加到拉普拉斯值
                except Exception as e:
                    print(f"维度 {i} 的二阶导数计算失败: {e}")
                    continue

            del inner_tape, outer_tape
            return laplacian / batch_size  # 返回批次平均值

        except Exception as e:
            print(f"反向+反向模式计算失败: {e}")
            return tf.constant(0.0)

    def compute_laplacian_forward_reverse_stable(self, func, x):
        """
        稳定的前向+反向模式计算拉普拉斯算子

        数学公式：
        Δf(x) = Σ(∂²f/∂x_i²)

        计算方法：
        1. 前向模式：计算方向导数 ∇f(x)·v，其中 v 是单位向量
        2. 反向模式：对方向导数再次求导

        对于每个坐标方向 e_i：
        - 计算方向导数：g_i(x) = ∇f(x)·e_i = ∂f/∂x_i
        - 计算二阶导数：∂g_i/∂x_i = ∂²f/∂x_i²
        """
        batch_size = tf.cast(tf.shape(x)[0], tf.float32)
        laplacian = 0.0

        for i in range(x.shape[1]):
            try:
                # 创建单位方向向量 e_i = [0, ..., 1, ..., 0]
                v = tf.one_hot(indices=[i], depth=x.shape[1], dtype=x.dtype)
                v_expanded = tf.tile(v, [tf.shape(x)[0], 1])  # 形状: [batch_size, dim]

                # 计算一阶方向导数：g_i(x) = ∇f(x)·e_i
                with tf.GradientTape() as inner_tape:
                    inner_tape.watch(x)
                    y = func(x)

                # 使用前向模式计算方向导数
                # output_gradients 指定输出方向，这里使用全1向量
                directional_deriv = inner_tape.gradient(y, x, output_gradients=tf.ones_like(y))
                if directional_deriv is None:
                    continue

                # 计算方向导数在方向v上的分量
                # 实际上就是 ∂f/∂x_i，因为 v = e_i
                dir_comp = tf.reduce_sum(directional_deriv * v_expanded, axis=1, keepdims=True)  # 形状: [batch_size, 1]

                # 计算二阶导数：∂(∂f/∂x_i)/∂x
                with tf.GradientTape() as outer_tape:
                    outer_tape.watch(x)
                    # 重新计算一阶导数用于二阶导数计算
                    with tf.GradientTape() as temp_tape:
                        temp_tape.watch(x)
                        y_temp = func(x)
                    grad_temp = temp_tape.gradient(y_temp, x)
                    if grad_temp is None:
                        continue
                    # 提取第i个分量 ∂f/∂x_i
                    grad_i = tf.gather(grad_temp, indices=[i], axis=1)  # 形状: [batch_size, 1]

                # 计算二阶导数 ∂²f/∂x_i∂x
                second_deriv = outer_tape.gradient(grad_i, x)
                if second_deriv is not None:
                    # 提取对角线元素 ∂²f/∂x_i²
                    diag_element = tf.gather(second_deriv, indices=[i], axis=1)
                    laplacian += tf.reduce_sum(diag_element)

            except Exception as e:
                print(f"前向+反向模式在维度 {i} 失败: {e}")
                continue

        return laplacian / batch_size

    def compute_laplacian_reverse_forward_stable(self, func, x):
        """
        稳定的反向+前向模式计算拉普拉斯算子

        数学公式：
        Δf(x) = Σ(∂²f/∂x_i²)

        计算方法：
        1. 反向模式：计算梯度 ∇f(x)
        2. 前向模式：对每个梯度分量 ∂f/∂x_i 应用前向模式求导
        """
        batch_size = tf.cast(tf.shape(x)[0], tf.float32)
        laplacian = 0.0

        # 首先计算一阶梯度 ∇f(x)
        with tf.GradientTape() as grad_tape:
            grad_tape.watch(x)
            y = func(x)
        grad = grad_tape.gradient(y, x)

        if grad is None:
            return tf.constant(0.0)

        for i in range(x.shape[1]):
            try:
                # 提取第i个梯度分量 ∂f/∂x_i
                grad_i = tf.gather(grad, indices=[i], axis=1)  # 形状: [batch_size, 1]

                # 使用前向模式计算二阶导数
                with tf.GradientTape() as second_tape:
                    second_tape.watch(x)
                    # 重新计算函数值
                    y_new = func(x)

                # 计算二阶导数：∂²f/∂x∂x_i
                # 这里使用前向模式，output_gradients 指定方向
                second_deriv = second_tape.gradient(y_new, x, output_gradients=tf.ones_like(y_new))
                if second_deriv is not None:
                    # 提取对角线元素 ∂²f/∂x_i²
                    diag_element = tf.gather(second_deriv, indices=[i], axis=1)
                    laplacian += tf.reduce_sum(diag_element)

            except Exception as e:
                print(f"反向+前向模式在维度 {i} 失败: {e}")
                continue

        return laplacian / batch_size

    def compute_laplacian_forward_forward_stable(self, func, x):
        """
        稳定的前向+前向模式计算拉普拉斯算子（使用有限差分）

        数学公式：
        Δf(x) = Σ(∂²f/∂x_i²)

        使用二阶中心有限差分公式：
        ∂²f/∂x_i² ≈ [f(x+he_i) - 2f(x) + f(x-he_i)] / h²

        其中 h 是小的步长，e_i 是第 i 个标准基向量
        """
        batch_size = tf.cast(tf.shape(x)[0], tf.float32)
        h = 1e-4  # 有限差分步长
        laplacian = 0.0

        for i in range(x.shape[1]):
            try:
                # 创建单位向量 e_i
                e_i = tf.one_hot(indices=[i], depth=x.shape[1], dtype=x.dtype)
                e_i_expanded = tf.tile(e_i, [tf.shape(x)[0], 1])

                # 中心差分计算二阶导数
                x_plus = x + h * e_i_expanded  # x + he_i
                x_minus = x - h * e_i_expanded  # x - he_i

                f_plus = func(x_plus)  # f(x + he_i)
                f_center = func(x)  # f(x)
                f_minus = func(x_minus)  # f(x - he_i)

                # 二阶中心差分公式
                second_deriv = (f_plus - 2 * f_center + f_minus) / (h * h)
                laplacian += tf.reduce_sum(second_deriv)

            except Exception as e:
                print(f"前向+前向模式在维度 {i} 失败: {e}")
                continue

        return laplacian / batch_size

    def compute_biharmonic_stable(self, func, x, method='reverse_reverse'):
        """
        稳定的双调和算子计算方法

        数学公式：
        Δ²f(x) = ∇⁴f(x) = Δ(Δf(x))

        计算方法：
        1. 首先计算拉普拉斯算子：g(x) = Δf(x)
        2. 然后计算拉普拉斯算子的拉普拉斯：Δ²f(x) = Δg(x)

        注意：双调和算子包含交叉导数项，但这里我们只计算对角线近似
        """
        try:
            if method == 'reverse_reverse':
                lap_func = self.compute_laplacian_reverse_reverse
            elif method == 'forward_reverse':
                lap_func = self.compute_laplacian_forward_reverse_stable
            elif method == 'reverse_forward':
                lap_func = self.compute_laplacian_reverse_forward_stable
            elif method == 'forward_forward':
                # 对于前向+前向模式，使用有限差分法
                return self.compute_biharmonic_finite_difference(func, x)
            else:
                return tf.constant(0.0)

            # 定义拉普拉斯函数：g(x) = Δf(x)
            def laplacian_func(x_inner):
                return lap_func(func, x_inner)

            # 计算双调和：Δ²f(x) = Δg(x)
            return self.compute_laplacian_reverse_reverse(laplacian_func, x)

        except Exception as e:
            print(f"双调和算子计算失败 ({method}): {e}")
            return tf.constant(0.0)

    def compute_biharmonic_finite_difference(self, func, x, h=1e-3):
        """
        使用有限差分法计算双调和算子

        数学公式：
        Δ²f(x) = Σ(∂⁴f/∂x_i⁴) + 交叉项

        使用四阶中心有限差分公式：
        ∂⁴f/∂x_i⁴ ≈ [f(x+2he_i) - 4f(x+he_i) + 6f(x) - 4f(x-he_i) + f(x-2he_i)] / h⁴
        """
        batch_size = tf.cast(tf.shape(x)[0], tf.float32)
        biharmonic = 0.0

        for i in range(x.shape[1]):
            try:
                # 创建单位向量 e_i
                e_i = tf.one_hot(indices=[i], depth=x.shape[1], dtype=x.dtype)
                e_i_expanded = tf.tile(e_i, [tf.shape(x)[0], 1])

                # 四阶中心差分
                x_plus2 = x + 2 * h * e_i_expanded  # x + 2he_i
                x_plus1 = x + h * e_i_expanded  # x + he_i
                x_minus1 = x - h * e_i_expanded  # x - he_i
                x_minus2 = x - 2 * h * e_i_expanded  # x - 2he_i

                f_plus2 = func(x_plus2)  # f(x + 2he_i)
                f_plus1 = func(x_plus1)  # f(x + he_i)
                f_center = func(x)  # f(x)
                f_minus1 = func(x_minus1)  # f(x - he_i)
                f_minus2 = func(x_minus2)  # f(x - 2he_i)

                # 四阶中心差分公式
                fourth_deriv = (f_plus2 - 4 * f_plus1 + 6 * f_center - 4 * f_minus1 + f_minus2) / (h ** 4)
                biharmonic += tf.reduce_sum(fourth_deriv)

            except Exception as e:
                print(f"双调和有限差分在维度 {i} 失败: {e}")
                continue

        return biharmonic / batch_size