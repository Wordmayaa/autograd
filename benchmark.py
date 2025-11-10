import tensorflow as tf
import numpy as np
import pandas as pd
import time
import os

# ============================================================================
# 评测指标
# ============================================================================

class BenchmarkMetrics:
    """评测指标计算"""

    @staticmethod
    def compute_mse(values1, values2):
        """计算均方误差"""
        if values1 is None or values2 is None:
            return float('inf')
        return np.mean((values1 - values2) ** 2)

    @staticmethod
    def compute_relative_error(values1, values2):
        """计算相对误差"""
        if values1 is None or values2 is None or np.all(values2 == 0):
            return float('inf')
        return np.mean(np.abs(values1 - values2) / (np.abs(values2) + 1e-8))

    @staticmethod
    def measure_computation_time(func, *args, num_repeats=3, warmup=1):
        """测量计算时间（包括预热）"""
        # 预热
        for _ in range(warmup):
            try:
                result = func(*args)
                if hasattr(result, 'numpy'):
                    result.numpy()
            except:
                pass

        # 正式测量
        times = []
        for _ in range(num_repeats):
            try:
                start_time = time.perf_counter()
                result = func(*args)
                if hasattr(result, 'numpy'):
                    result.numpy()  # 确保计算完成
                end_time = time.perf_counter()
                times.append(end_time - start_time)
            except Exception as e:
                print(f"时间测量失败: {e}")
                times.append(float('inf'))

        return np.median(times)