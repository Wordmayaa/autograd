import tensorflow as tf
import numpy as np
import pandas as pd
import time
import os
from model import MLP
from config import get_mlp_configurations
from operators import StableAutoDiffOperators
from benchmark import  BenchmarkMetrics
from results import export_results
import operators



# ============================================================================
# 主测试流程
# ============================================================================

def run_stable_benchmark():
    """运行稳定的基准测试"""
    configs = get_mlp_configurations()
    diff_operators = StableAutoDiffOperators()
    metrics = BenchmarkMetrics()

    all_results = {}
    detailed_data = {}

    # 定义自动微分方法组合
    laplacian_methods = [
        ('reverse_reverse', diff_operators.compute_laplacian_reverse_reverse),
        ('forward_reverse', diff_operators.compute_laplacian_forward_reverse_stable),
        ('reverse_forward', diff_operators.compute_laplacian_reverse_forward_stable),
        ('forward_forward', diff_operators.compute_laplacian_forward_forward_stable),
    ]

    for config_idx, config in enumerate(configs):
        print(f"\n{'=' * 60}")
        print(f"测试配置 {config_idx + 1}: 输入维度={config['input_dim']}")
        print(f"网络结构: {config['hidden_dims']}, 激活函数: {config['activation']}")
        print(f"{'=' * 60}")

        # 创建模型并初始化
        model = MLP(**config)

        # 使用更少的采样点以提高稳定性
        sample_points = min(100, 500 // config['input_dim'])
        x_sample = tf.random.uniform((sample_points, config['input_dim']), minval=-1.0, maxval=1.0)

        # 确保模型已构建
        _ = model(x_sample)

        config_key = f"config_{config_idx + 1}_dim_{config['input_dim']}"
        config_results = {}
        config_data = {}

        # 1. 基准计算：函数值 + 一阶梯度
        print("执行基准计算...")
        baseline_results = {}

        # 函数值计算
        func_time = metrics.measure_computation_time(model, x_sample)
        y_values = model(x_sample)
        baseline_results['function'] = {
            'computation_time': func_time,
            'values': y_values.numpy().flatten() if hasattr(y_values, 'numpy') else y_values
        }

        # 一阶梯度计算（基准）
        grad_time = metrics.measure_computation_time(diff_operators.compute_gradient_reverse, model, x_sample)
        gradients = diff_operators.compute_gradient_reverse(model, x_sample)
        baseline_results['gradient'] = {
            'computation_time': grad_time,
            'values': gradients.numpy() if hasattr(gradients, 'numpy') else gradients
        }

        config_results['baseline'] = baseline_results
        config_data['baseline'] = {
            'x': x_sample.numpy(),
            'function_values': baseline_results['function']['values'],
            'gradient_values': baseline_results['gradient']['values']
        }

        # 2. 高阶算子计算
        print("计算高阶微分算子...")
        operator_results = {}

        for method_name, lap_func in laplacian_methods:
            print(f"  方法: {method_name}")
            method_results = {}

            try:
                # 拉普拉斯算子
                lap_time = metrics.measure_computation_time(lap_func, model, x_sample)
                laplacian_value = lap_func(model, x_sample)

                method_results['laplacian'] = {
                    'computation_time': lap_time,
                    'value': laplacian_value.numpy() if hasattr(laplacian_value, 'numpy') else float(laplacian_value)
                }

                # 双调和算子
                biharm_time = metrics.measure_computation_time(
                    diff_operators.compute_biharmonic_stable, model, x_sample, method_name
                )
                biharmonic_value = diff_operators.compute_biharmonic_stable(model, x_sample, method_name)

                method_results['biharmonic'] = {
                    'computation_time': biharm_time,
                    'value': biharmonic_value.numpy() if hasattr(biharmonic_value, 'numpy') else float(biharmonic_value)
                }

                # 总计算时间
                method_results['total_time'] = lap_time + biharm_time

                # 存储详细数据
                config_data[method_name] = {
                    'laplacian_value': method_results['laplacian']['value'],
                    'biharmonic_value': method_results['biharmonic']['value']
                }

                print(f"    拉普拉斯: {method_results['laplacian']['value']:.6f}, "
                      f"双调和: {method_results['biharmonic']['value']:.6f}")

            except Exception as e:
                print(f"    {method_name} 计算失败: {e}")
                method_results['error'] = str(e)
                method_results['total_time'] = float('inf')

            operator_results[method_name] = method_results

        config_results['operators'] = operator_results

        all_results[config_key] = config_results
        detailed_data[config_key] = config_data

        # 打印当前配置结果摘要
        print(f"\n配置 {config_idx + 1} 结果摘要:")
        print(f"{'方法':<15} {'拉普拉斯时间(s)':<15} {'双调和时间(s)':<15} {'总时间(s)':<12} {'拉普拉斯值':<15}")
        for method_name, results in operator_results.items():
            if 'error' not in results:
                lap_time = results['laplacian']['computation_time']
                biharm_time = results['biharmonic']['computation_time']
                total_time = results['total_time']
                lap_value = results['laplacian']['value']
                print(
                    f"{method_name:<15} {lap_time:<15.4f} {biharm_time:<15.4f} {total_time:<12.4f} {lap_value:<15.6f}")
            else:
                print(f"{method_name:<15} {'失败':<15} {'失败':<15} {'失败':<12} {'失败':<15}")

    return all_results, detailed_data





# ============================================================================
# 主程序
# ============================================================================

if __name__ == "__main__":
    print("开始TensorFlow自动微分性能基准测试...")
    print("注意: 此测试使用稳定版本，避免了形状不匹配问题")

    start_time = time.time()

    try:
        # 运行基准测试
        all_results, detailed_data = run_stable_benchmark()

        # 导出结果
        summary_df = export_results(all_results, detailed_data, 'stable_benchmark_results')

        total_time = time.time() - start_time
        print(f"\n测试完成! 总耗时: {total_time:.2f} 秒")

        # 打印简要总结
        print("\n简要总结:")
        if not summary_df.empty:
            print(summary_df[['config', 'method', 'total_operator_time', 'laplacian_value']].head(10))

    except Exception as e:
        print(f"测试过程中发生错误: {e}")
        import traceback

        traceback.print_exc()


'''TensorFlow版本: 2.4.1
NumPy版本: 1.23.5
开始TensorFlow自动微分性能基准测试...
注意: 此测试使用稳定版本，避免了形状不匹配问题

============================================================
测试配置 1: 输入维度=2
网络结构: [32, 32], 激活函数: tanh
============================================================
2025-11-10 20:22:05.016069: I tensorflow/compiler/jit/xla_cpu_device.cc:41] Not creating XLA devices, tf_xla_enable_xla_devices not set
2025-11-10 20:22:05.016235: I tensorflow/core/platform/cpu_feature_guard.cc:142] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  SSE4.1 SSE4.2 AVX AVX2 FMA
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2025-11-10 20:22:05.016941: I tensorflow/core/common_runtime/process_util.cc:146] Creating new thread pool with default inter op setting: 2. Tune using inter_op_parallelism_threads for best performance.
执行基准计算...
计算高阶微分算子...
  方法: reverse_reverse
    拉普拉斯: 0.000000, 双调和: 0.000000
  方法: forward_reverse
    拉普拉斯: 0.007781, 双调和: 0.000000
  方法: reverse_forward
    拉普拉斯: -0.408058, 双调和: 0.000000
  方法: forward_forward
    拉普拉斯: -0.137836, 双调和: 28796.500000

配置 1 结果摘要:
方法              拉普拉斯时间(s)       双调和时间(s)        总时间(s)       拉普拉斯值          
reverse_reverse 0.0019          0.0024          0.0043       0.000000       
forward_reverse 0.0197          0.0277          0.0474       0.007781       
reverse_forward 0.0056          0.0118          0.0174       -0.408058      
forward_forward 0.0048          0.0077          0.0125       -0.137836      

============================================================
测试配置 2: 输入维度=5
网络结构: [64, 64, 32], 激活函数: tanh
============================================================
执行基准计算...
计算高阶微分算子...
  方法: reverse_reverse
    拉普拉斯: 0.000000, 双调和: 0.000000
  方法: forward_reverse
    拉普拉斯: -0.004912, 双调和: 0.000000
  方法: reverse_forward
    拉普拉斯: 0.038774, 双调和: 0.000000
  方法: forward_forward
    拉普拉斯: -7.215888, 双调和: 254530.453125

配置 2 结果摘要:
方法              拉普拉斯时间(s)       双调和时间(s)        总时间(s)       拉普拉斯值          
reverse_reverse 0.0024          0.0041          0.0066       0.000000       
forward_reverse 0.0338          0.0591          0.0929       -0.004912      
reverse_forward 0.0109          0.0270          0.0378       0.038774       
forward_forward 0.0150          0.0220          0.0371       -7.215888      

============================================================
测试配置 3: 输入维度=10
网络结构: [128, 64, 64, 32], 激活函数: tanh
============================================================
执行基准计算...
计算高阶微分算子...
  方法: reverse_reverse
    拉普拉斯: 0.000000, 双调和: 0.000000
  方法: forward_reverse
    拉普拉斯: -0.044500, 双调和: 0.000000
  方法: reverse_forward
    拉普拉斯: -0.642199, 双调和: 0.000000
  方法: forward_forward
    拉普拉斯: 0.745058, 双调和: -264793.625000

配置 3 结果摘要:
方法              拉普拉斯时间(s)       双调和时间(s)        总时间(s)       拉普拉斯值          
reverse_reverse 0.0032          0.0036          0.0068       0.000000       
forward_reverse 0.0756          0.1385          0.2141       -0.044500      
reverse_forward 0.0239          0.0590          0.0829       -0.642199      
forward_forward 0.0328          0.0513          0.0842       0.745058       
汇总结果已导出到: stable_benchmark_results/benchmark_summary.csv
详细数据已导出到: stable_benchmark_results/

测试完成! 总耗时: 3.44 秒

简要总结:
            config           method  total_operator_time  laplacian_value
0   config_1_dim_2  reverse_reverse             0.004332         0.000000
1   config_1_dim_2  forward_reverse             0.047442         0.007781
2   config_1_dim_2  reverse_forward             0.017398        -0.408058
3   config_1_dim_2  forward_forward             0.012535        -0.137836
4   config_2_dim_5  reverse_reverse             0.006567         0.000000
5   config_2_dim_5  forward_reverse             0.092884        -0.004912
6   config_2_dim_5  reverse_forward             0.037828         0.038774
7   config_2_dim_5  forward_forward             0.037065        -7.215888
8  config_3_dim_10  reverse_reverse             0.006771         0.000000
9  config_3_dim_10  forward_reverse             0.214093        -0.044500'''