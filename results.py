import tensorflow as tf
import numpy as np
import pandas as pd
import time
import os


# ============================================================================
# 数据导出
# ============================================================================

def export_results(all_results, detailed_data, export_dir='stable_results'):
    """导出结果数据"""
    os.makedirs(export_dir, exist_ok=True)

    # 1. 导出汇总结果 (CSV)
    summary_rows = []

    for config_key, config_results in all_results.items():
        baseline = config_results['baseline']
        operators = config_results.get('operators', {})

        for method_name, method_results in operators.items():
            if 'error' in method_results:
                continue

            row = {
                'config': config_key,
                'method': method_name,
                'function_time': baseline['function']['computation_time'],
                'gradient_time': baseline['gradient']['computation_time'],
                'laplacian_time': method_results['laplacian']['computation_time'],
                'biharmonic_time': method_results['biharmonic']['computation_time'],
                'total_operator_time': method_results['total_time'],
                'laplacian_value': method_results['laplacian']['value'],
                'biharmonic_value': method_results['biharmonic']['value'],
            }

            summary_rows.append(row)

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(f'{export_dir}/benchmark_summary.csv', index=False)
        print(f"汇总结果已导出到: {export_dir}/benchmark_summary.csv")

    # 2. 导出详细数据 (NPY)
    for config_key, config_data in detailed_data.items():
        config_export_dir = f'{export_dir}/{config_key}'
        os.makedirs(config_export_dir, exist_ok=True)

        # 保存基准数据
        if 'baseline' in config_data:
            baseline_data = config_data['baseline']
            np.save(f'{config_export_dir}/baseline_x.npy', baseline_data['x'])
            np.save(f'{config_export_dir}/baseline_function_values.npy', baseline_data['function_values'])
            np.save(f'{config_export_dir}/baseline_gradient_values.npy', baseline_data['gradient_values'])

        # 保存算子数据
        for method_name, method_data in config_data.items():
            if method_name == 'baseline':
                continue
            np.save(f'{config_export_dir}/{method_name}_laplacian.npy', np.array([method_data['laplacian_value']]))
            np.save(f'{config_export_dir}/{method_name}_biharmonic.npy', np.array([method_data['biharmonic_value']]))

    print(f"详细数据已导出到: {export_dir}/")

    return summary_df