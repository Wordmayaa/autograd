import tensorflow as tf
import numpy as np
import pandas as pd
import time
import os

# ============================================================================
# MLP模型参数搭配
# ============================================================================

def get_mlp_configurations():
    """定义不同复杂度的MLP配置，避免使用ReLU激活函数"""
    return [
        # 小型网络 - 低维输入
        {'input_dim': 3, 'output_dim': 1, 'hidden_dims': [64,64,64, 64], 'activation': 'relu'},

        # 中型网络 - 中等维输入
        {'input_dim': 10, 'output_dim': 3, 'hidden_dims': [128, 128,128,128,128,128], 'activation': 'relu'},

        # 大型网络 - 高维输入
        {'input_dim': 50, 'output_dim': 5, 'hidden_dims': [512,512,512,512], 'activation': 'tanh'},

        {'input_dim': 100, 'output_dim':100, 'hidden_dims': [128, 128,128,128,128,128,128,128], 'activation': 'relu'},

        {'input_dim': 50, 'output_dim': 5, 'hidden_dims': [512, 512, 512, 512,512, 512, 512, 512], 'activation': 'tanh'},
    ]

