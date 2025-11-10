import tensorflow as tf
import numpy as np
import pandas as pd
import time
import os
from typing import Dict, List, Tuple


# ============================================================================
# MLP模型 - 使用更稳定的激活函数
# ============================================================================

class MLP(tf.keras.Model):
    def __init__(self, input_dim, output_dim, hidden_dims, activation='tanh'):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # 构建网络层
        self.layers_list = tf.keras.Sequential()

        # 输入层
        self.layers_list.add(tf.keras.layers.Dense(
            hidden_dims[0],
            input_shape=(input_dim,),
            activation=activation,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros'
        ))

        # 隐藏层
        for i in range(len(hidden_dims) - 1):
            self.layers_list.add(tf.keras.layers.Dense(
                hidden_dims[i + 1],
                activation=activation,
                kernel_initializer='glorot_uniform',
                bias_initializer='zeros'
            ))

        # 输出层
        self.layers_list.add(tf.keras.layers.Dense(
            output_dim,
            activation='linear',  # 输出层使用线性激活，避免梯度问题
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros'
        ))

    def call(self, x):
        return self.layers_list(x)