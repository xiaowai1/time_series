# -*- coding = utf-8 -*-
# @Time : 2024/5/20 14:33
# @Author : ChiXiaoWai
# @File : calculate_error.py
# @Project : DLinear
# @Description : 误差计算
import pandas as pd
import numpy as np

# 从 CSV 文件中加载数据
# df = pd.read_csv('results/avgmem-ForecastResults.csv')  # 替换为你的 CSV 文件路径
df = pd.read_csv('results/avgcpu-ForecastResults.csv')  # 替换为你的 CSV 文件路径

# 提取实际值和预测值列，并转换为 NumPy 数组
# y_true = df['Avgmemreal'].values
y_true = df['Avgcpureal'].values
y_pred = df['forecast'].values

# 计算均方误差 (MSE)
mse = np.mean((y_true - y_pred)**2)

# 计算平均绝对误差 (MAE)
mae = np.mean(np.abs(y_true - y_pred))

# 计算均方根误差 (RMSE)
rmse = np.sqrt(mse)

# 计算对称平均绝对百分比误差 (SMAPE)
smape = np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

# 输出结果
print("Mean Squared Error (MSE):", mse)
print("Mean Absolute Error (MAE):", mae)
print("Root Mean Squared Error (RMSE):", rmse)
print("Symmetric Mean Absolute Percentage Error (SMAPE):", smape)
