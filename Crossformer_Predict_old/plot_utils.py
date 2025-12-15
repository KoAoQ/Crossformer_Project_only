import os
import matplotlib.pyplot as plt
import numpy as np

# 设置绘图风格 (可选)
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def plot_predictions(trues, preds, folder_path, target_col):
    """
    绘制整体预测对比图 (取最后一个时间步)

    Args:
        trues: 真实值数组 (numpy array)
        preds: 预测值数组 (numpy array)
        folder_path: 保存文件夹路径
        target_col: 目标变量名称 (用于标题)
    """
    plt.figure(figsize=(12, 6))

    # 为了图表清晰，只画前 300 个样本
    limit = min(300, len(preds))

    # 确保输入是 1维数组
    gt_plot = trues[:limit].flatten()
    pd_plot = preds[:limit].flatten()

    plt.plot(gt_plot, label='Actual (℃)', color='#1f77b4', linewidth=1.5)
    plt.plot(pd_plot, label='Prediction (℃)', color='#e377c2', linestyle='--', linewidth=1.5)

    plt.title(f'Prediction Overview: {target_col} (Last Step)')
    plt.xlabel('Time Steps (Samples)')
    plt.ylabel('Temperature (℃)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    save_path = os.path.join(folder_path, 'prediction_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"整体对比图已保存: {save_path}")


def plot_case_visuals(visual_samples, folder_path, mean, std):
    """
    绘制个例详细分析图 (历史+未来)

    Args:
        visual_samples: 包含 (hist, true, pred) 元组的列表
        folder_path: 保存文件夹路径
        mean: 反归一化均值
        std: 反归一化标准差
    """
    for idx, (hist, true, pred) in enumerate(visual_samples):
        # 手动反归一化
        hist = hist * std + mean
        true = true * std + mean
        pred = pred * std + mean

        # 准备坐标轴
        seq_len = len(hist)
        pred_len = len(true)
        # 历史部分坐标: 0 ~ 95
        x_hist = np.arange(seq_len)
        # 预测部分坐标: 96 ~ 119
        x_future = np.arange(seq_len, seq_len + pred_len)

        plt.figure(figsize=(10, 5))

        # 1. 画历史部分 (输入)
        plt.plot(x_hist, hist, label='History (Input)', color='black', alpha=0.6, linewidth=1.5)

        # 2. 画未来真实值
        plt.plot(x_future, true, label='Ground Truth', color='green', marker='.', linewidth=2)

        # 3. 画未来预测值
        plt.plot(x_future, pred, label='Prediction', color='red', marker='.', linewidth=2)

        # 4. 画分隔线
        plt.axvline(x=seq_len - 1, color='orange', linestyle=':', label='Current Time')

        plt.title(f'Case Study #{idx + 1}: Forecast Analysis')
        plt.xlabel('Time Steps')
        plt.ylabel('Temperature (℃)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 保存
        save_path = os.path.join(folder_path, f'case_study_{idx + 1}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"个例分析图已保存: {save_path}")