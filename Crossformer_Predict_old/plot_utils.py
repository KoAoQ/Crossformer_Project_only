import os
import matplotlib.pyplot as plt
import numpy as np

# 设置绘图风格 (可选)
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


# [plot_utils.py]

def plot_predictions(trues, preds, folder_path, target_col, step_label="overview"):
    """
    绘制整体预测对比图 (支持多步长)
    [修改版]：增加了 step_label 参数，用于生成不同的文件名和标题
    """
    plt.figure(figsize=(12, 6))

    # 为了图表清晰，只画前 300 个样本
    limit = min(300, len(preds))

    # 确保输入是 1维数组
    gt_plot = trues[:limit].flatten()
    pd_plot = preds[:limit].flatten()

    plt.plot(gt_plot, label='Actual (℃)', color='#1f77b4', linewidth=1.5)
    plt.plot(pd_plot, label='Prediction (℃)', color='#e377c2', linestyle='--', linewidth=1.5)

    # [修改点1]：标题动态化
    plt.title(f'Prediction Overview: {target_col} ({step_label})')
    plt.xlabel('Time Steps (Samples)')
    plt.ylabel('Temperature (℃)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # [修改点2]：文件名动态化，防止覆盖
    # 如果 step_label 是 "Step 6"，文件名变成 "prediction_Step_6.png"
    safe_label = step_label.replace(" ", "_")
    save_path = os.path.join(folder_path, f'prediction_{safe_label}.png')

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"整体对比图已保存: {save_path}")


def plot_case_visuals(visual_samples, folder_path, mean, std):
    """
    绘制个例详细分析图 (历史+未来)
    [优化版]：只展示最近的一部分历史数据，避免预测部分被压缩
    """
    # 设定你希望展示的历史长度
    # 你的预测是 24，建议这里设为 48 或 72，这样比例大约是 2:1 或 3:1，视觉效果最好
    SHOW_HISTORY_LEN = 72

    for idx, (hist, true, pred) in enumerate(visual_samples):
        # 手动反归一化
        hist = hist * std + mean
        true = true * std + mean
        pred = pred * std + mean

        seq_len = len(hist)
        pred_len = len(true)

        # === 核心修改逻辑 Start ===
        # 如果历史数据太长，我们只取最后一段来画
        if seq_len > SHOW_HISTORY_LEN:
            # 切片：取最后 SHOW_HISTORY_LEN 个点
            plot_hist = hist[-SHOW_HISTORY_LEN:]
            # 调整 X 轴坐标：保证时间轴是连续的 (例如从 120 到 192)
            x_hist = np.arange(seq_len - SHOW_HISTORY_LEN, seq_len)
        else:
            plot_hist = hist
            x_hist = np.arange(seq_len)
        # === 核心修改逻辑 End ===

        # 预测部分坐标: 192 ~ 216
        x_future = np.arange(seq_len, seq_len + pred_len)

        plt.figure(figsize=(10, 5))

        # 1. 画历史部分 (使用切片后的数据)
        plt.plot(x_hist, plot_hist, label='History (Recent)', color='black', alpha=0.6, linewidth=1.5)

        # 2. 画未来真实值
        plt.plot(x_future, true, label='Ground Truth', color='green', marker='.', linewidth=2)

        # 3. 画未来预测值
        plt.plot(x_future, pred, label='Prediction', color='red', marker='.', linewidth=2)

        # 4. 画分隔线
        plt.axvline(x=seq_len - 1, color='orange', linestyle=':', label='Current Time')

        plt.title(f'Case Study #{idx + 1}: Forecast Analysis (Last {SHOW_HISTORY_LEN} steps context)')
        plt.xlabel('Time Steps')
        plt.ylabel('Temperature (℃)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 保存
        save_path = os.path.join(folder_path, f'case_study_{idx + 1}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"个例分析图已保存: {save_path}")