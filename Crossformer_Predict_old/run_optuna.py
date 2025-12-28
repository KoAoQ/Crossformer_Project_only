import os
import time
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import optuna
# 【新增】显式导入 TPE 采样器
from optuna.samplers import TPESampler
import random
import logging
import sys

# 导入你的模块
from crossformer_model import Crossformer
from dataset import AutoColumnDataset

# 忽略警告
warnings.filterwarnings('ignore')

# 设置 Optuna 的日志级别
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------------------------------------------------------------------------------------------------
# 全局配置
# ---------------------------------------------------------------------------------------------------------------------
N_TRIALS = 20
TRAIN_EPOCHS = 10
DATA_PERCENTAGE = 0.2


# ---------------------------------------------------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------------------------------------------------
def fix_seed(seed=2025):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# ---------------------------------------------------------------------------------------------------------------------
# 配置类
# ---------------------------------------------------------------------------------------------------------------------
class OptunaConfig:
    def __init__(self):
        # 固定参数
        self.root_path = './datasets/'
        self.data_path = '23.3.10-4.10_4_24_x.csv'
        self.target_col = 'TEM'
        self.resample_step = 2  # 保持 8s 采样
        self.seq_len = 96  # 保持基线设定
        self.label_len = 48
        self.pred_len = 24
        self.seg_len = 6
        self.win_size = 2
        self.factor = 10
        self.baseline = False
        self.num_workers = 0
        self.use_gpu = True if torch.cuda.is_available() else False
        self.device = torch.device('cuda:0') if self.use_gpu else torch.device('cpu')

        # 调参专用设置
        self.train_epochs = TRAIN_EPOCHS
        self.patience = 3
        self.data_percentage = DATA_PERCENTAGE
        self.seed = 2025

        self.enc_in = 0
        self.c_out = 1

        # 待搜索参数占位
        self.learning_rate = 1e-4
        self.d_model = 256
        self.n_heads = 4
        self.e_layers = 3
        self.d_ff = 512
        self.dropout = 0.2
        self.batch_size = 32


# ---------------------------------------------------------------------------------------------------------------------
# Objective 函数 (已修改为 MAE 导向)
# ---------------------------------------------------------------------------------------------------------------------
def objective(trial):
    # 1. 初始化
    args = OptunaConfig()
    fix_seed(args.seed)

    # 2. 定义搜索空间
    args.learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    args.batch_size = trial.suggest_categorical("batch_size", [32, 64, 96])
    args.d_model = trial.suggest_categorical("d_model", [128, 256])
    args.e_layers = trial.suggest_int("e_layers", 1, 3)

    if args.d_model == 128:
        args.n_heads = trial.suggest_categorical("n_heads", [4, 8])
    elif args.d_model == 256:
        args.n_heads = trial.suggest_categorical("n_heads", [4, 8])

    args.dropout = trial.suggest_float("dropout", 0.1, 0.4, step=0.1)
    args.d_ff = trial.suggest_categorical("d_ff", [256, 512, 1024])

    # 3. 数据特征数自动获取
    try:
        df_tmp = pd.read_csv(os.path.join(args.root_path, args.data_path), nrows=5, encoding='gbk')
    except:
        df_tmp = pd.read_csv(os.path.join(args.root_path, args.data_path), nrows=5)
    args.enc_in = df_tmp.shape[1] - 1

    # 4. 准备数据
    train_data = AutoColumnDataset(args, flag='train')
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              drop_last=True)
    val_data = AutoColumnDataset(args, flag='val')
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                            drop_last=True)

    # 5. 模型与优化器
    model = Crossformer(
        data_dim=args.enc_in, in_len=args.seq_len, out_len=args.pred_len, seg_len=args.seg_len,
        win_size=args.win_size, factor=args.factor, d_model=args.d_model, d_ff=args.d_ff,
        n_heads=args.n_heads, e_layers=args.e_layers, dropout=args.dropout, baseline=args.baseline,
        device=args.device
    ).to(args.device)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    # 训练时依然用 MSE (因为它光滑好求导)，但评价时我们看 MAE
    criterion_train = nn.MSELoss()

    # 6. 训练循环
    best_mae_score = float('inf')

    print(
        f"\n⚡ [Trial {trial.number}] 开始训练... | Params: LR={args.learning_rate:.1e}, BS={args.batch_size}, L={args.e_layers}")
    epoch_pbar = tqdm(range(args.train_epochs), desc=f"Trial {trial.number} Progress", leave=False, unit="epoch")

    for epoch in epoch_pbar:
        # --- Train (使用 MSE 优化) ---
        model.train()
        train_loss = []
        for i, (batch_x, batch_y, _, _) in enumerate(train_loader):
            optimizer.zero_grad()
            batch_x = batch_x.float().to(args.device)
            batch_y = batch_y.float().to(args.device)
            outputs = model(batch_x)

            true_all = batch_y[:, -args.pred_len:, :]
            pred_target = outputs[:, :, -1:]
            true_target = true_all[:, :, -1:]

            loss = criterion_train(pred_target, true_target)
            train_loss.append(loss.item())
            loss.backward()
            optimizer.step()

        # --- Val (【关键修改】计算真实物理量纲的 MAE) ---
        model.eval()
        preds = []
        trues = []
        with torch.no_grad():
            for i, (batch_x, batch_y, _, _) in enumerate(val_loader):
                batch_x = batch_x.float().to(args.device)
                batch_y = batch_y.float().to(args.device)
                outputs = model(batch_x)

                f_dim = -1
                pred = outputs[:, :, f_dim:]
                true = batch_y[:, -args.pred_len:, f_dim:]

                # 收集预测值和真实值 (保持在 GPU 或转 CPU 均可，这里转 CPU 方便 numpy 计算)
                preds.append(pred.cpu().numpy())
                trues.append(true.cpu().numpy())

        # 1. 拼接
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)

        # 2. 【核心】反归一化 (还原为 ℃)
        # 只要 AutoColumnDataset 里有 inverse_transform_target 方法就可以直接用
        # 这样 Optuna 看到的分数就是真实的 "0.77" 这种 MAE
        preds = val_data.inverse_transform_target(preds)
        trues = val_data.inverse_transform_target(trues)

        # 3. 计算 MAE
        current_mae = np.mean(np.abs(preds - trues))
        avg_train_loss = np.average(train_loss)

        # 更新进度条
        epoch_pbar.set_postfix({'TrainMSE': f"{avg_train_loss:.4f}", 'ValMAE': f"{current_mae:.4f}"})

        # --- Pruning (基于 MAE 剪枝) ---
        trial.report(current_mae, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        best_mae_score = min(best_mae_score, current_mae)

    # 返回 MAE 给 Optuna，让它去找最小的 MAE
    return best_mae_score


# ---------------------------------------------------------------------------------------------------------------------
# 回调函数
# ---------------------------------------------------------------------------------------------------------------------
def tqdm_callback(study, trial):
    global pbar
    pbar.update(1)
    best_val = study.best_value

    if trial.state == optuna.trial.TrialState.COMPLETE:
        is_best = (trial.value == best_val)
        if is_best:
            tqdm.write(f"🔥 [Trial {trial.number}] New Best! MAE: {trial.value:.5f} (打破记录)")
        else:
            tqdm.write(f"✅ [Trial {trial.number}] Finished. MAE: {trial.value:.5f} (Best: {best_val:.5f})")
    elif trial.state == optuna.trial.TrialState.PRUNED:
        tqdm.write(f"✂️ [Trial {trial.number}] Pruned")
    pbar.set_postfix({"Best MAE": f"{best_val:.5f}"})


# ---------------------------------------------------------------------------------------------------------------------
# 主程序
# ---------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    # 1. 准备保存路径
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    save_dir = f'./optuna_results/{timestamp}'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print("=" * 60)
    print("🚀 Crossformer 自动超参数优化 (Optuna) - TPE + MAE版")
    print(f"   📂 结果将自动保存至: {save_dir}")
    print("=" * 60)

    # 2. 创建 Study
    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=2025),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2)
    )

    # 3. 开始优化
    with tqdm(total=N_TRIALS, desc="🏆 Total Progress", unit="trial") as pbar:
        try:
            study.optimize(objective, n_trials=N_TRIALS, callbacks=[tqdm_callback])
        except KeyboardInterrupt:
            print("\n🛑 用户强制停止！正在保存已有的结果...")
            pbar.close()

    # =========================================================
    # 4. 自动保存结果 (核心新增部分)
    # =========================================================
    print("\n" + "=" * 60)
    print(f"💾 正在保存结果到 {save_dir} ...")

    # A. 保存最佳参数到 txt
    best_params_path = os.path.join(save_dir, 'best_params.txt')
    with open(best_params_path, 'w', encoding='utf-8') as f:
        f.write(f"Experiment Time: {timestamp}\n")
        f.write(f"Best MAE Score: {study.best_value:.6f}\n")
        f.write("-" * 30 + "\n")
        f.write("Best Hyperparameters:\n")
        for key, value in study.best_params.items():
            f.write(f"  {key}: {value}\n")
    print(f"   ✅ 最佳参数已保存: best_params.txt")

    # B. 保存所有 Trial 的详细记录到 csv (方便用 Excel 分析)
    # study.trials_dataframe() 是 Optuna 自带的神器，能把所有记录转成 DataFrame
    df_trials = study.trials_dataframe()
    # 稍微清洗一下列名，把 "params_" 前缀去掉，看着更舒服
    df_trials.columns = [col.replace('params_', '') for col in df_trials.columns]

    csv_path = os.path.join(save_dir, 'all_trials.csv')
    df_trials.to_csv(csv_path, index=False)
    print(f"   ✅ 完整实验记录已保存: all_trials.csv")

    # C. (可选) 保存参数重要性分析图
    # 如果你安装了 matplotlib，这会生成一张图，告诉你哪个参数最重要
    try:
        from optuna.visualization import matplotlib as optuna_plt
        import matplotlib.pyplot as plt

        # 必须把 backend 设为 Agg，否则在没有屏幕的服务器上会报错
        plt.switch_backend('agg')

        fig = optuna_plt.plot_param_importances(study)
        fig.set_title("Hyperparameter Importances")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'param_importance.png'), dpi=300)
        print(f"   ✅ 参数重要性图表已保存: param_importance.png")
    except Exception as e:
        print(f"   ⚠️ 无法保存图片 (可能是缺少 matplotlib 或 plotly): {e}")

    print("=" * 60)
    print(f"🏆 最佳 MAE: {study.best_value:.6f} ℃")
    print("💎 最佳参数组合:")
    for key, value in study.best_params.items():
        print(f"   {key:<15}: {value}")
    print("=" * 60)