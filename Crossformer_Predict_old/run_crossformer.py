import os
import time
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from tqdm import tqdm

# 引入模型和数据集
from crossformer_model import Crossformer
from dataset import AutoColumnDataset
from plot_utils import plot_predictions, plot_case_visuals

#引入随机数
import random
import os

warnings.filterwarnings('ignore')
#随机数种子
def fix_seed(seed=2025):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    # 下面这两行会让卷积算法确定化，但可能会稍微降低训练速度
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    print(f">>> [Reproducibility] Random Seed Fixed: {seed} <<<")

# ================================================================
#  工具函数
# ================================================================
def metric(pred, true):
    # np.mean: 求平均值
    # np.abs: 求绝对值
    MAE = np.mean(np.abs(pred - true))
    MSE = np.mean((pred - true) ** 2)   # ** 2: 平方运算
    RMSE = np.sqrt(MSE)     # np.sqrt: 开根号
    MAPE = np.mean(np.abs((pred - true) / (true + 1e-5)))
    MSPE = np.mean(np.square((pred - true) / (true + 1e-5)))
    # R2_score: 直接调用 sklearn 库算拟合优度.flatten(): 把多维数组拍扁成一维，方便计算
    R2 = r2_score(true.flatten(), pred.flatten())
    return MAE, MSE, RMSE, MAPE, MSPE, R2


class TemporalWeightedMSE(nn.Module):
    def __init__(self, seq_len=24, start_weight=1.0, end_weight=2.0):
        super().__init__()
        self.seq_len = seq_len
        # 生成一个从 start 到 end 的线性权重向量
        # 例如: [1.0, 1.04, 1.08, ..., 2.0]
        self.weights = torch.linspace(start_weight, end_weight, seq_len)

    def forward(self, pred, true):
        # pred, true 形状: [Batch, 24, Dim]

        # 1. 计算每个点的平方误差 (不求平均)
        # shape: [Batch, 24, Dim]
        loss_pointwise = (pred - true) ** 2

        # 2. 把权重移到对应设备上 (GPU/CPU)
        w = self.weights.to(pred.device).view(1, -1, 1)  # 变成 [1, 24, 1] 以便广播

        # 3. 加权
        loss_weighted = loss_pointwise * w

        # 4. 求平均
        return loss_weighted.mean()

class EarlyStopping:
    # __init__ 是类的构造函数，创建对象时自动执行
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience    # 容忍度：甚至Loss不降，我还能忍几轮？
        self.verbose = verbose      # 啰嗦模式：要不要打印详细日志，True 表示每次 Loss 变好时都打印一条消息告诉我。
        self.counter = 0            # 计数器：已经忍了几轮了，初始为 None（还没开始考）。
        self.best_score = None      # 目前为止最好的成绩
        self.early_stop = False     # 开关：是否该停了，变成 True 时训练循环就会 break。
        self.val_loss_min = np.inf  # 初始化最小Loss为无穷大
        self.delta = delta          # 阈值 (delta)，判定“有提升”的最小门槛。通常设为 0，表示只要有一点点提升就算数。

    # __call__ 是一个魔法方法！允许你用 object() 的方式调用对象
    def __call__(self, val_loss, model, path):
        score = -val_loss           # 我们希望Loss越小越好，所以取负数变成“得分越高越好”
        # 情况一：第一次考试 (best_score 是 None)
        if self.best_score is None:
            self.best_score = score
            # 赶紧把这个模型存下来作为“初代冠军”
            self.save_checkpoint(val_loss, model, path)
        # 情况二：考砸了 (当前分数 < 历史最高分)
        # 注意：+ delta 是为了处理一些微小的波动，通常 delta=0
        elif score < self.best_score + self.delta:
            self.counter += 1
            # 如果开启啰嗦模式，打印一下：我又忍了一次 (1/7)
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            # 判断底线：忍无可忍 (计数器 >= 容忍度)
            if self.counter >= self.patience:
                self.early_stop = True
        # 情况三：考出新高 (当前分数 > 历史最高分)
        else:
            self.best_score = score
            # 只要有进步，就存盘，覆盖掉之前的模型
            self.save_checkpoint(val_loss, model, path)
            # 只要有进步，之前的忍耐一笔勾销，计数器清零
            self.counter = 0
    #当发现当前模型是“历史最佳”时，这个函数负责把模型参数保存到硬盘上。
    def save_checkpoint(self, val_loss, model, path):
        # 1. 打印好消息
        if self.verbose:
            # 告诉用户：Loss 从 多少 降到了 多少，正在保存...
            # f-string: {self.val_loss_min:.6f} 表示保留6位小数
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        # 2. 确保文件夹存在
        #    如果 results/RNN_... 文件夹不存在，就创建它，防止报错
        if not os.path.exists(path):
            os.makedirs(path)
        # 3. 核心：保存模型参数
        #    model.state_dict(): 获取模型所有层的参数（权重和偏置），这是一个字典。
        #    torch.save(): 把这个字典序列化并保存成 .pth 文件。
        #    os.path.join(path, 'checkpoint.pth'): 智能拼接路径，兼容 Windows/Linux。
        torch.save(model.state_dict(), os.path.join(path, 'checkpoint.pth'))
        # 4. 更新历史最低 Loss
        #    把当前的 Loss 记为新的“最低纪录”，供下一次比较用。
        self.val_loss_min = val_loss


def adjust_learning_rate(optimizer, epoch, lr):
    # 计算新的学习率
    # 策略：每个 epoch 学习率减半 (乘以 0.5)
    # epoch=1: new_lr = lr * 0.5^0 = lr (保持初始值)
    # epoch=2: new_lr = lr * 0.5^1 = 0.5 * lr
    # epoch=3: new_lr = lr * 0.5^2 = 0.25 * lr
    new_lr = lr * (0.5 ** ((epoch - 1) // 1))
    # 将新的学习率应用到优化器中
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
    print('Updating learning rate to {}'.format(new_lr))


# ================================================================
#  参数配置类 (Config)
# ================================================================
class Config:
    def __init__(self):
        # 1. 数据配置
        self.root_path = './datasets/'
        self.data_path = '23.3.10-4.10_4_24_x.csv'  # 替换为你的文件名
        self.target_col = 'TEM'

        # 【降采样】5表示20秒间隔
        self.resample_step = 2
        # 【数据比例】0.1=调试模式, 1.0=全量模式
        self.data_percentage = 0.1

        # 2. 预测任务设置
        self.seq_len = 192
        self.label_len = 48
        self.pred_len = 24  # 8分钟

        # 3. Crossformer 模型参数
        self.d_model = 256  #隐层维度。数越大，模型容量越大，但也越容易过拟合、显存占用越高。
        self.n_heads = 4    #多头注意力头数。
        self.e_layers = 3   #编码器层数。
        self.d_ff = 512     #前馈网络维度。
        self.dropout = 0.2  #Dropout 是深度学习中防止过拟合的机制，0.2 表示训练时随机丢弃 20% 的神经元

        self.seg_len = 6    #分段长度。Crossformer 的核心特性，将时间序列切成小段（Segment）来处理，这里每 6 个点切一段。
        self.win_size = 2   #窗口大小。用于跨维度注意力的窗口。
        self.factor = 10    #“路由器数量”。决定了模型处理 26 个变量之间复杂关系的“带宽”。保持默认即可。
        self.baseline = False#“保底策略开关”。决定是否在预测结果上强行加上历史均值。由于我们已经做了标准化，关掉（False）是没问题的。

        # 4. 训练超参数
        self.batch_size = 96 #批大小。一次喂给模型多少条数据。RTX 3060 显存较大，这里可以尝试调大（如 64 或 96）来加速训练。
        self.learning_rate = 1e-4   #初始学习率。
        self.train_epochs = 20      #训练轮数。
        self.patience = 5           #早停耐心值。如果验证集 Loss 连续 5 轮不下降，就提前结束训练。
        self.num_workers = 4

        # 5. 保存设置
        self.use_gpu = True if torch.cuda.is_available() else False
        self.device = torch.device('cuda:0') if self.use_gpu else torch.device('cpu')
        self.checkpoints = './checkpoints_crossformer/'
        self.save_folder = './results_crossformer/'

        # 自动填充
        # 这两个值初始化为 0 或 1，通常在主程序（run_crossformer.py）读取数据后，会根据 CSV 文件的实际列数来自动覆盖这些值。
        self.enc_in = 0        #输入特征数。
        self.c_out = 1         #输出特征数。


# ================================================================
#  Trainer 类
# ================================================================
class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = self.args.device  # 确定是用 CPU 还是 GPU

        # Crossformer 初始化
        # 1. 实例化模型 (Crossformer)
        # 把 Config 里的参数传给模型，并搬运到 GPU 上 (.to(self.device))
        self.model = Crossformer(
            data_dim=args.enc_in,
            in_len=args.seq_len,
            out_len=args.pred_len,
            seg_len=args.seg_len,
            win_size=args.win_size,
            factor=args.factor,
            d_model=args.d_model,
            d_ff=args.d_ff,
            n_heads=args.n_heads,
            e_layers=args.e_layers,
            dropout=args.dropout,
            baseline=args.baseline,
            device=self.device
        ).to(self.device)
        # 2. 定义优化器 (Optimizer)
        # Adam 是目前最流行的优化算法，它决定了参数更新的“步法”
        # model.parameters() 告诉优化器：你要负责更新这些参数
        # lr (Learning Rate) 是学习率，决定了步子迈多大
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.learning_rate)
        # 3. 定义损失函数 (Criterion)
        # MSELoss (均方误差) 是回归任务最常用的“打分器”
        # 它计算 (预测值 - 真实值)^2 的平均值
        self.criterion = TemporalWeightedMSE(seq_len=self.args.pred_len, start_weight=1.0, end_weight=10.0)

    def _get_data(self, flag):
        # 1. 决定要不要打乱数据 (Shuffle)
        # 训练集 (train) 必须打乱，防止模型死记硬背顺序。
        # 测试集 (test) 通常不打乱，按时间顺序预测。
        if flag == 'test':
            shuffle_flag = False;
            drop_last = True        #丢弃最后一个样本数不足 batch_size 的批次
        else:
            shuffle_flag = True;
            drop_last = True
        # 2. 实例化 Dataset
        # 这就是我们之前读过的 AutoColumnDataset
        data_set = AutoColumnDataset(self.args, flag=flag)

        # 3. 封装 DataLoader
        # batch_size: 一次拉多少货（比如 32 个样本）
        # num_workers: 雇几个搬运工（进程数）
        data_loader = DataLoader(
            data_set,
            batch_size=self.args.batch_size,    #把数据切成一块一块的（比如 32 个一组）。
            shuffle=shuffle_flag,               #洗牌，打乱顺序。
            num_workers=self.args.num_workers,  #多进程加载
            drop_last=drop_last)
        return data_set, data_loader

    def train(self, setting):
        # 1. 准备数据
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        #路径创建代码
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)
        # 初始化早停器
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        # 2. Epoch 循环：开始一轮轮刷题
        for epoch in range(self.args.train_epochs):
            train_loss = [] #初始化损失列表,在每一轮训练开始前创建空列表，用于存储当前轮中 “每个批次（batch）的训练损失”
            # 【关键】开启训练模式
            # 这会启用 Dropout 和 BatchNormalization 等只在训练时有效的层
            self.model.train()
            # 进度条 tqdm：让你看到训练进度
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.args.train_epochs}")

            # 3. Batch 循环：一批批吃数据
            # enumerate 会返回：索引 i 和 数据 (batch_x, batch_y, ...)
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_pbar):
                # A. 梯度清零：把上一次计算的梯度删掉，否则会累加
                self.optimizer.zero_grad()

                # B. 搬运数据到 GPU
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                # C. 前向传播 (Forward)：模型开始猜
                # Crossformer 不需要时间特征 mark，所以只传 batch_x
                outputs = self.model(batch_x)

                # ========================================================
                # [修改区域 Start]：实施 Hybrid Loss (混合损失)
                # ========================================================

                # 1. 准备真实标签 (全变量)
                # batch_y 包含了 历史(seq_len) + 未来(pred_len)，我们只取未来这一段
                # 形状: [Batch, Pred_Len, 24] (假设24维)
                true_all = batch_y[:, -self.args.pred_len:, :]

                # 2. 计算【主任务 Loss】：只看最后一列（水冷壁温度）
                # 我们希望模型在这一列上准之又准
                pred_target = outputs[:, :, -1:]
                true_target = true_all[:, :, -1:]
                loss_target = self.criterion(pred_target, true_target)

                # 3. 计算【辅助任务 Loss】：看所有变量
                # 强迫 Router 去理解 风量、给煤量、负荷 之间的物理联动
                loss_all = self.criterion(outputs, true_all)

                # 4. 混合 Loss
                # 0.5 是权重系数 (alpha)，表示分出一半精力兼顾全局物理规律
                loss = loss_target + 0.5 * loss_all

                # ========================================================
                # [修改区域 End]
                # ========================================================

                # E. 计算误差 (Loss)
                train_loss.append(loss.item())  #item()能提取张量的原生浮点数，是训练中存储损失的标准写法。
                # F. 反向传播 (Backward)：计算每个参数该怎么调
                loss.backward()
                # G. 参数更新 (Step)：根据梯度修改参数
                self.optimizer.step()
                # H. 更新进度条显示的 Loss
                train_pbar.set_postfix({'loss': f"{loss.item():.7f}"})

            # 4. 验证阶段 (Validation)
            # 一个 Epoch 结束后，用验证集考一下试
            train_loss_avg = np.average(train_loss)
            vali_loss = self.vali(vali_loader)

            print(f"Epoch: {epoch + 1} | Train Loss: {train_loss_avg:.7f} Vali Loss: {vali_loss:.7f}")

            # 5. 早停检查
            # 如果这次考得好，early_stopping 内部会保存模型 (save_checkpoint)
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            # 6. 学习率衰减 (可选策略)
            # 每 5 轮把学习率减半，让模型后期学得更细致
            if (epoch + 1) % 5 == 0:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.5

        # 训练结束，加载最好的那个模型返回
        if os.path.exists(os.path.join(path, 'checkpoint.pth')):
            self.model.load_state_dict(torch.load(os.path.join(path, 'checkpoint.pth')))
        return self.model
    #验证
    def vali(self, vali_loader):
        self.model.eval()
        total_loss = []
        #with 语句用来创建一个“临时环境”。当代码运行在这个缩进块里时，某些特殊规则生效；一旦跳出缩进，规则自动失效。
        #torch.no_grad()告诉 PyTorch：“接下来的计算不需要求导（Gradient）”。
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                outputs = self.model(batch_x)

                f_dim = -1
                pred = outputs[:, :, f_dim:]
                true = batch_y[:, -self.args.pred_len:, f_dim:]

                loss = self.criterion(pred, true)
                total_loss.append(loss.item())
        return np.average(total_loss)

    #测试
    def test(self, setting):
        # 1. 获取测试集数据 (flag='test')
        test_data, test_loader = self._get_data(flag='test')
        print('Loading best model...')
        # 2. 加载“最强大脑”
        # torch.load: 读取硬盘上的 .pth 文件
        # load_state_dict: 把读到的参数填进现在的模型壳子里
        self.model.load_state_dict(torch.load(os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')))
        # 3. 开启“考试模式”
        self.model.eval()
        preds = []  # 存预测结果
        trues = []  # 存标准答案
        visual_samples = [] # 存几个典型样本用来画图

        # 4. 关闭梯度 (只做题，不改错)
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                outputs = self.model(batch_x)

                f_dim = -1
                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:]

                # 个例抓取,(用于画 Case Study 图)
                # 逻辑：每隔 50 个 Batch，抓第 1 个样本存起来
                if i % 50 == 0 and len(visual_samples) < 6:
                    # batch_x: [B, L, D] -> 取最后一个特征(温度)
                    # .detach().cpu().numpy(): 把数据从 GPU 拿回 CPU，转成 numpy 数组
                    hist_data = batch_x[0, :, -1].detach().cpu().numpy()
                    true_data = batch_y[0, :, 0].detach().cpu().numpy()
                    pred_data = outputs[0, :, 0].detach().cpu().numpy()
                    visual_samples.append((hist_data, true_data, pred_data))
                # 收集结果
                preds.append(outputs.detach().cpu().numpy())
                trues.append(batch_y.detach().cpu().numpy())
        # 5. 拼接 (Concatenate)
        # 刚才的 preds 是一个列表，里面装着几十个 [Batch, 24, 1] 的小数组
        # 这一步把它们拼成一个巨大的 [Total_Samples, 24, 1] 数组
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)

        # 6. 反归一化 (还原真实物理量纲)
        print("正在进行数据反归一化...")
        # 调用 Dataset 里写好的反归一化函数
        preds = test_data.inverse_transform_target(preds)
        trues = test_data.inverse_transform_target(trues)
        # 7. 拍扁数据
        # 为了算 MSE/R2，通常要把 [样本数, 24] 这种形状拍扁成一维长条
        preds_flat = preds.reshape(-1)
        trues_flat = trues.reshape(-1)
        # 8. 调用工具函数算分
        mae, mse, rmse, mape, mspe, r2 = metric(preds_flat, trues_flat)
        # 打印成绩单
        print('=' * 40)
        print(f'  Crossformer 测试集性能评估')
        print(f'  MAE  : {mae:.4f} | MSE  : {mse:.4f} | RMSE : {rmse:.4f}')
        print(f'  R2   : {r2:.4f}')
        print('=' * 40)

        # 9. 创建结果文件夹
        folder_path = os.path.join(self.args.save_folder, setting)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # 10. 保存成绩单到 txt
        print(f">>> 正在计算分步指标并保存至: {folder_path}/metrics.txt ...")
        with open(os.path.join(folder_path, 'metrics.txt'), 'w') as f:
            # A. 写入核心指标
            f.write(f"Experiment Setting: {setting}\n")
            f.write(f"Resample Step: {self.args.resample_step}\n")
            f.write(f"Data Percentage: {self.args.data_percentage}\n")
            f.write("-" * 30 + "\n")
            f.write(f"MAE  : {mae:.4f}\n")
            f.write(f"MSE  : {mse:.4f}\n")
            f.write(f"RMSE : {rmse:.4f}\n")
            f.write(f"R2   : {r2:.4f}\n")
            f.write("-" * 30 + "\n")

            # B. [核心修改] 写入关键时间点(6, 12, 18, 24)的单独指标
            f.write("\n" + "=" * 15 + " Step-wise Performance " + "=" * 15 + "\n")
            key_steps = [6, 12, 18, 24]

            for step in key_steps:
                if step > self.args.pred_len:
                    continue

                # 取出第 step 个时刻的数据 (索引是 step-1)
                step_idx = step - 1
                # 切片形状: [Batch, 1] -> 拍扁 -> [Batch]
                curr_pred = preds[:, step_idx, :].reshape(-1)
                curr_true = trues[:, step_idx, :].reshape(-1)

                # 单独算分
                s_mae, s_mse, s_rmse, s_mape, s_mspe, s_r2 = metric(curr_pred, curr_true)

                f.write(f"Step {step:<2}: MAE={s_mae:.4f} | MSE={s_mse:.4f} | RMSE={s_rmse:.4f} | R2={s_r2:.4f}\n")

            # c. 写入所有模型配置参数 (自动遍历 Config)
            f.write("\n" + "-" * 30 + " Configuration " + "-" * 30 + "\n")
            # vars(obj) 可以把对象的所有属性变成一个字典
            for key, value in vars(self.args).items():
                # 过滤掉一些不需要打印的内部对象（比如 device 对象, 或者私有属性）
                if not key.startswith('_') and not isinstance(value, torch.device):
                    f.write(f"{key:<20} : {value}\n")  # <20 表示左对齐占20格，排版更整齐
            f.write("-" * 75 + "\n")

        # 11. 画图 (调用外部 plot_utils)

        # A. 整体对比图
        # 定义你想看的关键时间点 (第6, 12, 18, 24个预测点)
        # 注意：如果 args.pred_len 小于 24，代码会自动跳过不存在的点
        key_steps = [6, 12, 18, 24]

        print("-" * 20 + " 开始绘制分步预测图 " + "-" * 20)

        for step in key_steps:
            # 检查 step 是否越界 (比如你只预测了 12 步，就画不了 24)
            if step > self.args.pred_len:
                    continue

            # --- 核心切片逻辑 ---
            # 数组索引从 0 开始，所以第 6 个点索引是 5 (step - 1)
            # trues 形状: [Sample_Num, Pred_Len, 1]
            step_idx = step - 1

            # 取出所有样本在这一时刻的真实值和预测值
            current_step_trues = trues[:, step_idx, 0]
            current_step_preds = preds[:, step_idx, 0]

            # 调用画图函数，传入标签 "Step 6" 等
            plot_predictions(
                current_step_trues,
                current_step_preds,
                folder_path,
                self.args.target_col,
                step_label=f"Step {step}"
            )

        # B. 个例分析图：把刚才抓拍的 visual_samples 画出来
        # 需要获取均值和方差，因为 plot_case_visuals 里可能会再次反归一化(取决于你的实现)
        # 注意：这里我们传入的是 dataset 里的 scaler 参数
        mean = test_data.scaler.mean[-1]
        std = test_data.scaler.std[-1]
        plot_case_visuals(visual_samples, folder_path, mean, std)



if __name__ == '__main__':
    # 1. 固定随机种子
    SEED = 2025
    fix_seed(SEED)

    args = Config()

    # 2. 生成随机实验ID
    import random
    rand_id = random.randint(1000, 9999)

    args.random_seed = SEED  # 记录固定的种子
    args.experiment_id = rand_id  # 记录本次的随机ID


    # 安全检查与特征数自动计算 (保持不变)
    if not os.path.exists(os.path.join(args.root_path, args.data_path)):
        print(f"错误：找不到文件 {args.data_path}")
        exit()

    try:
        df_tmp = pd.read_csv(os.path.join(args.root_path, args.data_path), nrows=5, encoding='gbk')
    except:
        df_tmp = pd.read_csv(os.path.join(args.root_path, args.data_path), nrows=5)
    args.enc_in = df_tmp.shape[1] - 1
    print(f"【Crossformer】变量数: {args.enc_in} ...")

    # 实例化 Trainer
    trainer = Trainer(args)

    import re

    safe_target = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9_]', '', args.target_col)
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    setting = f'Crossformer_{safe_target[:4]}_sl{args.seq_len}_pl{args.pred_len}_rs{args.resample_step}_dp{int(args.data_percentage * 100)}_{timestamp}_{rand_id}'

    print(f">>> 本次实验唯一标识符: {setting}")
    print('>>>>>>> 开始训练 Crossformer >>>>>>>')
    trainer.train(setting)

    print('>>>>>>> 开始测试 Crossformer >>>>>>>')
    trainer.test(setting)