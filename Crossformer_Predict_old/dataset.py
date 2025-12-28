import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import warnings

# 忽略 pandas 的一些未来版本警告
warnings.filterwarnings('ignore')


class StandardScaler:
    def __init__(self, mean=0., std=1.):
        self.mean = mean
        self.std = std

    def fit(self, data):
        # 沿着行维度计算均值和标准差
        self.mean = data.mean(0)
        self.std = data.std(0)

    def transform(self, data):
        # 【改进点1】加入 eps (epsilon)，防止标准差为 0 导致除以 0 报错
        # 1e-7 是一个极小的数，不影响精度但能救命
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data - mean) / (std + 1e-7)

    def inverse_transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data * (std + 1e-7)) + mean


class AutoColumnDataset(Dataset):
    def __init__(self, args, flag='train'):
        self.seq_len = args.seq_len
        self.label_len = args.label_len
        self.pred_len = args.pred_len

        # 获取配置
        self.resample_step = getattr(args, 'resample_step', 1)
        self.data_percentage = getattr(args, 'data_percentage', 1.0)

        # 数据集划分逻辑
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.target = args.target_col
        self.data_path = args.data_path
        self.root_path = args.root_path

        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        file_path = os.path.join(self.root_path, self.data_path)

        # 1. 鲁棒读取 (Smart Loading)
        try:
            df_raw = pd.read_csv(file_path, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                print("UTF-8 读取失败，尝试 GBK...")
                df_raw = pd.read_csv(file_path, encoding='gbk')
            except UnicodeDecodeError:
                print("GBK 读取失败，尝试 GB18030...")
                df_raw = pd.read_csv(file_path, encoding='gb18030')

        # 2. 列名清洗 (Column Cleaning)
        df_raw.columns = df_raw.columns.str.strip().str.replace('\ufeff', '')

        # 3. 降采样 (Resampling)
        if self.resample_step > 1:
            origin_len = len(df_raw)
            # 使用 iloc 切片降采样 (每隔 step 取一行)
            df_raw = df_raw.iloc[::self.resample_step].reset_index(drop=True)
            if self.set_type == 0:
                print(f">>> [降采样] 步长={self.resample_step} | 数据: {origin_len} -> {len(df_raw)}")

        # 4. 数据截断调试 (Debugging Cut)
        if self.data_percentage < 1.0:
            total_len = len(df_raw)
            cut_len = int(total_len * self.data_percentage)
            df_raw = df_raw.iloc[:cut_len].reset_index(drop=True)
            if self.set_type == 0:
                print(f">>> [调试模式] 使用前 {self.data_percentage * 100}% 数据: {total_len} -> {len(df_raw)}")

        # 5. 数据预处理与填充 (Preprocessing)
        cols = list(df_raw.columns)
        date_col = cols[0]  # 假设第一列是时间

        # 强制将非时间列转为数字，无法转换的变为 NaN
        # 【改进点2】先转换 numeric，再处理缺失值，更安全
        for col in cols[1:]:
            df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')

        # 填充缺失值 (先前向填充，再后向填充，最后填0)
        df_raw = df_raw.fillna(method='ffill').fillna(method='bfill').fillna(0)

        # 6. 列对齐 (把 Target 移到最后)
        if self.target not in cols:
            raise ValueError(f"目标列 '{self.target}' 不在数据列中！现有列: {cols}")

        cols.remove(self.target)
        cols.remove(date_col)
        # 最终列序：[时间, ...特征..., 目标]
        df_raw = df_raw[[date_col] + cols + [self.target]]

        # 准备数值数据 (去掉时间列)
        df_data = df_raw.iloc[:, 1:]

        # =========================================================
        # 【新增功能】 打印 "索引 - 列名" 映射表
        # =========================================================
        # 只在训练集加载时打印，避免 Test/Val 刷屏
        if self.set_type == 0:
            print("\n" + "=" * 50)
            print(" 🔍 [特征索引映射表] Feature Index Mapping ")
            print(" 请依据此表设置 PINT 模块中的 load_index")
            print("=" * 50)
            print(f"{'Index':<8} | {'Column Name'}")
            print("-" * 50)

            # 将列名列表保存到 self 变量中，方便外部调用核实
            self.feature_names = df_data.columns.tolist()

            for idx, col_name in enumerate(self.feature_names):
                # 标记一下哪个是负荷(猜测)，哪个是目标
                note = ""
                if col_name == self.target:
                    note = " <--- TARGET (预测目标)"
                elif "负荷" in col_name or "煤" in col_name or "Load" in col_name:
                    note = " <--- 可能是负荷列 (Check This!)"

                print(f" {idx:<8} | {col_name}{note}")
            print("=" * 50 + "\n")
        # =========================================================
        # 7. 划分数据集 (Splitting) - 7:1:2
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_val = len(df_raw) - num_train - num_test

        # 计算切分边界
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_val, len(df_raw)]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        # 8. 归一化 (Scaling)

        # 只在训练集上 fit Scaler (严防数据泄露)
        train_data = df_data[border1s[0]:border2s[0]]
        self.scaler.fit(train_data.values)
        data = self.scaler.transform(df_data.values)

        # 记录 Target 的统计量，用于后续单独反归一化
        # 因为 target 已经被移到了最后一列，所以取 [-1]
        self.target_mean = self.scaler.mean[-1]
        self.target_std = self.scaler.std[-1] + 1e-7  # 别忘了 epsilon

        # 9. 时间特征提取 (Time Features) - 【改进点3：极速向量化】
        # 转换时间列为 datetime 对象
        df_stamp = df_raw[[date_col]].iloc[border1:border2]
        df_stamp[date_col] = pd.to_datetime(df_stamp[date_col])

        # 使用 .dt 访问器进行向量化操作，比 .apply(lambda) 快 100 倍以上
        data_stamp = np.zeros((len(df_stamp), 4))
        data_stamp[:, 0] = df_stamp[date_col].dt.month.values
        data_stamp[:, 1] = df_stamp[date_col].dt.day.values
        data_stamp[:, 2] = df_stamp[date_col].dt.weekday.values
        data_stamp[:, 3] = df_stamp[date_col].dt.hour.values

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

        # 打印当前数据集信息
        if self.set_type == 0:
            print(f"数据处理完毕: Train={num_train}, Val={num_val}, Test={num_test}")

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return (torch.tensor(seq_x, dtype=torch.float32),
                torch.tensor(seq_y, dtype=torch.float32),
                torch.tensor(seq_x_mark, dtype=torch.float32),
                torch.tensor(seq_y_mark, dtype=torch.float32))

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        """反归一化整个数据矩阵"""
        return self.scaler.inverse_transform(data)

    def inverse_transform_target(self, data):
        """
        仅反归一化目标列 (用于计算 MSE/MAE 时还原真实温度)
        data: (Batch, Len, 1) 或者 (Batch, Len)
        """
        return data * self.target_std + self.target_mean