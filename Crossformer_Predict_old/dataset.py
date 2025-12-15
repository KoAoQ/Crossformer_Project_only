import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


class StandardScaler:
    def __init__(self, mean=0., std=1.):
        self.mean = mean
        self.std = std

    def fit(self, data):
        # 计算数据的“平均长相”（均值）和“胖瘦程度”（标准差）
        self.mean = data.mean(0)    # 0 表示沿着第0维（行）计算，即计算每一列的均值
        self.std = data.std(0)      # 计算每一列的标准差

    def transform(self, data):
        # 核心公式：z = (x - mean) / std
        # 作用：把数据变成均值为0，方差为1的分布，方便神经网络消化

        # 下面这一大串是在判断 data 是 numpy 数组还是 torch 张量
        # 如果是张量，就把 mean/std 也转成张量放到同一个设备(GPU)上
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data - mean) / std

    def inverse_transform(self, data):
        # 反归一化：x = z * std + mean
        # 作用：模型输出的是归一化后的数字，我们需要把它还原成真实的温度
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data * std) + mean


class AutoColumnDataset(Dataset):
    def __init__(self, args, flag='train'):
        # 接收参数
        self.seq_len = args.seq_len
        self.label_len = args.label_len
        self.pred_len = args.pred_len

        # 获取配置：降采样步长 & 数据量比例
        # getattr(obj, 'name', default): 安全获取属性，如果 args 里没这个属性，就返回默认值 1
        self.resample_step = getattr(args, 'resample_step', 1)
        # 【数据量控制】
        self.data_percentage = getattr(args, 'data_percentage', 1.0)

        # 确定当前是训练集、验证集还是测试集
        # type_map 是一个字典，把字符串映射成数字 ID
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.target = args.target_col
        self.data_path = args.data_path
        self.root_path = args.root_path
        # 【重点】立刻执行数据读取和处理
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        file_path = os.path.join(self.root_path, self.data_path)

        # Step 1: 智能读取 (尝试多种编码)
        # 很多人死在第一步：CSV 文件乱码。这里用 try-except 挨个试，直到读出来为止
        try:
            df_raw = pd.read_csv(file_path, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                print("UTF-8 读取失败，正在尝试 GBK 编码...")           # ... 尝试 gbk ...
                df_raw = pd.read_csv(file_path, encoding='gbk')
            except UnicodeDecodeError:
                print("GBK 读取也失败，尝试 GB18030...")
                df_raw = pd.read_csv(file_path, encoding='gb18030') # ... 尝试 gb18030 ...

        # Step 2: 强力清洗 (去除列名里的隐形空格)
        # 如果列名是 " TEM "，代码里写 "TEM" 就会报错。这一步把它修剪干净。
        df_raw.columns = df_raw.columns.str.strip()
        # 某些文件会有 BOM 头 (\ufeff)，也要删掉
        df_raw.columns = [col.replace('\ufeff', '') for col in df_raw.columns]

        # Step 3: 降采样 (加速神器)
        if self.resample_step > 1:
            origin_len = len(df_raw)
            # iloc[::step]: 切片操作，每隔 step 行取一行
            df_raw = df_raw.iloc[::self.resample_step, :].reset_index(drop=True)
            if self.set_type == 0:
                print(f">>> [降采样生效] Step={self.resample_step} | 数据量: {origin_len} -> {len(df_raw)}")

        # Step 4: 数据截断 (调试神器)
        if self.data_percentage < 1.0:
            total_len = len(df_raw)
            # 只保留前百分之几的数据
            cut_len = int(total_len * self.data_percentage)
            df_raw = df_raw.iloc[:cut_len].reset_index(drop=True)
            if self.set_type == 0:
                print(f">>> [调试模式] 仅使用前 {self.data_percentage * 100}% 数据: {total_len} -> {len(df_raw)} 条")

        # Step 5: 填充缺失值
        # 工业数据常有空值 (NaN)。ffill 是用前一个值填，bfill 是用后一个值填。
        # 确保数据是完整的数字，否则模型会由 NaN 算出 Loss=NaN。
        cols = list(df_raw.columns)
        date_col = cols[0]
        # 将非时间列强制转为数字，非数字变成 NaN
        df_raw[cols[1:]] = df_raw[cols[1:]].apply(pd.to_numeric, errors='coerce')
        df_raw = df_raw.fillna(method='ffill').fillna(method='bfill').fillna(0)

        # Step 6: 自动对齐列 (把你想要预测的目标移到最后一列)
        # ... (检查 target 是否存在) ...
        if self.target not in cols:
            print(f"\n{'!' * 40}")
            print(f"【错误】找不到目标列: [{self.target}]")
            print(f"现有列名: {cols}")
            print(f"{'!' * 40}\n")
            import sys;
            sys.exit(1)

        cols.remove(self.target)
        cols.remove(date_col)
        df_raw = df_raw[[date_col] + cols + [self.target]]

        # Step 7: 划分数据集 (7:1:2)
        # 计算切分点索引
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_val = len(df_raw) - num_train - num_test

        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_val, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        # Step 8: 标准化
        # iloc[:, 1:]: 取除了时间列以外的所有数据
        df_data = df_raw.iloc[:, 1:]
        # 【关键】只在训练集上 fit！防止数据泄露 (Data Leakage)
        train_data = df_data[border1s[0]:border2s[0]]
        self.scaler.fit(train_data.values)
        data = self.scaler.transform(df_data.values)
        # 记录目标的均值方差，方便以后反归一化
        self.target_mean = self.scaler.mean[-1]
        self.target_scale = self.scaler.std[-1]

        # Step 9: 时间特征提取 (Time Features)
        # 提取 月、日、周、时，做成额外的特征喂给 Transformer
        # ... (这部分逻辑主要是 pandas 的日期处理) ...
        df_stamp = df_raw[[date_col]][border1:border2]
        df_stamp[date_col] = pd.to_datetime(df_stamp[date_col])

        data_stamp = np.zeros((len(df_stamp), 4))
        data_stamp[:, 0] = df_stamp[date_col].apply(lambda row: row.month, 1).values
        data_stamp[:, 1] = df_stamp[date_col].apply(lambda row: row.day, 1).values
        data_stamp[:, 2] = df_stamp[date_col].apply(lambda row: row.weekday(), 1).values
        data_stamp[:, 3] = df_stamp[date_col].apply(lambda row: row.hour, 1).values

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index                     # 计算滑动窗口的起止点
        s_end = s_begin + self.seq_len      # 历史结束点
        r_begin = s_end - self.label_len    # 标签开始点 (为了包含先验引子，往回退一点)
        r_end = r_begin + self.label_len + self.pred_len    # 标签结束点
        # 切片取值
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        # 取对应的时间戳
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        # 包装成 Tensor 返回
        return (torch.tensor(seq_x, dtype=torch.float32),
                torch.tensor(seq_y, dtype=torch.float32),
                torch.tensor(seq_x_mark, dtype=torch.float32),
                torch.tensor(seq_y_mark, dtype=torch.float32))

    def __len__(self):
        # 计算还能切出多少个完整的窗口
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform_target(self, data):
        return data * self.target_scale + self.target_mean