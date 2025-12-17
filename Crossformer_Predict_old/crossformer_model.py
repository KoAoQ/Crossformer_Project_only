import torch
import torch.nn as nn
import math
from einops import rearrange, repeat

#DSW嵌入
class DSW_embedding(nn.Module):
    def __init__(self, seg_len, d_model):
        super(DSW_embedding, self).__init__()
        self.seg_len = seg_len
        # 定义一个全连接层 (Linear Layer)
        # 输入大小: seg_len (比如 6)
        # 输出大小: d_model (比如 256)
        self.linear = nn.Linear(seg_len, d_model)

    def forward(self, x):
        # 1. 解包形状 (Unpacking)
        batch, ts_len, ts_dim = x.shape
        # 2. 第一次变形：切片 (Segmentation)
        x_segment = rearrange(x, 'b (seg_num seg_len) d -> (b d seg_num) seg_len', seg_len=self.seg_len)
        # 3. 嵌入 (Embedding)
        x_embed = self.linear(x_segment)
        # 4. 第二次变形：还原 (Restoration)
        x_embed = rearrange(x_embed, '(b d seg_num) d_model -> b d seg_num d_model', b=batch, d=ts_dim)
        return x_embed

#全注意力机制
class FullAttention(nn.Module):
    def __init__(self, scale=None, attention_dropout=0.1):
        super(FullAttention, self).__init__()
        # scale: 缩放因子。对应公式里的分母 sqrt(d_k)。
        # 如果不传，后面代码会自动计算。
        self.scale = scale
        # 定义 Dropout 层，防止注意力过于集中在某一点，增强泛化能力
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values):
        # 1. 解包维度 (Unpacking)
        # B: Batch Size (批次大小):表示这一批次有多少个独立的样本（例如 32）。
        # L: Length (Query 的时间长度):表示 Query 有多少个时间步（例如预测未来时，可能是 24）。
        # H: Heads (多头注意力的头数):表示多头注意力分成了几个头（例如 4 或 8）。
        # E: Embedding (每个头的特征维度，即 d_k):表示每个头负责处理多少维特征（例如总特征 256，分 4 个头，E 就是 64）。
        # S: 源序列的长度(Length of Source/Key/Value):表示 Key 和 Value 有多少个时间步。
        # D: Value 的特征维度。通常 D = E。
        # _: 占位符
        B, L, H, E = queries.shape  # queries 是一个 4 维张量：[Batch, Length, Heads, Head_Dim]
        _, S, _, D = values.shape   # values 也是一个 4 维张量，但有些维度我们不关心（用 _ 忽略）

        # 2. 计算缩放因子 (Scaling Factor)
        # 对应公式中的 1 / sqrt(d_k)
        scale = self.scale or 1. / math.sqrt(E)

        # 3. 计算注意力分数 (Score Calculation)
        # 核心公式: Q * K^T
        # 使用了爱因斯坦求和约定 (einsum)，极其优雅！
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)
        return V.contiguous()


class AttentionLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_values=None, dropout=0.1):
        super(AttentionLayer, self).__init__()

        # 1. 计算每个头的维度
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        # 2. 实例化核心计算引擎 (QK^T)
        self.inner_attention = FullAttention(scale=None, attention_dropout=dropout)

        # 3. 投影层
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)

        # ============================================================
        # [新增] Gated Attention 核心模块 (论文方案 G1)
        # 论文引用: "applying an head-specific sigmoid gate after the SDPA" [cite: 9]
        # ============================================================
        self.n_heads = n_heads
        self.total_value_dim = d_values * n_heads

        # 定义门控网络: Linear -> Sigmoid
        # 输入维度: 所有头的输出拼接 (Head * d_values)
        # 输出维度: 同上 (Element-wise Gating)
        # 这里的 Linear 层权重对于每个维度都是独立的，满足 "Head-Specific" 要求
        self.gate_projection = nn.Linear(self.total_value_dim, self.total_value_dim)

        # 4. 最终输出投影层
        self.out_projection = nn.Linear(d_values * n_heads, d_model)

    def forward(self, queries, keys, values):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        # 1. 投影 + 分头
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        # 2. 核心注意力计算 (SDPA)
        # out 形状: [B, L, H, d_values]
        out = self.inner_attention(queries, keys, values)

        # 3. 合并多头 (Concatenate Heads)
        # 形状变化: [B, L, H, d_values] -> [B, L, H * d_values]
        out = out.view(B, L, -1)

        # ============================================================
        # [新增] 执行门控机制 (Gating Mechanism)
        # 公式: Y' = Y * Sigmoid(Y * W_gate)
        # ============================================================
        # 1. 计算门控分数 (Gating Score)
        # 论文指出 SDPA output gating score 具有极强的稀疏性 [cite: 311]
        gate_score = torch.sigmoid(self.gate_projection(out))

        # 2. 施加门控 (Element-wise Multiplication)
        out = out * gate_score

        # 4. 输出投影 (Final Projection)
        # 此时的 out 已经是经过门控过滤的“干净”特征
        return self.out_projection(out)


class TwoStageAttentionLayer(nn.Module):
    def __init__(self, seg_num, factor, d_model, n_heads, d_ff=None, dropout=0.1):
        super(TwoStageAttentionLayer, self).__init__()
        d_ff = d_ff or 4 * d_model

        # 1. 定义三个注意力层
        # time_attention: 用于第一阶段，算时间相关性
        self.time_attention = AttentionLayer(d_model, n_heads, dropout=dropout)
        # dim_sender & dim_receiver: 用于第二阶段，算变量间相关性
        self.dim_sender = AttentionLayer(d_model, n_heads, dropout=dropout)
        self.dim_receiver = AttentionLayer(d_model, n_heads, dropout=dropout)

        # 2. 定义“路由器” (Router) —— 核心创新！
        # 这是一个可学习的参数矩阵，充当变量之间交换信息的“中介”
        # 形状: [段数, 因子, 维度]
        self.router = nn.Parameter(torch.randn(seg_num, factor, d_model))

        # 3. 定义常规组件 (Dropout, Norm, MLP)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)

        # 定义两个前馈网络 (Feed Forward Network)
        # nn.Sequential: 把几个层串联起来，数据进去流水线式处理
        # d_model输入维度（例如 256）。这是模型的主干特征维度。
        # d_ff输出维度（Feed Forward dimension）。通常 d_ff 是 d_model 的 4倍
        # nn.GELU()：激活函数（注入灵魂）
        self.MLP1 = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
        self.MLP2 = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))

    def forward(self, x):
        batch = x.shape[0]

        # 1. 维度变形：把 Batch 和 Dimension 捏在一起
        # 原始 x: [Batch, Time Series Dimension, Seg_Num, d_model]
        # 变形后 time_in: [Batch*Dimension, Seg_Num, d_model]
        time_in = rearrange(x, 'b ts_d seg_num d_model -> (b ts_d) seg_num d_model')

        # 2. 自注意力计算
        # Attention 看到的“样本数”变多了(B*D)，但每个样本的“序列长度”是 seg_num
        # 物理意义：它在计算“第1段”和“第3段”有多大关系。
        time_enc = self.time_attention(time_in, time_in, time_in)

        # 3. 残差连接 + 归一化 + MLP (Transformer 的标准操作)
        dim_in = time_in + self.dropout(time_enc)           # Add (残差连接)
        dim_in = self.norm1(dim_in)                         # Norm (层归一化)
        dim_in = dim_in + self.dropout(self.MLP1(dim_in))   # MLP + Add
        dim_in = self.norm2(dim_in)                         # Norm
        #跨维度注意力阶段 (Cross-Dimension Stage)。
        # 1. 维度重排：准备让变量们见面
        # 输入 dim_in: [(Batch * Time Series Dimension), Seg_Num, d_model]  <-- 上一阶段的状态
        dim_send = rearrange(dim_in, '(b ts_d) seg_num d_model -> (b seg_num) ts_d d_model', b=batch)

        # 2. 复制路由器：给每个样本发一套路由器
        # self.router: [Seg_Num, Factor, d_model]
        batch_router = repeat(self.router, 'seg_num factor d_model -> (repeat seg_num) factor d_model', repeat=batch)

        # 3. 变量 -> 路由器
        # Q = batch_router (路由器)
        # K = dim_send (所有变量)
        # V = dim_send (所有变量)
        dim_buffer = self.dim_sender(batch_router, dim_send, dim_send)

        # 4. 路由器 -> 变量
        # Q = dim_send (所有变量)
        # K = dim_buffer (路由器摘要)
        # V = dim_buffer (路由器摘要)
        dim_receive = self.dim_receiver(dim_send, dim_buffer, dim_buffer)

        # 5. 残差连接 + 归一化 (Add & Norm)
        dim_enc = dim_send + self.dropout(dim_receive)

        # 6. 前馈网络深度加工 (MLP)
        dim_enc = self.norm3(dim_enc)
        dim_enc = dim_enc + self.dropout(self.MLP2(dim_enc))
        dim_enc = self.norm4(dim_enc)

        # 7. 最终还原 (Restore Shape)
        final_out = rearrange(dim_enc, '(b seg_num) ts_d d_model -> b ts_d seg_num d_model', b=batch)
        return final_out

#片段合并
class SegMerging(nn.Module):
    def __init__(self, d_model, win_size, norm_layer=nn.LayerNorm):
        super().__init__()
        self.d_model = d_model
        self.win_size = win_size

        # 1. 线性层 (压缩机)
        # 输入维度: win_size * d_model (因为我们要把 win_size 个向量拼起来)
        # 输出维度: d_model (压缩回原来的维度，保持接口统一)
        self.linear_trans = nn.Linear(win_size * d_model, d_model)

        # 2. 归一化层
        self.norm = norm_layer(win_size * d_model)

    def forward(self, x):
        # 1. 解包维度
        # x: [Batch, Vars, Seg_Num, d_model] (例如 [32, 26, 16, 256])
        batch_size, ts_d, seg_num, d_model = x.shape

        # 2. 自动补齐 (Padding)
        # 假设 Seg_Num=15, win_size=2。15 除以 2 余 1。
        # 那个落单的片段没人跟它合并，所以我们要补齐。
        pad_num = seg_num % self.win_size
        if pad_num != 0:
            pad_num = self.win_size - pad_num
            # 策略：复制最后 pad_num 个片段补在后面
            # torch.cat(dim=-2) 表示在 Seg_Num 维度拼接
            x = torch.cat((x, x[:, :, -pad_num:, :]), dim=-2)

        # 3. 切片分组 (核心魔法!)
        seg_to_merge = []
        for i in range(self.win_size):
            # i::self.win_size 是 Python 切片语法 [start : end : step]
            # 假设 win_size = 2:
            # i=0 -> 取索引 0, 2, 4, 6... (偶数位)
            # i=1 -> 取索引 1, 3, 5, 7... (奇数位)
            seg_to_merge.append(x[:, :, i::self.win_size, :])

        # 4. 特征拼接
        # 把偶数位片段和奇数位片段，在“特征维度(-1)”拼在一起
        # 此时形状: [Batch, Vars, Seg_Num/2, 2*d_model]
        x = torch.cat(seg_to_merge, -1)
        x = self.norm(x)
        x = self.linear_trans(x)
        return x


class scale_block(nn.Module):
    def __init__(self, win_size, d_model, n_heads, d_ff, depth, dropout, seg_num=10, factor=10):
        super(scale_block, self).__init__()

        # 1. 判断是否需要合并 (Pooling)
        # win_size: 合并窗口大小。
        # 如果 win_size > 1 (比如 2)，说明这一层需要把时间段两两合并。
        # 如果 win_size = 1，说明这一层保持原样，不进行合并（通常是第一层）。
        if (win_size > 1):
            self.merge_layer = SegMerging(d_model, win_size, nn.LayerNorm)
        else:
            self.merge_layer = None

        # 2. 堆叠注意力层 (Processing)
        # depth: 这一层级内部，需要重复几次注意力计算？
        # nn.ModuleList: 这是一个容器，专门用来存一堆层。
        self.encode_layers = nn.ModuleList()
        for i in range(depth):
            # 循环 depth 次，往列表里塞 TwoStageAttentionLayer
            self.encode_layers.append(TwoStageAttentionLayer(seg_num, factor, d_model, n_heads, d_ff, dropout))

    def forward(self, x):
        # 1. 尝试合并 (Merge)
        # 如果定义了合并层，先执行合并。
        # 效果：Seg_Num 减半，感受野变大。
        if self.merge_layer is not None:
            x = self.merge_layer(x)

        # 2. 循环计算注意力 (Attend)
        # 拿着（可能合并过的）数据，依次经过 depth 个注意力层。
        # 效果：在当前的时间尺度下，反复进行“跨时间+跨维度”的信息交互。
        for layer in self.encode_layers:
            x = layer(x)
        return x


class Encoder(nn.Module):
    def __init__(self, e_blocks, win_size, d_model, n_heads, d_ff, block_depth, dropout, in_seg_num=10, factor=10):
        super(Encoder, self).__init__()

        # 容器：用来存每一层 scale_block
        self.encode_blocks = nn.ModuleList()
        # --- 第 1 层 (基座) ---
        # 关键点：这里强行传入 win_size = 1
        # 含义：第一层绝对不进行合并。它要负责“原汁原味”地提取最细粒度的特征。
        self.encode_blocks.append(scale_block(1, d_model, n_heads, d_ff, block_depth, dropout, in_seg_num, factor))
        # --- 第 2 到 N 层 (上层建筑) ---
        # 循环 range(1, e_blocks)，构建剩下的层
        for i in range(1, e_blocks):
            # 1. 计算这一层应该有多少个时间段 (seg_num)
            # 公式：原始段数 / (窗口大小 ^ 层数)
            # 比如：初始16段，win_size=2。
            # i=1: 16 / 2^1 = 8 段
            # i=2: 16 / 2^2 = 4 段
            # math.ceil: 向上取整，防止除不尽
            self.encode_blocks.append(scale_block(win_size, d_model, n_heads, d_ff, block_depth, dropout,
                                                  math.ceil(in_seg_num / win_size ** i), factor))

    def forward(self, x):
        encode_x = []

        # 1. 先把原始输入存起来 (第0级)
        encode_x.append(x)

        # 2. 逐层闯关
        for block in self.encode_blocks:
            # 数据流过这一层 (可能发生合并、Attention计算)
            x = block(x)

            # 3. 关键动作：每过一层，就把结果存下来！
            encode_x.append(x)

        # 4. 返回整个列表
        # 列表里包含了：[原始输入, 第1层输出, 第2层输出, ...]
        return encode_x


class DecoderLayer(nn.Module):
    def __init__(self, seg_len, d_model, n_heads, d_ff=None, dropout=0.1, out_seg_num=10, factor=10):
        super(DecoderLayer, self).__init__()
        # 1. 自注意力 (Self-Attention)
        # 这里的 Decoder 居然也用了 TwoStageAttentionLayer！
        # 作用：让 Decoder 自己的输入（未来的占位符）先在内部理清“时间”和“维度”的关系。
        self.self_attention = TwoStageAttentionLayer(out_seg_num, factor, d_model, n_heads, d_ff, dropout)
        # 2. 交叉注意力 (Cross-Attention) —— 也就是 Encoder-Decoder Attention
        # 作用：这是 Decoder 和 Encoder 唯一的沟通桥梁。
        # 它用普通的 AttentionLayer 即可，不需要 Two-Stage。
        self.cross_attention = AttentionLayer(d_model, n_heads, dropout=dropout)
        # 3. 其他常规组件
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.MLP1 = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        # 4. 预测头 (Prediction Head) —— 关键！
        # 作用：把抽象特征 (d_model=256) 变回具体的数值 (seg_len=6)。
        self.linear_pred = nn.Linear(d_model, seg_len)

    def forward(self, x, cross):
        batch = x.shape[0]

        # ==========================================
        # 阶段 1: 自我反思 (Self-Attention)
        # ==========================================
        # Decoder 先处理自己的内部关系。
        # 比如：预测第 24 小时的时候，要考虑和第 12 小时的逻辑连贯性。
        x = self.self_attention(x)

        # ==========================================
        # 阶段 2: 查阅资料 (Cross-Attention)
        # ==========================================

        # 1. 维度调整：把 Batch 和 变量数(ts_d) 合并
        # 目的：确保 Cross-Attention 是“变量对变量”的（Temperature 查 Temperature 的历史）。
        # x (Decoder) 形状: [(B*D), Out_Seg_Num, d_model]
        x = rearrange(x, 'b ts_d out_seg_num d_model -> (b ts_d) out_seg_num d_model')

        # cross (Encoder) 形状: [(B*D), In_Seg_Num, d_model]
        cross = rearrange(cross, 'b ts_d in_seg_num d_model -> (b ts_d) in_seg_num d_model')

        # 2. 核心交互
        # Query = x (我想预测的未来)
        # Key/Value = cross (已知的历史)
        # 含义：“根据我现在的状态(Q)，去历史(K)里找相关的线索，提取出来(V)”。
        tmp = self.cross_attention(x, cross, cross)

        # 3. 残差 + 归一化
        x = x + self.dropout(tmp)
        y = x = self.norm1(x)

        # ==========================================
        # 阶段 3: 消化理解 (MLP)
        # ==========================================
        y = self.MLP1(y)
        dec_output = self.norm2(x + y)

        # 还原形状：把 Batch 和 变量数 拆开
        dec_output = rearrange(dec_output, '(b ts_d) seg_dec_num d_model -> b ts_d seg_dec_num d_model', b=batch)

        # ==========================================
        # 阶段 4: 尝试预测 (Linear Prediction)
        # ==========================================
        # 这一步是 Crossformer 的特色！每一层都要输出一个预测结果。

        # 输入: dec_output (特征维度 256)
        # 输出: layer_predict (数值维度 6)
        layer_predict = self.linear_pred(dec_output)

        # 调整形状，准备输出
        layer_predict = rearrange(layer_predict, 'b out_d seg_num seg_len -> b (out_d seg_num) seg_len')

        # 返回两个东西：
        # 1. dec_output: 给下一层 Decoder 用（接力棒）。
        # 2. layer_predict:这一层的预测结果（作业本）。
        return dec_output, layer_predict


class Decoder(nn.Module):
    def __init__(self, seg_len, d_layers, d_model, n_heads, d_ff, dropout, router=False, out_seg_num=10, factor=10):
        super(Decoder, self).__init__()
        self.router = router

        # 1. 创建层级列表
        self.decode_layers = nn.ModuleList()

        # 2. 循环堆叠 DecoderLayer
        # d_layers: 解码器有多少层 (例如 3 层)
        for i in range(d_layers):
            self.decode_layers.append(DecoderLayer(seg_len, d_model, n_heads, d_ff, dropout, out_seg_num, factor))

    def forward(self, x, cross):
        # x: Decoder 的初始输入 (带位置/维度嵌入的空壳)
        # cross: Encoder 返回的列表！(还记得 Encoder 返回的是 [原始x, 第1层out, 第2层out...] 吗？)

        final_predict = None
        i = 0
        ts_d = x.shape[1]

        # 循环遍历每一层 DecoderLayer
        for layer in self.decode_layers:

            # Key Point 1: 对应的 Encoder 输出
            # cross 是一个列表。这里用 i 来索引。
            # 这意味着：第 1 层 Decoder，看的是 Encoder 的第 1 个输出；第 2 层看第 2 个。
            # 这就是“层级对应”！
            cross_enc = cross[i]
            # 执行单层计算
            # x: 更新后的隐状态 (传给下一层)
            # layer_predict:这一层预测出来的具体数值
            x, layer_predict = layer(x, cross_enc)

            # Key Point 2: 预测结果累加 (残差生成)
            if final_predict is None:
                final_predict = layer_predict
            else:
                final_predict = final_predict + layer_predict
            i += 1

        # 当前 final_predict 形状:
        # [Batch, (Vars * Seg_Num), Seg_Len]
        # 比如: [32, 26*4=104, 6]
        # 这里的 104 是把变量和分段混在一起了，6 是每个分段的长度。
        final_predict = rearrange(final_predict, 'b (out_d seg_num) seg_len -> b (seg_num seg_len) out_d', out_d=ts_d)

        # 变换后形状:
        # b: Batch (32)
        # (seg_num seg_len): 总预测时长 (4 * 6 = 24秒)
        # out_d: 变量数 (26)

        # 最终形状: [32, 24, 26]
        return final_predict


class Crossformer(nn.Module):
    def __init__(self, data_dim, in_len, out_len, seg_len, win_size=4, factor=10, d_model=512, d_ff=1024, n_heads=8,
                 e_layers=3, dropout=0.0, baseline=False, device=torch.device('cuda:0')):
        super(Crossformer, self).__init__()
        # ... (保存参数)

        self.data_dim = data_dim
        self.in_len = in_len
        self.out_len = out_len
        self.seg_len = seg_len
        self.merge_win = win_size
        self.baseline = baseline
        self.device = device
        # ==========================================
        # 1. 补齐计算 (Padding Logic)
        # ==========================================
        # 假设 in_len=22, seg_len=6
        # 22 / 6 = 3.66 -> ceil -> 4 -> 4 * 6 = 24
        # 所以我们需要把输入补齐到 24。

        self.pad_in_len = math.ceil(1.0 * in_len / seg_len) * seg_len
        self.pad_out_len = math.ceil(1.0 * out_len / seg_len) * seg_len
        # 计算需要补多少：24 - 22 = 2
        self.in_len_add = self.pad_in_len - self.in_len

        # ==========================================
        # 2. 编码器的嵌入层 (Encoder Embeddings)
        # ==========================================
        # DSW_embedding: 把 6 秒的原始数值切片并投影成 256 维向量
        self.enc_value_embedding = DSW_embedding(seg_len, d_model)

        # 位置编码: 这是一个可学习参数
        # 形状: [1, 变量数, 输入段数, 维度]
        # 注意：这里用的段数是补齐后的 pad_in_len // seg_len
        self.enc_pos_embedding = nn.Parameter(torch.randn(1, data_dim, (self.pad_in_len // seg_len), d_model))
        self.pre_norm = nn.LayerNorm(d_model)
        # ==========================================
        # 3. 实例化 Encoder 和 Decoder
        # ==========================================
        # 把之前讲过的 Encoder 搬过来

        self.encoder = Encoder(e_layers, win_size, d_model, n_heads, d_ff, block_depth=1, dropout=dropout,
                               in_seg_num=(self.pad_in_len // seg_len), factor=factor)
        # 解码器的位置编码 (关键！这就是 Decoder 的输入来源)
        # 形状: [1, 变量数, 输出段数, 维度]
        # 注意：这里包含了“变量维度(data_dim)”和“时间维度(out_seg_num)”的信息
        self.dec_pos_embedding = nn.Parameter(torch.randn(1, data_dim, (self.pad_out_len // seg_len), d_model))
        self.decoder = Decoder(seg_len, e_layers + 1, d_model, n_heads, d_ff, dropout,
                               out_seg_num=(self.pad_out_len // seg_len), factor=factor)

    def forward(self, x_seq):
        # x_seq 输入形状: [Batch, Time(in_len), Vars]

        # 1. 设定基准 (Baseline)
        # 有些时间序列预测任务，预测“增量”比预测“绝对值”更容易。
        # 如果 baseline=True，我们先算出均值，后面只预测波动，最后再加回来。
        if (self.baseline):
            base = x_seq.mean(dim=1, keepdim=True)
        else:
            base = 0
        batch_size = x_seq.shape[0]

        # ==========================================
        # 2. 补齐输入 (Padding Input)
        # ==========================================
        # 如果原始长度 22，需要补到 24 (差 2 个)。
        # 做法：把第 1 个时间点的数据 (x[:, :1, :]) 复制 2 次，拼在最前面。
        # 为什么拼前面？为了让最后的输出对齐。

        if (self.in_len_add != 0):
            x_seq = torch.cat((x_seq[:, :1, :].expand(-1, self.in_len_add, -1), x_seq), dim=1)

        # ==========================================
        # 3. 编码阶段 (Encoding)
        # ==========================================
        # A. 嵌入 (DSW Embedding)
        # [Batch, Time, Vars] -> [Batch, Vars, Seg_Num, d_model]
        # 此时物理数值变成了抽象特征。
        x_seq = self.enc_value_embedding(x_seq)

        # B. 加位置编码
        # 这里的 enc_pos_embedding 包含了“变量身份”和“时间身份”。
        x_seq += self.enc_pos_embedding
        x_seq = self.pre_norm(x_seq)

        # C. Encoder 提取特征
        # 返回的是一个列表 enc_out = [Layer0, Layer1, Layer2...]
        enc_out = self.encoder(x_seq)

        # ==========================================
        # 4. 解码准备 (Decoding Preparation) <--- 你的疑惑点！
        # ==========================================
        # 构造 Decoder 的输入 dec_in。
        # 这里的 dec_pos_embedding 就是之前说的“带身份证的空壳”。
        # 它的形状是 [1, Vars, Seg_Num, d_model]。
        # 我们用 repeat 把它复制 Batch 份。

        dec_in = repeat(self.dec_pos_embedding, 'b ts_d l d -> (repeat b) ts_d l d', repeat=batch_size)
        predict_y = self.decoder(dec_in, enc_out)

        return base + predict_y[:, :self.out_len, :]