# Assumptions

所有普通计算以当前一个 token 为主例。Configuration 是模型固有设置，Scenario 是演示场景，Approximation 是计数约定，Unverified 为当前证据边界。

## Configuration

| 项目 | 条件 |
|---|---|
| Model | DeepSeek-V4.1-Flash；正式 config 与 inference/config 32 项逐项对应 |
| Hidden / Layers | 5120；文本主干 40 层，CED 教学分段 0–19 / 20–39 |
| Query / KV | 64 个 Query Head × 512；KV 一份 512 维向量 |
| MoE | 每层存 384 个路由专家 + 1 个共享专家；当前选择 6 个路由专家 |
| KV / Index source | KV 源层 2 / 8 / 14 / 20；Indexer 层 2 / 8 / 14 / 20 / 24 / 28 / 32 / 36 |
| 窗口 / 选址上限 | Local 窗口 128；Indexer 最多选择 512 个 Global 位置 |
| 绑定与偏置 | Token Embedding 与 LM Head 不绑权；文本 Linear 无加性 Bias；Router 选择偏置及 Vision 的真实 Bias 单独计数 |
| 额外分支 | 3 个 DSpark 草稿阶段；32 层 Vision + Aligner；主图没有新增这些模块 |

## Scenario

| 项目 | 条件 |
|---|---|
| 当前 token | Batch Size 1；一次 Decode 更新 1 个 token，不是 Prefill 整段输入 |
| 历史 | 演示固定此前 1024 个 token；当前为第 1025 个。128 / 512 是此场景实际读取数，不能与配置上限混为一谈 |
| 主计算范围 | 文本 40 层 + 两次 Engram + 最终 LM Head；不执行 Vision 或 DSpark |
| 缓存 | 历史 KV 已存在；本次新的 Local KV 每层生成，Global KV 只在源层生成。ratio=2 当前半组等待，ratio=1 当前产生一行 |
| 反向 | 实数数学的局部梯度示例；不代表整批训练、历史缓存的完整反向、优化器更新或官方训练内核 |
| 共享 | 重用一个权重或缓存不会增加存储参数；草稿引用的主 Embedding / LM Head 统计一次 |

## Approximation

| 项目 | 条件 |
|---|---|
| 矩阵 FLOPs | 主结果采用 2MKN 近似：乘、加各 1 次；单 token 的 M=1。展开审计另列精确乘加 MN(2K−1) |
| 线性反向 | dX 与 dW 分别计算；近似分别 2MNK 和 2KMN。精确分别 MK(2N−1) 和 KN(2M−1)，因此单 token 外积不应称作精确 2KN |
| 非矩阵操作 | 加、减、乘各 1 次；rsqrt、exp、sigmoid、softplus、sqrt、除法各简化计 1 次；比较、Top-K、索引、abs/copysign 的符号处理和类型转换不计为浮点 FLOPs |
| Attention | 矩阵小计按实际读取位置；在线 Softmax 的展开估计按源码 64 位置分块，含 sink。它不是 GPU 实测指令数 |
| 量化与内存 | 仅显式计 Engram 查表反量化的逐项乘法；其他 FP4/FP8 量化、GEMM 内部反量化、缓存搬运、通信和分词耗时不计 |
| 参数与字节 | 统计逻辑权重数字；FP4 两值打包、FP8/FP4 scale、缓存和整数哈希 buffer 不无条件当作新增数学学习权重，字节数另算 |

## Unverified

| 项目 | 条件 |
|---|---|
| 完整反向 | 参考实现为推理，不含训练反向；非线性总量、量化权重训练规则、选择偏置的更新方式未验证 |
| 训练状态 | 技术报告 §2.5 说明预训练早期冻结 Vision 主体，保留最终归一化与对齐层可训练；本页面仍不能确定用户当前任务的 requires_grad 或量化权重更新规则 |
| 完整 Prefill / Draft / Vision FLOPs | 未运行完整图片、Prefill、DSpark 调度，当前未给出这些场景的精确 FLOPs |
| 本地差异 | 本地 model.py 落后于固定官方 revision：共享索引缓存发布条件、输出投影前量化有差异。参数维度不变；证据以官方固定版本为准 |

## query

### Configuration

- 权重：参数表按源码实际定义的 shape 与拥有关系统计，公式采用 XW 行向量

### Scenario

- 计算行数：普通运算只处理当前 1 个 token；反向为局部理论示例

### Approximation

- FLOPs：线性 2MKN 近似；非矩阵特殊函数按全局简化计数；缓存与访存另计

## kv

### Configuration

- 权重：参数表按源码实际定义的 shape 与拥有关系统计，公式采用 XW 行向量

### Scenario

- 计算行数：普通运算只处理当前 1 个 token；反向为局部理论示例

### Approximation

- FLOPs：线性 2MKN 近似；非矩阵特殊函数按全局简化计数；缓存与访存另计

## multi

### Configuration

- Head / Window / Top-K：64 / 128 / 至多 512；都是配置值
- Learnable Sink：每层 64 个；只改变归一化分母

### Scenario

- 计算行数：普通运算只处理当前 1 个 token；反向为局部理论示例
- 历史与读取：此前 1024 个 token；本例 640 个位置，其中 128 Local + 512 Global
- 边界：Attention 读取结果不包含 Query/KV/输出投影、Compressor 或 Indexer；完整 Attention 参数表会列实际拥有的权重

### Approximation

- FLOPs：线性 2MKN 近似；非矩阵特殊函数按全局简化计数；缓存与访存另计

### Unverified

- 反向：历史 KV 梯度及跨 Head/Query 的累加、sink 反传未包含在四个矩阵小计里

## compressor

### Configuration

- source：KV owners 2/8/14 为 ratio=2，20 为 ratio=1；后续层复用
- Indexer：8 套 Query / Head mixing，仅 4 套拥有 Key 投影与归一化

### Scenario

- 计算行数：普通运算只处理当前 1 个 token；反向为局部理论示例
- 本次位置：第 1025 个 token：ratio=2 等待，ratio=1 新增 KV；Indexer 主例为 source layer 20 的 1025 个候选
- 量化：已执行位置处理；普通量化和搬运没有算入所示 FLOPs

### Approximation

- FLOPs：线性 2MKN 近似；非矩阵特殊函数按全局简化计数；缓存与访存另计

## indexer

### Configuration

- source：KV owners 2/8/14 为 ratio=2，20 为 ratio=1；后续层复用
- Indexer：8 套 Query / Head mixing，仅 4 套拥有 Key 投影与归一化

### Scenario

- 计算行数：普通运算只处理当前 1 个 token；反向为局部理论示例
- 本次位置：第 1025 个 token：ratio=2 等待，ratio=1 新增 KV；Indexer 主例为 source layer 20 的 1025 个候选
- 量化：已执行位置处理；普通量化和搬运没有算入所示 FLOPs

### Approximation

- FLOPs：线性 2MKN 近似；非矩阵特殊函数按全局简化计数；缓存与访存另计

## hc

### Configuration

- 四路：hc_mult=4；24 个混合分数，base 24、scale 3，Sinkhorn 20 轮
- 次序：本次生成 pre 给下一子模块；当前输入用前一子模块的 pre_mix
- 方向：comb[i,j] 表示旧输入路 i 向新输出路 j；Y_j=Σ_i R_i C_ij+b_j U

### Scenario

- 计算行数：普通运算只处理当前 1 个 token；反向为局部理论示例

### Approximation

- FLOPs：线性 2MKN 近似；非矩阵特殊函数按全局简化计数；缓存与访存另计

## moe

### Configuration

- 专家：384 routed，Top-6，1 shared；每专家三套 5120↔2304 矩阵
- 层与共享：主干 40 层各自有权重；Shared Expert 为所有 token 执行，不是跨层绑权

### Scenario

- 计算行数：普通运算只处理当前 1 个 token；反向为局部理论示例
- Activated：一个 token 用 6+1 个专家；展示的 I₁…I₆ 为符号，没有运行真实路由 ID
- Router bias：文本 token 使用文本选择偏置；图像选择偏置存储但本次不激活

### Approximation

- FLOPs：线性 2MKN 近似；非矩阵特殊函数按全局简化计数；缓存与访存另计

### Unverified

- Top-K 训练：离散选择不求普通梯度；如何学习选择偏置未由推理代码给出

## engram

### Configuration

- 记忆：layer 1 / 14 两张表；24 次查表 × 256 维，投影 6144→25600
- 权重：q_weight、k_weight 为 [4,5120]；不是额外 Token 或普通 RMSNorm γ

### Scenario

- 计算行数：普通运算只处理当前 1 个 token；反向为局部理论示例
- 输入：当前 token 和最多前三个历史编号；主例 layer 1，另一张表另列

### Approximation

- FLOPs：线性 2MKN 近似；非矩阵特殊函数按全局简化计数；缓存与访存另计
- 反量化：仅本次 6144 个查表数值乘 scale 计入；整数哈希、查行不算矩阵 FLOPs

## head

### Configuration

- 权重：参数表按源码实际定义的 shape 与拥有关系统计，公式采用 XW 行向量

### Scenario

- 计算行数：普通运算只处理当前 1 个 token；反向为局部理论示例

### Approximation

- FLOPs：线性 2MKN 近似；非矩阵特殊函数按全局简化计数；缓存与访存另计

