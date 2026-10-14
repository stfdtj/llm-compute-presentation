# 增量验证：Speedup、连续 Decode 与 Roofline

验证日期：2026-10-13。旧单 Token 公式及范围见 `validation.md`。本轮在原有计算和页面上增量开发；默认 G=1024、S=8192，G 为生成数量，N 为每步可见位置，避免符号混淆。

## 求和过程

第 i 步（从 0 开始）历史 S+i，可见 S+i+1；最后一步历史 S+G−1。生成是串行 Decode，每步 X=[1,D]，不是 [G,D] 的 Prefill。

Σvisible=G[2(S+1)+G−1]/2。默认值为 1024×(2×8193+1023)/2=8,913,408。

普通注意力核心：C(L)=2H(dk+dv)L+scalar×H(5L−1)。累计 = [2H(dk+dv)+5scalarH]ΣL−scalarHG。MHA/GQA 的 L=N；MLA 的核心为 [2H(2r+dr)+6scalarH]ΣN−scalarHG；GDA 按已知递归核心乘 G，完整模块仍未知。

SWA 前缀和 A(t)=Σₙ₌₁ᵗ min(n,W)：a=min(t,W)，A(t)=a(a+1)/2+(t−a)W。区间 ΣL=A(S+G)−A(S)。已经过窗口后每步核心不再增长。

MSA 的 L(n) 在 n≤k×block 时为 n；之后为 (k−1)block+[(n−1) mod block+1]。周期前缀 Q(t)=floor(t/block)×block(block+1)/2+(t mod block)(t mod block+1)/2。令 cutoff=k×block，则超过 cutoff 的 A(t)=cutoff(cutoff+1)/2+(t−cutoff)(k−1)block+Q(t)−Q(cutoff)。区间仍取前缀差。Index Branch 的额外 FLOPs 按 Hkv(2di+scalar)ΣN 累加，不按稀疏 L 缩减。

每步 Projection 与当前缓存写入相同，乘 G 合理；依赖历史的项使用自己的 ΣN 或 ΣL。实现没有用首步完整 FLOPs 直接乘 G。所有累计主结果都是**单层、每序列**；整批另乘 Batch。

默认配置 / 标量启用 / MLA 保留因子的生成 1024 Tokens 模块小计（FLOPs）：

| 架构 | First Token | Last Token | Total | Average / Token |
|---|---:|---:|---:|---:|
| MHA | 269,762,688 | 286,687,200 | 284,902,342,656 | 278,224,944 |
| GQA | 219,431,040 | 236,355,552 | 233,362,735,104 | 227,893,296 |
| MLA | 639,701,152 | 711,131,104 | 691,626,115,072 | 675,416,128 |
| SWA | 100,827,104 | 100,827,104 | 103,246,954,496 | 100,827,104 |
| MSA | 105,345,672 | 107,978,720 | 109,222,088,704 | 106,662,196 |
| GDA | 完整模块未知 | 完整模块未知 | 完整模块未知 | 完整模块未知 |

数值仅作可复核快照，页面不硬编码此表。GDA 累计已知核心为 3,766,484,992 FLOPs。

## Speedup 与 MLA 审核

Speedup=F_MHA/F_arch，Reduction=(1−F_arch/F_MHA)×100%=100(1−1/Speedup)。Core 和 Module 都以当前生成区间比较。Speedup<1 时 Reduction<0，显示计算量增加；GDA 未知 Module 不用于排名。

原 MLA 在 Query content up 后对每头做 Query→latent，Value up 只恢复当前聚合结果；没有重新展开所有历史 K/V，也没有把同一矩阵投影算两次。

在行向量布局下，WUQ[h] 为 [rq,dk]，WUK[h] 为 [r,dk]，WUV[h] 为 [r,dv]，WO[h] 为 [dv,D]：

- Query absorption：WUQ[h] WUK[h]ᵀ → [rq,r]。替换 Query content up 与 Query→latent。投影 FLOPs 差为 2H(rq r−rq dk−dk r)。
- O absorption：WUV[h] WO[h] → [r,D]。替换 Value up 与 O。投影 FLOPs 差为 2H(rD−rdv−dvD)。
- 融合矩阵由离线预计算得到，预计算、临时构造与存储原权重不计每 Token 运行量。权重 HBM 模型只读本路径实际执行的矩阵。
- 已用非方阵、不同头权重校验实数代数等价（相对容差 1e−10），包含输出各头求和。有限精度舍入不保证等价；融合矩阵可能更大，FLOPs 也可能增加。

依据：[DeepSeek-V2 §2.1.2](https://arxiv.org/html/2405.04434v5#S2.SS1.SSS2)。仍排除 RMSNorm / RoPE 旋转，dk 内容分支另加 dr 位置维度，不称为完整官方模块或等容量对比。

## HBM 流量模型

所有估算限定当前**单层整批生成区间**，来源标签为 `modeled-estimate`；用户总量为 `user-estimate`，缺失为 `unknown`。逻辑 Cache Capacity 与区间 Bytes Moved 分开保存。默认条件：权重每步从 HBM 读取、当前 token K/V 在片上使用、每步缓存写入、组内查询头理想共享、无 score/probability 回写、无量化 scales 或通信。

令 Vhist=ΣN−G，Ahist=ΣL−G，B=Batch，b=数据字节，ηw/ηkv=HBM 未命中比例，R=实际读取的 KV 头数。共享时 R=Hkv（MHA 为 H），不共享时 R=H。

| 项 | HBM 字节公式 |
|---|---|
| Projection weights | 执行矩阵元素总数 × b × G × Device Count × ηw；Core Only 为 0 |
| MHA/GQA/SWA 历史 KV | Ahist × R(dk+dv)bBηkv |
| MSA 主 KV + Index | [Ahist×R(dk+dv)+Vhist×di×IndexRepeat]bBηkv；IndexRepeat 为共享 1 / 按 group 重读 Hkv |
| MLA 历史 latent | Vhist×(2r+dr)×LatentRepeat×bBηkv；LatentRepeat 为共享 1 / 按头重读 H |
| 新 KV / latent 写入 | 每步新增缓存元素数 × bBG；MSA Core 不包含 Index Cache 写入 |
| GDA 递归核心状态 | 2Hdkdv bstate BG，每步读与写；仅此状态，不含 ShortConv |
| Module 输入与输出 | 2DbBG；Core 边界使用片上接口，计 0 HBM |
| 可选 score / probability 回写 | 4HΣL bB；MSA Module 另加 2HkvΣN bB 的 index scores |

MSA Core Only 不计 Index Branch 流量和 FLOPs。MLA 的两遍 latent 读取默认都到 HBM；可用历史命中率表达片上缓存。GDA 状态独立按完整读写建模，不将 KV 命中率套给状态。MSA 路由索引假定片上完成；选择比较、内核启动与排序时间不由此浮点 Roofline 描述。

Weight hit=100% 表示已经驻留片上的理想暖缓存，不能解释成 1024 步必然只加载一次。权重不乘每设备 Batch，因为假定同一步各序列共享权重读取。设备数要求整除 Batch，设备持有独立序列与复制权重；不做 Tensor Parallel 估算。

用户可输入完整区间的总 MiB，不强行推断分项。即使提供 GDA 字节，完整模块 FLOPs 未知也不绘 Module 点。改变实验或范围会清空旧自定义字节 / Runtime；仅改 Peak / Bandwidth 保留同一工作量的字节，并清除旧 Runtime；改设备数也清空字节。

## Roofline 与测试依据

TFLOP/s=10¹² FLOP/s，GB/s=10⁹ B/s，TB/s=10¹² B/s，用户 MiB=2²⁰ B。多设备峰值和带宽在独立序列假设下理想求和。

AI=F/Bytes，Ridge=Peak/BW；Pupper=min(Peak,AI×BW)；Tcompute=F/Peak；Tmemory=Bytes/BW；Tlower=max(Tcompute,Tmemory)。恒等式 F/Tlower=Pupper 已测试。区间合并模型是宽松理想下界，不是把 G 次 Decode 变成一次大 GEMM；没有逐内核阶段、调度与延迟。混合 GEMM / 标量使用单一峰值是粗略模型，不能保证 exp 或递归状态指令达到 Tensor Core 峰值。

合成测试输入（不是实际设备）：Peak=100 TFLOP/s、BW=1000 GB/s。两个工作量 300M/100M FLOPs，流量都为 1GB，FLOPs Ratio=3，但二者 Tmemory 都是 1ms，时间下界比值为 1。硬件 UI 默认留空，合成数字仅用于验证与截图。

实测 Runtime 作为额外输入，算 F/t_measured，与 Pupper 分别展示。若实测速率超过估算上界，提示检查计数范围、流量假设、精度与测量；不用于训练 MFU。

Roofline 依据：[Berkeley Lab — Roofline Model](https://amcr.lbl.gov/departments/computer-science-department/ppan/roofline-performance-model/)。MSA/GDA 结构仍沿用初版的论文与边界。

## 验证与限制

- 原有 `tests.js`：23 项通过，原有单 Token、内存、退化和接口保持有效。
- 新增 `extension_tests.js`：25 项通过。解析前缀和与显式逐 Token 循环在 1,620 组架构 / 场景中核验（标量开关、3 种 MLA 路径、9 个初始 Context、5 种生成数、6 种架构）；窗口 / block 边界、严格计数、G=1、默认 1024、等价吸收、Speedup、单位、HBM、未知输入与设备配置均覆盖。
- 浏览器检查：38 项全部通过，记录在 `qa/browser-results.json`；覆盖默认 1024、Single/G=1、累计图、Speedup、MLA 路径、硬件 / 精度、未知点移除、用户字节 / Runtime、场景变化清空、MFU 独立性、Attention 与 Roofline 桌面 / 390px 布局。
- Prefill 本轮未实现，API 拒绝输入；MFU 仍保留训练接口，未利用推理 Roofline 估算 MFU。
- GDA 完整模块、ShortConv 缓冲、真实 HBM 流量及实际硬件速度未核验。MLA 和 MSA 仍是声明路径和计数范围，不能作为所有实现的精确硬件操作数。

新增逻辑主要在 `model.js` 和 `extension.js`，页面仍通过 `build.py` 生成独立 `index.html`，无需外部依赖。
