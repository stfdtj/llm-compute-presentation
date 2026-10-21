"""Semantic slide source and paragraph mapping, independent of drawing coordinates."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
def item(scene,title,section,paras,clicks,spoken,**extra):
    return dict(type='content',role='evidence',variant='standard',math_scene=scene,title=title,
      section=section,script_paragraphs=paras,clicks=clicks,spoken=spoken,**extra)

S=[
item('architecture','DeepSeek V4.1 Flash：计算成本从架构开始','一.1 整体架构',[0,17],
 ['文本 / 图像进入各自输入模块','显示 40 层 Transformer 主干与 LM Head','显示 DSpark 独立草稿分支'],
 '沿架构 → 参数量 → FLOPs → 硬件效率展开。DSpark 的 3 个 Draft Blocks 是辅助分支，不计为 43 个同样的主层。'),
item('ced','20 层 + 20 层：两部分都遵守 Causal Mask','一.2 Causal Encoder 与 Decoder',[18,25],
 ['显示 Prefill 主要处理输入的路径','显示 Encoder 表示构建 Decoder Global KV','显示 Decode 当前 token 经过 40 层'],
 'Encoder / Decoder 的区别是计算职责和 Global KV 来源，不是传统双向 Encoder 与 Cross-Attention Decoder。'),
item('local','局部窗口读取近处，全局路径补充远处','一.3 Encoder Attention',[26,42],
 ['高亮最近最多 128 个 Local 位置','显示历史 token 的 2:1 全局压缩','连接 Global KV 与当前 Query'],
 '图中只画部分示意位置。Encoder 第 0、1 层只有 SWA；从第 2 层开始增加 Global KV。128 是配置窗口上限。'),
item('indexer','Indexer 先选位置，Main Attention 再取信息','一.3 Global KV 与 Indexer',[43,47],
 ['Indexer Query 为全局位置评分','高亮最多 Top-512 位置','Main Query 读取选中 Global KV 与 Local KV','显示共同 Softmax 与输出'],
 'Indexer 与 Main Attention 使用不同 Query。Top-512 是上限；实际数量取决于可用历史。评分和筛选本身也有开销。'),
item('reuse','CSA2：分清 Full、Reindex 与 Reuse','一.4 CSA2 与层间复用',[48,70],
 ['显示 Encoder 全局源层 2 / 8 / 14','显示 Decoder 第 20 层提供 Global KV','显示 24 / 28 / 32 / 36 层重新选址'],
 'Full 建立 Global KV 和 Index Keys 并选择；Reindex 保留缓存重新选择；Reuse 连最近选址也复用。Decoder 候选池每块 8 位置、最多 2048 块，限制后续搜索。层号从 0 开始。'),
item('mhc','mHC：四路残差信息，一路主模块计算','一.5 mHC',[71,82],
 ['四路 Hidden States 通过 Pre 混为一路','Attention / MoE 执行一次主计算','Post 分配、Comb 混合回四路'],
 '每路维度 5120。四路状态不等于 Attention 和 MoE 直接执行四遍；混合权重和混合过程另有参数和成本。'),
item('moe','Dense 使用同一组 FFN；MoE 选择执行路径','一.6 MoE 与 Dense',[83,96],
 ['Token 进入 Router','显示 384 个 Routed Experts 的评分示意','高亮其中 6 个选中专家','显示始终执行的 1 个 Shared Expert','选中专家执行 FFN','路由加权求和，再加共享专家输出'],
 '专家墙真实包含 384 格。分数与选中的具体专家编号只作教学示意，不是运行轨迹；共享专家不参与这次 Top-6 竞争。'),
item('swiglu','一个 Expert：Gate、Up、Down 三个投影','一.6 Expert 内部计算',[93,96],
 ['X 分别进入 Gate 与 Up','Gate 经过 SiLU 后与 Up 逐元素相乘','Down 将 2304 维投影回 5120 维'],
 'Expert(x)=Down(SiLU(Gate(x)) ⊙ Up(x))。三个 Linear 均无 bias，SiLU 和逐元素乘法无新增权重但有计算。'),
item('engram','Engram：按局部 token 组合查找条件记忆','一.7 Engram',[97,103],
 ['N-gram token 组合进入哈希查找','从大型表中读取少量条目','读取的记忆参与后续处理'],
 'Engram 位于 Encoder 第 1 和 14 层。大量存储参数不等于每 token 都执行整表的稠密乘法；查表仍有访存成本。'),
item('expert_params','从矩阵形状数参数：一个 Expert 约 3540 万','二.1 一个 Expert',[104,124],
 ['显示 Gate 权重 5120 × 2304','显示 Up 和 Down 的矩阵形状','三个矩阵参数量求和'],
 '以单 token 输入 [1,5120] 说明形状，参数量本身不随本次输入 token 数增加。Down 形状按 XW 记法写作 [2304,5120]。'),
item('active','保存 385 个专家，每个 token 使用其中 7 个','二.2–3 Total vs Active',[125,144],
 ['显示全部专家的总参数量','仅高亮 6 Routed + 1 Shared','分别加上 Router 矩阵权重'],
 '单层专家总权重 13,624,934,400；激活专家 247,726,080。Router 矩阵 1,966,080。主数值不含选择偏置，且不是全模型参数量。'),
item('matmul','Matrix Multiplication：一行 × 一列，生成一个数','三.前向 1 从参数到 FLOPs',[145,155],
 ['高亮 A 第一行 [1,2,3]','高亮 B 第一列 [1,3,5]','显示 1×1 = 1','显示 2×3 = 6','显示 3×5 = 15','将乘积相加：1+6+15 = 22','将 22 写入 C 第一行第一列'],
 '小矩阵为教学例。A=[[1,2,3],[4,5,6]]，B=[[1,2],[3,4],[5,6]]。每次点击只推进一个乘法或求和动作，等待听众看清再继续。'),
item('matmul_next','换一列，得到下一个输出；重复同样的规则','三.前向 1 矩阵乘法与计数',[151,155],
 ['高亮 A 第一行与 B 第二列','显示 1×2 + 2×4 + 3×6 = 28','将 28 写入 C，并补齐第二行','显示 3 次乘法 + 2 次加法，以及 2MKN 近似'],
 'C=[[22,28],[49,64]]。每个输出精确 5 次乘加，全矩阵 20 FLOPs；2×2×3×2=24 为常用近似。不是精确 24。'),
item('lowrank','Query Low-Rank Projection：先压缩，再展开','三.前向 2 Query 投影',[156,188],
 ['W_QA：5120 → 1280，13.11 MFLOPs','RMSNorm：维度仍为 1280','W_QB：1280 → 32768，83.89 MFLOPs','重排为 64 Heads × 512，汇总 96.99 MFLOPs'],
 '形状真实，向量宽度为示意，不按维度严格等比。主 FLOPs 仅两个 Linear，排除 RMSNorm / RoPE。Reshape 不增加浮点乘加。'),
item('layers','40 层 Query 投影：约 3.88 GFLOPs','三.前向 3 从单模块到多层',[189,194],
 ['显示一层 Query 的 96.99 MFLOPs','按 40 个主干层累加','显示 3.8797 GFLOPs，并指出其余模块仍需另计'],
 '该乘法针对正常主干 Decode 的两次 Query Linear；不是 Prefill、不是完整全模型前向、也不含 DSpark。'),
item('backward','Forward 保存输入；Backward 沿图反向传梯度','三.Backward 1–2 五步与顺序',[196,214],
 ['显示前向 X → W_QA → RMSNorm → W_QB → Q','① 从 Loss 获得 dQ','② 取回本层保存的输入和权重','反向从 W_QB 经 RMSNorm 回到 W_QA'],
 '原稿用 dZ 指末端梯度；此处聚焦 Query 子图，以 dQ 表示后续 Attention 已传回的梯度。整个模型的 Loss 不直接等于 Query 局部 Loss。'),
item('gradients','每个 Linear 同时产生两种去向的 Gradient','三.Backward 3–5 Weight / Input Gradient',[215,234],
 ['W_QB 收到 dQ','③ dW_QB = H′ᵀ dQ，向权重分支流动','④ dH′ = dQ W_QBᵀ，继续向前','RMSNorm 将 dH′ 转为 dH','W_QA 分支产生 dW_QA 与 dX','⑤ dX 继续传给前一模块'],
 'dW 使用洋红色，dX / 继续传播梯度使用橙色。输入形状、输出形状均列出。RMSNorm 反向必须经过，不是恒等；γ 的梯度未在主图展开，成本小计另排除。'),
item('backward_flops','两次梯度矩阵乘法，让 Linear 反向约为前向两倍','三.Backward 6 反向 FLOPs 汇总',[235,241],
 ['前向一次矩阵乘法 F','反向分别计算 dW 和 dX','汇总 Query Linear Backward 193.99 MFLOPs'],
 '采用 2MKN 近似且输入、权重都需要梯度，排除 RMSNorm、优化器与额外重计算。单 token 精确外积没有跨行求和，因此精确反向不严格等于两倍；见核对报告。'),
item('roofline','Roofline：两条上限，共同限定可达性能','四.1–3 概念与屋顶线',[243,266],
 ['只显示 Arithmetic Intensity 与 Performance 坐标轴','绘制 Memory Bandwidth Limit：AI × BW','加入 Peak Compute 水平线','标出 Ridge Point = Peak / BW','标注 Memory-bound 与 Compute-bound'],
 '双对数坐标。斜线与水平线的较小值构成屋顶；点的纵坐标是模型给出的性能上界，不是实测。H100 SXM BF16 Dense 989 TFLOP/s，3.35 TB/s。'),
item('roofline_batch','权重复用提高 Arithmetic Intensity，点向右移动','四.3–5 Query 流量与 Batch',[267,285],
 ['显示 BF16 权重和激活流量拆分','显示 M=1：AI≈1，Memory-bound','显示 M=1024：AI≈552.3，Compute-bound','显示向右移动轨迹与优化方向'],
 'F(M)=96,993,280M；B(M)=96,993,280+80,896M。每个权重矩阵按批次读取一次，中间 H 写一次读一次；不含 RMSNorm、真实缓存与融合。M 是一次共同处理的行数，不是同一请求未来 token 可任意并行。'),
item('mfu','MFU：把有效模型工作量与可用峰值放在同一尺度','五.MFU 基本公式与示例',[286,320],
 ['Useful FLOPs：100.6M × 60 GFLOPs = 6.036 EFLOPs','Total Peak：2048 × 989.5 TFLOP/s = 2.026496 EFLOP/s','Available FLOPs：乘 6 s；进度条自动在约 0.9 秒内填到 49.6%'],
 'Illustrative Example / Not Measured。报告 batch 100.6M tokens；60 GFLOPs/token、2048 GPUs 和 6 s 均为假设；989.5 为示例硬件规格。60 GFLOPs 不是已验证的全模型训练值。MFU 分子不计额外激活重计算；更高 MFU 不自动表示质量更高或总成本更低。'),
item('summary','沿着计算路径，把成本与效率接起来','五.6–7 综合与总结',[321,328],
 ['架构决定执行路径与可学习权重','前向与反向决定声明范围的 FLOPs','Roofline 解释潜在瓶颈，MFU 衡量训练系统效率'],
 'Roofline 看潜在算力 / 带宽上限，MFU 结合有效模型 FLOPs 与真实吞吐量。仍需 profiling 核实端到端瓶颈；教学估算不是实测。')
]

def main():
  for i,s in enumerate(S,1):
    s['slide_id']=f's{i:02d}-{s["math_scene"]}'
    s['notes']='对应 Word：'+s['section']+'。\n段落索引（从 0 起）：'+str(s['script_paragraphs'])+'。\n讲述：'+s['spoken']+'\n点击顺序：\n'+'\n'.join(f'{n}. {v}' for n,v in enumerate(s['clicks'],1))+'\n本页完成后再次点击切换下一页。'
  d=dict(title='DeepSeek V4.1 Flash 计算成本可视化',deck_style=dict(style_preset='lab-report',composition_grammar='scientific-evidence-plate',style_seed='deepseek-math-lesson',header_variant='plain',font_pair='system_clean_v1',footer_page_numbers=True,readability_contract=dict(min_title_pt=28,min_body_pt=16,min_support_pt=13,min_caption_pt=9,min_footer_pt=9,min_metadata_pt=9,max_title_lines=2)),slides=S)
  (ROOT/'outline.json').write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8')
  (ROOT/'slide_mapping.md').write_text('# 页面与原稿映射\n\n以原稿顺序讲述；不需要重新背一套讲稿。每次点击对应原生 Appear 显现；同一次点击中的箭头和标签一起出现。\n\n|页|标题|Word 章节|提取段落（0 起，含空段）|点击数|\n|---|---|---|---|---|\n'+'\n'.join(f'|{i}|{s["title"]}|{s["section"]}|{s["script_paragraphs"]}|{len(s["clicks"])}|' for i,s in enumerate(S,1))+'\n\n## 逐页提示\n\n'+'\n\n'.join(f'### {i:02d} {s["title"]}\n\n'+s['notes'].replace('\n','\n\n') for i,s in enumerate(S,1)),encoding='utf-8')

if __name__=='__main__': main()
