# 动态 CP vs 静态 CP:变长序列训练的对比发现

> 工具:`llm-perf`(LLM 性能建模)
> 模型:`compare_cp_strategies`(`src/llm_perf/dynamic_cp.py`)+ 变长 1F1B 流水模拟器(`src/llm_perf/pp_pipeline.py`)
> 演示:`python examples/demo_dynamic_cp.py`
> 日期:2026-06-26

---

## 1. 我们在比什么

在**变长序列**训练里,序列长度服从重尾分布(大量短序列 + 少量超长序列)。Context Parallelism(CP)把一条序列沿序列维切到多张卡上,缓解长序列的显存/算力压力。问题是 CP 度怎么选:

| 方案 | CP 分配策略 | 一句话 |
|------|------------|--------|
| **静态 CP(static)** | 所有序列统一用 `cp = max_cp`(按最长序列定档) | 短序列被**过度切分**,陪着长序列一起付 CP 通信代价 |
| **动态 CP(dynamic)** | 每个长度桶按需选 CP:短序列 `cp=1`,长序列 `cp=max_cp` | 短序列不切分,免掉无谓的 ring 通信;长序列才上大 CP |

两个方案在**同一个 global batch**上比较,都叠加 packing + PP bubble,用同一个变长流水模拟器跑出真实 step time。

### 关键抽象:池宽 microbatch(pool-wide unit)

把 CP/DP 卡池(本例 `R=8`)看成一个整体。一个"池宽单元"占满整个池:

- `cp=1` 的单元 ⇒ `dp = R/cp = 8`,一次并行处理 **8 条短序列**,墙钟只算单条 `cp=1` 的时间(快、宽)。
- `cp=8` 的单元 ⇒ `dp = 1`,处理 **1 条长序列**,沿 8 卡切分(慢、窄)。

这把"二维调度(流水级 × 池内 rank packing)"降成一维,可直接复用现有多流模拟引擎。动态 CP 的收益来自两件**相互独立**的事:

1. **吞吐**:短序列走 `dp=8`,一次并行处理 8 条 ⇒ 同样的 batch 只需**更少的池宽单元 `m`**,且每个短单元免掉 over-shard 的 ring 通信。
2. **流水 bubble 更小**:长序列被 CP 切分后,每卡工作量被压到与短序列相当 ⇒ **单元时长更均匀** ⇒ 1F1B bubble 更接近闭式、更低。

> ⚠️ 注意:bubble 变小**不是因为 `m` 变小**。经典 1F1B 公式 `(p-1)/(m+p-1)` 里 `m` 越大 bubble 越小——方向恰好相反。在**变长**场景里,bubble 由**单元时长的方差**主导,而非 `m`:静态 CP 虽然 `m` 更大,但单元时长差 ~15×(短序列 vs 60k 超长尾,全用 cp=8),一条超长单元的串行关键路径撑大 makespan、设备大量空转,反而 bubble 更高。详见 §3 的来源分解与 §5 的 batch 扫描(静态 bubble 随 m 增大并非单调下降,印证 m 不是主因)。

---

## 2. 怎么造"变长分布":log-normal 离散成桶

整个对比的输入是一条**序列长度分布**。真实语料的长度几乎都是**重尾**的:绝大多数样本短(几 k token),少量样本极长(几十 k),且长尾衰减缓慢。我们用 **log-normal(对数正态)** 来刻画它——这是建模正值、重尾长度最常用的分布:支撑集恒正、右尾厚、且只需 `std > avg` 就能制造出"短序列扎堆 + 长尾稀疏"的过离散形态(正是让 CP 分配产生差异的根源)。

### 三步:参数化 → 矩匹配 → 分桶

`lognormal_buckets(avg, std, max_len, n_buckets)`(`src/llm_perf/dynamic_cp.py`)做三件事:

1. **用真实 token 单位参数化**。直接给 `avg`(均值)、`std`(标准差)、`max_len`(截断上限),不让用户去碰对数空间里反直觉的 μ/σ。
2. **矩匹配回 log-space**。把 (avg, std) 解析地换算成底层正态的参数:

   ```
   μ = ln( avg² / √(std² + avg²) )
   σ = √( ln(1 + std² / avg²) )
   ```

   这保证生成分布的真实均值/方差**恰好**等于用户给的 `avg`/`std`(默认场景:μ=7.513, σ=1.269)。

3. **等宽分桶 + CDF 取质量**。在 `(0, max_len]` 上切 `n_buckets` 个等宽区间,每桶用 log-normal CDF(`erf` 闭式)的**增量**作为概率质量,桶中心长度作为代表长度。超过 `max_len` 的尾巴被**截断并归一化**进最后的分桶里(对应真实训练里的 `max_seq_len` 硬截断)。

### 默认场景长这样

`avg=4096, std=8192, max_len=65536, n_buckets=8` → 8 个 `(代表长度, 占比)` 桶:

| 代表长度 | 占比 | 动态 CP 给的档 |
|---------|------|--------------|
| 4096 | **88.3%** | `cp=1`(dp=8,不切) |
| 12288 | 7.7% | `cp=4` |
| 20480 | 2.2% | `cp=8` |
| 28672 | 0.9% | `cp=8` |
| 36864 | 0.4% | `cp=8` |
| 45056 | 0.2% | `cp=8` |
| 53248 | 0.1% | `cp=8` |
| 61440 | 0.1% | `cp=8` |

这张表把"为什么动态 CP 有收益"摆得很直白:**~88% 的样本是短序列**,静态 CP 却把它们全部按最长尾的 `cp=8` 来切——8 张卡绑一条 4k 序列、白付 ring 通信;动态 CP 让这 88% 走 `dp=8`(一次并行 8 条),只对那不到 12% 的长尾上大 CP。每个桶的**代表长度**喂给 `assign_cp` 决定每桶 CP 档,**占比 × global_batch_seqs** 决定该桶的池宽单元数。

### 退化与可调

- **`std=0`(或缺省)→ 单桶定长**:退回到"所有序列等长",动态 CP 此时**没有任何收益**(`speedup ≈ 1.0`,由 `test_compare_pipeline_uniform_no_gain` 锁定)。这既是健全性下界,也说明**收益完全来自长度方差**。
- **可调旋钮**(demo `--avg / --std / --max-len / --buckets`):`std/avg` 比值越大、`max_len` 越高,重尾越极端,动态 CP 收益越大。§5 的敏感性分析就是扫这些维度得到的。

```bash
python examples/demo_dynamic_cp.py --avg 8192 --std 16384 --max-len 131072  # 更极端的重尾
```

---

## 3. 核心结论

**动态 CP 在变长分布下显著快于静态 CP。** 参考场景(Llama-3.1-8B,128 卡,`tp=2 × pp=8 × pool R=8`,`max_cp=8`,序列分布 avg=4096 / std=8192 / max=65536):

| 指标 | 静态 CP | 动态 CP | 动态优势 |
|------|---------|---------|----------|
| **Step time** | 2499.6 ms | 1338.9 ms | **1.87× 更快** |
| 池宽单元数 `m` | 69 | 16 | 4.3× 更少 |
| PP bubble | 76.9% | 46.3% | -30.6 pp |
| **MFU** | 17.6% | 37.5% | **2.1×** |
| TFLOPS/GPU | 141.1 | 299.9 | 2.12× |
| 峰值显存/卡 | 7.1 GB | 7.4 GB | 略高(见下) |
| 可行性 | OK | OK | 都能装下 |

> *(`--global-batch-seqs 64`,默认配置。下方有敏感性分析。)*

加速来自三个可分解的来源:

1. **CP 分配(rank-seconds)**:短序列用 `cp=1` 而非 `cp=8`,不再把 8 张卡绑在一条短序列上,也免掉短序列的 ring 通信。
2. **PP bubble 形状**:由**单元时长的均匀度**决定,而非单元数 `m`。静态把所有序列(含 60k 超长尾)都用 cp=8 塞进同一条流水,单元时长差 ~15×,超长单元的串行关键路径撑大 makespan、设备大量空转 → bubble 76.9%;动态把长序列 CP 切到每卡 ~quota,单元时长拉均匀 → bubble 46.3%。(`m=16` 是 dp 并行的副产品,计入第 1 项的吞吐收益,不是 bubble 低的原因。)
3. **comm overlap**:CP-ring 通信与 attention 计算重叠(`fwd = max(compute, cp_comm) + tp_comm`),静态过度切分暴露的通信更多。

---

## 4. 显存的反直觉之处(O(S) 轴)

动态 CP 虽然更快,但**峰值显存反而略高**(7.4 vs 7.1 GB):

- 静态 `cp=8`:每条序列被切成 8 份,每卡只扛 `S/8` 的激活 ⇒ 单卡显存低,但**浪费了卡**(短序列也占满 8 卡)。
- 动态 `cp=1`(短序列):一条序列整个落在单卡上 ⇒ 单卡激活更高。

这正是论文里"FLOPs 和 memory 无法同时均衡"的体现:动态 CP 用**略高的单卡显存**换**高得多的吞吐**。在本例两者都装得下(`usable_HBM` 足够);**在更大模型(如 DeepSeek-V3 671B)上,显存会成为约束**,动态 CP 的 solver 会被迫对长序列提高 CP(`cp_memory` 项),此时收益收窄——这是建模里 `assign_bin_cp = clamp(max(cp_workload, cp_memory), 1, max_cp)` 捕捉的张力。

---

## 5. 加速比对 batch 大小敏感

单一数字会误导。扫 `--global-batch-seqs`:

| global_batch_seqs | 静态 step | 动态 step | **Speedup** | 静态 bubble | 动态 bubble |
|------|----------|----------|------------|------------|------------|
| 16 | 1357.9 ms | 930.2 ms | **1.46×** | 73.8% | 51.0% |
| 64 | 2499.6 ms | 1338.9 ms | **1.87×** | 76.9% | 46.3% |
| 256 | 4645.8 ms | 2922.5 ms | **1.59×** | 65.2% | 38.2% |

**加速比在 ~1.5–1.9× 之间非单调波动**,但**始终 > 1.4×**:

- batch 越大,两个方案的 bubble 都被更多 microbatch 摊薄,纯算力比成为主导,加速收窄;
- batch 很小,单元数太少、bubble 占比高,动态的单元数优势(m 更少)反而更突出。

**结论的稳健形态是"动态 CP 稳定快 1.4×以上",而非"恰好 1.87×"。** 报告单一数字时务必标注 batch 配置。

---

## 6. 建模口径与已知简化

为避免过度解读,明确建模假设:

**口径(诚实可分解)**
- `speedup = static.step / dynamic.step`,由模拟器实测,含变长 bubble + comm overlap 的真实差异(不是纯解析比)。
- **MFU 用不可约算力(cp=1 基准)**:CP 分片复制的固定开销算 overhead 而非有用功,所以 MFU 有界 ≤ `compute_eff`(本硬件 0.5)。分子分母用同一 backward 口径(`bwd = bwd_factor × fwd`,默认 2.0),保证 MFU 物理合法。
- PP bubble 由**硬调度边强制的真 1F1B 模拟**得出,等长退化时精确复现闭式 `(p-1)/(m+p-1)`。
- 显存为**流水级在飞激活**(1F1B ≈ p 个单元,不随 m 增长)。

**已知简化(待后续)**
- **v>1 interleaved 暂缓**:`simulate_pipeline` 对 v≠1 报 `NotImplementedError`,当前只建模 v=1 标准 1F1B。
- **packing 效率 η 未进单元计数**:单元数现由 dp 倍数主导(`n_b = ceil(bin_seqs / (R//cp))`),碎片打包损耗未建模。
- **backward = 2×forward 启发式**:模拟器未用真实反向算子图(分子分母一致,故 MFU 合法,但绝对时间是估计)。
- **CP 仅建 ring**,Ulysses(all-to-all)未单独建模。

---

## 7. 一句话总结

> 在重尾变长分布下,**静态 CP 为最长序列买单、让短序列陪绑**,动态 CP 按需分配把短序列的卡和通信解放出来 —— 在 Llama-8B/128卡参考场景下带来 **~1.5–1.9× 的 step time 加速**和 **~2× 的 MFU/TFLOPS-per-GPU 提升**,代价是单卡显存略升(在显存吃紧的更大模型上收益会被 solver 的显存约束收窄)。

---

## 复现

```bash
source .venv/bin/activate
python examples/demo_dynamic_cp.py                      # 默认场景
python examples/demo_dynamic_cp.py --global-batch-seqs 256   # 敏感性
python examples/demo_dynamic_cp.py --avg 8192 --std 16384 --max-len 131072  # 换分布
```

相关代码:`src/llm_perf/dynamic_cp.py`(solver + 对比)、`src/llm_perf/pp_pipeline.py`(变长 1F1B 模拟器)、设计文档 `docs/superpowers/specs/2026-06-26-dynamic-cp-pipeline-design.md`。
