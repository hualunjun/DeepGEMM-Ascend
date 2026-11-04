# Mega MoE Linear1 W4A4 改造方案

## 1. 目标与范围

目标：

- 将 Mega MoE 的 **第一个 GMM（Linear1）** 从 W4A8 改为 **W4A4**；
- 第二个 GMM（Linear2）保持 **W4A8**；
- routed experts 和 shared experts 的 Linear1 都使用 W4A4；
- 外部输入 token 直接使用 FP4；
- routed / shared 的全部权重均使用 FP4 打包存储。

最终精度组合：

| 部分 | Linear1（第一个 GMM） | Linear2（第二个 GMM） |
| --- | --- | --- |
| routed experts | A=FP4, B=FP4 | A=FP8, B=FP4→FP8 |
| shared experts | A=FP4, B=FP4 | A=FP8, B=FP4→FP8 |

说明：

- Linear2 仍然沿用现在的“AIV0 把 FP4 权重量化成 FP8，再执行 FP8×FP8 MMA”的路径；
- Linear1 不再反量化权重，A/B 都直接以 `float4_e2m1x2_t` 进入 Cube，执行原生 FP4 MX MMA；
- 由于输入和 shared Linear1 都使用 FP4，不再需要保留“输入 FP8 副本”。

---

## 2. 当前实现（W4A8）分析

### 2.1 当前数据流

```text
input.acts (FP8 + SF)
   │
   ├─ dispatch 重排 ──> routed.l1_acts (FP8 + SF)
   │                         │
   │                         └─ AIC:
   │                              A = FP8
   │                              B = AIV0 把 FP4 权重反量化成 FP8
   │                              └─ FP8 × FP8 MMA
   │
   └─ shared Linear1:
        A = input.acts (FP8)
        B = shared_l1_weights (FP8)
        └─ FP8 × FP8 MMA

Linear1 输出
   └─ AIV1: SwiGLU + 量化 ──> routed.l2_acts / shared_l2_acts (FP8 + SF)

Linear2:
   A = FP8
   B = AIV0 把 FP4 权重反量化成 FP8
   └─ FP8 × FP8 MMA
```

### 2.2 当前 W4A8 的本质

当前并不是硬件原生的 W4A8 混合 MMA，而是：

1. 权重以 FP4 打包存储在 GM；
2. AIV0 用 `vf_fp4_to_fp8_nz` 把 FP4 反量化成 FP8，并排成 NZ 写入 L1B；
3. AIC 用 FP8 × FP8 执行 `asc_mmad_mx`；
4. 权重 SF 在 host transform 时统一 `+6`，用来抵消反量化中的 `2^-6`。

关键代码：

- `deep_gemm/include/deep_gemm/simd/mega_moe.hpp:10-55`：`vf_fp4_to_fp8_nz`
- `deep_gemm/include/deep_gemm/mega_moe.hpp:352-392`：AIV0 权重加载、反量化、`ub2l1`
- `deep_gemm/include/deep_gemm/mega_moe.hpp:501-506`：L1/L0 操作数声明为 FP8
- `deep_gemm/include/deep_gemm/mega_moe.hpp:598-600`：`mad()` 执行 FP8 MMA
- `deep_gemm/mega/__init__.py:116-126`、`:151-153`：L1/L2 权重 SF 都 `+6`

### 2.3 当前 shared experts 的特殊性

当前 shared experts：

- 激活来自 `input.acts`（FP8）；
- 权重是 FP8，并且不带 expert 维度；
- 由 AIC 直接 `copy_gm_to_l1_align` 加载到 shared B 区；
- shared B 和 routed B 复用同一块 L1 区域（`layout/mega_moe.hpp` / `mega_moe.hpp:50-59` 的注释）。

本次改造后 shared 权重也改为 FP4，因此 shared 的 B 生产路径需要一起统一。

---

## 3. 目标数据流

```text
input.acts (FP4 packed + UE8M0 SF)
   │
   ├─ dispatch 重排 ──> routed.l1_acts (FP4 packed + SF)
   │                         │
   │                         └─ AIC Linear1:
   │                              A = FP4
   │                              B = AIV0 把 FP4 权重排成 NZ（不反量化）
   │                              └─ FP4 × FP4 原生 MMA
   │
   └─ shared Linear1:
        A = input.acts (FP4)
        B = AIV0 把 shared_l1_weights (FP4) 排成 NZ
        └─ FP4 × FP4 原生 MMA

Linear1 输出
   └─ AIV1: SwiGLU + 量化 ──> routed.l2_acts / shared_l2_acts (FP8 + SF)

Linear2（routed / shared 都一样）:
   A = FP8
   B = AIV0 把 FP4 权重反量化成 FP8
   └─ FP8 × FP8 MMA
```

核心变化：

1. `input.acts` 从 FP8 改为 FP4；
2. `routed.l1_acts` 从 FP8 改为 FP4；
3. shared L1 权重从 FP8 改为 FP4；
4. Linear1 的 B 生产从“反量化”改为“只做 NZ 重排”；
5. AIC 的 Linear1 compute 使用 FP4 操作数；
6. Linear2 和 SwiGLU 的 FP8 路径不变。

---

## 4. 总体设计

### 4.1 统一 B 生产流水线

推荐把 **所有 B（routed/shared、Linear1/Linear2）都交给 AIV0 生产**，AIC 只负责消费。

原因：

- Linear1 和 Linear2 的 B 共用同一块 routed B L1 区域；
- AIV0 按任务顺序串行生产，天然避免 Linear1/Linear2 的 B 竞争；
- 现有 routed B 的 stage 数和事件协议可以保留；
- 可以删除 shared B 的 AIC 直接加载路径，减少一套 shared 专用事件和布局。

改造后：

- AIV0 对 `kLinear1` / `kSharedLinear1`：FP4 → FP4 NZ 重排；
- AIV0 对 `kLinear2` / `kSharedLinear2`：FP4 → FP8 NZ 反量化；
- AIC 对所有任务统一等待 `kL1BFull`，不再区分 shared B 和 routed B。

### 4.2 Linear1 使用原生 FP4 MMA

当 A/B 都是 `float4_e2m1x2_t` 时，`mad()` 会自动选择 `asc_mmad_mx`：

```cpp
// ascend/copy.hpp:370-381
if constexpr (sizeof(TA) == 1 || sizeof(TB) == 1) {
    asc_mmad_mx(...);
} else {
    asc_mmad(...);
}
```

因此不需要新增 MMA 接口，只需要把 Linear1 的 L1/L0 操作数类型改为 FP4。

### 4.3 L1/L0 布局：padded-K 保持 64KB stage

FP4 每个元素 4 bit。如果 L1 视图直接使用真实 `BLOCK_K`，每个 stage 会只有 FP8 的一半，现有 4-stage B 区和事件索引会错位。

解决方式：**L1 视图的 `shape_k` 写成 `2 * BLOCK_K`，仅用于计算 stage 偏移**。

```cpp
// FP8（现有）
l1_ptr<float8_e4m3_t> l1b(MAD_N, BLOCK_K, base);
// size_per_stage = MAD_N * BLOCK_K

// FP4（新）
l1_ptr<float4_e2m1x2_t> l1b_fp4(MAD_N, 2 * BLOCK_K, base);
// size_per_stage = MAD_N * (2 * BLOCK_K) / 2 = MAD_N * BLOCK_K
```

这样 FP4 和 FP8 的 stage 大小完全一致，现有：

- `kNumL1BStages = 4`
- `kL1BFull / kL1BEmpty`
- `l1b_stage_idx` 推进逻辑
- AIV0/AIC 的 stage 对应关系

都可以保留。

安全性说明：

- `copy_gm_to_l1` 的 K-major 路径只使用 `dst.shape_mn` 和 `actual_k/2`，不使用 `dst.shape_k`；
- `copy_l1_to_l0a/b` 的 K-major 路径只使用 `src.shape_mn` 和 `k_idx / FRAC_K`，不使用 `src.shape_k`；
- AIV0 的 `asc_copy_ub2l1` 只写紧凑的 `MAD_N * BLOCK_K / 2` 字节。

因此每个 64KB stage 实际只使用前 32KB，但 stage 基址、事件索引、stage 轮转都与 FP8 完全一致。

L0A/L0B 使用真实形状：

```cpp
l0a_ptr<float4_e2m1x2_t> l0a(MAD_M, MAD_K);
l0b_ptr<float4_e2m1x2_t> l0b(MAD_N, MAD_K);
```

L0 区域是独立的，不需要 padded。

### 4.4 SF 规则

| 权重 | 是否 `+6` | 原因 |
| --- | --- | --- |
| routed Linear1 | 否 | 原生 FP4 MMA，直接使用真实 E2M1 |
| shared Linear1 | 否 | 原生 FP4 MMA |
| routed Linear2 | 是 | 仍然 FP4→FP8，反量化产生 `2^-6` |
| shared Linear2 | 是 | 仍然 FP4→FP8 |

因此 `transform_weights_for_mega_moe` 中：

```python
if l1_w.dtype == torch.int8:
    # 只补偿 Linear2
    l2_sf = _scale_fp4_weight_sf(l2_sf)
```

L1 的 SF 必须保持原值。

---

## 5. 具体代码改动

### 5.1 `deep_gemm/include/deep_gemm/layout/mega_moe.hpp`

#### 5.1.1 新增 `FP4Buffer`

```cpp
struct FP4Buffer {
    Buffer<uint8_t> data;   // rows * cols / 2 bytes
    Buffer<int16_t> sf;     // rows * ceil(cols / MX_SF_DIVISOR) int16

    constexpr __aicore__ FP4Buffer(uint64_t& offset, uint64_t rows, uint32_t cols):
        data{offset, rows * (cols / 2), kNumBufferAlignmentBytes},
        sf{offset, rows * ceil_div(cols, MX_SF_DIVISOR), kNumBufferAlignmentBytes} {}
};
```

#### 5.1.2 `input.acts` 改为 FP4

```cpp
struct {
    FP4Buffer acts;
    Buffer<int64_t> topk_idx;
    Buffer<float> topk_weights;
} input{
    .acts = {next_offset, config.num_max_tokens_per_rank, config.hidden},
    ...
};
```

`shared_l1_acts` 继续等于 `acts`（在 `csrc/apis/mega_moe.hpp` 中建立视图）。

#### 5.1.3 `routed.l1_acts` 改为 FP4，`l2_acts` 保持 FP8

```cpp
struct {
    FP4Buffer l1_acts;

    Buffer<float> topk_weights;
    Buffer<uint32_t> src_metadata;

    FP8Buffer l2_acts;
    Buffer<bf16_t> output;
} routed{
    .l1_acts = {next_offset, num_ring_rows, config.hidden},
    ...
    .l2_acts = {next_offset, num_ring_rows, config.intermediate_hidden},
    ...
};
```

`shared_l2_acts` 保持 `FP8Buffer`。

#### 5.1.4 L1 布局注释更新

统一 B 生产后，shared B 不再单独占用/复用一套阶段。L1 布局仍然是：

```text
L1  A       0      3 x 64KB
    B       192    4 x 64KB
    A SF    448    2 x 16KB
    B SF    480    2 x 16KB
```

FP4 的 B 每个 64KB slot 实际使用 32KB，这是预期行为。

### 5.2 `deep_gemm/include/deep_gemm/simd/mega_moe.hpp`

#### 5.2.1 新增 `vf_fp4_to_fp4_nz`

用途：Linear1 权重只做 NZ 重排，不反量化。

要点：

- 输入是 `MAD_N x (BLOCK_K/2)` 的 packed FP4；
- 输出是 NZ 布局，C0 = 32 bytes，对应 64 个 FP4 值；
- 不需要 nibble 扩展，只需要把 32-byte 块按 NZ 顺序 scatter；
- 复用现有 `vf_fp4_to_fp8_nz` 的索引/步长逻辑，但保留 packed nibble。

#### 5.2.2 保留 `vf_fp4_to_fp8_nz`

Linear2 的 AIV0 反量化继续使用，不做修改。

#### 5.2.3 `vf_swiglu_quant` 不变

SwiGLU 输出仍然量化成 FP8，供 Linear2 使用。

### 5.3 `deep_gemm/include/deep_gemm/mega_moe.hpp`

#### 5.3.1 AIV0：支持 shared 任务 + Linear1 FP4 NZ

`aiv0_weight_pipeline` 当前调用：

```cpp
sched::take_linear_task<workspace>(0, num_total_m_blocks, ...);
```

改为传入真实 token 数，使 AIV0 也能看到 shared 任务：

```cpp
sched::take_linear_task<workspace>(
    num_tokens, num_total_m_blocks, schedule, core_idx, block_phase);
```

任务解码需要同时处理 shared / routed：

```cpp
const bool is_shared = block_phase == sched::BlockPhase::kSharedLinear1 ||
                       block_phase == sched::BlockPhase::kSharedLinear2;
const bool is_linear1 = block_phase == sched::BlockPhase::kLinear1 ||
                        block_phase == sched::BlockPhase::kSharedLinear1;

const uint32_t intermediate_hidden =
    is_shared ? config.shared_intermediate_hidden : config.intermediate_hidden;
const uint32_t num_n_blocks =
    is_linear1 ? intermediate_hidden / (BLOCK_N / 2) : config.hidden / BLOCK_N;
const uint32_t num_k = is_linear1 ? config.hidden : intermediate_hidden;

const uint32_t pool_block_idx = task_idx / num_n_blocks;
const uint32_t n_block_idx = task_idx % num_n_blocks;
const uint32_t layer_idx = is_linear1 ? 0 : 1;

const uint32_t valid_m = is_shared
    ? min(num_tokens - pool_block_idx * BLOCK_M, BLOCK_M)
    : expert_cursors[layer_idx].advance_to(buffer, pool_block_idx);

const uint64_t weight_block_idx = is_shared
    ? n_block_idx
    : static_cast<uint64_t>(expert_cursors[layer_idx].local_expert_idx) * num_n_blocks + n_block_idx;

const auto task_weights = is_linear1
    ? (is_shared ? weights.shared_l1 : weights.l1)
    : (is_shared ? weights.shared_l2 : weights.l2);
```

`load_weights` 分支：

```cpp
if (is_linear1) {
    // FP4 -> FP4 NZ，不反量化
    constexpr uint32_t kNumColGroups = BLOCK_K / get_frac_k<float4_e2m1x2_t>();  // BLOCK_K / 64
    constexpr l1_ptr<float4_e2m1x2_t, Major::K> l1b(
        MAD_N, 2 * BLOCK_K, kNumL1AStages * L0ASizeBytes);

    // copy_gm_to_ub -> vf_fp4_to_fp4_nz -> asc_copy_ub2l1
} else {
    // 保持现有 FP4 -> FP8 NZ 反量化
}
```

注意：

- Linear1 的 `kNumColGroups` 是 `BLOCK_K / 64`，不是 `BLOCK_K / 32`；
- `l1b` 使用 `2 * BLOCK_K` 保持 64KB stage；
- `is_shared` 的早期 reduce 条件必须排除：

```cpp
if (not is_shared and kOverlapReduce and ...) {
    advance_combine_reduce(...);
}
```

- shared 任务的 L2 cache hint 与 AIC 保持一致：

```cpp
const bool reuse_shared_weights = ceil_div(num_tokens, BLOCK_M) > 1;
const auto weight_l2_hint = is_shared
    ? (reuse_shared_weights ? asc_load_l2_cache_mode::NORMAL_LAST_VICTIM
                            : asc_load_l2_cache_mode::NOTALLOC_KEEP)
    : (expert_cursors[layer_idx].num_expert_tokens > BLOCK_M
           ? asc_load_l2_cache_mode::NORMAL_LAST_VICTIM
           : asc_load_l2_cache_mode::NOTALLOC_KEEP);
```

- `valid_m` 的模板选择必须与 AIC 一致：

```cpp
const uint32_t shape_m = is_shared ? num_tokens : valid_m;
if (shape_m <= 128)
    load_weights.template operator()<128>();
else
    load_weights.template operator()<256>();
```

#### 5.3.2 AIC：Linear1 使用 FP4 compute

建议把当前单一的 `compute` 拆成：

- `compute_fp8<MAD_M>()`：处理 Linear2（shared + routed），保持现有逻辑；
- `compute_fp4<MAD_M>()`：处理 Linear1（shared + routed），使用 FP4 A/B。

`compute_fp4` 的关键声明：

```cpp
constexpr l1_ptr<float4_e2m1x2_t, Major::K> l1a(MAD_M, 2 * BLOCK_K, 0);
constexpr l1_ptr<float4_e2m1x2_t, Major::K> l1b(
    MAD_N, 2 * BLOCK_K, l1a.offset(kNumL1AStages));
constexpr l0a_ptr<float4_e2m1x2_t> l0a(MAD_M, MAD_K);
constexpr l0b_ptr<float4_e2m1x2_t> l0b(MAD_N, MAD_K);
constexpr l0c_ptr<float> l0c(MAD_M, MAD_N);
```

A 加载：

```cpp
// input 的基址按字节定位：Linear1 每行 hidden/2 字节
const auto input = input_layout.data.get_ptr(
    buffer, input_m_idx * (num_k / 2));

copy_gm_to_l1(
    l1a[l1a_stage_idx],
    gm_ptr<float4_e2m1x2_t, Major::K>(num_k / 2, reinterpret_cast<uintptr_t>(input)),
    valid_m, valid_k,
    asc_load_l2_cache_mode::NORMAL_LAST_VICTIM);
```

B 使用 AIV0 生产的 FP4 NZ：

```cpp
const auto l1b_src = l1b[
    (l1b_stage_idx + n_mad_idx / MAD_N) % kNumL1BStages];

copy_l1_to_l0b(l0b[l0b_stage_idx], l1b_src, 0, k_mad_idx);
```

`mad()` 不需要改，FP4 自动走 `asc_mmad_mx`。

#### 5.3.3 AIC：删除 shared B 直接加载路径

删除：

- `MMAConfig::get_num_shared_l1b_stages()`
- `kNumSharedL1BStages`
- `kSharedL1BFull / kSharedL1BEmpty`
- `shared_l1b` 指针
- `is_shared` 下的 `copy_gm_to_l1_align` 加载
- shared B 的 `wait / notify`
- `notify<PIPE_MTE1, PIPE_MTE2, 3>(kSharedL1BEmpty)`

所有任务都统一使用 routed B 区域和 `kL1BFull / kL1BEmpty`：

```cpp
// 等待 B
for (uint32_t n_mad_idx = 0; n_mad_idx < BLOCK_N; n_mad_idx += MAD_N)
    wait_intra_block<PIPE_MTE1, 0>(
        kL1BFull, (l1b_stage_idx + n_mad_idx / MAD_N) % kNumL1BStages);

// 释放 B
for (uint32_t n_mad_idx = 0; n_mad_idx < BLOCK_N; n_mad_idx += MAD_N)
    notify_intra_block<PIPE_MTE1, 0>(
        kL1BEmpty, (l1b_stage_idx + n_mad_idx / MAD_N) % kNumL1BStages);
l1b_stage_idx = (l1b_stage_idx + (BLOCK_N / MAD_N)) % kNumL1BStages;
```

`copy_l1_to_l0b` 的 `mn_idx` 固定为 `0`，因为每个 stage 已经只包含一个 `MAD_N` 子块。

#### 5.3.4 AIC：L1B 初始 credit 提前

当前在第一个 routed 任务处才给 AIV0 初始 `kL1BEmpty`：

```cpp
// 现有：mega_moe.hpp:480-484
if (not is_shared and not routed_l1b_initialized) { ... }
```

由于 shared 任务现在也由 AIV0 生产 B，必须在 kernel 启动时就初始化：

```cpp
notify_intra_block<PIPE_MTE1, 0, kNumL1BStages>(kL1BEmpty);
```

并删除 `routed_l1b_initialized`。

#### 5.3.5 AIC：shared Linear1 的 A 也使用 FP4

`input_layout` 选择逻辑保持：

```cpp
const auto input_layout = is_linear1
    ? (is_shared ? workspace.input.acts : workspace.routed.l1_acts)
    : (is_shared ? workspace.shared_l2_acts : workspace.routed.l2_acts);
```

但 Linear1 的 `input` 字节偏移改成 `input_m_idx * num_k / 2`；Linear2 仍是 `input_m_idx * num_k`。

#### 5.3.6 AIV1：dispatch 改为 FP4

`advance_dispatch_pull` 中：

```cpp
// 原：hidden 字节
copy_gm_to_ub(..., remote_token, 1, config.hidden, ...);
// 新：hidden/2 字节
copy_gm_to_ub(..., remote_token, 1, config.hidden / 2, ...);
```

写回 `routed.l1_acts` 同样改成 `config.hidden / 2`。

`ub_pull_token` 可以改成 `config.hidden / 2` 大小。SF 的拉取和写回不变。

SwiGLU、combine、Linear2 完全不变。

### 5.4 `csrc/apis/mega_moe.hpp`

#### 5.4.1 输入视图

```cpp
auto x = slice_buffer(
    buffer, workspace.input.acts.data.offset,
    {num_tokens, hidden / 2}, torch::kInt8);
auto x_sf = slice_buffer(
    buffer, workspace.input.acts.sf.offset,
    {num_tokens, config.num_input_sf_pairs}, torch::kInt16);
```

`shared_l1_acts = x`，`shared_l1_acts_sf = x_sf`（如果有 shared experts）。

`l1_acts` 视图：

```cpp
auto l1_acts = slice_buffer(
    buffer, workspace.routed.l1_acts.data.offset,
    {num_ring_tokens, hidden / 2}, torch::kInt8);
```

`shared_l2_acts`、`l2_acts` 保持 FP8。

#### 5.4.2 权重检查

shared 权重不再特殊：

```cpp
DJ_HOST_ASSERT(weights.scalar_type() == torch::kInt8 and weights.is_contiguous());
DJ_HOST_ASSERT(weights_sf.scalar_type() == torch::kInt16 and weights_sf.is_contiguous());

const int64_t weight_shape[]{
    num_experts_per_rank, num_n_blocks, layout::BLOCK_N, num_k / 2};
const int64_t sf_shape[]{
    num_experts_per_rank, num_n_blocks, num_k / MX_SF_DIVISOR, layout::BLOCK_N};

DJ_HOST_ASSERT(weights.sizes() == torch::IntArrayRef(weight_shape + is_shared, 4 - is_shared));
DJ_HOST_ASSERT(weights_sf.sizes() == torch::IntArrayRef(sf_shape + is_shared, 4 - is_shared));
```

即 shared 也是 int8 packed，shape 为 `[num_n_blocks, BLOCK_N, num_k/2]`。

#### 5.4.3 API / mma_type

建议新增/调整 `mma_type`，例如：

```text
fp4xfp4+fp8xfp4
```

含义：

- Linear1：FP4 × FP4
- Linear2：FP8 × FP4

`SymmBuffer` 和 `get_symm_buffer_size_for_mega_moe` 都使用这个值做校验和 buffer sizing。

第一版可以直接替换旧路径；如果必须保留旧的 `fp8xfp4` 路径，需要给 `Workspace` 增加编译期开关，分别实例化两套 layout，成本较高，不建议首版实现。

### 5.5 `deep_gemm/mega/__init__.py`

#### 5.5.1 权重 SF 规则

```python
if l1_w.dtype == torch.int8:
    # Linear1 使用原生 FP4，不做 +6
    # Linear2 仍然 FP4 -> FP8，需要 +6
    l2_sf = _scale_fp4_weight_sf(l2_sf)
```

#### 5.5.2 shared 权重

shared L1/L2 的权重也走 `transform_weights_for_mega_moe`，输入为 FP4 packed：

```python
shared_l1_weights = make_npu_fp4((2 * shared_intermediate_hidden, hidden), seed)
shared_l2_weights = make_npu_fp4((hidden, shared_intermediate_hidden), seed)
```

`transform_weights_for_mega_moe` 的 `_interleave_weights` / `_reshape_weights` 已经支持 2D 输入。

#### 5.5.3 `fp8_fp4_mega_moe` 命名

建议新增：

```python
fp4_fp4_mega_moe(...)
```

或保留 `fp8_fp4_mega_moe` 作为内部实现，但文档和测试明确它现在执行的是 `Linear1 W4A4 + Linear2 W4A8`。

### 5.6 `tests/test_mega_moe.py`

#### 5.6.1 输入和 shared 权重改为 FP4

```python
self.l1_weights = make_npu_fp4((num_experts_per_rank, 2 * intermediate_hidden, hidden), seed)
self.l2_weights = make_npu_fp4((num_experts_per_rank, hidden, intermediate_hidden), seed)

if num_shared_experts:
    shared_intermediate_hidden = num_shared_experts * intermediate_hidden
    self.shared_l1_weights = make_npu_fp4(
        (2 * shared_intermediate_hidden, hidden), 24000)
    self.shared_l2_weights = make_npu_fp4(
        (hidden, shared_intermediate_hidden), 25000)
```

`create_inputs`：

```python
self.x = make_npu_fp4((num_tokens, buffer.hidden), 22000 + seed_offset + rank_idx)
```

`buffer.x` 现在是 `[num_tokens, hidden/2]` 的 int8 视图。

#### 5.6.2 baseline 改为 FP4 Linear1

Routed Linear1：

```python
deep_gemm.m_grouped_fp4_gemm_nt_contiguous(
    recv_x_fp4, self.l1_weights, l1_y, layout,
    recipe=(1, 1, 32), use_psum_layout=True)
```

Routed Linear2 保持：

```python
deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous(
    l2_x, self.l2_weights, l2_y, layout,
    recipe=(1, 1, 32), use_psum_layout=True)
```

Shared Linear1：

```python
deep_gemm.fp4_gemm_nt(
    self.x, self.shared_l1_weights, shared_l1_y, recipe=(1, 1, 32))
```

Shared Linear2：

```python
deep_gemm.fp8_fp4_gemm_nt(
    shared_l2_x, self.shared_l2_weights, shared_y, recipe=(1, 1, 32))
```

如果 DeepEP dispatch 不支持 FP4，可以继续用 FP8 dispatch，然后在调用 `m_grouped_fp4_gemm` 前把 `recv_x` 量化成 FP4。由于 E2M1 的值都能被 E4M3 精确表示，使用相同的 UE8M0 规则重新量化应能恢复相同 FP4 code；需要增加一次 round-trip 校验。

#### 5.6.3 benchmark 统计

- Linear1 的 A 从 1 byte/元素改成 0.5 byte/元素；
- Linear2 的 A 仍为 1 byte/元素；
- remote pull 的 token 字节数从 `hidden` 改成 `hidden/2`；
- shared L1 权重从 FP8 改成 FP4，shared L2 权重的反量化开销仍在 AIV0。

---

## 6. 正确性关键点

### 6.1 SF `+6` 必须只作用于 Linear2

这是最容易出错的地方：

- Linear1 权重 SF：原值；
- Linear2 权重 SF：`+6`。

如果 Linear1 误加 `+6`，输出会整体差 `2^6`。

### 6.2 AIV0 和 AIC 的 MAD_M 模板必须一致

AIV0 和 AIC 都根据 `valid_m` 选择 `MAD_M = 128/256`。对于 shared 任务，AIC 使用的是：

```cpp
(is_shared ? num_tokens : valid_m) <= 128
```

AIV0 必须使用同样的判断，否则：

- `BLOCK_K`、`MAD_N` 不一致；
- B 的 stage 解释不一致；
- 会读写错误的 NZ 布局。

### 6.3 L1B stage 必须保持 64KB

FP4 Linear1 的 L1 视图必须使用 `2 * BLOCK_K` 的 `shape_k`。否则：

- stage 大小变成 32KB；
- `l1b_stage_idx` 和 AIV0/AIC 的事件索引错位；
- 可能出现覆盖或读到旧数据。

### 6.4 shared 任务必须全部在 routed 任务之前

`sched::take_linear_task` 对 shared 任务按 `layer_idx = 0, 1` 顺序处理，即：

1. 所有 shared Linear1 任务；
2. 所有 shared Linear2 任务；
3. routed 任务。

因此共享 B 区域（现在统一为 routed B 区域）可以按任务顺序复用。不要修改这个顺序。

### 6.5 AIV0 的初始 L1B credit

统一 B 生产后，shared 任务也需要 AIV0 写 B，因此 AIC 必须在 kernel 启动时就 `notify kL1BEmpty`，而不是等到第一个 routed 任务。

### 6.6 AIV0 早期 reduce 不能作用于 shared 任务

```cpp
if (not is_shared and kOverlapReduce and ...) { ... }
```

### 6.7 dispatch 的字节数

`input.acts` 和 `routed.l1_acts` 都是 FP4：

- 数据：`hidden / 2` 字节；
- SF：`hidden / 64` 个 int16，保持不变。

### 6.8 Linear2 完全不变

`routed.l2_acts`、`shared_l2_acts` 仍是 FP8；`vf_swiglu_quant` 不变；AIV0 的 Linear2 反量化不变。

---

## 7. 验证计划

### 7.1 单元验证

1. `vf_fp4_to_fp4_nz`：
   - 输入随机 packed FP4；
   - 输出与 AIC `copy_gm_to_l1` 生成的 NZ 布局逐字节对比。

2. `transform_weights_for_mega_moe`：
   - Linear1 SF 不变；
   - Linear2 SF 每个 exponent `+6`；
   - routed/shared 都覆盖。

3. 权重检查：
   - routed Linear1/L2：`[E, nblocks, BLOCK_N, K/2]`；
   - shared Linear1/L2：`[nblocks, BLOCK_N, K/2]`。

### 7.2 单算子验证

- Linear1：`deep_gemm.fp4_gemm_nt` / `m_grouped_fp4_gemm_nt_contiguous` 对比；
- Linear2：`deep_gemm.fp8_fp4_gemm_nt` / `m_grouped_fp8_fp4_gemm_nt_contiguous` 对比；
- shared 和 routed 分别覆盖。

### 7.3 端到端验证

```bash
python tests/test_mega_moe.py --skip-prof
```

重点检查：

- `check_correctness` 的 bit-exact repeated runs；
- routed/shared 输出与 baseline 的 diff；
- 空 token、小 token、大 token；
- shared experts 数量 0 / 1 / 2；
- `cumulative_local_expert_recv_stats` 仍然正确。

### 7.4 性能验证

对比改造前后：

- Linear1 TFLOPS；
- L1B / L0B 流量；
- HBM 带宽；
- dispatch/combine 通信带宽；
- 小 token 延迟（shared 任务串行化可能影响启动）。

---

## 8. 风险与备选方案

### 8.1 AIV0 启动依赖 metadata

统一 B 生产后，AIV0 需要先执行 `wait_dispatch_metadata`，然后才能生产 shared B；而 AIC 的 shared Linear1 现在依赖 AIV0。

影响：小 token 场景下，shared Linear1 会被 metadata 等待串行化。

备选：

1. 首版接受该开销，先保证正确性；
2. 把 shared 任务拆到 metadata 等待之前执行：
   - AIV0 先用 `metadata_ready = false` 跑 shared-only 任务；
   - 完成后再 `wait_dispatch_metadata`，继续 routed 任务；
   - 需要把 `aiv0_weight_pipeline` 改成有状态的两阶段调用。
3. 保留 shared Linear1 由 AIC 直接加载 FP4，仅 shared Linear2 由 AIV0 反量化；需要额外的 shared B 区域/事件，L1 容量和同步复杂度更高。

### 8.2 shared Linear2 的 B 处理

本方案统一由 AIV0 反量化 shared Linear2 权重。若实现时发现 shared 任务的调度/事件处理过于复杂，可以退化为：

- shared Linear2 权重在 host 侧预反量化成 FP8（一次性），kernel 保持 AIC 直接加载；
- 代价是 shared Linear2 权重存储从 FP4 变成 FP8，失去这部分权重带宽收益；
- shared experts 数量通常很小，这个折中可以作为临时方案。

### 8.3 旧 W4A8 输入路径的兼容性

本方案直接替换 `input.acts` 为 FP4，旧的 `fp8xfp4` 路径不能复用同一套 layout。

如果必须同时支持：

- 给 `Workspace` 增加编译期 `kLinear1AIsFP4`；
- 分别实例化 FP8/FP4 两套 buffer 类型；
- 增加两套 JIT 实例化和 API 校验。

首版不建议做双路径，先完成 W4A4。

---

## 9. 实施顺序

建议按以下里程碑推进：

### M1：Host / 权重 / SF

- `transform_weights_for_mega_moe` 只对 Linear2 `+6`；
- shared 权重支持 FP4 packed；
- `check_weights` 支持 shared int8 packed；
- 增加权重布局单元测试。

验收：routed/shared 权重和 SF 的 shape、exponent 正确。

### M2：Layout / Buffer

- 新增 `FP4Buffer`；
- `input.acts`、`routed.l1_acts` 改 FP4；
- `shared_l2_acts`、`routed.l2_acts` 保持 FP8；
- 更新 `get_symm_buffer_size_for_mega_moe` 和 Python 视图。

验收：buffer size、offset、view shape 正确；测试能构造输入。

### M3：AIV0

- 支持 shared 任务调度；
- 新增 `vf_fp4_to_fp4_nz`；
- Linear1 走 FP4 NZ 重排；
- Linear2 保持 FP4→FP8；
- 验证 L1B stage 布局。

验收：AIV0 生产的 B 与 AIC `copy_gm_to_l1` 的 NZ 布局逐字节一致。

### M4：AIC

- `compute_fp4` 处理 Linear1；
- `compute_fp8` 处理 Linear2；
- 删除 shared B 直接加载路径；
- 初始 L1B credit 提前；
- 验证 Linear1/Linear2 单算子结果。

验收：`fp4_gemm` / `fp8_fp4_gemm` 对比通过。

### M5：端到端

- dispatch 改 FP4 字节数；
- 更新 baseline 和测试；
- 跑 `test_mega_moe.py`；
- 修正精度/阈值问题。

验收：routed/shared 正确性通过，重复运行 bit-exact。

### M6：性能

- 调整 `MAD_M/MAD_N/BLOCK_K` 和 L1 stage 数；
- 评估 AIV0 shared 串行化影响；
- 对比 W4A8 基线的 TFLOPS / HBM / Comm。

---

## 10. 验收标准

- Linear1（routed + shared）使用 FP4×FP4 原生 MMA；
- Linear2（routed + shared）保持 FP8×FP8 + FP4 反量化；
- shared Linear1 直接消费 FP4 输入，不再需要 FP8 副本；
- `transform_weights_for_mega_moe` 的 SF 规则正确；
- `test_mega_moe.py` 正确性通过；
- 性能报告中 Linear1 的权重 L1 流量和输入通信量下降；
- 无 shared/routed 任务顺序、L1B 事件或 stage 错位导致的偶发错误。

---

## 11. 实施记录（当前代码状态）

本节记录实际落地时相对前面方案的调整，以及已经在环境中完成的验证。

### 11.1 实际实现的 B 流水线

实施时采用了比“AIV0 做 FP4 NZ 重排”更简单、也更容易保证正确性的方案：

- **Linear1（routed + shared）的 B 由 AIC 直接加载**：
  - AIC 使用 `copy_gm_to_l1<float4_e2m1x2_t>` 从 GM 直接做 nd2nz；
  - 不需要新增 `vf_fp4_to_fp4_nz`；
  - Linear1 B 使用独立的 L1 direct 区域，避免与 AIV0 生产的 Linear2 B 竞争。
- **Linear2（routed + shared）的 B 仍由 AIV0 反量化**：
  - 保持 `vf_fp4_to_fp8_nz`；
  - routed 和 shared Linear2 都由 AIV0 处理；
  - shared Linear1 任务在 AIV0 中直接跳过。

### 11.2 实际 L1 布局

为同时容纳 Linear1 direct B 和 Linear2 AIV0 B，L1 布局调整为：

```text
L1  A                 0      2 x 64KB
    Linear1 B       128      2 x 64KB  (AIC direct)
    Linear2 B       256      3 x 64KB  (AIV0)
    A SF            448      2 x 16KB
    B SF            480      2 x 16KB
```

对应常量：

```cpp
constexpr uint32_t kNumL1AStages = 2, kNumL1BStages = 3, kNumSFStages = 2;
constexpr uint32_t kNumL1BDirectStages = 2;
constexpr uint32_t kL1BL2BaseBytes = kNumL1AStages * L0ASizeBytes + kNumL1BDirectStages * L0BSizeBytes;
constexpr uint32_t kL1SfBaseBytes = kL1BL2BaseBytes + kNumL1BStages * L0BSizeBytes;
```

Linear1 的 A 使用 `l1_ptr<float4_e2m1x2_t>(MAD_M, 2 * BLOCK_K)` 的 padded-K 视图，保持
Linear1 的 A stage 与 Linear2 一样是 64KB。

### 11.3 实际代码改动

已完成：

- `layout/mega_moe.hpp`：
  - 新增 `FP4Buffer`；
  - `input.acts`、`routed.l1_acts` 改为 FP4；
  - `routed.l2_acts`、`shared_l2_acts` 保持 FP8。
- `mega_moe.hpp`：
  - AIV0 只处理 Linear2，支持 routed/shared；
  - AIC 增加 `compute<MAD_M, kIsFP4>`，Linear1 走 FP4 direct B，Linear2 走 AIV0 B；
  - 删除 shared B 直接加载和 `kSharedL1B*` 事件；
  - dispatch 的 token 数据改为 `hidden/2` 字节；
  - L1B 初始 credit 提前到 kernel 启动。
- `csrc/apis/mega_moe.hpp`：
  - `check_weights` 对所有 routed/shared 权重统一要求 int8 packed；
  - 输入 `x` 和 `l1_acts` 视图改为 `[rows, hidden/2]` int8；
  - `mma_type` 改为 `"fp4xfp4+fp8xfp4"`。
- `deep_gemm/mega/__init__.py`：
  - 权重 SF 只对 Linear2 `+6`；
  - `mma_type` 默认值更新。
- `tests/test_mega_moe.py`：
  - 输入 x 和 shared 权重改为 `make_npu_fp4`；
  - shared Linear1/Linear2 baseline 分别改为 `fp4_gemm_nt` / `fp8_fp4_gemm_nt`；
  - symmetric buffer 的 `expert_recv_count` 读取改为 `clone().cpu()`。

### 11.4 CANN 9.2.0 兼容性修改

使用 `/mnt/xds/sfs/z00956057/cann-9.2.0` 时，补充了两个编译兼容性修改：

- `common.hpp`：`#include <utility>` 改为 `<type_traits>`，避免 `.asc` 编译时
  libstdc++ 13 的 `stl_pair.h` 在 C++20 下报 `operator=` 重定义。
- `comm/mega_moe.hpp`：显式包含
  `utils/debug/asc_assert.h` 和 `utils/debug/asc_printf.h`，提供 `__trap` / `printf`。

### 11.5 已完成的运行验证

构建：

```bash
source /mnt/xds/sfs/z00956057/cann-9.2.0/set_env.sh
./develop.sh
```

设备侧 JIT 编译验证：

```bash
bisheng -O2 --npu-arch=dav-3510 --cce-aicore-only \
  -mllvm -enable-hiipu-vf-loop-unroll \
  -mllvm -cce-aicore-dcpreload-args=false \
  --cce-disable-vf-stack-reserved-ubuf -std=c++20 \
  -I deep_gemm/include \
  -I /mnt/xds/sfs/z00956057/cann-9.2.0/aarch64-linux/asc/include/adv_api \
  kernel.asc -o kernel.rel.o
```

运行 smoke：

| 场景 | routed | shared | H/I | E/topk | tokens | 结果 |
| --- | --- | --- | --- | --- | --- | --- |
| 小 shape | 是 | 否 | 5120/512 | 4/2 | 64 | PASS, finite |
| 小 shape + shared | 是 | 是 | 5120/512 | 4/2 | 64 | PASS, finite |
| 中等 N | 是 | 否 | 5120/1024 | 4/2 | 64 | PASS, finite |
| 中等 I | 是 | 否 | 5120/2304 | 4/2 | 64 | PASS, finite |
| 中等 I + shared | 是 | 是 | 5120/2304 | 4/2 | 64 | PASS, finite |
| M=256 路径 | 是 | 否 | 5120/512 | 4/2 | 512 | PASS, finite |
| M=256 + shared | 是 | 是 | 5120/512 | 4/2 | 512 | PASS, finite |

`tests/test_mega_moe.py` 单进程 smoke：

```bash
python3 tests/test_mega_moe.py --num-processes 1 \
  --hidden 5120 --intermediate-hidden 512 \
  --num-experts 4 --num-topk 2 --num-tokens 64 \
  --num-shared-experts 1 --skip-baseline --num-correctness-tests 0
```

输出：

```text
[1/1] H/I=5120/ 512 | E=4 | topk=2 | shared=1 | tokens=64 |
38.21 us | speedup N/A | 79.0 TFLOPS/rank | HBM 699.8 GB/s/rank
```

### 11.6 当前限制与后续工作

- 按要求未接入 DeepEP baseline，因此尚未做端到端数值 diff；当前验证为“设备侧编译通过 + 多配置 fused kernel 运行通过 + 输出 finite”。
- 运行环境中另一进程占用了大部分 NPU HBM，原始 96 experts / 7168 hidden 等大配置暂时无法运行；小配置已经覆盖 routed/shared、MAD_M=128/256、非 2 次幂 intermediate 等情况。
- 后续可在 NPU 空闲时补：
  1. 原 96 experts / 5120 或 7168 配置的 smoke；
  2. 不依赖 DeepEP 的轻量 PyTorch 参考核对；
  3. L1 stage 数、`MAD_M/MAD_N/BLOCK_K` 的性能调优。

### 11.7 AIV0 精度问题根因与修复

AIV0 版本在 `valid_m >= 2` 时出现非确定性误差，而同 shape 的 AIC direct-B 版本稳定。定位过程：

- AIV0 的 `l2_acts` 数据与 SF 在 T=2 下与 golden 完全一致，说明 FP4 布局、host-NZ、`asc_copy_ub2l1` 和 Linear1 MMA 本身正确；
- 同一进程重复运行 T=1，L1 输出确定，但最终 `y` 每次不同，确认是同步问题而非数值问题；
- 去掉 AIV0 的 Linear1 B 生产后 `y` 恢复确定，锁定 Linear1/Linear2 B 的 intra-block 事件。

根因：AIV0 需要在同一核上交替生产 Linear1 B 和 Linear2 B，而两者的 intra-block flag 基址重叠：

- Linear1 B 4 个 stage：flag `0..3`
- Linear2 B 3 个 stage：flag `1..3`（与 Linear1 重叠）
- metadata：flag `2`（也与两者重叠）

AIV0 处理下一个 M-block 的 Linear1 B 时，可能满足 AIC 对当前 M-block Linear2 B 的等待，导致 AIC 读取尚未写完的 L2 B，表现为非确定性误差。

修复：给三类事件分配不相交的 flag 区间：

```cpp
constexpr IntraBlockEvent<0> kMetadataFull{8};
constexpr IntraBlockEvent<0> kL1BL1Full{0}, kL1BL1Empty{0};
constexpr IntraBlockEvent<0> kL1BL2Full{4}, kL1BL2Empty{4};
```

验证（AIV0 + W4A4 Linear1 + W4A8 Linear2）：

| E | topk | shared | tokens | golden diff |
| --- | --- | --- | --- | --- |
| 1 | 1 | 0 | 1 | 0 |
| 1 | 1 | 0 | 2 | 0 |
| 1 | 1 | 0 | 64 | 0 |
| 48 | 6 | 1 | 64 | 1.788e-7 |

T=1024（4 个 M-block）仍存在另一处非确定性误差，且在 AIC direct-B 版本中同样复现，与本次 AIV0 flag 修复无关。定位结果：

- T=512（2 个 M-block）确定且 golden PASS；
- T=1024 时 `l1_acts` / `l2_acts` 的 M-block 2–3 在多次运行间出现不同排列，最终 `y` 随之变化，部分运行会触发 aicore exception；
- `l2_acts_sf` 稳定，说明 FP4/FP8 布局与 AIV0 Linear1 权重生产已经正确；
- 该问题只在 `num_tokens * topk` 跨越 512 个 dispatch route 后出现；
- 可复现的最小配置：`H=512, I=2304, E=1, topk=1, T=1024`，此时只有 pool M-block 3 的 L1 输出错误，具体集中在 N-block 10/11（输出列 1280–1535）；
- 该 M-block 的 `kDispatched` 计数为 32、`kLinear1Ready` 计数为 18，说明完成信号都已到达；
- 用最终 `l1_acts` 反推 L1 输出时，wrong 列仍与 kernel 输出不符；逐 N-block 匹配显示 kernel 用的 B 数据对应正确的 N-block；
- 进一步交叉验证：A 数据取自 pool block 3、A SF 取自 pool block 3、B 数据取 N-block 10/11、B SF 取 N-block 10/11 时结果最优，但仍残余约 24% 的 FP8 code mismatch，说明不是单一 A/B/SF 索引错误，而更像是该次 L1 task 读到了**部分旧数据/部分新数据的混合**；
- `NOTALLOC_KEEP` 读缓存模式、AIC 侧 DSB、MTE3/ALL pipe drain、AIC 侧 delay、AIV0 任务级 drain 均未修复，部分尝试会触发 aicore exception；
- 当前最可能的方向是：pool M-block 3 的 L1 dispatch / B 生产完成后，最早进入该 M-block 的 AIC core（对应 N-block 10/11）仍可能看到未完全可见的数据；需要为该边界设计显式的跨核 release/acquire 或 consumer-side cache invalidate。

### 11.8 跨核 GM 握手尝试

尝试用 `route_m_blocks` 作为 GM 信号，对 Linear1 B 做 per-task release/acquire：

- AIV0 生产完一个 L1 task 的 B 后写 GM done；AIC 消费完该 task 后写 consumed，AIV0 才进入下一个 task；
- 全局清空 handshake 区并加 `clear_done`，避免跨 launch 旧值。

结果：

- `H=512, I=2304, E=1, topk=1, T=1024`：连续 3 次 golden PASS；
- `H=5120, I=2304, E=1, topk=1, T=64`：golden PASS；
- `H=5120, I=2304, E=1, topk=1, T=1024`：仍有 `1.5e-3` 残余误差；
- 进一步做 per-stage GM ring（4 个 stage 的 written/consumed 序列）会在多 stage 配置下死锁，已回退。

当前代码仍保持 AIV0 flag 修复版本；上述 GM 握手是有效的诊断/方向，但尚未作为最终实现保留。


### 11.9 后续尝试与结论

- 将 metadata flag 从 8 改为 7（避免可能的 3-bit flag 越界）：`H=5120/T=64` PASS，`H=5120/T=1024` 仍 aicore exception。
- 在 AIC direct-B 版本上应用同样的 disjoint flag（metadata 8、L2 B 4）：`H=5120/T=64` PASS，`H=5120/T=1024` 仍 aicore exception。
- 因此 T=1024 问题不是单纯的 intra-block flag 基址越界；direct-B 路径同样存在。
- per-stage GM ring 仍因 AIV0/AIC 的 stage sequence 在 task/`k_mad` 边界无法对齐而死锁，未保留。

当前代码为 AIV0 flag 修复版本，T=64 配置 golden PASS。


### 11.10 aicore exception 的根因

`H=5120/I=2304/E=1/T=1024` 在 NPU0 上的 aicore exception 已确认是设备硬件 ECC 故障，不是 AIV0 flag：

- NPU0 报错 core 2：`error code = 8, A multi-bit ECC error occurs when CUBE reads L0A`；
- 其他 core 随之等待超时；
- 同一份 kernel、同一配置：
  - NPU1：无 aicore exception（golden diff 1.23e-2，剩余精度问题）；
  - NPU2：无 aicore exception（golden diff 1.25e-2）；
  - NPU3：同样 aicore exception。

因此 NPU0/NPU3 存在 L0A 多 bit ECC 硬件故障；后续精度调试应使用 NPU1/NPU2。AIV0 flag 修复本身不再需要为 aicore exception 负责。


### 11.11 per-task GM 握手 + PIPE_ALL 的阶段性结果

在 NPU1/NPU2（无 ECC 故障）上继续优化，当前版本增加了：

- Linear1 B per-task GM done/consumed 握手；
- 每个 L1 B stage `asc_copy_ub2l1` 后 `asc_sync_pipe(PIPE_ALL)` + `asc_sync_data_barrier(DSB_DDR)`；
- AIC 在第一个 L1 task 前等待最后一个 M-block 的 `kDispatched`。

结果：

- `H=512/I=2304/T=1024`：golden PASS；
- `H=5120/T=64`、`E=48/topk=6/T=64/shared=1`、`T=2`：PASS；
- `H=5120/I=2304/T=1024`：误差从约 `1.2e-2` 降到 `1e-3 ~ 4e-3`，但仍不稳定通过。

说明 per-task 握手 + 全 pipe drain 能显著降低竞争，但 20 个 L1 B stage/task 的 within-task 覆盖窗口仍未完全解决。下一步可尝试减少 L2 B stage 数、给 L1 B 增加第 5/6 个 stage，或把 within-task 覆盖也改成 GM ring。


### 11.12 FP4 BLOCK_K/MAD_K 加倍 + A8W4 共享 B 布局（已修复 T=1024）

按建议将 Linear1 原生 FP4 的 `BLOCK_K` / `MAD_K` 加倍，使每个 FP4 B stage 的字节数与 W4A8 的 FP8 B stage 一致：

- `FP4MMAConfig<MAD_M>`：
  - `BLOCK_K = 2 * MMAConfig<MAD_M>::BLOCK_K`
  - `MAD_K = BLOCK_K / 2`
  - `MAD_N` 保持与 FP8 相同
- 每个 FP4 B stage 变成 64 KiB，和 Linear2 B 一致；
- 每个 Linear1 task 的 B stage 数从 20 降到 10，和 Linear2 对齐；
- L1 恢复 A8W4 布局：

```text
0            192K                     448K  480K  512K
| A 3x64K     | B 4x64K (L1/L2 共用)  | A SF | B SF |
```

- Linear1 和 Linear2 共用 `kL1BBaseBytes`、同一个 `l1b_stage_idx`、同一套 `kL1BFull/Empty`；
- FP4 B 的 64 KiB stage 由两个 32 KiB half 写入（UB 仍用 32 KiB buffer）；
- `kMetadataFull` 使用 flag 4，避免和 B 的 0–3 重叠。

验证（NPU1/NPU2，健康设备）：

| H | E/topk | shared | T | golden diff |
|---|---|---|---|---|
| 5120 | 1/1 | 0 | 1 | 0 |
| 5120 | 1/1 | 0 | 2 | 0 |
| 5120 | 1/1 | 0 | 64 | 0 |
| 5120 | 1/1 | 0 | 512 | 0 |
| 5120 | 1/1 | 0 | 1024 | 1.192e-7 |
| 512 | 1/1 | 0 | 1024 | 4.172e-7 |
| 5120 | 48/6 | 1 | 64 | 1.788e-7 |
| 5120 | 48/6 | 1 | 256 | 1.788e-7 |

当前版本已保存为 `/tmp/opencode/mega_moe_av0_fp4k2_shared.hpp`。
