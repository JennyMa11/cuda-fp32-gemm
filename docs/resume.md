# 马建麟｜个人简历素材库

- 电话：17373604033
- 邮箱：majinalin26@stu.pku.edu.cn
- 求职意向：AI Infra / 大模型推理优化实习
- 到岗安排：5 天/周，随时到岗，3 个月及以上

## 教育背景

### 北京大学｜软件工程（硕士）

2026.09—2029.06（预计）

### 北京交通大学｜计算机科学与技术（本科）

2022.09—2026.06

- 专业成绩：GPA 3.47/4，均分 85.1/100

## 核心项目

### nano-vLLM：轻量级大模型推理引擎优化

2026.03—2026.07

**技术栈：** Python、PyTorch、Triton、CUDA Graph、Qwen3

- **算子优化：** 基于 Triton 实现 GQA Paged Attention 与变长 Prefill，并按 Shape 自动选择 Triton/SDPA；BF16 典型 Shape（16Q/8KV Heads，Dim=128）下，Paged Attention 相对 v1 在 Batch=8/32 加速 1.28～1.30×/1.20～1.87×；8×32/128-token Prefill 相对 SDPA 吞吐提升 4.20×/1.62×，TTFT 从 200.0/169.5 ms 降至 47.7/104.4 ms。
- **调度与缓存：** 设计 Decode-first 统一 Token Budget 调度，支持 Chunked Prefill、Prefix Cache、增量 KV Block 分配、Round-robin 与 Preemption/Recompute；RTX 3060 + Qwen3-0.6B 混合请求并发 2 达 155.9 output tok/s，TTFT P50/P90 为 50.5/73.4 ms；12 请求公平性与抢占测试后无 KV Block 泄漏。
- **INT8 KV Cache：** 实现 Per-token/Per-KV-head 对称量化，在 Triton KV Store/Decode Paged Attention 中融合量化/反量化；单 Block 存储降至 BF16 的 50.78%，KV Blocks 由 92 增至 179（1.95×）；Batch=8、Context=511 下，融合 Decode Kernel 局部相对 FP16 加速 1.22×，Attention 输出与 BF16 Reference 的余弦相似度达 0.999958、最大绝对误差 0.002747。

### LLM Decode CUDA Kernel 实现与性能优化

2026.07—2026.09

**技术栈：** CUDA C++、PyTorch CUDA Extension、Warp Shuffle、Vectorized I/O、Kernel Fusion

- **版本化算子：** 基于 PyTorch CUDA Extension 从零实现 GEMV、RMSNorm、Fused Add + RMSNorm、SwiGLU 与 Softmax，保留显式 Version API 以支持逐版本对照和回归定位。
- **计算与访存优化：** 将 GEMV 由 Thread-per-row 重构为 Warp-per-row/Multi-warp Block，结合 Warp Shuffle 和 Pack-4；采用 Block/Warp Reduction 将 RMSNorm Shared Memory 从 1024 B 降至 32 B，并融合 Residual Add + RMSNorm，减少中间张量流量与一次 Kernel Launch。
- **评测与调度：** 基于 CUDA Event 统一 Cold-cache/Repeat Median 口径，并按 Shape、Dtype 与 Alignment 动态调度；RTX 3060 FP16 下，GEMV 相对 `torch.mv` 达 1.01～1.29×，RMSNorm 相对 PyTorch 实现达 1.2～1.5×，融合算子相对 PyTorch Eager 达 1.2～2.0×；建立 Nsight Compute/cuobjdump 工作流，正确性覆盖三种精度、边界 Shape、非对齐输入与 Current Stream。

## 个人技能

- **编程语言：** 熟悉 Python、C++ 与 CUDA C++，具备 PyTorch CUDA Extension 和 Triton 算子开发经验。
- **大模型推理：** 熟悉 PyTorch、Triton 与 nano-vLLM，理解 vLLM 的 Continuous Batching、Paged Attention、Chunked Prefill、CUDA Graph 等核心机制。
- **CUDA 优化：** 熟悉 Warp Shuffle、向量化访存、算子融合、混合精度及 CUDA Event 性能评测。
- **开发工具与语言：** 熟悉 Linux、Git；CET-6，能够熟练阅读英文论文与技术文档。

## 获奖情况

- 2025 全国大学生数学竞赛全国二等奖
- 2025 北京市大学生数学竞赛二等奖

## 补充项目素材

### CUDA FP32 GEMM 高性能算子

**技术栈：** CUDA C++17、cuBLAS、CUDA Runtime、PTX `cp.async`、Shared Memory、Register Blocking

- 从零实现支持 Row-major `C = alpha * A * B + beta * C` 的 CUDA FP32 GEMM，提供统一调用接口，并构建基于 `cublasSgemm` 的正确性验证和性能对照框架。
- 设计 CTA Tile、Warp Tile、Thread Tile 三级计算映射，采用 Shared Memory Tiling、每线程 `8x8` Register Blocking、外积 FMA 累加及 K 维循环展开，提高数据复用率和指令吞吐。
- 面向 Ampere 架构使用 16 B `cp.async` 异步搬运和两级 Shared Memory 双缓冲，实现下一 K-Tile 预取与当前计算重叠；通过 `float4` 向量化加载与写回优化全局内存访问。
- 实现三组 Fast Kernel（`128x128x8`、`128x64x8`、`64x128x8`）及通用 Edge Kernel，根据矩阵长宽比和对齐条件执行 Shape-aware Dispatch，同时覆盖规则尺寸与非对齐边界尺寸。
- 使用 CUDA Event 完成预热、多轮计时、GFLOPS 和相对 cuBLAS 吞吐统计。在 RTX 3060 Laptop GPU、CUDA 12.8、Pedantic FP32 模式下，38 组测试全部通过；逐 Shape 吞吐比算术平均达到 cuBLAS 的 **72.29%**，对齐 Fast 路径最高达到 **91.6%**。

#### 测试口径

- 编译架构：`sm_86`
- 预热次数：5
- 测量次数：20
- cuBLAS 模式：`CUBLAS_PEDANTIC_MATH`，不使用 TF32 Tensor Core
- 平均值定义：38 组矩阵各自相对 cuBLAS 吞吐比例的算术平均
- 正确性用例：`alpha = 1.25`、`beta = -0.5`
