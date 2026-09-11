# Jetson AGX Orin 上的大模型本地部署：从 vLLM INT4 到 llama.cpp Q4_K_M 的实测与思考

> 这不是一篇“哪个框架一定更好”的结论帖，而是一份来自 Jetson AGX Orin 本地部署过程中的实验记录。  
> 我更关心的问题是：在边缘设备上，模型参数量、量化方式、推理框架、内存占用、生成速度和任务准确率之间，到底应该怎么权衡？

## 1. 背景

最近我在 Jetson AGX Orin 上做本地大模型部署和模型选型，主要面向自然语言理解、意图识别和槽位填充等任务。

一开始使用的是 vLLM + INT4，主要测试 Qwen2.5-7B-Instruct / Qwen3.5-27B；后来为了进一步降低边缘设备上的内存占用，又尝试了 llama.cpp + GGUF Q4_K_M，并测试了从 2B 到 27B 的多个 Qwen 模型。

这个过程中一个非常明显的现象是：

> “都是 4 bit 量化”，并不意味着它们会有相同的内存占用，也不意味着参数量越大，任务效果就一定越好。

尤其是在 Jetson 这种 CPU/GPU 共享物理内存的设备上，仅仅看“INT4”“Q4_K_M”几个字，很容易对真实部署成本产生误判。

## 2. 当前实测结果

当前测试任务使用 ATIS 数据集，主要关注单轮意图识别准确率、槽位填充准确率、TTFT、生成速度和峰值内存占用。

| 模型 | 数据集 | 平均生成速度 (tok/s) | Mean TTFT (ms) | 峰值内存 (GB) | 量化策略 | 意图识别准确率 | 槽填充准确率 |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: |
| Qwen3.5-2B | ATIS | 50.088 | 97.563 | 14.660 | Q4_K_M | 77.60% | 0.473035 |
| Qwen3.5-4B | ATIS | 25.625 | 171.770 | 14.731 | Q4_K_M | 78.50% | 0.629538 |
| Qwen3.5-9B | ATIS | 18.020 | 215.094 | 15.432 | Q4_K_M | 77.60% | 0.663377 |
| Qwen2.5-14B-Instruct | ATIS | 12.708 | 143.596 | 19.143 | Q4_K_M | **84.55%** | 0.594290 |
| Qwen3.5-27B | ATIS | 6.294 | 573.258 | 26.394 | Q4_K_M | 78.95% | **0.757490** |

此外，之前使用 vLLM 部署 Qwen2.5-7B-Instruct INT4 时，整机观测到的内存占用曾达到约 **51 GB**。这个数字也是我后来转向 llama.cpp / GGUF 做对照实验的重要原因之一。

## 3. 第一个体会：不要把“INT4”当成完整的部署描述

刚开始做量化部署时，很容易形成一个直觉：FP16 是 16 bit，INT4 是 4 bit，所以 INT4 模型应该只有 FP16 的四分之一大小。

这个估算只能用来理解“权重存储量级”，不能直接用来预测运行时内存。

“INT4”更多描述的是权重精度或某种量化方案的基本位宽。在实际部署里，还必须继续问：用的是什么量化算法？推理框架是什么？KV Cache 用什么精度？Context Length 多大？并发是多少？推理框架会预留多少内存？是否启用了 CUDA Graph、Prefix Cache、Paged Attention 等机制？

如果这些条件不同，仅比较“INT4”三个字，其实没有太大意义。

## 4. Q4_K_M 也不是简单的“所有权重统一 INT4”

Q4_K_M 是 llama.cpp / GGUF 生态中非常常见的一种 K-Quant 量化方式。

它和“统一把所有 tensor 强制压成同一种 4 bit 表示”并不是完全相同的概念。llama.cpp 的量化体系允许不同 tensor 使用不同量化类型，也支持对部分 tensor 使用不同精度。Q4_K_M 的目标更接近于：

> 在模型体积、推理速度和精度损失之间找到一个比较实用的平衡点。

这也是为什么我在边缘端测试中优先选择 Q4_K_M，而不是单纯追求最低 bit 数。

需要强调的是：Q4_K_M 的优势不能只归结为“它是 4 bit”。真正影响运行内存的，还有 GGUF 的模型组织方式以及 llama.cpp 自身的加载和内存管理策略。

## 5. 为什么 vLLM INT4 的 7B 模型会看到约 51 GB，而 llama.cpp Q4_K_M 低很多？

这是这次实验里最值得思考的问题。

首先，一个 7B 模型即使完全按照理想 4 bit 权重粗略估算：

```text
7B × 4 bit ≈ 3.5 GB
```

再考虑量化 scale、metadata 和部分高精度 tensor，权重文件会更大一些。但无论如何：

> **“7B INT4 权重本身”并不能解释 51 GB 的运行内存。**

因此，51 GB 更应该理解为：

> 当时那套 vLLM + CUDA + Jetson 统一内存环境下，整个模型服务的运行时 memory footprint。

而不是：

> Qwen2.5-7B 的 INT4 权重有 51 GB。

这是两个完全不同的概念。

## 6. vLLM 更像一个“高吞吐服务引擎”

vLLM 的设计目标并不只是“用最少内存跑起来一个模型”。它更强调高吞吐、连续批处理、多并发、KV Cache 管理、Paged Attention、Prefix Caching 和服务化推理。

vLLM 中的 `gpu_memory_utilization` 是一个非常关键的参数。它代表模型执行器可以使用的 GPU memory 比例，并会影响模型权重之外可供 KV Cache 等组件使用的空间。

也就是说，当我们看到 vLLM 启动后占用了非常大的内存时，里面可能同时包含：

```text
Model Weights
+ KV Cache
+ CUDA / Runtime Buffers
+ CUDA Graph / Workspace
+ Allocator Reserved Memory
+ Framework Runtime
+ 其他缓存和中间 buffer
```

所以：

> “vLLM 占了 51 GB”并不等于“模型需要 51 GB”。

甚至值得检查一个很有意思的数字：

```text
64 GB × 0.8 = 51.2 GB
```

这和我曾经观察到的约 51 GB 非常接近。

这并不能直接证明当时就是 `gpu_memory_utilization=0.8` 导致的，因为还需要结合实际启动参数和运行日志才能确认；但它至少提醒我：

> 在 vLLM 中，首先应该检查框架的内存预算和预留策略，而不是看到一个大数字就认为是模型权重太大。

## 7. llama.cpp 的思路明显不同

llama.cpp 更偏向“尽可能简单、高效地在本地设备上执行量化模型”。GGUF 模型可以直接由 llama.cpp 加载，当前 llama.cpp 默认支持 memory mapping（mmap）方式加载模型。

从操作系统角度看，mmap 的一个特点是：映射到虚拟地址空间中的文件，并不意味着所有模型页面都已经同时成为常驻物理内存。

因此：

```text
文件大小
≠ 虚拟地址空间大小
≠ RSS
≠ CUDA buffer
≠ Jetson 上 tegrastats 看到的总 RAM
```

这些数字本来就不是同一个概念。

这也是为什么比较 vLLM 和 llama.cpp 时，我现在不会再简单写：

> “vLLM 51 GB，llama.cpp 15 GB，所以 Q4_K_M 节省了 70% 显存。”

这种说法太粗糙。

更准确的说法应该是：

> **在相同 Jetson 设备、相近任务条件下，我观测到 llama.cpp + GGUF Q4_K_M 的整机峰值内存明显低于此前的 vLLM + INT4 服务。差异不仅来自量化格式，也来自两个框架完全不同的模型加载、缓存、内存预留和服务化设计。**

## 8. Jetson 上的“显存”最好换一个说法

在桌面 NVIDIA GPU 上，我们习惯区分 System RAM 和 VRAM。

但是 Jetson AGX Orin 是集成 GPU 架构，CPU 和 GPU 使用的是同一套 LPDDR5 物理内存。因此，在 Jetson 上写实验报告时，我现在更倾向于使用：

> **Peak Unified Memory / 峰值统一内存**

而不是简单写“显存占用”。

## 9. 为什么 2B、4B、9B 的内存看起来差别并不大？

这组数据也很有意思：

| 模型 | 峰值内存 |
| --- | ---: |
| 2B | 14.660 GB |
| 4B | 14.731 GB |
| 9B | 15.432 GB |
| 14B | 19.143 GB |
| 27B | 26.394 GB |

直觉上看，2B 和 9B 参数量差了四倍多，为什么总内存只差不到 1 GB？

一个很重要的原因是：我现在记录的是“整机/服务峰值内存”，而不是“纯模型权重占用”。里面还包含操作系统基础占用、Benchmark 进程、llama.cpp Runtime、CUDA Buffer、KV Cache、Page Cache、模型权重常驻页面以及其他后台服务。

当模型较小时，固定基础开销占整个数字的比例非常高。

因此，后续我准备把内存测量方式进一步改成：

```text
Idle Memory
Model Loaded Memory
Inference Peak Memory
```

然后额外计算：

```text
Model Load Delta = Model Loaded - Idle
Inference Delta  = Inference Peak - Idle
```

我认为这种统计方式会比只记录一个“Peak Memory”更有可比性。

## 10. 第二个体会：模型参数量和任务准确率不是单调关系

如果只看参数量，我们可能会猜：

```text
2B < 4B < 9B < 14B < 27B
```

模型越大，ATIS 意图识别应该越准。

实测却不是这样：

| 模型 | Intent Accuracy |
| --- | ---: |
| Qwen3.5-2B | 77.60% |
| Qwen3.5-4B | 78.50% |
| Qwen3.5-9B | 77.60% |
| Qwen2.5-14B | **84.55%** |
| Qwen3.5-27B | 78.95% |

14B 反而取得了当前最高的意图识别准确率。

这说明在一个具体下游任务上，参数量不是唯一变量。模型代际、Tokenizer、训练数据、Instruction Following、Prompt 形式、输出约束以及量化后的行为，都可能影响最终分类结果。

因此边缘端模型选型不能变成：

> “设备能塞进去最大的模型就是最好的模型。”

真正应该关注的是：

> **单位资源带来的任务收益。**

## 11. 槽位填充又呈现出了另一条曲线

| 模型 | Slot Accuracy |
| --- | ---: |
| Qwen3.5-2B | 0.473035 |
| Qwen3.5-4B | 0.629538 |
| Qwen3.5-9B | 0.663377 |
| Qwen2.5-14B | 0.594290 |
| Qwen3.5-27B | **0.757490** |

这里 27B 的优势明显得多。

这其实符合一个比较直观的经验：简单的意图分类和细粒度的信息抽取，不一定依赖同样的模型能力。

Intent Classification 最终可能只需要输出 `flight`，Slot Filling 却需要正确理解整句话，并完成细粒度语义边界和实体类型映射。

所以后续选型时，我不会再使用一个单一的“Accuracy”代表所有 NLU 能力。

## 12. 第三个体会：边缘端真正要找的是 Pareto Point

如果项目存在“生成速度至少 20 tok/s”这样的硬约束，那么当前 Q4_K_M 测试里：

| 模型 | tok/s | Intent | Slot |
| --- | ---: | ---: | ---: |
| Qwen3.5-2B | **50.09** | 77.60% | 0.473 |
| Qwen3.5-4B | **25.63** | **78.50%** | **0.630** |
| Qwen3.5-9B | 18.02 | 77.60% | 0.663 |
| Qwen2.5-14B | 12.71 | **84.55%** | 0.594 |
| Qwen3.5-27B | 6.29 | 78.95% | **0.757** |

那么结论开始变得很有意思。

2B 很快，但是槽位填充能力明显下降；9B 相比 4B 更慢，Intent Accuracy 没有提升，Slot Accuracy 只提升了一部分；14B 的 Intent Accuracy 当前最好，但生成速度低于 20 tok/s；27B 的 Slot Filling 最好，但速度只有约 6.3 tok/s，TTFT 和内存开销也明显上升。

所以在当前“未针对任务微调”的阶段，如果必须同时考虑速度 >20 tok/s、尽可能好的 NLU 能力和较低边缘端资源占用，那么：

> **Qwen3.5-4B 是当前非常值得继续关注的 Pareto Candidate。**

这不是说 4B 一定是最终模型，而是说在当前约束下，继续把 4B 做领域微调，可能比直接把模型换成 14B/27B 更符合边缘部署的系统目标。

## 13. ATIS 的 tok/s 不能直接代表自然对话速度

ATIS 意图识别的输出可能只有 `flight` 或 `ground_service`，通常只有几个 token。

因此一次请求的时间主要花在 Prompt Prefill、TTFT，以及输出 1～几个 Token。这个时候直接统计 tok/s 会受到 TTFT、输出长度和计时误差非常大的影响。

自然对话则完全不同。例如我在前端测试过一条较长回答：

```text
TTFT: 0.77 s
Total Latency: 12.65 s
Output: 262 tokens
End-to-End Throughput: 20.7 tok/s
```

如果去掉 TTFT，只计算 Decode 阶段，则纯生成速度还会更高一些。

所以现在我更倾向于把 Benchmark 拆成两类：ATIS 主要测试 Intent Accuracy、Slot Accuracy、TTFT 和 End-to-End Classification Latency；另外准备固定的长文本 Prompt，固定 `max_tokens=256`、`temperature=0`，专门测试 Decode Throughput、End-to-End Throughput、TTFT 和 P50/P95 Latency。

这样才能真正回答：

> 这个模型能不能稳定达到 20 tok/s？

而不是用一个只输出 2 个 token 的分类任务来判断长文本生成性能。

## 14. TTFT 也不能只看模型参数量

当前结果中还有一个值得继续调查的现象：

| 模型 | TTFT |
| --- | ---: |
| Qwen3.5-2B | 97.6 ms |
| Qwen3.5-4B | 171.8 ms |
| Qwen3.5-9B | 215.1 ms |
| Qwen2.5-14B | **143.6 ms** |
| Qwen3.5-27B | 573.3 ms |

14B 的 TTFT 低于 9B。

这再次说明：TTFT 不是参数量的简单线性函数。它还受到 Prompt Length、Prompt Cache、模型架构、CUDA Kernel、Flash Attention、Tokenizer、llama.cpp 版本、Power Mode、GPU Frequency、CPU Scheduling 和 Memory Bandwidth 等因素影响。

因此这类结果不能直接解释成“14B 比 9B 响应更快”。更严谨的方法应该是在完全相同条件下多次重复，记录 Mean、P50、P95，并严格控制 Context Size、GPU Layers、Flash Attention、Parallel Slots、Prompt、Power Mode、Jetson Clocks 和 llama.cpp 版本。

## 15. 我现在对边缘端模型选型的理解

做完这一轮实验后，我越来越觉得，边缘端大模型部署不应该从：

> “我能不能把一个 27B 模型塞进设备？”

开始。

而应该从：

> “我的任务真正需要什么能力？”

开始。

一个更合理的路径是：

```text
任务定义
↓
确定准确率 / 速度 / 内存硬指标
↓
从小模型开始
↓
逐步增加参数规模
↓
找到第一个满足任务要求的模型
↓
做 LoRA / QLoRA 或领域数据微调
↓
再次评估
↓
最后才考虑是否需要更大的模型
```

如果 4B 经过领域微调后就能满足任务准确率，那么部署 14B/27B 带来的额外参数，很可能只是在消耗内存、功耗、TTFT、Decode Speed 和热设计余量，而没有带来足够的业务收益。

这也是我现在对“边缘智能”的一个基本理解：

> **边缘端追求的不是最大的模型，而是在受限资源下最合适的模型。**

## 16. 下一步准备继续做什么

| 方向 | 目标 |
| --- | --- |
| 统一长文本 Generation Benchmark | 更严格验证 >20 tok/s |
| Idle / Loaded / Peak 三阶段内存采样 | 分离模型和系统基础占用 |
| LoRA / QLoRA | 验证小模型领域微调后的上限 |
| ATIS 多轮扩展 | 测试 1 / 3 / 5 / 7 轮上下文 |
| 多轮 TTFT | 观察 KV Cache 复用和长上下文带来的影响 |
| vLLM INT4 vs llama.cpp Q4_K_M 控制变量实验 | 同一个模型比较不同框架 |
| 功耗模式测试 | 比较不同功耗模式下吞吐、温度和功耗 |
| ASR + LLM + TTS | 测试完整端侧语音交互链路 |

后面如果有新的数据，我会继续更新这个仓库。

## 17. 当前阶段的几个结论

**第一，4 bit 不是一个足够完整的部署描述。** 量化算法、推理框架、KV Cache、Context Length、并发和内存预留策略都必须一起考虑。

**第二，vLLM 的大内存占用不能简单归因于模型权重。** vLLM 是面向高吞吐服务设计的推理引擎；权重之外的 KV Cache、runtime buffer 和内存预算都可能占据大量空间。

**第三，llama.cpp + GGUF Q4_K_M 在单路边缘推理场景中非常有吸引力。** 至少在我的 Jetson AGX Orin 实验里，它显著降低了整体内存压力，也让更多模型能够在同一设备上进行横向测试。

**第四，参数量和下游任务效果不是单调关系。** 14B 当前 Intent Accuracy 最好，27B Slot Filling 最好，而 9B 并没有在所有指标上超过 4B。

**第五，边缘设备应该寻找 Pareto Optimal，而不是最大模型。** 速度、TTFT、内存、准确率和功耗之间，没有单一最优解。

## References

1. vLLM CacheConfig documentation: https://docs.vllm.ai/en/stable/api/vllm/config/cache/
2. llama.cpp quantization documentation: https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md
3. llama.cpp server / model loading documentation: https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md
4. NVIDIA Jetson AGX Orin product specifications: https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/
5. NVIDIA Jetson AGX Orin technical brief: https://www.nvidia.com/content/dam/en-zz/Solutions/gtcf21/jetson-orin/nvidia-jetson-agx-orin-technical-brief.pdf

## Disclaimer

以上数据来自我当前 Jetson AGX Orin 环境中的实际测试，主要用于模型选型和工程分析。

不同 JetPack、CUDA、llama.cpp/vLLM 版本、功耗模式、Context Size、Prompt、量化模型来源和后台服务都可能影响最终结果，因此这些数字不应被理解为对应模型的通用官方性能。

更有价值的是测试方法和趋势，而不是某一个孤立的 token/s 数字。
