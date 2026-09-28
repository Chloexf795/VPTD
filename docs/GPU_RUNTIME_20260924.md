# 克隆实例运行记录（2026-09-24）

克隆实例保留了代码、数据、模型和两个 Python 环境。GPU 为 RTX PRO 6000
Blackwell Server Edition 96GB，驱动 595.58.03，容器内存上限 120GiB，
数据盘 150GB。项目仍位于 `/root/autodl-tmp/VPTD`。本文不保存登录凭据。

## 克隆验收

- torch 2.10.0+cu128、FlashAttention 2.8.3.post1 可导入。
- FlashAttention BF16 前向、反向在新 GPU 上通过，梯度均有限。
- 上游 HEAD 仍为 `c8a8fdd1f88eef1b5ef4fe6a8d64eb0272917471`，无跟踪文件修改。
- 既有 ACE/SWiG dev 基线不重复运行，见 `GPU_RUNTIME_20260923.md`。

## 单卡训练问题

`qwen3vl_ace_static_smoke_v2` 已实际进入生成与 actor update 阶段，随后失败。
服务器 `logs/opd_static_smoke_v2.raylet.log` 记录：工作进程
`WorkerDict.actor_rollout_ref_update_actor` 占用 103.32GiB 主机内存；
Ray 观察到总内存 114.45 / 120GiB，超过 95% 阈值并开始终止工作进程。
后续内存达到约 116.6GiB。没有成功的训练步指标或 checkpoint。
不能把 RPC Socket closed 的外层异常简单解释为网络故障。

针对该证据，下一次运行使用：冻结教师 BF16、学生 FP32 主参数驻留 GPU、
优化器状态继续 CPU offload、rollout 显存比例 0.35，以及 `MALLOC_ARENA_MAX=2`。
保留 Ray 的内存保护。教师加载为模型原始 BF16 精度，不改训练目标或 gold 标签。

同时通过 `engine_kwargs.vllm.max_model_len` 限制实际引擎长度为 2304，
避免上游服务器从 HF 配置重新覆盖成 262144。短跑设置 eager 和最多 8 个并发序列。

## 运行命令

```bash
cd /root/autodl-tmp/VPTD
export PATH=/root/autodl-tmp/vptd-envs/opd/bin:/usr/local/cuda-12.8/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.8
export HF_HOME=/root/autodl-tmp/vptd-cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false MALLOC_ARENA_MAX=2
python scripts/train_opd.py \
  --mode ace-static-opd --data data/opd/ace_static.parquet \
  --output-dir checkpoints/qwen3vl_ace_static_smoke_v3 \
  --model models/Qwen3-VL-4B-Instruct --gpus 1 --cpus 16 \
  --batch-size 2 --rollouts 2 --steps 2 --save-freq 2 \
  --max-prompt-tokens 2048 --max-response-tokens 256 --max-teacher-tokens 4096 \
  --eager-rollout --keep-actor-on-gpu --rollout-memory-utilization 0.35
```

## 续训约束

`--resume` 保持模型、数据校验和、框架和训练参数一致，允许增加 `--steps`，
并调整 checkpoint 保存频率及保留数量。
原始 `launch.json` 保留，每次恢复的命令与 checkpoint 步数写入
`resume_history.jsonl`。已达到目标步数时直接退出，防止上游多执行一步。
延长步数后从保存的优化器和调度器状态恢复，不宣称与一开始就采用更长预算的
训练在所有调度策略下严格等价。

## 已完成的训练与恢复验收

| Step | OPD loss | 梯度范数 | 学习率 |
| --- | --- | --- | --- |
| 1 | 0.01376663 | 6.15617 | 0（预热起点） |
| 2 | 0.01078034 | 8.06271 | 2e-7 |
| 3（恢复后） | 0.00050446 | 0.67068 | 4e-7 |

三步教师覆盖率、图像替换率均为 1，policy fallback 为 0。不同步对应不同样本，
这些 loss 不能直接当作全量性能改善。
第 2、3 步均保存了完整模型、优化器、调度器、RNG 和 dataloader 状态；
实际恢复日志从 step 2 开始并完成 step 3。
新实例再次加载验证：714 个模型张量、61 个优化器状态，优化器步数分别为 2、3。
抽查第 0 层 q_proj 权重，相对初始模型最大变化分别为 2.086e-7、6.035e-7，
第 3 步相对第 2 步仍有 4.023e-7 的变化。已验证非零参数更新。

训练返回后 Ray 清理 DataLoader 子进程时出现 Killed 提示；初次训练主命令退出码为 0，
checkpoint tracker、data.pt 和各权重文件均完整，后续实际恢复通过。
上游 perf 内存指标含宿主机统计/分配器逻辑统计，不作为本容器的物理内存峰值报告。

本地证据位于 `outputs/gpu_20260924/training_verification.json` 和同目录日志。
25 项本地测试通过。2026-09-25 迁移后清理了已被替代的第 2 步两份大型权重，
保留其元数据和验证报告，第 3 步完整 checkpoint 保留。
