# 新磁盘实例续训（服务器日期 2026-09-25）

用户提供了迁移后的新实例。项目路径、数据、模型、Python 环境均保留；
GPU 仍为 RTX PRO 6000 Blackwell 96GB，驱动 595.58.03，RAM 配额 120GiB。
数据盘扩为 200GB。登录凭据不写入项目。

## 已验证与磁盘处理

- 再次加载第 2、3 步 checkpoint，确认模型、优化器、调度器、数据游标完整。
- 第 3 步参数相对第 2 步确实变化；见 `GPU_RUNTIME_20260924.md`。
- 日志、运行配置、小型 checkpoint 元数据及验证报告已备份到本机。
- 删除被第 3 步替代的第 2 步 model/optim 两份大文件，保留元数据；
  可用空间从约 79GB 增至约 130GB。第 3 步完整 checkpoint 保留。

## 50 步短实验

从第 3 步恢复到总共 50 步；batch 2，rollout 2，lr 2e-6，10 步预热，
训练样本与原型池均来自 train。固定终点，不按 dev 成绩挑步数。
只在第 25、50 步保存，新保存 checkpoint 保留 1 份；保存过程中允许新旧两份暂存。
本次会额外保留之前的第 3 步迁移恢复点。

全部 5,091 条静态训练 student prompt 经模型 tokenizer 审计，最长 1,268 tokens，
没有超过设置的 2,048 tokens。教师预算 4,096，响应预算 256。

首次尝试在第 4 步的 AdamW 更新中 CUDA OOM，没有推进 checkpoint。
堆栈位于 PyTorch 的 Adam 批量更新，不是数据损坏或磁盘耗尽。
保留日志 `logs/opd_static50_attempt1.log` 和退出码 1。

随后使用 `--low-memory-adam`：这是官方 PyTorch AdamW 的薄适配层，强制
`foreach=False, fused=False`，减少全参数规模的临时张量；恢复 checkpoint 后再次
应用这两个 kernel 标记，其余优化器状态原样保留。依据
[PyTorch 2.10 AdamW 文档](https://docs.pytorch.org/docs/2.10/generated/torch.optim.AdamW.html)。
原始 Vision-OPD checkout 不修改，算法、学习率、数据与蒸馏损失均保留。
测试覆盖旧 foreach checkpoint 恢复后状态连续、下一步结果与官方逐参数实现一致。
本地 27 项测试及服务器 PyTorch 2.10 下对应测试通过。

续训允许增加总步数、调整保存策略，或切换到该等价 AdamW kernel；其他训练配置
仍须一致。每次尝试完整命令写入 `resume_history.jsonl`。

```bash
python scripts/train_opd.py \
  --mode ace-static-opd --data data/opd/ace_static.parquet \
  --output-dir checkpoints/qwen3vl_ace_static_smoke_v3 \
  --model models/Qwen3-VL-4B-Instruct --gpus 1 --cpus 16 \
  --batch-size 2 --rollouts 2 --steps 50 --save-freq 25 --max-checkpoints 1 \
  --max-prompt-tokens 2048 --max-response-tokens 256 --max-teacher-tokens 4096 \
  --eager-rollout --keep-actor-on-gpu --rollout-memory-utilization 0.35 \
  --low-memory-adam --resume
```

环境变量沿用上一份记录。服务器实际脚本为 `logs/run_static50.sh`，
日志为 `logs/opd_static50.log`，退出码为 `logs/opd_static50.exit`。
训练后将最终 checkpoint 转为 Hugging Face 权重，使用与基线一致的设置分别评估
ACE dev 747 条、SWiG dev 3,263 条。本实验是 static OPD 的短跑对比，不能替代完整
训练轮次、正式 test split 或完整视频 VPTD 的实验。

全量评测与本机独立复算已完成，结果见下文。

## 再次迁移（服务器日期 2026-09-26）

新实例数据盘 350GB，恢复前可用约 209GB。原训练日志已运行到第 49 步，
第 50 步保存时中断：model 文件存在，但 optim 仅约 1.9GB，缺少其余状态，
checkpoint tracker 仍为 25。因此没有将残缺目录作为可恢复检查点。

第 25 步 model、optim、extra_state、data.pt 完整；实际读取优化器确认所有
step 为 25，调度器 last_epoch 为 25，foreach=False。残缺的第 50 步目录改名
为 `global_step_50.incomplete_20260926`，旧日志备份为
`logs/opd_static50_interrupted_20260925.log`，从第 25 步原生恢复到 50。

恢复后 26–50 共 25 次更新完成，训练退出码 0，tracker 更新为 50。
最终 loss=0.008499710820615292，grad_norm=3.6805810928344727。
实际读取最终 checkpoint：714 个模型张量，61 个优化器状态条目，所有 optimizer
step=50，scheduler.last_epoch=50，data.pt 可读取。检查首层 q_proj 参数有限，
相对初始权重最大绝对变化 6.174854934215546e-05。

模型文件 19,307,451,691 bytes，优化器 35,502,581,759 bytes；不再是残缺保存。
训练结束仍出现与前次相同的 DataLoader 子进程清理警告，但主进程退出 0，
最终模型、优化器、调度器与数据状态均独立读取通过。
验收报告和训练日志已下载到 `outputs/gpu_20260925/`；实际服务器源码备份位于
其 `source_used/`，含 SHA-256 清单，区别于较早的本地源码快照。

官方 FSDP merger 已成功导出 `models/Qwen3-VL-4B-VPTD-static50`，随后启动
同配置全量开发集评测。

## 完整开发集对比（已验收）

训练与评测退出码均为 0。ACE 747 条、SWiG 3,263 条完整预测已回传，
本机按原始 gold 独立重算与服务器 metrics 完全一致。数据与图片 SHA-256、
输入和输出预算、temperature=0、seed=42、workers=16 等评测设置与基线一致，
仅模型替换为训练后的权重。没有依据 dev 分数挑选 checkpoint。

| 开发集 | 指标 | 基线 F1 (%) | static OPD 50 步 F1 (%) | 变化（百分点） |
|---|---|---:|---:|---:|
| ACE | Arg-I | 45.3857 | 46.6186 | +1.2329 |
| ACE | Arg-C | 32.9558 | 33.2729 | +0.3171 |
| SWiG | Arg-I | 47.6565 | 47.6414 | -0.0151 |
| SWiG | Arg-C | 34.5954 | 34.3092 | -0.2863 |

ACE 无效与截断输出仍均为 0；SWiG 无效输出 110（基线 103），其中截断
63（基线 72），全部保留完整 gold 分母。ACE Arg-C 召回由 55.85% 增至
58.68%，精确率由 23.37% 降至 23.22%；SWiG Arg-C 召回由 39.89% 增至
40.30%，精确率由 30.54% 降至 29.87%。目前表现主要是更多预测和更高召回，
没有证据说明角色混淆已被解决，也没有进行多随机种子显著性验证。

ACE 使用给定事件与实体的 surface-string EAE 协议；SWiG 使用 ACE 映射
事件的 grounding 协议，IoU ≥ 0.5。不可把这些值当作标准 ACE span-level
或原始 SWiG 504-verb 的官方测试成绩。

本地结果：

- `outputs/gpu_20260925/comparison.json`：完整 P/R/F1、分母、无效输出和校验结果。
- `outputs/gpu_20260925/qwen_static50/{ace_dev,swig_dev}/`：逐条预测、metrics、run 配置。
- `outputs/gpu_20260925/opd_static50_checkpoint_verification.json`：检查点实际读取验收。
- `outputs/gpu_20260925/training_metrics.json`：本次恢复后的 26–50 步更新记录。
- `outputs/gpu_20260925/logs/opd_static50_model.sha256`：导出权重校验值。

服务器保留完整 checkpoint 与导出权重；大权重未下载本机。
评测结束后已停止本次推理服务，验收时 GPU 显存占用 0MiB、利用率 0%。
租用实例未关机，仍可能计费。50 步只是短实验，完整训练轮次、视频教师接入、
正式 test 与消融仍待后续完成。
