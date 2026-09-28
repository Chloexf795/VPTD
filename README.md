# VPTD-EAE

VPTD-EAE 是一个独立的事件论元抽取（Event Argument Extraction, EAE）研究原型。
项目目标是：用 ACE 提供文本中的真实事件论元标签，用 SWiG 提供同事件类型的视觉原型，
再利用多段合成视频之间一致的时间变化信号，帮助模型减少角色混淆并提高 Precision、Recall
和 F1。

这是 `Chloexf795/VPTD` 自己的实现。核心包不内嵌 VAD/verl 源码，也不会修改或推送到
`York-Gold/EMNLP2026`。数据格式和结果适配器与 EMNLP 风格的 EAE baseline 兼容，
但运行本仓库的单元测试、数据准备和评测脚本不需要克隆 EMNLP 仓库。

## 当前阶段

已经完成：

- ACE 与 SWiG 的独立数据转换；
- 按事件类型匹配 SWiG 视觉原型，明确禁止把图片框当作 ACE 实例标签；
- 角色级 temporal attribution / distillation loss；
- 多视频一致性门控和 support/refutation 分支；
- Argument Identification、Argument Classification 的 P/R/F1 评测；
- 角色反转诊断和 dev 阈值校准；
- 单元测试和可运行的小型示例。
- ACE/SWiG 本地真实导出接入、跨 split 审计与训练原型匹配；
- Qwen3-VL Transformers / OpenAI-compatible API 推理、断点续跑和完整性评分；
- 固定 commit 的官方 Vision-OPD/verl 外部运行时及三种 OPD 启动配置。
- 在 RTX PRO 6000 96GB 上完成 Qwen3-VL-4B 的 ACE dev 全量基线：
  747 条，Arg-I F1 45.39%，Arg-C F1 32.96%（surface-string 协议）。
- SWiG dev 全量 3,263 条：Arg-I F1 47.66%，Arg-C F1 34.60%
  （ACE 映射的事件 grounding，IoU ≥ 0.5，失败输出保留完整分母）。
- 静态原型 OPD 的两步真实训练、完整 checkpoint 保存，以及恢复后的第 3 步更新。
- 静态 OPD 已续训至 50 步并完成两套全量 dev 对比：ACE Arg-I/Arg-C F1
  为 46.62%/33.27%，SWiG 为 47.64%/34.31%；已在本机独立复算。
  详细对比与实验限制见 [`docs/GPU_RUNTIME_20260925.md`](docs/GPU_RUNTIME_20260925.md)。

尚未完成：

- 把 Qwen3-VL 的输出接成 `student/static teacher/video teacher` 三组 role logits；
- 在真实 ACE/SWiG 数据上完成完整训练轮次；
- 完整训练后的 dev/test 对比和消融实验。

服务器、正式 Qwen 权重和 5,573 张 SWiG 图片已就绪；两套全量基线、静态 OPD 50 步训练、续训验收和训练后 dev 对比完成。
目前评测使用已有 dev split，正式 test split 尚未提供。
当前 static OPD 使用原版 token distillation，视频时间归因损失尚未接入该训练器。

最新数据审计发现旧 ACE 导出缺少 trigger，且 dev 有 67 组同输入异标签；
新运行入口已默认拦截这类数据。原始事件标注仍待补齐，已实现的输入审计和输出约束
见 [`docs/LOCAL_REPAIRS_20260926.md`](docs/LOCAL_REPAIRS_20260926.md)。

**开始跑实验：[`docs/RUN_EXPERIMENTS.md`](docs/RUN_EXPERIMENTS.md)。**

## 安装与检查

```bash
python -m pip install -e '.[inference]'
python -m unittest discover -s tests -p 'test_vptd*.py' -v
python scripts/smoke_vptd_eae.py
```

## 跑通演示数据

```bash
python scripts/prepare_vptd_eae_data.py \
  --ace examples/vptd_eae/ace_processed.demo.json \
  --swig examples/vptd_eae/swig_processed.demo.json \
  --mapping examples/vptd_eae/ace_sr_mapping.demo.txt \
  --split train \
  --output outputs/demo_train.jsonl \
  --stats outputs/demo_stats.json

python scripts/eval_vptd_eae.py \
  --result examples/vptd_eae/result.demo.json \
  --mode text \
  --output outputs/demo_metrics.json
```

详细设计、数据关系、损失输入和下一步实验顺序见
[`docs/VPTD_EAE.md`](docs/VPTD_EAE.md)。

## 代码边界

VAD 的时间反事实/归因思想属于相关工作，应在论文中引用。本仓库重新实现的是面向 EAE
角色分布的版本，没有复制 VAD 的训练运行时。`scripts/setup_framework.py` 会下载原版
Vision-OPD 到被 Git 忽略的 `external/Vision-OPD`，并核对 commit；许可证随上游保留。
EMNLP2026 可作为 baseline 和数据格式参考，
但不是本仓库的远程目标或运行依赖。更完整的边界说明见
[`NOTICE_VPTD_EAE.md`](NOTICE_VPTD_EAE.md)。
