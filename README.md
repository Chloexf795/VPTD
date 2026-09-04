# VPTD-EAE

VPTD-EAE 是一个独立的事件论元抽取（Event Argument Extraction, EAE）研究原型。
项目目标是：用 ACE 提供文本中的真实事件论元标签，用 SWiG 提供同事件类型的视觉原型，
再利用多段合成视频之间一致的时间变化信号，帮助模型减少角色混淆并提高 Precision、Recall
和 F1。

这是 `Chloexf795/VPTD` 自己的实现，不包含 VAD/verl 源码，也不会修改或推送到
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

尚未完成：

- 把 Qwen3-VL 的输出接成 `student/static teacher/video teacher` 三组 role logits；
- 在真实 ACE/SWiG 数据上训练；
- 报告真实 dev/test P、R、F1 和消融实验。

所以当前成果是“算法、数据接口和评测链路可运行的研究原型”，不是已经得到正式实验分数的
完整论文系统。

## 安装与检查

```bash
python -m pip install -e .
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
角色分布的版本，没有复制 VAD 的训练运行时。EMNLP2026 可作为 baseline 和数据格式参考，
但不是本仓库的远程目标或运行依赖。更完整的边界说明见
[`NOTICE_VPTD_EAE.md`](NOTICE_VPTD_EAE.md)。
