# VPTD-EAE：从研究想法到可运行原型

## 一句话解释

普通 EAE 模型只看一句话，容易把 `Attacker/Target`、`Giver/Recipient`、
`Origin/Destination` 这类方向相反的角色弄混。VPTD-EAE 让训练阶段额外查看与事件类型
匹配的视觉原型和多段合成视频；只有多个视频都支持同一个变化方向时，才把这部分时间证据
蒸馏给最终的文本学生模型。

## 数据关系

ACE 和 SWiG 没有一一对应关系。因此本项目不会声称“这张 SWiG 图片就是这句 ACE 文本的
现场图片”，也不会把 SWiG 的框直接贴到 ACE 人物上。

```text
ACE sentence ────────────────> 真实 span/entity + role 标签
       │
       └── event type 匹配 ──> SWiG visual prototypes
                                      │
                                      └── 生成多个 video hypotheses
```

`scripts/prepare_vptd_eae_data.py` 在本仓库内完成 ACE/SWiG 转换和事件类型匹配：

```bash
python scripts/prepare_vptd_eae_data.py \
  --ace /path/to/ACE_processed_train.json \
  --swig /path/to/SWiG_processed_train.json \
  --mapping /path/to/ace_sr_mapping.txt \
  --image-root /path/to/SWiG/images_512 \
  --split train \
  --output outputs/vptd_eae_train.jsonl \
  --stats outputs/vptd_eae_train.stats.json
```

每条输出都会保存以下声明，防止后续代码误用数据：

```json
{
  "alignment": {
    "kind": "event_type_prototype",
    "instance_aligned": false,
    "label_source": "ACE",
    "visual_source": "SWiG"
  }
}
```

## 核心损失吃什么数据？

对 ACE 句子里的每个候选实体，模型需要输出“它属于每个事件角色的分数”：

```text
student_logits         [batch, roles]       最终文本学生
static_teacher_logits  [batch, roles]       文本 + 静态视觉原型
video_teacher_logits   [K, batch, roles]    文本 + K 个视频假设
role_mask              [batch, roles]       当前事件允许哪些角色
gold_role              [batch]              ACE 真实标签
```

角色集合包含当前事件允许的角色和 `NONE`。例如 Attack 可以允许
`Attacker/Target/Instrument/NONE`，其余角色会被 mask 掉。

```python
from vptd_eae.temporal_attribution import compute_temporal_distillation_loss

loss, diagnostics, target = compute_temporal_distillation_loss(
    student_logits,
    static_teacher_logits,
    video_teacher_logits,
    labels=gold_role,
    role_mask=role_mask,
)
```

总损失包含三部分：

```text
ACE gold role 的交叉熵
+ 时间证据重构目标与学生分布的 JS 散度
+ 时间证据不可靠时，学生对 static teacher 的弱锚定
```

多视频一致性门控的作用是：如果生成视频彼此矛盾，就不让这批视觉信号强行改变学生。
support/refutation 分支则同时描述“哪个角色获得支持”和“哪个错误角色应被压低”。

## 评测

```bash
python scripts/eval_vptd_eae.py \
  --result outputs/vptd_eae.json \
  --baseline-result outputs/text_baseline.json \
  --mode text \
  --output outputs/vptd_eae_metrics.json
```

报告包含 Argument Identification 和 Argument Classification 的 Precision、Recall、F1，
以及方向角色的反转纠正率。如果 role scorer 输出
`best_non_NONE_logit - NONE_logit`，只在 ACE dev 上选择一次阈值，然后固定用于 test：

```bash
python scripts/calibrate_vptd_eae.py \
  --gold outputs/dev_gold.jsonl \
  --scored-predictions outputs/dev_scored.jsonl \
  --output-predictions outputs/dev_filtered.jsonl \
  --output-calibration outputs/dev_calibration.json
```

## 目录说明

```text
vptd_eae/
├── converters.py            ACE/SWiG 独立转换器
├── event_schema.py          本地 ACE 事件角色定义
├── data.py                  SWiG 事件类型视觉原型匹配
├── role_support.py          角色词表和 event-specific mask
├── temporal_attribution.py  时间证据目标与训练损失
├── result_parsing.py        模型 JSON 输出解析
├── emnlp_adapter.py         EMNLP 风格结果兼容层
├── metrics.py               P/R/F1 与角色反转指标
└── calibration.py           dev 阈值选择
```

## 下一步实验顺序

1. 在自己的环境中用 Qwen3-VL 复现 ACE text-only baseline；
2. 写 role scorer，把生成文本输出改成稳定的候选实体 × 角色 logits；
3. 加 static SWiG prototype teacher；
4. 生成 K 个视频假设并得到 video teacher logits；
5. 先检查 `F1(video teacher) > F1(static teacher)`，确认视频真的提供信息；
6. 接入 `compute_temporal_distillation_loss` 训练学生；
7. 比较 text baseline、static image、直接 KD、无一致性门控和完整 VPTD-EAE；
8. 跑“打乱 SWiG event type”负对照，排除只是多看图片就涨分的解释。

真实 P/R/F1 必须来自 ACE 官方 split。演示数据和 smoke test 只能证明代码能运行，不能作为
论文实验结果。
