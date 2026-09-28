# 本地修复与下一次实验准备

2026-09-26。未连接 GPU、未启动训练、未改写已有数据和分数。

## 已完成

1. `audit_inputs` 按实际 prompt 与图片路径检查同输入异标签，忽略隐藏的样本 ID，
   gold 比较不受排列顺序或重复论元影响。另检查 ACE 缺失 trigger、重复出现却没有定位信息的 trigger。
2. 数据准备、Parquet 导出、推理入口默认阻止有歧义的输入；训练入口要求 ACE Parquet
   带有审计记录。旧 Parquet 不会静默绕过检查。
3. `--allow-ambiguous-ace` 仅用于明确复现旧协议，记录在 manifest/run/训练策略历史中。
   该参数不会解除 SWiG 标签冲突检查。`--score-only` 仍可复算历史实验。
4. 原始 processed ACE 导入会保留文档、句子、事件 ID、原始 trigger 与论元位置等
   `source_annotations`，并把 trigger 和已有 start/end 信息写入学生提示词。
   新协议名为 `ace-processed-surface-eae-given-trigger-v2`。论元标签不进入提示词，
   论元偏移仅作为来源记录保留；本轮未实现标准 span-level ACE 官方评测。
5. API 推理增加可选 `--output-policy schema-v1`：角色枚举来自提示词；ACE 实体枚举
   来自输入候选实体；SWiG 要求四个 0–1000 的坐标。不从 gold 推断实体或框数量。
6. 增加显式 `--repetition-penalty`，默认 1.0，保持旧解码。策略和参数写入运行配置；
   任意改变都会阻止续接同目录旧预测。原始响应完整保留，不做事后删框或修补答案。

结构化输出和 repetition_penalty 接口按照固定版本的
[vLLM 0.18 结构化输出文档](https://docs.vllm.ai/en/v0.18.0/features/structured_outputs/)与
[Chat API 参数文档](https://docs.vllm.ai/en/v0.18.0/serving/openai_compatible_server/)接入。
这两项当前只支持 API 后端，Transformers 后端会明确拒绝该配置。
结构约束不能证明论元语义正确，也不能保证消除重复；坐标大小关系仍由现有评分器检查。
不添加按 dev gold 数量设置的输出上限，不自动提高重复惩罚。

## 验证

- 40 项单元/边界测试通过，包括真实 OpenAI Python 客户端发往本地 HTTP 测试响应的请求检查。
  HTTP 测试不是 GPU 推理，未宣称 vLLM grammar 编译或新设置下模型质量已验收。
- 全部 747 条 ACE、3,263 条 SWiG 的 schema 均能在本机构建；替换 gold 后 schema 不变。
- 带 trigger 的示例完成准备、输入检查、Parquet 往返读取与 API 推理 dry-run。
- 真实旧 ACE 导出默认被准备入口拒绝，退出码 1；独立审计发现歧义返回 2。
- 当前处理后 ACE train：390 组冲突、933 条涉及样本、5,828 条缺少 trigger。
  ACE dev：67 组冲突、152 条涉及样本、747 条缺少 trigger。
  SWiG dev：0 组同输入异标签。

证据在 `outputs/local_repairs_20260926/`：`tests.log`、`schema_preflight.json`、
`source_inventory.json`、各 split 的 audit、`rejected_legacy/audit.failed.json`、示例 Parquet。

```bash
python3 -m unittest discover -s tests -p 'test_vptd*.py' -v
python3 scripts/audit_benchmark_data.py \
  --data data/processed/ace_dev.jsonl \
  --output outputs/ace_input_audit.json
```

## 尚缺的原始数据

本地项目范围内检查了 141 个候选 JSON/JSONL/XML 文件的首条记录，只有示例文件
保留原始 event_mentions；真实 ACE 为丢失 trigger 的 SFT/normalized 导出。
上游 `EMNLP2026/trainer/utils/convert.py` 的 ACEConverter 读取 trigger，
但 `trainer/utils/construct.py` 的 ACEDatasetConstructor.construct_prompt 没有插入 trigger。
上游脚本指向师兄环境中的 `/ceph100t/.../dataset/ACE/ACE_processed_train.json`。
本次未修改该上游仓库，也没有用模型或 gold 猜测缺失 trigger。

请提供同一数据版本与划分的 `ACE_processed_train.json`、`ACE_processed_dev.json`，
如果有，也提供 test。需要包含：

- sentence、doc_id/sent_id；
- entity_mentions（独立于特定事件 gold 的候选实体）；
- event_mentions 下的 event_type、事件 ID、trigger.text；
- trigger 与 argument 的 start/end，以及偏移单位（字符或 token、end 是否排他）；
- 若位置基于 token，提供对应 tokens，避免把不同分词的坐标当字符偏移。

导入器保留原始位置，不猜测或转换未知偏移单位。收到真实数据后仍要核对 schema、
位置约定、跨 split 重复和同输入异标签，生成新的 processed 目录与 manifest。
不能把新 protocol 的分数直接替换到此前旧协议表格中。

## 何时开 GPU

先恢复真实 ACE 输入并通过审计。之后开卡做少量 train 样本上的 vLLM schema 编译、
推理与重复行为检查；如端点不支持约束，直接报错，不静默退回普通生成。
用训练侧样本确定解码设置，固定后对原始 Qwen 与 static50 同时跑新协议评测。
再决定是否扩大训练和接入视频教师。

新设置的运行模板（示例惩罚值只是待验证配置，不是根据 dev 挑出的最优值）：

```bash
python scripts/run_qwen_benchmark.py \
  --data data/processed_v2/ace_dev.jsonl \
  --output-dir outputs/qwen_v2_schema/ace_dev \
  --backend api --base-url http://127.0.0.1:8000/v1 --model qwen-vptd \
  --workers 16 --output-policy schema-v1 --repetition-penalty 1.05
```

这个命令尚未执行，`data/processed_v2/ace_dev.jsonl` 需要先由原始标注重新生成。
