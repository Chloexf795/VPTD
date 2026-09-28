# 运行 ACE / SWiG / Qwen 实验

本次实现位于 `VPTD-owned`。所有命令都在该目录执行。
默认模型是 `Qwen/Qwen3-VL-4B-Instruct`；同一命令的 `--model` 可改为本地模型目录
或合并后的 checkpoint。外部框架锁在 `configs/framework.lock.json`。

## 已验证与待运行

本机已完成数据转换、Parquet 往返校验、官方 Hydra 配置组合、单元测试与
Qwen API 请求/续跑/评分的 HTTP 边界测试。HTTP 测试使用测试响应，不是模型分数。
Qwen 官方 processor 已实际处理全部 747 条 ACE dev（738–1,129 tokens，均未超限），
并通过合成小图的图像张量处理测试；记录在 `outputs/qwen_processor_check.json`。
2026-09-23 已部署到用户提供的单卡 RTX PRO 6000 Blackwell 96GB 服务器。
Qwen3-VL-4B-Instruct 的 15 个文件通过 ModelScope 官方 SHA-256 校验，
5,573 张 SWiG 图片通过 ZIP CRC 和图像解码校验。ACE dev 全部 747 条已真实完成：
Arg-I F1 45.3857%，Arg-C F1 32.9558%，无无效输出。
SWiG dev 全部 3,263 条已完成：Arg-I F1 47.6565%，Arg-C F1 34.5954%，
无效输出 103 条（包含 72 条截断），均保留在完整分母。
结果位于 `outputs/gpu_20260923/qwen_baseline/{ace_dev,swig_dev}/`；
本机重新评分与服务器结果完全一致，校验记录为 `outputs/gpu_20260923/verification.json`。
静态 OPD 的两步 CUDA 训练、checkpoint 保存及恢复后的第 3 步更新已验收。
50 步短训练与训练后两套全量 dev 对比也已完成并独立复算：ACE Arg-I/Arg-C F1
为 46.62%/33.27%，SWiG 为 47.64%/34.31%。记录在 `GPU_RUNTIME_20260925.md`，
完整结果位于 `outputs/gpu_20260925/comparison.json`。这仍是短实验，未完成整轮训练。

2026-09-26 错误分析发现：ACE dev 有 67 组相同输入却标签不同，涉及 152 条样本，
上游导出已存在该问题且缺少实例 trigger。扩大训练前应优先回溯原始事件标注；
证据、角色误差及后续优先级见 [`ERROR_ANALYSIS_STATIC50_20260926.md`](ERROR_ANALYSIS_STATIC50_20260926.md)。

输入审计、默认阻止歧义数据、保留 trigger 的 v2 导入及可选输出约束已实现。
旧 ACE 示例命令在新代码下会被预检阻止；只有明确复现旧协议才添加
`--allow-ambiguous-ace`，该参数不会修复数据。新一轮实验应先恢复原始事件标注。
本地验证与所需原始文件见 [`LOCAL_REPAIRS_20260926.md`](LOCAL_REPAIRS_20260926.md)。

| 数据 | 本地输入 | 准备后 | 备注 |
|---|---:|---:|---|
| ACE train | 5,830 | 5,828 | 排除与 dev 重复的 1 条句子对应的 2 条事件样本 |
| ACE dev | 747 | 747 | 保持原 split |
| SWiG train | 2,310 | 2,310 | 服务器图片已齐 |
| SWiG dev | 3,263 | 3,263 | 服务器图片已齐 |

ACE train 有 5,091 条可匹配 SWiG train 原型，dev 为 666 条。
静态蒸馏训练只导出有原型的 train 样本，报告覆盖率；文本基线评测保留全部 dev 样本。

## 1. 推理与数据准备环境

```bash
python -m pip install -e '.[inference]'
python -m unittest discover -s tests -p 'test_vptd*.py' -v
python scripts/setup_framework.py
```

从当前工作区已有的 Alpaca 导出准备数据：

```bash
python scripts/prepare_benchmarks.py \
  --ace-train ../EMNLP2026/dataset/sft_ace_train.json \
  --ace-dev ../EMNLP2026/dataset/sft_ace_dev.json \
  --swig-train ../EMNLP2026/dataset/sft_swig_train.json \
  --swig-dev ../EMNLP2026/dataset/sft_swig_dev.json \
  --drop-train-overlap --allow-missing-images
```

`--allow-missing-images` 仅允许生成数据审计和文本基线输入。图像推理、图像 OPD
导出仍然拒绝缺图。数据、输入校验和、排除记录与统计在 `data/processed/manifest.json`。
`audit.failed.json` 是历史失败审计，成功产物以 `manifest.json` 为准。
服务器上要重新运行准备命令，使用该服务器的图片路径，不能直接沿用 Mac 绝对路径。

SWiG 图片来源是 [allenai/swig 官方仓库](https://github.com/allenai/swig#1-swig-dataset)
列出的 `images_512.zip`。可以按需下载已有标注引用的图片，避免下载整个 13.6GB 包：

```bash
python -m pip install -e '.[data]'
python scripts/fetch_swig_images.py \
  --data data/source/sft_swig_train.json data/source/sft_swig_dev.json \
  --index data/raw/swig/archive_index.json --workers 16
```

然后重跑准备命令，去掉 `--allow-missing-images`，加上
`--image-root /absolute/path/to/images_512`（指向直接包含 JPG 的目录）。
SWiG 原始 `verb/width/height/bb` JSON 也可作为 `--swig-*` 输入，但须提供
`--mapping /path/to/ace_sr_mapping.txt`。ACE processed `event_mentions` JSON 也受支持。
ACE 原始 APF/XML 不直接支持；使用已有合法预处理数据，原始语料来源见
[LDC2006T06](https://catalog.ldc.upenn.edu/LDC2006T06)。

正式 test split 通过 `--ace-test` / `--swig-test` 显式加入，不从 train 自动切分。

## 2. Qwen 基线跑分

Hugging Face 不可达时，可以使用 Qwen 官方 ModelScope 仓库。下载脚本逐文件验证
官方大小与 SHA-256，并保存 `vptd_snapshot.json`，记录实际文件 revision 和校验和：

```bash
python -m pip install modelscope==1.40.1
python scripts/fetch_qwen_model.py --output-dir models/Qwen3-VL-4B-Instruct
```

后续 `--model` 可指向该本地目录。API 服务启动时设置 `MODEL_PATH` 为它的绝对路径，
并把快照清单与评测结果一起保留。

可先做输入检查，不下载模型：

```bash
python scripts/run_qwen_benchmark.py --data data/processed/ace_dev.jsonl \
  --output-dir outputs/qwen_baseline/ace_dev --dry-run
```

GPU 服务器直接加载 Hugging Face 模型：

```bash
python scripts/run_qwen_benchmark.py --data data/processed/ace_dev.jsonl \
  --output-dir outputs/qwen_baseline/ace_dev \
  --model Qwen/Qwen3-VL-4B-Instruct --backend transformers --device auto
```

也可以在 GPU 服务器用 vLLM 部署，并让本机调用该服务：

```bash
# 在已安装兼容 vLLM 的 GPU 环境启动
bash scripts/serve_qwen.sh

# 与服务同机时用 localhost；远程时替换为你的可访问服务地址。
python scripts/run_qwen_benchmark.py --data data/processed/ace_dev.jsonl \
  --output-dir outputs/qwen_api/ace_dev --backend api \
  --base-url http://127.0.0.1:8000/v1 --model qwen-vptd
```

默认服务监听 localhost；远程访问可通过 SSH 端口转发。服务密钥从 `QWEN_API_KEY`
读取。脚本不会自动调用第三方模型服务。SWiG 评测把 `--data` 改为
`data/processed/swig_dev.jsonl`，换一个 `--output-dir`。

`--limit 8` 做少量样本诊断，**不生成整集分数**。之后同一命令加 `--resume`
并去掉 `--limit`，会继续剩余样本。修改模型、数据或生成预算必须使用新目录。
运行完整后生成 `predictions.jsonl`、`run.json`、`metrics.json`。
HTTP 错误会中止并保留已完成预测；无效 JSON 与 token 预算耗尽的输出保留原文，
计入 `invalid_outputs`（后者另计 `truncated_outputs`），对应 gold 仍进入完整分母。
即使被截断的文本恰好是合法 JSON，也不当作有效预测。API 可加 `--workers 16` 并发请求，
写入仍在单线程完成，续跑要求生成参数一致。
发布实验时用 `--revision` 固定 HF commit，或固定本地模型快照；API 侧需自行固定服务权重。

## 3. 官方 Vision-OPD 训练环境

在 **Linux + CUDA** 单独建 Python 3.12 环境；不要在本机推理环境安装这一整套栈。
固定源码来自 [VisionOPD/Vision-OPD](https://github.com/VisionOPD/Vision-OPD/tree/c8a8fdd1f88eef1b5ef4fe6a8d64eb0272917471)。
遵循其官方冻结环境安装方式：

```bash
conda create -n vptd-opd python=3.12 -y
conda activate vptd-opd
python scripts/setup_framework.py
python -m pip install --upgrade pip
python -m pip install --no-deps -r external/Vision-OPD/requirements.txt
python -m pip install -e external/Vision-OPD --no-deps
python -m pip install flash-attn==2.8.3.post1 --no-build-isolation
# Qwen3.5 路径还需要 causal-conv1d；当前 Qwen3-VL 不使用它。
# python -m pip install causal-conv1d==1.6.1 --no-build-isolation
python -m pip install -e . --no-deps
```

核心版本为 torch 2.10.0、transformers 5.5.0、vLLM 0.18.0、Ray 2.53.0。
需要服务器驱动匹配该 CUDA 栈。上游使用多 GPU，本地入口的 `--gpus 1` 是可改配置，
不能直接沿用上游多卡参数。96GB GPU / 120GiB RAM 已通过的单卡配置见
`GPU_RUNTIME_20260924.md`：batch 2、rollout 2、冻结教师 BF16、学生参数驻留 GPU、
rollout 显存比例 0.35。其他大小/配置需要单独验收。

先导出 ACE 文本 OPSD 数据并核对训练配置：

```bash
python scripts/export_opd_data.py --data data/processed/ace_train.jsonl \
  --mode ace-opsd --output data/opd/ace_opsd.parquet
python scripts/train_opd.py --mode ace-opsd --data data/opd/ace_opsd.parquet \
  --output-dir checkpoints/qwen3vl_ace_opsd --check-config
python scripts/train_opd.py --mode ace-opsd --data data/opd/ace_opsd.parquet \
  --output-dir checkpoints/qwen3vl_ace_opsd_smoke --gpus 1 --steps 2
```

两步 CUDA smoke 通过后，用新目录去掉 `--steps 2` 正式训练。默认只记录 console /
TensorBoard，不启用 W&B。checkpoint 按 `--save-freq` 保存。
4B 全参数 checkpoint（含优化器）约 51GiB；长跑建议显式设置
`--save-freq 25 --max-checkpoints 1`，并为保存期间的新旧两份预留空间。
默认保留 1 份新 checkpoint。恢复时可增加总步数、调整 checkpoint 存储策略，
或使用 `--low-memory-adam` 切换到保留原状态的逐参数 AdamW；其他训练设置必须一致；
原始 launch 和每次恢复命令均保留。
teacher 从初始模型初始化，EMA update rate 为 0，保持固定教师；student 用自己的 rollout。

补齐图片后运行静态原型教师实验：

```bash
python scripts/export_opd_data.py --data data/processed/ace_train.jsonl \
  --mode ace-static-opd --output data/opd/ace_static.parquet
python scripts/train_opd.py --mode ace-static-opd --data data/opd/ace_static.parquet \
  --output-dir checkpoints/qwen3vl_ace_static --gpus 1
```

SWiG 视觉任务可用 `--mode swig-opsd`、`data/processed/swig_train.jsonl` 单独导出和训练。
三种模式均复用官方 `verl.trainer.main_ppo`；不修改上游损失函数。

合并 checkpoint 后用同一基线命令跑 dev/test：

```bash
PYTHONPATH=external/Vision-OPD python -m verl.model_merger merge --backend fsdp \
  --local_dir checkpoints/qwen3vl_ace_opsd/global_step_10/actor \
  --target_dir checkpoints/qwen3vl_ace_opsd_step10_hf
```

这里 step_10 是示例，替换为实际选中的 checkpoint；评测 `--model` 指向合并目录。

## 评分口径与后续研究边界

- ACE：given event / entities 的 surface-string Arg-I、Arg-C micro P/R/F1。大小写忽略，
  同一样本重复项去重。原导出没有 trigger/document/span offsets，不能宣称标准 ACE span F1。
  固定 schema 修复了源 prompt 中 Transfer-Money 角色列表为空，以及 Meet 的
  Participant/Entity 不一致；预测角色别名沿用原项目适配器。不会从当前样本答案构造角色列表。
  原 gold 保留不改，少数超出 schema 的源标注问题记录在 manifest 中。
- SWiG：ACE 映射的 8 类事件、given event 的 bbox grounding，IoU ≥ 0.5，最大一对一匹配，
  输出 Arg-I、Arg-C。不是原始 504-verb SWiG 的 verb/value/grounded-value 官方指标。
- SWiG train 的 objects 与 dev 的 role/bboxes 被归一为同一个视觉任务，重新构造提示词，
  不把已有 CI 或候选框混进模型输入。视觉 Transfer-Money 的 Instrument 保留其视觉语义。
- 结果的协议、分割和完整性一起记录。用 dev 选 checkpoint，正式 test 保留到最终报告。
- static OPD 是文本 student + 同类型图片 teacher 的对照；原型不能证明 ACE 句子的事实。
  完整 VPTD 视频教师、三路 role logits 和时间增量 loss 仍需后续接入，不能把 static 分数
  写成完整 VPTD 的分数。

模型调用方式依据 [Qwen 官方模型卡](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)。
