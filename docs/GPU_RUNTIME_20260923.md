# GPU 运行记录（2026-09-23）

用户授权的租用服务器：RTX PRO 6000 Blackwell 96GB × 1，25 vCPU，
容器内存上限 120GiB，数据盘 150GB，Ubuntu 22.04，驱动 580.142，CUDA Toolkit 12.8。
SSH 凭据不保存到项目。服务器项目目录为 `/root/autodl-tmp/VPTD`。

## 环境

- `/root/autodl-tmp/vptd-envs/inference`：Python 3.12，继承镜像的 torch 2.8.0+cu128，
  Transformers 5.5.0、Accelerate 1.12.0、OpenAI SDK 2.54.0。
- `/root/autodl-tmp/vptd-envs/opd`：Python 3.12，torch 2.10.0+cu128，
  vLLM 0.18.0、Transformers 5.5.0、Ray 2.53.0；按锁定上游的 requirements 安装。
- CUDA 编译使用 `/usr/local/cuda-12.8`，临时目录与 pip 缓存均放在数据盘。
- FlashAttention 2.8.3.post1 为 SM120 编译：`FLASH_ATTN_CUDA_ARCHS=120 MAX_JOBS=8 NVCC_THREADS=2`。

两个 venv 使用 `--system-site-packages` 复用镜像中的 CUDA 库，升级包安装在各自 venv。
上游冻结清单的依赖声明存在冲突，记录在服务器 `logs/opd_pip_check.txt`；
没有把 `pip check` 宣称为通过。实际模型推理和训练要分别用运行日志验收。

## 数据与权重

- ACE train 5,828、dev 747；排除 train 中与 dev 重复的两个事件样本。
- SWiG train 2,310、dev 3,263；5,573 张官方图片均已验证。
- 静态原型 OPD 可用 ACE train 5,091 条。全部原型来自 SWiG train。
- 模型：`models/Qwen3-VL-4B-Instruct`，15 个文件经官方 SHA-256 验证。
- 数据、图片、权重的清单分别在 `data/processed/manifest.json`、
  `data/raw/swig/download_manifest.json`、`models/Qwen3-VL-4B-Instruct/vptd_snapshot.json`。
- 完整正式 test split 尚未提供，当前结果仅为 dev。

SWiG ZIP 的远程索引读取在服务器较慢；本机用 `--index-only` 准备索引后上传，
服务器通过 HTTP Range 只下载约 611MB 所需图片。索引有确定的文件偏移、大小和 CRC，
图片落盘前验证 CRC 和 PIL 解码，续传还会校验已有文件。

## 本次推理服务

```bash
cd /root/autodl-tmp/VPTD
export PATH=/root/autodl-tmp/vptd-envs/opd/bin:/usr/local/cuda-12.8/bin:$PATH
export HF_HOME=/root/autodl-tmp/vptd-cache/huggingface
export OMP_NUM_THREADS=8
MODEL_PATH=/root/autodl-tmp/VPTD/models/Qwen3-VL-4B-Instruct \
MAX_MODEL_LEN=8192 GPU_MEMORY_UTILIZATION=0.7 \
bash scripts/serve_qwen.sh --enforce-eager --max-num-seqs 32 \
  --limit-mm-per-prompt '{"image":1,"video":0}'
```

服务仅监听 127.0.0.1。用 inference 环境运行评测脚本，两个数据集均使用
`--backend api --base-url http://127.0.0.1:8000/v1 --model qwen-vptd --workers 16`，
temperature=0、seed=42、max_tokens=1024。实际预测与运行配置保存在
`outputs/qwen_baseline/{ace_dev,swig_dev}/`；模型快照清单与结果一同保存。

所有 token 预算耗尽的响应保留原文并计为无效，gold 不从分母移除。
传输失败会中止，由 `--resume` 续跑。基线服务与 OPD 训练顺序执行，避免相互争用显存。

## 已完成的验收

- GPU BF16 矩阵运算通过；上游训练器导入和实际 Parquet 的 Hydra 配置组合通过。
- Qwen 官方权重在 vLLM 中成功加载，文本和图片输入均实际生成了结果。
- ACE dev 747/747：Arg-I F1 45.385694%，Arg-C F1 32.955771%，无效输出 0。
- SWiG dev 3,263/3,263：Arg-I F1 47.656515%，Arg-C F1 34.595416%；
  无效输出 103（其中截断 72），完整分母计分。本机独立重新评分与服务器一致。
- FlashAttention 在 SM120 上的 BF16 前向与反向通过，梯度全部有限。
- 本地 23 项单元/接口测试通过；HTTP 测试与这里的真实模型评测分开记录。

OPD 两步训练结果待运行结束后补记。本记录不把仅成功启动配置
或导入训练器等同于完成训练，也不把静态图片蒸馏等同于完整视频 VPTD 方法。
