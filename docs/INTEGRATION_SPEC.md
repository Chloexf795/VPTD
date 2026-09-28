# ACE / SWiG / Qwen / Vision-OPD 接入

目标：复用官方 Vision-OPD 的 verl 运行时，提供真实数据准备、Qwen 基线推理、
完整性审计和 OPD 启动入口。先建立基线，再运行 ACE answer-hint OPSD 和
ACE 文本 student / SWiG 原型 teacher 的 static OPD。后者不是视频时间归因实验。

数据 → 推理与评分 → 框架导出与训练，按此顺序实现。默认模型
`Qwen/Qwen3-VL-4B-Instruct`，模型目录和 GPU 数量均可覆盖。

## 数据契约

- 支持本地 Alpaca ACE/SWiG 导出及已有 processed ACE/SWiG 格式。
- train/dev/test 来自独立输入；只使用 SWiG train 作 ACE 的视觉原型库。
- prompt 不包含目标答案。SWiG 图像任务重新构建统一 prompt，避免已有候选框/CI 带来的协议差异。
- 明确标注现有 ACE 导出无 trigger offsets、document IDs，不能宣称标准 ACE span benchmark。
- 输入 SHA256、数量、缺图和跨 split 重复内容均进入 manifest；缺图可以做数据审计，但训练/图像推理必须拒绝。

## 代码与验证

`vptd_eae/` 放适配器，`scripts/` 放 CLI，`tests/test_vptd_integration.py` 放 unittest。
沿用 Python 类型标注、纯函数与延迟加载重依赖，例如
`def parse_answer(text: str) -> list[dict]: ...`。

验证命令：`python -m unittest discover -s tests -p 'test_vptd*.py' -v`。
测试覆盖 gold 隔离、确定性匹配、split 泄漏、异常输出、断点续跑身份、评分完整性、
框匹配和 Parquet 契约。真实 CUDA 训练是独立验收项，必须留下实际日志才能标为通过。

## 边界

- 已授权：新增依赖声明、数据转换、官方源码下载和训练/评测脚本。
- 保留工作目录内已有暂存修改；不把它们混入新增提交。
- ACE、权重和实验数据仅部署到用户授权的 GPU 服务器，不提交或公开发布；密钥不落盘。不伪造资源或跑分。
- 外部源码单独存放并锁定 commit，许可证随原仓库保留。
- 2026-09-23：用户已提供单卡 RTX PRO 6000 96GB 主机；官方 SWiG 图片与 Qwen 权重已补齐。正式 test split 仍未提供。
