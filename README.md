# VPTD-EAE

VPTD-EAE 用 ACE 提供文本中的真实事件论元标签，用 SWiG 提供同事件类型的视觉原型，
再利用多段合成视频之间一致的时间变化信号，帮助模型减少角色混淆并提高 Precision、Recall
和 F1。

## 安装与检查

```bash
python -m pip install -e .
python -m unittest discover -s tests -p 'test_vptd*.py' -v
python scripts/smoke_vptd_eae.py
```
