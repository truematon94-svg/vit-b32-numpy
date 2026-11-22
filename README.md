# vit-b32-numpy

OpenCLIP **ViT-B/32** 图像塔（vision tower）的纯 NumPy 实现。不依赖 PyTorch、
`open_clip` 或任何深度学习框架，全部前向计算由手写数组运算完成。

输入 `[B,3,224,224]` 的预处理图像，输出 `[B,512]` 的图像 token，每行 L2 长度为 1。

## 前向链路

```
[B,3,224,224]
   ├─ patchify ───────────────► [B,49,3072]   7×7 个 32×32 图块
   ├─ 投影 + 拼 CLS + 位置编码 ─► [B,50,768]
   ├─ 12 × TransformerBlock ──► [B,50,768]
   └─ 取 CLS + post-Norm + proj ► [B,512]
```

## 目录

| 文件 | 说明 |
| --- | --- |
| `image_encoder.py` | 核心实现：`ImageEncoder`、`TransformerBlock`、`MultiHeadAttention` |
| `helpers.py` | LayerNorm、GELU、softmax、单位化 |
| `preprocess_image.py` | 图像预处理（resize + center crop + 归一化） |
| `runtime.py` | 权重与图像读取 |
| `run.py` | 命令行入口：图像 → token |
| `config.json` | 模型结构参数 |

## 安装

依赖 NumPy 与 Pillow：

```bash
pip install -r requirements.txt
```

## 准备权重

权重不随仓库分发（`visual.*` 共 152 个张量，约 312 MB）。从 OpenCLIP ViT-B/32 检查点
导出 `visual.*` 到 `weights/image_encoder.npz` 即可。

## 运行

```bash
python run.py --output output_tokens.npz              # 使用 assets 中的示例图
python run.py --images a.jpg b.png --output out.npz   # 指定图片
```

输出 `[B,512]` 的 float32 数组，顺序与输入一致，每行长度 1。10 张图约 5 秒。

## 三处容易写错的地方

1. **线性层权重的方向**。`in_proj_weight`、`out_proj.weight`、`mlp.c_fc.weight`、
   `mlp.c_proj.weight` 均按 `[out,in]` 存放，投影须乘 `.T`；`conv1.weight` 先 reshape 为
   `[768,3072]`，同样是 `[out,in]`。
2. **`visual.proj` 是例外**。它已是 `[in,out]=[768,512]`，直接相乘，**不能转置**。
3. **四维张量的转置**。注意力里 `k` 是 `[B,H,N,64]`，`k.T` 会同时翻转 batch 与 head，
   必须写 `k.swapaxes(-1,-2)`。

另外：图块投影无 bias，12 层 block 均无 LayerScale，GELU 用精确的 `erf` 而非 `tanh` 近似。

## 数值验证

与独立的 PyTorch FP32 参考逐阶段比对（容差：内部 token `atol=2e-4`，输出 `2e-5`，
`rtol=2e-4`）：

| 阶段 | 形状 | max_abs | mean_abs | 结果 |
| --- | --- | --- | --- | --- |
| preprocess | (10,3,224,224) | 0 | 0 | PASS |
| patches | (2,49,3072) | 0 | 0 | PASS |
| tokens | (2,50,768) | 4.387e-05 | 1.335e-06 | PASS |
| block0 | (2,50,768) | 7.147e-06 | 3.348e-07 | PASS |
| last_tokens | (2,50,768) | 4.196e-05 | 2.633e-06 | PASS |
| output | (10,512) | 5.662e-07 | 8.688e-08 | PASS |
| batch | (3,512) | 4.061e-07 | 7.848e-08 | PASS |

`preprocess` 与 `patches` 仅做索引重排，误差严格为 0；进入 3072 维矩阵乘后，NumPy 与
PyTorch 的 BLAS 分块与累加顺序不同，产生 1e-5 量级绝对误差，相对误差约 1e-6，属 float32
正常范围。

## 复用

`ImageEncoder`、`LayerNorm`、`TransformerBlock(weights, prefix, heads)` 三个接口与 `causal`
开关均保留，文本塔可直接复用同一个 `TransformerBlock`。
