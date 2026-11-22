# vit-b32-numpy

OpenCLIP **ViT-B/32** 图像塔（vision tower）的纯 NumPy 实现。不使用 PyTorch、
`open_clip` 或任何深度学习框架，全部前向计算由数组运算手写完成。

输入 `[B,3,224,224]` 的预处理图像，输出 `[B,512]` 的图像 token，每行 L2 长度为 1。

## 前向链路

```
[B,3,224,224] --patchify--> [B,49,3072] --投影+拼CLS+位置编码--> [B,50,768]
              --12 x TransformerBlock--> [B,50,768] --CLS+post-Norm+proj--> [B,512]
```

一张 224×224 的图按 32×32 切分得到 7×7 = 49 个图块，每个图块展平为 3072 维；
拼接一行 CLS 后为 50 个 token，经 12 层 pre-LN Transformer 取 CLS 行，归一化后
投影到 512 维并逐行单位化。

## 文件

| 文件 | 说明 |
| --- | --- |
| `image_encoder.py` | 核心实现：`ImageEncoder`、`TransformerBlock`、`MultiHeadAttention` |
| `helpers.py` | LayerNorm、GELU、softmax、单位化等数值辅助函数 |
| `preprocess_image.py` | 图像预处理（resize + center crop + 归一化） |
| `runtime.py` | 权重与图像的读取入口 |
| `run.py` | 命令行：图像 → 输出 token |
| `config.json` | 模型结构参数 |

## 运行

```bash
python -m pip install -r requirements.txt
```

权重不在仓库内（`visual.*` 共 152 个张量，约 312 MB）。将 OpenCLIP ViT-B/32
检查点中的 `visual.*` 导出为 `weights/image_encoder.npz` 后即可运行：

```bash
python run.py --output output_tokens.npz
python run.py --images a.jpg b.png --output my_tokens.npz
```

输出 `[B,512]` 的 float32 数组，顺序与输入一致，每行长度 1。10 张图约 5 秒。

## 三处容易写错的地方

1. **线性层权重的方向**。检查点里 `in_proj_weight`、`out_proj.weight`、
   `mlp.c_fc.weight`、`mlp.c_proj.weight` 均按 `[out,in]` 存放，做投影时必须乘 `.T`；
   `conv1.weight` 需先 reshape 为 `[768,3072]`，同样是 `[out,in]`。
2. **`visual.proj` 是例外**。它的形状已经是 `[in,out]=[768,512]`，直接相乘，
   **不能转置**。
3. **四维张量的转置**。注意力里 `k` 的形状是 `[B,H,N,64]`，用 `k.T` 会同时翻转
   batch 与 head 两轴，必须写 `k.swapaxes(-1,-2)` 只交换最后两轴。

另外：图块投影没有 bias，12 层 block 均无 LayerScale，GELU 使用精确的 `erf`
实现而非 `tanh` 近似（后者在 float32 下偏差约 1e-4）。

## 数值验证

与独立的 PyTorch FP32 参考实现逐阶段比对，7 个阶段误差如下
（容差：内部 token `atol=2e-4`，输出 `2e-5`，`rtol=2e-4`）：

| 阶段 | 形状 | max_abs | mean_abs | 结果 |
| --- | --- | --- | --- | --- |
| preprocess | (10,3,224,224) | 0 | 0 | PASS |
| patches | (2,49,3072) | 0 | 0 | PASS |
| tokens | (2,50,768) | 4.387e-05 | 1.335e-06 | PASS |
| block0 | (2,50,768) | 7.147e-06 | 3.348e-07 | PASS |
| last_tokens | (2,50,768) | 4.196e-05 | 2.633e-06 | PASS |
| output | (10,512) | 5.662e-07 | 8.688e-08 | PASS |
| batch | (3,512) | 4.061e-07 | 7.848e-08 | PASS |

`preprocess` 与 `patches` 只做索引重排，误差严格为 0。进入 3072 维矩阵乘后，
NumPy 与 PyTorch 的 BLAS 分块与累加顺序不同，产生 1e-5 量级的绝对误差，
相对误差约 1e-6，属 float32 正常范围。

## 作为组件复用

`ImageEncoder`、`LayerNorm`、`TransformerBlock(weights, prefix, heads)` 三个接口
与 `causal` 开关都保留，文本塔可以直接复用同一个 `TransformerBlock`。
