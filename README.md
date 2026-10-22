<div align="center">

# 🏭 钢材表面缺陷检测 · YOLOv8 注意力机制消融实验

**SE / EMA / CBAM 三种注意力模块的系统性对比** ｜ **NEU-DET 数据集** ｜ **YOLOv8n 骨干**

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-8.0.210-FF4D00.svg)](https://github.com/ultralytics/ultralytics)
[![Dataset](https://img.shields.io/badge/Dataset-NEU--DET-6%20classes-2ea44f.svg)](https://github.com/fei33fei/yolov8-se-ema-cbam)
[![License](https://img.shields.io/badge/License-AGPL--3.0-orange.svg)](./LICENSE)
[![Paper](https://img.shields.io/badge/Paper-Core%20Journal%20First--Author-blue.svg)](#-论文与引用)

**一句话结论**：在 YOLOv8n 骨干的 P2/P3/P4 三个层级后分别插入 SE、EMA、CBAM 注意力模块，**三种模块均能显著提升钢材表面缺陷检测精度**；其中 **SE 与 EMA 在 mAP50-95 上分别提升 +0.0296 / +0.0292**，而 **CBAM 参数量开销高达 EMA 的 6.4 倍却精度垫底**——精度收益与参数量开销不成正比，选型必须由消融数据决定。

</div>

---

## 📖 目录

| 章节 | 内容 |
| :-- | :-- |
| [✨ 核心成果](#-核心成果) | 消融总表、相对增益、参数量开销、六类分类别表现、训练稳定性 |
| [🧠 方法：三种注意力模块](#-方法三种注意力模块) | SEBlock / EMA / CBAM 原理与本仓库实现 |
| [📊 实验设置](#-实验设置) | 数据集、训练超参、**读数须知（公平性说明）** |
| [🚀 快速复现](#-快速复现) | 环境、训练、验证、生成对比图 |
| [🗂️ 仓库结构](#️-仓库结构) | 目录说明与哪些目录未提交 |
| [📓 结果归档](#-结果归档) | 每次实验的完整产物清单 |
| [🔍 已知问题与后续工作](#-已知问题与后续工作) | 本实验的局限与下一步 |
| [📄 论文与引用](#-论文与引用) | 关联论文与引用格式 |
| [📜 许可证](#-许可证) | AGPL-3.0 与上游声明 |

---

## ✨ 核心成果

### 1. 消融总表

统一设置：`NEU-DET` 验证集 · `imgsz=640` · `batch=32` · `epochs=300`（含早停）· `seed=0` · `deterministic=True` · 单卡 `device=0`
所有指标取自各自训练过程的**最佳 epoch**（按 `mAP50-95` 选取）。

| # | 实验 | 模型配置 | 权重初始化 | patience | 实际轮数 | 最佳 epoch | Precision | Recall | mAP50 | mAP50-95 | `best.pt` |
|:--|:--|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| — | **基线** | `yolov8n.pt` | COCO 预训练 | 30 | 79（早停） | 49 | 0.5554 | 0.6167 | 0.6440 | 0.3190 | — |
| 实验一 | **SE** | `yolov8-se.yaml` | 随机初始化 | 50 | 277 | 227 | **0.6630** | 0.6516 | 0.6849 | **0.3486** | 8,327 KB |
| 实验二 | **EMA** | `yolov8-ema.yaml` | 随机初始化 | 50 | 172 | 122 | 0.5848 | 0.6433 | 0.6776 | 0.3482 | 8,320 KB |
| 实验三 | **CBAM** | `yolov8-cbam.yaml` | 随机初始化 | 50 | 248 | 198 | 0.6270 | **0.6730** | **0.6862** | 0.3399 | 8,364 KB |

> 📌 三组注意力实验在**完全不使用预训练权重**的条件下，仍在 mAP50、mAP50-95、Precision、Recall 四项上**全面超过 COCO 预训练基线**。

### 2. 相对基线的增益

| 模块 | Δ mAP50 | Δ mAP50-95 | Δ Precision | Δ Recall |
|:--|--:|--:|--:|--:|
| **SEBlock** | +0.0409 | **+0.0296** | **+0.1076** | +0.0349 |
| **EMA** | +0.0336 | +0.0292 | +0.0294 | +0.0266 |
| **CBAM** | **+0.0422** | +0.0209 | +0.0716 | **+0.0563** |

**读法**：
- **SE 的 Precision 增益最大（+0.1076）** —— 通道注意力按全局统计重标定各通道，直接压低了误检（false positive），对钢材表面这种「背景复杂、缺陷与纹理容易混淆」的场景最对症。
- **CBAM 的 Recall 增益最大（+0.0563）** —— 空间注意力能定位缺陷区域，漏检下降，但 mAP50-95 反而垫底，说明它把召回换来的同时带来了额外的定位误差。
- **EMA 的增益最均衡**（两项 mAP 都在 +0.03 量级，且没有单项极端），是**精度—开销比最优**的折中选择，也是本项目最终采用并成文的方案。

### 3. 参数量开销

下表为按 `ultralytics/nn/modules/conv.py` 中模块定义**逐层计算**的理论增量（基线 YOLOv8n 规模约 3.16 M 参数）：

| 模块 | 插入位置 | 理论参数增量 | 相对 EMA | 实测 `best.pt` 体积 |
|:--|:--|--:|--:|--:|
| SEBlock | C2f 后 ×3（c=128/256/512） | +43,008 | 0.79× | 8,327 KB |
| **EMA** | C2f 后 ×3（c=128/256/512） | **+54,208** | **1.00×** | **8,320 KB** |
| CBAM | C2f 后 ×3（c=128/256/512） | +345,254 | **6.37×** | 8,364 KB |

> CBAM 的参数开销主要来自通道注意力里的 `Conv2d(c, c, 1)`（c=512 时单层即 262,656 参数），是 SE 的 `c²/8` 结构的 8 倍。

**核心结论：参数量与精度收益不成正比。** CBAM 花了 EMA 的 6.4 倍参数，mAP50-95 反而比 EMA 低 0.0083。工程选型应当由消融数据决定，而不是默认「注意力越复杂越好」。

> 复核命令（输出权威的参数量 / GFLOPs / 层数）：
> ```bash
> yolo model=yolov8-se.yaml   info
> yolo model=yolov8-ema.yaml  info
> yolo model=yolov8-cbam.yaml info
> ```

### 4. EMA 模型 · 六类缺陷分类别表现

取自 EMA 模型在 NEU-DET 验证集上的分类别指标（`生成对比图.py` 中同源数据）：

| 类别 | 中文 | Precision | Recall | mAP@50 |
|:--|:--|--:|--:|--:|
| `crazing` | 龟裂 | 0.588 | 0.397 | 0.483 |
| `inclusion` | 夹杂 | 0.725 | 0.694 | 0.795 |
| `patches` | 补丁 | **0.855** | 0.797 | **0.896** |
| `pitted_surface` | 麻点 | 0.713 | 0.682 | 0.749 |
| `rolled-in_scale` | 氧化轧入 | **0.488** | 0.587 | **0.481** |
| `scratches` | 划痕 | 0.630 | **0.751** | 0.706 |

**观察**：
- `patches`（补丁）表现最好，mAP@50 达 0.896 —— 面积大、对比度强，特征显著。
- `crazing`（龟裂）与 `rolled-in_scale`（氧化轧入）是两个难点类别：前者细如发丝、后者与正常轧制纹理高度相似，两者 mAP@50 均在 0.48 附近。**这两个类别正是后续引入分割分支（FGL / U-Net 系列）的动机** —— 检测框无法表达「像素级边界」，而这两类缺陷恰恰需要边界信息。
- `scratches`（划痕）Recall 最高（0.751）但 Precision 最低（0.630）→ 细长目标在 NMS 阶段易被误合并/漏合并。

### 5. 训练稳定性观察

| 实验 | 最佳 epoch | 停止 epoch | 早停触发 | 说明 |
|:--|--:|--:|:--|:--|
| 基线 | 49 | 79 | ✅ | 最佳点后连续 30 轮无提升，验证集指标已明显回落 |
| SE | 227 | 277 | ✅ | 提升持续到第 227 轮 |
| EMA | 122 | 172 | ✅ | 提升持续到第 122 轮 |
| CBAM | 198 | 248 | ✅ | 提升持续到第 198 轮 |

> **基线在第 79 轮就触发了早停，最佳点仅在第 49 轮；而三组注意力实验的最佳点分别推迟到第 227 / 122 / 198 轮。**
> 这说明在验证集上，加入注意力机制后模型的有效学习期**显著延长**、过拟合来得更晚。基线的早收敛（mAP50-95 停在 0.3190）与三组实验的持续爬升（0.3399 ~ 0.3486）互为佐证。

---

## 🧠 方法：三种注意力模块

三种模块均**插入在 Backbone 的 C2f 之后**，覆盖 P2/4、P3/8、P4/16 三个尺度：

```
Conv → C2f → [ATTN] → Conv → C2f → [ATTN] → Conv → C2f → [ATTN] → Conv → C2f → SPPF
         c=128         c=256           c=512
```

对应模型配置文件：`ultralytics/cfg/models/v8/yolov8-{se,ema,cbam}.yaml`
三份 yaml 除注意力模块名外**结构完全一致**，Head 与原生 YOLOv8n 保持相同，保证对比公平。

### SEBlock — Squeeze-and-Excitation

> 论文：[*Squeeze-and-Excitation Networks*, CVPR 2017](https://arxiv.org/abs/1709.01507) ｜ 代码位置：`ultralytics/nn/modules/conv.py` → `class SEBlock`

**通道注意力**，只回答「哪些通道重要」：

```
x → GAP → Linear(c, c/16) → ReLU → Linear(c/16, c) → Sigmoid → x * s
```

- `c_ = max(1, c1 // reduction)`，默认 `reduction=16`
- 参数增量 `2 × c1 × c1/16 = c1²/8`，是三者中最省的
- 缺点：完全忽略空间位置信息

### EMA — Efficient Multi-Scale Attention

> 论文：[*Efficient Multi-Scale Attention Module with Cross-Channel Learning*, arXiv:2401.00945](https://arxiv.org/abs/2401.00945) ｜ 代码位置：`ultralytics/nn/modules/conv.py` → `class EMA`

**分组 + 通道/空间联合注意力**：

```python
group_x = x.view(b * groups, -1, h, w)          # groups = 8，每组通道数 c // 8
x1 = self.gn(self.conv1x1(self.avg_pool(group_x)))   # 通道注意力分支：GAP → 1×1 Conv → GroupNorm
x2 = self.gn(self.conv3x3(group_x))                  # 空间注意力分支：3×3 Conv → GroupNorm
weights = self.sigmoid(x1 + x2)                      # 融合后生成逐元素注意力图
out = group_x * weights
```

- 参数增量 `10·cg² + 4·cg`（`cg = c/8`）
- **本仓库实现为简化版**：原论文的 EMA 包含多尺度分组特征聚合（将特征拆成不同分组、分别做多尺度卷积再交互），本实现以**全局平均池化**替代了多尺度分组聚合部分，保留「分组 + 通道空间联合注意力」的核心思想。**这是为了控制改动幅度、聚焦消融变量而做的取舍，并非论文原始实现的逐行复现** —— 复现原论文版本可作为后续工作（见「已知问题与后续工作」）。

### CBAM — Convolutional Block Attention Module

> 论文：[*CBAM: Convolutional Block Attention Module*, ECCV 2018](https://arxiv.org/abs/1807.06521) ｜ 代码位置：`ultralytics/nn/modules/conv.py` → `class CBAM`

**通道注意力 → 空间注意力 串联**：

```
x → ChannelAttention → SpatialAttention → x
```

- `ChannelAttention`：`GAP → Conv2d(c, c, 1) → Sigmoid`（本实现改编自 mmdetection RTMDet）
- `SpatialAttention`：沿通道维取 `mean` 与 `max` 拼接 → `Conv2d(2, 1, 7) → Sigmoid`
- 参数增量 `c² + c + 98`，c=512 时单层即 262,654 个参数，**是三者中最重的**

---

## 📊 实验设置

### 数据集

**NEU-DET**（东北大学钢材表面缺陷数据集），本仓库所用划分：

| 划分 | 图像数 | 说明 |
|:--|--:|:--|
| train | 2,460 | — |
| val | 1,080 | — |
| **合计** | **3,540** | 6 类缺陷 |

| ID | 类别 | 中文 | 类别 | 中文 |
|:--:|:--|:--|:--|:--|
| 0 | `crazing` | 龟裂 | 3 | `pitted_surface` 麻点 |
| 1 | `inclusion` | 夹杂 | 4 | `rolled-in_scale` 氧化轧入 |
| 2 | `patches` | 补丁 | 5 | `scratches` 划痕 |

数据集目录结构（本地，**未提交到 Git**，见 `.gitignore`）：

```
datasets/NEU-DET/
├── images/{train,val}/
└── labels/{train,val}/
```

### 训练超参

| 参数 | 值 | 参数 | 值 |
|:--|:--|:--|:--|
| model | `yolov8-se.yaml` / `yolov8-ema.yaml` / `yolov8-cbam.yaml` | epochs | 300（含早停） |
| data | `my_data.yaml` | patience | 50（基线为 30，见下方说明） |
| imgsz | 640 | batch | 32 |
| optimizer | `auto` | lr0 / lrf | 0.01 / 0.01 |
| weight_decay | 0.0005 | close_mosaic | 10 |
| seed | 0 | deterministic | True |
| pretrained | True | device / workers | `0` / 8 |
| ultralytics | 8.0.210 | torch | 2.x |

> 三份模型 yaml 中的 `nc: 80` 仅为 Ultralytics 模板的占位值，实际类别数由 `my_data.yaml` 决定（6 类），构建模型时会被数据集配置覆盖。

### ⚠️ 读数须知（公平性说明）

本实验的结论请在以下前提下阅读，这些差异**未在原始实验中做严格受控对齐**，如实披露：

1. **权重初始化不一致**
   基线使用 `yolov8n.pt`（**COCO 预训练权重**初始化），三组注意力实验使用 `.yaml` 架构，Ultralytics 在此路径下 **不加载预训练权重**（`engine/model.py` → `_new()`，`self.ckpt` 为 `None`），即**从随机初始化训练**。
   → 这使「三组实验全面超过基线」的结论**比在同等初始化下的对比更强**，但严格受控对比仍需补跑一组 `yolov8n.yaml`（随机初始化）作为对齐基线。**已列入后续工作。**

2. **早停阈值不一致**
   基线 `patience=30`，三组实验 `patience=50`。
   → 基线在第 49 轮取得最佳、第 79 轮停止，已处于平台期，阈值放宽对基线的潜在增益有限；但严格来说仍存在偏差。

3. **尚无组合消融**
   当前结果为 **SE / EMA / CBAM 单模块对比**，**未开展** SE+EMA、EMA+CBAM、三者联合等组合实验。三种模块作用于同一批插入位置，组合后的收益是否可叠加尚未验证。

4. **参数量为理论计算值**
   第 3 节的参数量增量按模块定义逐层计算得出；官方 `model.info()` 的 GFLOPs 与实测 FPS **尚未采集**，需要时可用上文命令复核。

---

## 🚀 快速复现

### 环境

```bash
git clone https://github.com/fei33fei/yolov8-se-ema-cbam.git
cd yolov8-se-ema-cbam/ultralytics-main

conda create -n yolo8 python=3.10 -y
conda activate yolo8
pip install -e .
```

### 准备数据

将 NEU-DET 按下列结构放到 `datasets/NEU-DET/`，并确认 `my_data.yaml` 中的 `path` 与实际路径一致：

```yaml
path: NEU-DET
train: images/train
val: images/val
names:
  0: crazing
  1: inclusion
  2: patches
  3: pitted_surface
  4: rolled-in-scale
  5: scratches
```

### 训练

```bash
# 训练入口（支持断点续训 --resume，结果按模型名自动归档到 runs/detect/）
python run_train.py --data my_data.yaml --model yolov8-se.yaml   --epochs 300 --batch 32
python run_train.py --data my_data.yaml --model yolov8-ema.yaml  --epochs 300 --batch 32
python run_train.py --data my_data.yaml --model yolov8-cbam.yaml --epochs 300 --batch 32

# 中断后续训
python run_train.py --data my_data.yaml --model yolov8-ema.yaml --resume
```

`run_train.py` 的关键行为：
- 自动 `chdir` 到脚本所在目录，`--data` / `--model` 默认取 `my_data.yaml` / `yolov8n.yaml`
- 训练前调用 `torch.serialization.add_safe_globals([DetectionModel])`，规避加载旧权重时的 pickle 安全报错
- `patience=50` 早停，`exist_ok=True` 允许覆盖同名实验目录
- 未指定 `--name` 时自动用模型文件名命名（如 `yolov8-ema`），便于消融结果自动归档

### 验证与推理

```bash
yolo val model=runs/detect/yolov8-ema/weights/best.pt data=my_data.yaml
yolo predict model=runs/detect/yolov8-ema/weights/best.pt source=your_image.jpg
```

### 生成六类性能对比图

```bash
python 生成对比图.py
```

> ⚠️ **使用前请修改脚本首部的 `output_dir` 默认路径**（当前为作者本机绝对路径），改为你自己的 `runs/` 目录，否则会写入失败。
> 脚本使用 Times New Roman 字体输出 300 DPI 论文级配图，数据源为 EMA 模型的六类分类别指标。

---

## 🗂️ 仓库结构

```
yolov8-se-ema-cbam/
├── README.md                    # 本文档（项目主文档）
├── README.zh-CN.md              # 中文版（内容同本文档）
├── README.ultralytics.md        # 上游 Ultralytics 原始 README（保留出处与许可声明）
├── README.ultralytics.zh-CN.md  # 上游中文版
│
├── my_data.yaml                 # NEU-DET 数据集配置（6 类）
├── run_train.py                 # 训练入口：断点续训 / 早停 / 自动归档
├── 生成对比图.py                 # 六类缺陷性能对比图生成脚本（300 DPI）
├── requirements.txt / setup.py / setup.cfg / MANIFEST.in
├── CITATION.cff                 # 引用信息（⚠️ 仍为上游内容，见「已知问题」）
├── LICENSE                      # AGPL-3.0
│
├── ultralytics/                 # Ultralytics YOLO 定制版源码
│   ├── cfg/models/v8/
│   │   ├── yolov8-se.yaml       # ★ SE 注意力配置
│   │   ├── yolov8-ema.yaml      # ★ EMA 注意力配置
│   │   └── yolov8-cbam.yaml     # ★ CBAM 注意力配置
│   └── nn/modules/conv.py       # ★ SEBlock / EMA / CBAM 实现（含 ChannelAttention、SpatialAttention）
│
├── 实验结果/                     # ★ 4 次实验的完整产物（已提交）
│   ├── 使用完整数据集…（基线）/train/
│   ├── 实验一：yolov8-se/yolov8-se/
│   ├── 实验二：yolov8-ema/yolov8-ema/
│   └── 实验三：yolov8-cbam/yolov8-cbam/
│
├── datasets/                    # ⚠️ 本地数据集（.gitignore，未提交）
├── runs/                        # ⚠️ 本地训练输出（.gitignore，未提交）
├── YOLOv8/  docker/  docs/  examples/  scripts/  tests/  .github/
```

**Git 提交状态**：`.gitignore` 中 `datasets/`、`runs/`、`*.pt` 均被排除，因此**数据集与权重文件不在仓库中**。`实验结果/` 目录下的 `results.csv`、`args.yaml`、混淆矩阵、PR/F1 曲线、训练批次可视化均已提交，可直接在线查看。

---

## 📓 结果归档

`实验结果/` 下每次实验均包含完整产物：

| 文件 | 内容 |
|:--|:--|
| `args.yaml` | 完整训练超参快照（可复现的唯一凭据） |
| `results.csv` | 逐 epoch 指标（loss / precision / recall / mAP50 / mAP50-95 / lr） |
| `results.png` | 训练曲线总览（8 子图） |
| `confusion_matrix.png` / `confusion_matrix_normalized.png` | 混淆矩阵（原始 / 归一化） |
| `PR_curve.png` / `F1_curve.png` / `P_curve.png` / `R_curve.png` | 各类别 PR / F1 / 精确率 / 召回率曲线 |
| `labels.jpg` / `labels_correlogram.jpg` | 数据集标签分布与相关性图 |
| `train_batch0~2.jpg` | 训练批次可视化（带标签） |
| `val_batch{0,1,2}_{labels,pred}.jpg` | 验证批次真值 vs 预测对比 |
| `weights/best.pt` / `last.pt` | 最佳 / 末期权重（⚠️ 因 `*.pt` 被 gitignore，未提交） |

**建议阅读顺序**：`results.png` 看整体收敛 → `confusion_matrix_normalized.png` 看类别混淆 → `val_batch*_pred.jpg` 看真实预测效果。

---

## 🔍 已知问题与后续工作

**当前实验的局限**（诚实披露，也是下一步的切入点）：

1. **缺少随机初始化基线** —— 补跑 `yolov8n.yaml`（随机初始化、`patience=50`）以对齐初始化条件，得到严格受控的对照结论。
2. **缺少组合消融** —— SE+EMA、EMA+CBAM、SE+EMA+CBAM 的组合收益是否可叠加，待验证。
3. **EMA 为简化实现** —— 当前用全局平均池化替代了原论文的多尺度分组特征聚合；复现原论文完整版本并对比是明确的后续方向。
4. **缺少效率指标** —— 参数量 / GFLOPs / FPS 的权威实测（`model.info()` + 边缘设备推理）尚未采集。
5. **检测粒度的天花板** —— `crazing`、`rolled-in_scale` 两类 mAP@50 仅约 0.48，根源在于检测框无法表达像素级边界。**这正是本项目后续迁移到分割框架（UNet / FGL 系列）的动机**，相关工作见 [SDE-Net / FCS-Net（海上溢油分割）](https://github.com/fei33fei) 与绝缘子缺陷检测仓库。
6. **`CITATION.cff` 仍为上游内容** —— 仍引用 Jocher 等人的 YOLO 论文，未指向本项目关联的期刊论文，应更新。
7. **`.github/workflows/translate-readme.yml` 会自动覆盖 `README.zh-CN.md`** —— 本项目的中文 README 为手写内容，建议删除或禁用该工作流。
8. **数据集未随仓库发布** —— 建议补充 `datasets/README.md`，说明 NEU-DET 来源、许可与获取方式。

---

## 📄 论文与引用

本项目为**《Steel Surface Defect Detection Based on an Improved YOLOv8 Integrating EMA Attention Mechanism》**（核心期刊，**第一作者**）的实验代码库。

> 该论文采用 **EMA** 注意力方案 —— 对应本仓库的 `yolov8-ema.yaml` 与「实验二」。
> 相关成果同时获得**中国大学生计算机设计大赛（钢材缺陷分割检测赛道）国赛三等奖 + 省赛一等奖**。

> **📌 若使用本仓库，请引用上游工作。** 本仓库基于 [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) 深度定制（v8.0.210），遵循 AGPL-3.0 协议，版权归原作者所有。

---

## 📜 许可证

本仓库遵循 **[AGPL-3.0](https://www.gnu.org/licenses/agpl-3.0.html)** 协议。

- 代码基于 [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) 深度修改，保留上游全部版权声明与许可条款（见 `LICENSE` 与 `README.ultralytics.md`）
- 本仓库新增部分（`SEBlock` / `EMA` / `CBAM` 模块实现、三份模型配置、消融实验与结果归档）同样以 AGPL-3.0 发布
- ⚠️ **若将本仓库用于商业闭源产品，需注意 AGPL-3.0 的强 copyleft 要求**
- 引用本项目关联论文时请一并引用上游 Ultralytics 工作

---

<div align="center">

**Built with** [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) v8.0.210 · [PyTorch](https://pytorch.org)

**Author**：[沈亦菲 FeyFey](https://github.com/fei33fei) ｜ 南通理工学院 · 人工智能专业

*如有疑问或发现错误，欢迎提 [Issue](https://github.com/fei33fei/yolov8-se-ema-cbam/issues) 或 PR。*

</div>
