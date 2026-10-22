# 消融实验结果汇总 · SE / EMA / CBAM

> 钢材表面缺陷检测 · NEU-DET · YOLOv8n 骨干 · imgsz=640 · batch=32 · seed=0 · ultralytics 8.0.210  
> 指标取各实验 **mAP50-95 最高的一轮**（最佳 epoch）；全部数字来自仓库内 `results.csv`，无手工填数。  
> 由 `实验结果/生成消融汇总.py` 生成于 2026-10-22 13:30

## 1. 消融总表

| 实验 | 模型配置 | 权重初始化 | patience | 实际轮数 | 最佳 epoch | Precision | Recall | mAP50 | mAP50-95 |
|:--|:--|:--|--:|--:|--:|--:|--:|--:|--:|
| 基线 YOLOv8n | `yolov8n.pt` | COCO 预训练 | 30 | 79 | 49 | 0.5554 | 0.6167 | 0.6440 | 0.3190 |
| 实验一 + SEBlock | `yolov8-se.yaml` | 随机初始化 | 50 | 277 | 227 | 0.6630 | 0.6516 | 0.6849 | 0.3486 |
| 实验二 + EMA | `yolov8-ema.yaml` | 随机初始化 | 50 | 172 | 122 | 0.5848 | 0.6433 | 0.6776 | 0.3482 |
| 实验三 + CBAM | `yolov8-cbam.yaml` | 随机初始化 | 50 | 248 | 198 | 0.6270 | 0.6730 | 0.6862 | 0.3399 |

## 2. 相对基线的增益

| 模块 | Δ mAP50 | Δ mAP50-95 | Δ Precision | Δ Recall |
|:--|--:|--:|--:|--:|
| 实验一 + SEBlock | +0.0409 | +0.0296 | +0.1076 | +0.0349 |
| 实验二 + EMA | +0.0336 | +0.0292 | +0.0294 | +0.0266 |
| 实验三 + CBAM | +0.0422 | +0.0209 | +0.0716 | +0.0563 |

## 3. 末轮指标参考

| 实验 | 末轮 epoch | Precision | Recall | mAP50 | mAP50-95 |
|:--|--:|--:|--:|--:|--:|
| 基线 YOLOv8n | 79 | 0.5557 | 0.6345 | 0.6050 | 0.3014 |
| 实验一 + SEBlock | 277 | 0.6298 | 0.6514 | 0.6669 | 0.3320 |
| 实验二 + EMA | 172 | 0.6370 | 0.6011 | 0.6546 | 0.3271 |
| 实验三 + CBAM | 248 | 0.6406 | 0.6397 | 0.6730 | 0.3309 |

## 4. 参数量开销与权重体积

| 实验 | 理论参数增量 | 相对 EMA | `best.pt` 实测体积 |
|:--|--:|--:|--:|
| 基线 YOLOv8n | 0（基线） | --（基线） | 6105 KB |
| 实验一 + SEBlock | +43008 | 0.79x | 8327 KB |
| 实验二 + EMA | +54208 | 1.00x | 8320 KB |
| 实验三 + CBAM | +345254 | 6.37x | 8364 KB |

## 5. EMA 模型 · 六类缺陷分类别表现

| 类别 | 中文 | Precision | Recall | mAP@50 |
|:--|:--|--:|--:|--:|
| `crazing` | 龟裂 | 0.588 | 0.397 | 0.483 |
| `inclusion` | 夹杂 | 0.725 | 0.694 | 0.795 |
| `patches` | 补丁 | 0.855 | 0.797 | 0.896 |
| `pitted_surface` | 麻点 | 0.713 | 0.682 | 0.749 |
| `rolled-in_scale` | 氧化轧入 | 0.488 | 0.587 | 0.481 |
| `scratches` | 划痕 | 0.630 | 0.751 | 0.706 |

> 数据源：`生成对比图.py` 内嵌常量，与论文配图同源，非本次 val 日志直接导出。

## 6. 数据来源回溯

- `基线 YOLOv8n` → `实验结果/使用完整数据集。设置300轮，30次，提前终止（49次最好，79轮停止）/train/results.csv`
- `实验一 + SEBlock` → `实验结果/实验一：yolov8-se/yolov8-se/results.csv`
- `实验二 + EMA` → `实验结果/实验二：yolov8-ema/yolov8-ema/results.csv`
- `实验三 + CBAM` → `实验结果/实验三：yolov8-cbam/yolov8-cbam/results.csv`

## 7. 读数须知（诚实披露）

1. **权重初始化不一致** —— 基线用 `yolov8n.pt`（COCO 预训练），三组注意力实验用 `.yaml` 架构从随机初始化训练；严格受控对比需补跑 `yolov8n.yaml` 随机初始化基线。
2. **早停阈值不一致** —— 基线 `patience=30`，三组实验 `patience=50`。
3. **尚无组合消融** —— SE+EMA / EMA+CBAM / 三者联合未验证。
4. **参数量为理论计算值** —— GFLOPs 与 FPS 未实测。
5. **分类别指标为脚本内嵌常量** —— 复核需重跑 val。

## 8. 复核命令

```bash
# 重新生成本汇总
python 实验结果/生成消融汇总.py

# 权威参数量 / GFLOPs / 层数
yolo model=yolov8-se.yaml   info
yolo model=yolov8-ema.yaml  info
yolo model=yolov8-cbam.yaml info

# 复现分类别指标
yolo val model=runs/detect/yolov8-ema/weights/best.pt data=my_data.yaml
```

