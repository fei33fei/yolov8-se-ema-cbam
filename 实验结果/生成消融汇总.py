# -*- coding: utf-8 -*-
"""生成 SE / EMA / CBAM 消融实验汇总表。

输出:
    实验结果/ablation_summary_latest.txt   (等宽表格, 与 Insulator_defect 仓库同名文件同规范)
    实验结果/ablation_summary_latest.md    (Markdown 版, 附数据来源与读数须知)

数据源(全部为仓库内真实产物, 无手工填数):
    实验结果/<实验目录>/results.csv   逐 epoch 指标  -> 取 mAP50-95 最高的一轮作为"最佳 epoch"
    实验结果/<实验目录>/args.yaml     训练超参快照
    实验结果/<实验目录>/weights/best.pt  权重体积

用法:
    python 实验结果/生成消融汇总.py
"""

import csv
import os
import unicodedata
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP

BASE = os.path.dirname(os.path.abspath(__file__))          # .../实验结果
REPO = os.path.dirname(BASE)                               # .../ultralytics-main

EXPERIMENTS = [
    {
        "name": "基线 YOLOv8n",
        "dir": "使用完整数据集。设置300轮，30次，提前终止（49次最好，79轮停止）/train",
        "init": "COCO 预训练",
        "params_inc": 0,
        "params_note": "基线, 无增量",
    },
    {
        "name": "实验一 + SEBlock",
        "dir": "实验一：yolov8-se/yolov8-se",
        "init": "随机初始化",
        "params_inc": 43008,
        "params_note": "C2f 后 x3 (c=128/256/512), c^2/8 结构",
    },
    {
        "name": "实验二 + EMA",
        "dir": "实验二：yolov8-ema/yolov8-ema",
        "init": "随机初始化",
        "params_inc": 54208,
        "params_note": "C2f 后 x3 (c=128/256/512), 10cg^2+4cg",
    },
    {
        "name": "实验三 + CBAM",
        "dir": "实验三：yolov8-cbam/yolov8-cbam",
        "init": "随机初始化",
        "params_inc": 345254,
        "params_note": "C2f 后 x3 (c=128/256/512), c^2+c+98",
    },
]

# EMA 模型在 NEU-DET 验证集上的分类别指标, 与 生成对比图.py 内嵌数据同源
PER_CLASS = [
    ("crazing", "龟裂", 0.588, 0.397, 0.483),
    ("inclusion", "夹杂", 0.725, 0.694, 0.795),
    ("patches", "补丁", 0.855, 0.797, 0.896),
    ("pitted_surface", "麻点", 0.713, 0.682, 0.749),
    ("rolled-in_scale", "氧化轧入", 0.488, 0.587, 0.481),
    ("scratches", "划痕", 0.630, 0.751, 0.706),
]

DATASET = "NEU-DET (train 2,460 / val 1,080, 6 类)"


def dwidth(text):
    """显示宽度: 全角字符按 2 计。"""
    return sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1 for ch in str(text))


def pad(text, width, align="<"):
    text = str(text)
    space = width - dwidth(text)
    if space < 0:
        space = 0
    if align == ">":
        return " " * space + text
    if align == "^":
        left = space // 2
        return " " * left + text + " " * (space - left)
    return text + " " * space


def render_table(headers, rows, aligns=None):
    """按显示宽度渲染等宽表格。"""
    cols = len(headers)
    aligns = aligns or ["<"] * cols
    widths = [dwidth(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], dwidth(cell))

    sep = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    out = [sep]
    out.append("|" + "|".join(" " + pad(h, widths[i], "^") + " " for i, h in enumerate(headers)) + "|")
    out.append(sep)
    for row in rows:
        out.append("|" + "|".join(" " + pad(c, widths[i], aligns[i]) + " " for i, c in enumerate(row)) + "|")
    out.append(sep)
    return "\n".join(out)


def read_results(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = [h.strip() for h in next(reader)]
        idx = {name: header.index(name) for name in header}
        rows = []
        for raw in reader:
            if not raw or not raw[0].strip():
                continue
            rows.append({
                "epoch": int(float(raw[idx["epoch"]])),
                "p": float(raw[idx["metrics/precision(B)"]]),
                "r": float(raw[idx["metrics/recall(B)"]]),
                "m50": float(raw[idx["metrics/mAP50(B)"]]),
                "m5095": float(raw[idx["metrics/mAP50-95(B)"]]),
            })
    return rows


def read_args(path):
    out = {}
    if not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if ":" not in line or line.startswith("#"):
                continue
            k, _, v = line.partition(":")
            out[k.strip()] = v.strip()
    return out


def load():
    data = []
    for spec in EXPERIMENTS:
        d = os.path.join(BASE, spec["dir"].replace("/", os.sep))
        csv_path = os.path.join(d, "results.csv")
        if not os.path.exists(csv_path):
            print("[WARN] 缺少 results.csv, 跳过: %s" % csv_path)
            continue
        rows = read_results(csv_path)
        best = max(rows, key=lambda x: x["m5095"])
        args = read_args(os.path.join(d, "args.yaml"))
        weight_path = os.path.join(d, "weights", "best.pt")
        weight_kb = os.path.getsize(weight_path) / 1024 if os.path.exists(weight_path) else None
        data.append({
            "spec": spec,
            "dir": spec["dir"],
            "csv_rel": "实验结果/%s/results.csv" % spec["dir"],
            "args": args,
            "rows": rows,
            "best": best,
            "last": rows[-1],
            "weight_kb": weight_kb,
        })
    return data


def r4(x):
    """四位小数四舍五入(ROUND_HALF_UP), 与 README 表格取整口径一致。"""
    return Decimal(repr(float(x))).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)


def fmt(x, n=4):
    """格式化到 n 位小数(ROUND_HALF_UP)。"""
    if n == 4:
        return str(r4(x))
    q = Decimal(1).scaleb(-n)
    return str(Decimal(repr(float(x))).quantize(q, rounding=ROUND_HALF_UP))


def signed(x, n=4):
    """带符号的差值; 入参已是四舍五入后的 Decimal。"""
    v = r4(x) if not isinstance(x, Decimal) else x
    return ("+%.4f" % v) if v >= 0 else ("%.4f" % v)


def delta(a, b):
    """由已四舍五入的数值相减, 保证表内可手工核对。"""
    return r4(a) - r4(b)


def build(data):
    base = data[0]
    lines = []
    W = 118

    lines.append("=" * W)
    lines.append(" YOLOv8 注意力机制消融实验结果对比 (NEU-DET / YOLOv8n 骨干 / imgsz=640 / batch=32 / seed=0)")
    lines.append(" 指标口径: 每个实验取 mAP50-95 最高的一轮作为最佳 epoch; 数据源为各自 results.csv, 无手工填数")
    lines.append(" 生成脚本: 实验结果/生成消融汇总.py    生成时间: %s" % datetime.now().strftime("%Y-%m-%d %H:%M"))
    lines.append("=" * W)
    lines.append("")

    rows = []
    for d in data:
        b = d["best"]
        rows.append([
            d["spec"]["name"],
            d["args"].get("model", "-"),
            d["spec"]["init"],
            d["args"].get("patience", "-"),
            len(d["rows"]),
            b["epoch"],
            fmt(b["p"]),
            fmt(b["r"]),
            fmt(b["m50"]),
            fmt(b["m5095"]),
            signed(delta(b["m5095"], base["best"]["m5095"])),
        ])
    lines.append(render_table(
        ["实验名称", "模型配置", "权重初始化", "patience", "实际轮数", "最佳epoch",
         "Precision", "Recall", "mAP50", "mAP50-95", "ΔmAP50-95"],
        rows,
        ["<", "<", "<", ">", ">", ">", ">", ">", ">", ">", ">"],
    ))
    lines.append("")

    lines.append("[相对基线的增益]")
    lines.append("")
    rows = []
    for d in data[1:]:
        b, bb = d["best"], base["best"]
        rows.append([
            d["spec"]["name"],
            signed(delta(b["m50"], bb["m50"])),
            signed(delta(b["m5095"], bb["m5095"])),
            signed(delta(b["p"], bb["p"])),
            signed(delta(b["r"], bb["r"])),
        ])
    lines.append(render_table(
        ["实验名称", "Δ mAP50", "Δ mAP50-95", "Δ Precision", "Δ Recall"],
        rows,
        ["<", ">", ">", ">", ">"],
    ))
    lines.append("")
    lines.append(" 注: 增益由表内已四舍五入的数值相减得出, 可手工核对; 与 README《核心成果》第 2 节一致。")
    lines.append("")

    lines.append("[末轮 (最后一个 epoch) 指标参考 -- 说明曲线后期回落, 汇报应统一用最佳 epoch]")
    lines.append("")
    rows = []
    for d in data:
        l = d["last"]
        rows.append([
            d["spec"]["name"], l["epoch"], fmt(l["p"]), fmt(l["r"]), fmt(l["m50"]), fmt(l["m5095"]),
        ])
    lines.append(render_table(
        ["实验名称", "末轮 epoch", "Precision", "Recall", "mAP50", "mAP50-95"],
        rows,
        ["<", ">", ">", ">", ">", ">"],
    ))
    lines.append("")

    lines.append("[参数量开销与权重体积]")
    lines.append("")
    ema_inc = next(d["spec"]["params_inc"] for d in data if d["spec"]["name"].endswith("EMA"))
    rows = []
    for d in data:
        inc = d["spec"]["params_inc"]
        if d is data[0]:
            rel = "-- (基线)"
        else:
            rel = "%.2fx" % (inc / ema_inc)
        size = "%d KB" % round(d["weight_kb"]) if d["weight_kb"] else "(未提交, *.pt 已被 .gitignore 排除)"
        rows.append([
            d["spec"]["name"],
            "+%d" % inc if inc else "0 (基线)",
            rel,
            size,
            d["spec"]["params_note"],
        ])
    lines.append(render_table(
        ["实验名称", "理论参数增量", "相对 EMA", "best.pt 实测体积", "模块结构"],
        rows,
        ["<", ">", ">", "<", "<"],
    ))
    lines.append("")
    lines.append(" 注: 参数增量为按 ultralytics/nn/modules/conv.py 中模块定义逐层计算的理论值;")
    lines.append("     官方 model.info() 的 GFLOPs / 实测 FPS 尚未采集, 可用 'yolo model=yolov8-se.yaml info' 复核。")
    lines.append("")

    lines.append("[EMA 模型 · 六类缺陷分类别表现]  (数据源: 生成对比图.py, 与该脚本内嵌常量同源)")
    lines.append("")
    rows = []
    for en, zh, p, r, m in PER_CLASS:
        rows.append([en, zh, "%.3f" % p, "%.3f" % r, "%.3f" % m])
    lines.append(render_table(
        ["类别", "中文", "Precision", "Recall", "mAP@50"],
        rows,
        ["<", "<", ">", ">", ">"],
    ))
    lines.append("")

    lines.append("[数据来源]  每个数字都可回溯到仓库内文件")
    lines.append("")
    for d in data:
        lines.append("   %-16s -> 实验结果/%s/results.csv" % (d["spec"]["name"], d["dir"]))
    lines.append("")

    lines.append("[读数须知 / 诚实披露]")
    lines.append("   1. 权重初始化不一致: 基线用 yolov8n.pt (COCO 预训练), 三组注意力实验用 .yaml 架构从随机初始化训练。")
    lines.append("      这使'三组全面超过基线'的结论在同等初始化下更强, 但严格受控对比仍需补跑 yolov8n.yaml 随机初始化基线。")
    lines.append("   2. 早停阈值不一致: 基线 patience=30, 三组实验 patience=50。")
    lines.append("   3. 尚无组合消融: 未开展 SE+EMA / EMA+CBAM / 三者联合, 收益是否可叠加未验证。")
    lines.append("   4. 参数量为理论计算值, GFLOPs 与 FPS 未实测。")
    lines.append("   5. 分类别指标为脚本内嵌常量, 非本次 val 日志直接导出, 复核时需重跑 val。")
    lines.append("")
    lines.append("=" * W)
    return "\n".join(lines)


def build_md(data):
    base = data[0]
    out = []
    out.append("# 消融实验结果汇总 · SE / EMA / CBAM")
    out.append("")
    out.append("> 钢材表面缺陷检测 · NEU-DET · YOLOv8n 骨干 · imgsz=640 · batch=32 · seed=0 · ultralytics 8.0.210  ")
    out.append("> 指标取各实验 **mAP50-95 最高的一轮**（最佳 epoch）；全部数字来自仓库内 `results.csv`，无手工填数。  ")
    out.append("> 由 `实验结果/生成消融汇总.py` 生成于 %s" % datetime.now().strftime("%Y-%m-%d %H:%M"))
    out.append("")
    out.append("## 1. 消融总表")
    out.append("")
    out.append("| 实验 | 模型配置 | 权重初始化 | patience | 实际轮数 | 最佳 epoch | Precision | Recall | mAP50 | mAP50-95 |")
    out.append("|:--|:--|:--|--:|--:|--:|--:|--:|--:|--:|")
    for d in data:
        b = d["best"]
        out.append("| %s | `%s` | %s | %s | %d | %d | %s | %s | %s | %s |" % (
            d["spec"]["name"], d["args"].get("model", "-"), d["spec"]["init"],
            d["args"].get("patience", "-"), len(d["rows"]), b["epoch"],
            fmt(b["p"]), fmt(b["r"]), fmt(b["m50"]), fmt(b["m5095"])))
    out.append("")
    out.append("## 2. 相对基线的增益")
    out.append("")
    out.append("| 模块 | Δ mAP50 | Δ mAP50-95 | Δ Precision | Δ Recall |")
    out.append("|:--|--:|--:|--:|--:|")
    for d in data[1:]:
        b, bb = d["best"], base["best"]
        out.append("| %s | %s | %s | %s | %s |" % (
            d["spec"]["name"],
            signed(delta(b["m50"], bb["m50"])),
            signed(delta(b["m5095"], bb["m5095"])),
            signed(delta(b["p"], bb["p"])),
            signed(delta(b["r"], bb["r"]))))
    out.append("")
    out.append("## 3. 末轮指标参考")
    out.append("")
    out.append("| 实验 | 末轮 epoch | Precision | Recall | mAP50 | mAP50-95 |")
    out.append("|:--|--:|--:|--:|--:|--:|")
    for d in data:
        l = d["last"]
        out.append("| %s | %d | %s | %s | %s | %s |" % (
            d["spec"]["name"], l["epoch"], fmt(l["p"]), fmt(l["r"]), fmt(l["m50"]), fmt(l["m5095"])))
    out.append("")
    out.append("## 4. 参数量开销与权重体积")
    out.append("")
    out.append("| 实验 | 理论参数增量 | 相对 EMA | `best.pt` 实测体积 |")
    out.append("|:--|--:|--:|--:|")
    ema_inc = next(d["spec"]["params_inc"] for d in data if d["spec"]["name"].endswith("EMA"))
    for i, d in enumerate(data):
        inc = d["spec"]["params_inc"]
        rel = "--（基线）" if i == 0 else "%.2fx" % (inc / ema_inc)
        size = "%d KB" % round(d["weight_kb"]) if d["weight_kb"] else "（未提交，`*.pt` 被 `.gitignore` 排除）"
        out.append("| %s | %s | %s | %s |" % (
            d["spec"]["name"], "+%d" % inc if inc else "0（基线）", rel, size))
    out.append("")
    out.append("## 5. EMA 模型 · 六类缺陷分类别表现")
    out.append("")
    out.append("| 类别 | 中文 | Precision | Recall | mAP@50 |")
    out.append("|:--|:--|--:|--:|--:|")
    for en, zh, p, r, m in PER_CLASS:
        out.append("| `%s` | %s | %.3f | %.3f | %.3f |" % (en, zh, p, r, m))
    out.append("")
    out.append("> 数据源：`生成对比图.py` 内嵌常量，与论文配图同源，非本次 val 日志直接导出。")
    out.append("")
    out.append("## 6. 数据来源回溯")
    out.append("")
    for d in data:
        out.append("- `%s` → `实验结果/%s/results.csv`" % (d["spec"]["name"], d["dir"]))
    out.append("")
    out.append("## 7. 读数须知（诚实披露）")
    out.append("")
    out.append("1. **权重初始化不一致** —— 基线用 `yolov8n.pt`（COCO 预训练），三组注意力实验用 `.yaml` 架构从随机初始化训练；严格受控对比需补跑 `yolov8n.yaml` 随机初始化基线。")
    out.append("2. **早停阈值不一致** —— 基线 `patience=30`，三组实验 `patience=50`。")
    out.append("3. **尚无组合消融** —— SE+EMA / EMA+CBAM / 三者联合未验证。")
    out.append("4. **参数量为理论计算值** —— GFLOPs 与 FPS 未实测。")
    out.append("5. **分类别指标为脚本内嵌常量** —— 复核需重跑 val。")
    out.append("")
    out.append("## 8. 复核命令")
    out.append("")
    out.append("```bash")
    out.append("# 重新生成本汇总")
    out.append("python 实验结果/生成消融汇总.py")
    out.append("")
    out.append("# 权威参数量 / GFLOPs / 层数")
    out.append("yolo model=yolov8-se.yaml   info")
    out.append("yolo model=yolov8-ema.yaml  info")
    out.append("yolo model=yolov8-cbam.yaml info")
    out.append("")
    out.append("# 复现分类别指标")
    out.append("yolo val model=runs/detect/yolov8-ema/weights/best.pt data=my_data.yaml")
    out.append("```")
    out.append("")
    return "\n".join(out)


def build_csv(data):
    """机器可读的全指标表。"""
    base = data[0]
    buf = []
    w = buf.append
    w("实验名称,模型配置,权重初始化,patience,实际轮数,最佳epoch,Precision,Recall,mAP50,mAP50-95,"
      "ΔmAP50,ΔmAP50-95,ΔPrecision,ΔRecall,理论参数增量,best.pt体积KB,"
      "末轮epoch,末轮Precision,末轮Recall,末轮mAP50,末轮mAP50-95,数据来源")
    for d in data:
        b, l = d["best"], d["last"]
        bb = base["best"]
        w(",".join([
            d["spec"]["name"], d["args"].get("model", "-"), d["spec"]["init"],
            d["args"].get("patience", "-"), str(len(d["rows"])), str(b["epoch"]),
            fmt(b["p"]), fmt(b["r"]), fmt(b["m50"]), fmt(b["m5095"]),
            fmt(delta(b["m50"], bb["m50"])), fmt(delta(b["m5095"], bb["m5095"])),
            fmt(delta(b["p"], bb["p"])), fmt(delta(b["r"], bb["r"])),
            str(d["spec"]["params_inc"]),
            str(round(d["weight_kb"])) if d["weight_kb"] else "",
            str(l["epoch"]), fmt(l["p"]), fmt(l["r"]), fmt(l["m50"]), fmt(l["m5095"]),
            "实验结果/%s/results.csv" % d["dir"],
        ]))
    return "\n".join(buf)


def main():
    data = load()
    if not data:
        print("[ERROR] 未读到任何 results.csv")
        return
    txt_path = os.path.join(BASE, "ablation_summary_latest.txt")
    md_path = os.path.join(BASE, "ablation_summary_latest.md")
    csv_path = os.path.join(BASE, "ablation_summary_latest.csv")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(build(data) + "\n")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(build_md(data) + "\n")
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        f.write(build_csv(data) + "\n")
    print("已生成:\n  %s\n  %s\n  %s" % (txt_path, md_path, csv_path))


if __name__ == "__main__":
    main()
