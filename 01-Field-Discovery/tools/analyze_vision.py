"""
课题 09 视觉模型实测 · 结果统计与出图
======================================
输入：raw_data/课题09-视觉模型-评分表.csv
输出：photos/chart-09-accuracy.png、photos/chart-09-hallucination.png，并打印统计表

用法：python analyze_vision.py
"""

import csv
import os
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CSV = os.path.join(ROOT, "raw_data", "课题09-视觉模型-评分表.csv")
FIG = os.path.join(ROOT, "photos")
os.makedirs(FIG, exist_ok=True)

MODELS = ["豆包", "通义千问"]
JUDGE_COL = {"豆包": ("豆包_判定", "豆包_幻觉类型"), "通义千问": ("通义千问_判定", "通义千问_幻觉类型")}
CATS = ["对", "部分对", "错"]
HALLU = ["部分漏读", "数字错位", "张冠李戴", "无中生有", "过度推断", "无输出"]


def load():
    with open(CSV, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def normalize(h):
    """把 '张冠李戴+数字错位' 这类复合标签拆开计数"""
    if not h or h == "无":
        return []
    return [p.strip() for p in h.replace("＋", "+").split("+") if p.strip()]


def main():
    rows = load()
    print(f"共 {len(rows)} 条记录\n")

    # ---- 判定统计 ----
    stats = {m: Counter() for m in MODELS}
    for r in rows:
        for m in MODELS:
            stats[m][r[JUDGE_COL[m][0]].strip()] += 1

    print(f"{'模型':<8}{'完全正确':>8}{'部分对':>8}{'错':>6}{'正确率':>10}{'有效识别率':>12}")
    print("-" * 56)
    for m in MODELS:
        ok, part, bad = stats[m]["对"], stats[m]["部分对"], stats[m]["错"]
        n = len(rows)
        print(f"{m:<8}{ok:>8}{part:>8}{bad:>6}{ok / n:>9.1%}{(ok + part) / n:>11.1%}")

    # ---- 幻觉类型统计 ----
    hall = {m: Counter() for m in MODELS}
    for r in rows:
        for m in MODELS:
            for t in normalize(r[JUDGE_COL[m][1]]):
                hall[m][t] += 1

    print(f"\n{'幻觉/失败类型':<14}{'豆包':>8}{'通义千问':>10}")
    print("-" * 34)
    for t in HALLU:
        if hall["豆包"][t] or hall["通义千问"][t]:
            print(f"{t:<14}{hall['豆包'][t]:>8}{hall['通义千问'][t]:>10}")

    # ---- 图 1：判定分布 ----
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(len(CATS))
    w = 0.36
    colors = {"豆包": "#1a73e8", "通义千问": "#e37400"}
    for i, m in enumerate(MODELS):
        vals = [stats[m][c] for c in CATS]
        bars = ax.bar(x + (i - 0.5) * w, vals, w, label=m, color=colors[m])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.15, str(v),
                    ha="center", va="bottom", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(CATS, fontsize=11)
    ax.set_ylabel("张数")
    ax.set_ylim(0, max(max(stats[m][c] for c in CATS) for m in MODELS) * 1.18)
    ax.set_title(f"课题 09　两款视觉模型逐张判定结果（n={len(rows)}）", fontsize=11, loc="left")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    p1 = os.path.join(FIG, "chart-09-accuracy.png")
    fig.tight_layout()
    fig.savefig(p1)
    plt.close(fig)

    # ---- 图 2：幻觉类型分布 ----
    types = [t for t in HALLU if hall["豆包"][t] or hall["通义千问"][t]]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    x = np.arange(len(types))
    for i, m in enumerate(MODELS):
        vals = [hall[m][t] for t in types]
        bars = ax.bar(x + (i - 0.5) * w, vals, w, label=m, color=colors[m])
        for b, v in zip(bars, vals):
            if v:
                ax.text(b.get_x() + b.get_width() / 2, v + 0.1, str(v),
                        ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(types, fontsize=10)
    ax.set_ylabel("出现次数")
    ax.set_title("课题 09　幻觉与失败类型分布（复合标签已拆分计数）", fontsize=11, loc="left")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    p2 = os.path.join(FIG, "chart-09-hallucination.png")
    fig.tight_layout()
    fig.savefig(p2)
    plt.close(fig)

    print(f"\n图1 → {os.path.relpath(p1, ROOT)}")
    print(f"图2 → {os.path.relpath(p2, ROOT)}")

    # ---- 逐条差异清单 ----
    print("\n两模型判定不一致的样本：")
    for r in rows:
        a, b = r["豆包_判定"], r["通义千问_判定"]
        if a != b:
            print(f"  #{r['序号']:>2} {r['场景维度']:<14} 豆包={a:<4} 千问={b:<4} "
                  f"（{r['通义千问_幻觉类型']}）")


if __name__ == "__main__":
    main()
