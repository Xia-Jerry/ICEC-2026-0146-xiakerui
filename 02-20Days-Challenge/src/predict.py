"""
项目三 · 单段推理（盲测用）
===========================
对一个未知的 wav 文件判断它属于哪种工况，输出判定结果与投票明细。
演示视频就是录这个脚本跑起来的样子。

用法：
    python predict.py 某个录音.wav
    python predict.py data/wav-blind/        # 整个目录批量，输出逐条结果与总准确率

模型说明：kNN 无需训练权重，"模型"就是参考集本身。
参考集取 data/features.csv（75 段正式样本），特征取贪心筛选后的 3 个：
    rms_db（电平）、entropy（谱熵）、peak（峰值）
这三个是在留一交叉验证里达到 97.3% 的组合，见 report.md 实验 A。
"""

import argparse
import csv
import glob
import os
import sys

import numpy as np

from features import frame_features, load_wav

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
DATA = os.path.join(PROJ, "data")

FEAT_SEL = ["rms_db", "entropy", "peak"]
LABEL = {"C1": "正常·低档", "C2": "正常·高档", "C3": "扇叶失衡",
         "C4": "进风遮挡", "C5": "停机静默"}
K = 3


def load_ref():
    with open(os.path.join(DATA, "features.csv"), encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    X = np.array([[float(r[k]) for k in FEAT_SEL] for r in rows])
    y = np.array([r["cond"] for r in rows])
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd[sd < 1e-9] = 1.0
    return (X - mu) / sd, y, mu, sd


def predict_one(path, Xz, y, mu, sd):
    x, fs = load_wav(path)
    f = frame_features(x, fs)
    v = np.array([f[k] for k in FEAT_SEL])
    vz = (v - mu) / sd
    d = np.sqrt(((Xz - vz) ** 2).sum(axis=1))
    idx = np.argsort(d)[:K]
    votes = {}
    for i in idx:
        w = 1.0 / (d[i] + 1e-6)
        votes[y[i]] = votes.get(y[i], 0.0) + w
    total = sum(votes.values())
    top = max(votes.items(), key=lambda kv: kv[1])
    return top[0], top[1] / total, idx, d, f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target", help="wav 文件或目录")
    args = ap.parse_args()

    Xz, y, mu, sd = load_ref()
    print(f"参考集：{len(y)} 段　特征：{', '.join(FEAT_SEL)}　k={K}\n")

    if os.path.isdir(args.target):
        files = sorted(glob.glob(os.path.join(args.target, "*.wav")))
    else:
        files = [args.target]
    if not files:
        sys.exit("没有找到 wav 文件")

    hit = 0
    for p in files:
        name = os.path.basename(p)
        pred, conf, idx, d, f = predict_one(p, Xz, y, mu, sd)
        truth = None
        for c in LABEL:
            if c.lower() in name.lower():
                truth = c
                break
        mark = ""
        if truth:
            mark = "✅" if pred == truth else "❌"
            hit += int(pred == truth)

        print(f"■ {name}")
        print(f"   电平 {f['rms_db']:.1f} dB　谱熵 {f['entropy']:.3f}　峰值 {f['peak']:.3f}")
        print(f"   判定 → {pred} {LABEL[pred]}　置信度 {conf:.0%}　{mark}")
        nb = "、".join(f"{y[i]}({d[i]:.2f})" for i in idx)
        print(f"   最近邻：{nb}\n")

    if len(files) > 1 and hit:
        print(f"=== 共 {len(files)} 段，命中 {hit} 段，准确率 {hit / len(files):.1%} ===")


if __name__ == "__main__":
    main()
