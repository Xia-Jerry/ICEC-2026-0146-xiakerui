"""
项目三 · 盲测与少样本重标定评估
================================
盲测集（data/wav-blind，25 段）是在**另一时间、另一位置**另录的，用来检验跨场次泛化。

三组结果：
  1. 零样本：只用 75 段训练集当参考，直接判盲测集 —— 真实部署时的最坏情况
  2. 少样本重标定：每类只加 1 段现场样本进参考集，判剩余 4 段 —— 工程上标准的现场校准
  3. 全量重标定：每类加 4 段，判剩下 1 段 —— 上限参考

每种都用两组特征分别跑：
  level3（rms_db/entropy/peak，含电平） vs shape4（kurtosis/bandwidth/zcr/band_20_100，与电平无关）

用法：python eval_blind.py
"""

import csv
import itertools
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
DATA = os.path.join(PROJ, "data")

FEAT_SETS = {
    "level3": ["rms_db", "entropy", "peak"],
    "shape4": ["kurtosis", "bandwidth", "zcr", "band_20_100"],
}
CLASSES = ["C1", "C2", "C3", "C4", "C5"]


def load(path, feats):
    with open(path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    X = np.array([[float(r[k]) for k in feats] for r in rows])
    y = np.array([r["cond"] for r in rows])
    return X, y


def knn(Xtr, ytr, x, k=3):
    d = np.sqrt(((Xtr - x) ** 2).sum(axis=1))
    idx = np.argsort(d)[:k]
    votes = {}
    for i in idx:
        w = 1.0 / (d[i] + 1e-6)
        votes[ytr[i]] = votes.get(ytr[i], 0.0) + w
    return max(votes.items(), key=lambda kv: kv[1])[0]


def evaluate(Xtr, ytr, Xte, yte, feats):
    """标准化只用参考集统计量"""
    mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0)
    sd[sd < 1e-9] = 1.0
    Ztr, Zte = (Xtr - mu) / sd, (Xte - mu) / sd
    pred = np.array([knn(Ztr, ytr, z) for z in Zte])
    return float((pred == yte).mean()), pred


def main():
    print(f"{'特征集':<9}{'零样本':>10}{'每类+1段':>12}{'每类+4段':>12}")
    print("-" * 45)
    out = []
    for fname, feats in FEAT_SETS.items():
        Xtr, ytr = load(os.path.join(DATA, "features.csv"), feats)
        Xb, yb = load(os.path.join(DATA, "_blind_feat.csv"), feats)
        idx = {c: np.where(yb == c)[0] for c in CLASSES}

        # 1) 零样本
        acc0, _ = evaluate(Xtr, ytr, Xb, yb, feats)

        # 2) 每类 +1 段标定，5 折轮换（每折用不同的那 1 段做标定）
        accs1 = []
        for k in range(5):
            cal = np.concatenate([[idx[c][k]] for c in CLASSES])
            te = np.concatenate([[j for j in idx[c] if j != idx[c][k]] for c in CLASSES])
            a, _ = evaluate(np.vstack([Xtr, Xb[cal]]), np.concatenate([ytr, yb[cal]]),
                            Xb[te], yb[te], feats)
            accs1.append(a)
        acc1 = float(np.mean(accs1))

        # 3) 每类 +4 段标定（上限参考）
        accs4 = []
        for k in range(5):
            te = np.concatenate([[idx[c][k]] for c in CLASSES])
            cal = np.concatenate([[j for j in idx[c] if j != idx[c][k]] for c in CLASSES])
            a, _ = evaluate(np.vstack([Xtr, Xb[cal]]), np.concatenate([ytr, yb[cal]]),
                            Xb[te], yb[te], feats)
            accs4.append(a)
        acc4 = float(np.mean(accs4))

        print(f"{fname:<9}{acc0:>9.1%}{acc1:>12.1%}{acc4:>12.1%}")
        out.append([fname, acc0, acc1, acc4])

    with open(os.path.join(DATA, "blind_summary.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["特征集", "零样本", "每类加1段标定", "每类加4段标定"])
        for r in out:
            w.writerow([r[0], f"{r[1]:.4f}", f"{r[2]:.4f}", f"{r[3]:.4f}"])

    # ---- 出图：同场次 / 跨场次 / 标定 三档对比 ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 150

    with open(os.path.join(DATA, "eval_summary.csv"), encoding="utf-8-sig") as f:
        loo = {r["实验"]: float(r["准确率"]) for r in csv.DictReader(f)}
    same = {"level3": loo.get("A_全部特征_LOO", 0), "shape4": loo.get("B_去音量特征_LOO", 0)}

    scen = ["同场次\n留一交叉验证", "跨场次\n零样本", "跨场次\n每类+1段标定", "跨场次\n每类+4段标定"]
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    x = np.arange(len(scen)); w = 0.36
    for i, (r, col) in enumerate(zip(out, ["#1a73e8", "#e37400"])):
        vals = [same[r[0]] * 100, r[1] * 100, r[2] * 100, r[3] * 100]
        bars = ax.bar(x + (i - 0.5) * w, vals, w,
                      label=f"{r[0]}（{'含电平' if r[0]=='level3' else '与电平无关'}）", color=col)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f}%",
                    ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks(x); ax.set_xticklabels(scen, fontsize=9)
    ax.set_ylabel("准确率 (%)"); ax.set_ylim(0, 115)
    ax.axhline(20, color="#9aa0a6", ls="--", lw=1)
    ax.text(-0.45, 23, "随机猜测 20%", fontsize=7.5, color="#5f6368", ha="left")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("项目三　泛化能力：同场次 vs 跨场次（盲测）", fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(PROJ, "photos", "chart-vib-generalization.png"))
    print("图 → photos/chart-vib-generalization.png")

    print("\n注：零样本 = 直接部署的最坏情况；每类 +1 段 = 现场校准一次即可达到的水平。")


if __name__ == "__main__":
    main()
