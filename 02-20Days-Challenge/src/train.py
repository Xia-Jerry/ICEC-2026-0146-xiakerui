"""
项目三 · 分类器训练与评估
==========================
三组实验，回答三个不同的问题：

  实验 A  全部特征 + 留一交叉验证        —— 能不能分？能分到多少？
  实验 B  去掉音量类特征（rms_db/peak）   —— 扣除"音量计效应"后，频谱本身还有多少信息？
  实验 C  叠加真实环境噪声               —— 抗噪能力如何？

实验 B 是关键：如果去掉音量后准确率崩到随机水平，说明模型只是在测响度，
不是在做声振诊断。实验 C 的噪声取自 C5（停机静默）的真实录音，而非白噪声合成。

用法：python train.py
"""

import csv
import glob
import os
import random

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from features import FEATURES, frame_features, load_wav

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
DATA = os.path.join(PROJ, "data")
FIG = os.path.join(PROJ, "photos")
os.makedirs(FIG, exist_ok=True)

CLASSES = ["C1", "C2", "C3", "C4", "C5"]
LABEL = {"C1": "1档", "C2": "3档", "C3": "3档+失衡", "C4": "3档+遮挡", "C5": "停机"}
LEVEL_FEATS = ["rms_db", "peak"]


# ---------------------------------------------------------------- 数据
def load_features():
    with open(os.path.join(DATA, "features.csv"), encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    y = [r["cond"] for r in rows]
    files = [r["file"] for r in rows]
    X = np.array([[float(r[k]) for k in FEATURES] for r in rows])
    return X, np.array(y), files


def zscore_fit(X):
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-9] = 1.0
    return mu, sd


# ---------------------------------------------------------------- kNN
def knn_predict(Xtr, ytr, x, k=3):
    d = np.sqrt(((Xtr - x) ** 2).sum(axis=1))
    idx = np.argsort(d)[:k]
    votes = {}
    for i in idx:
        w = 1.0 / (d[i] + 1e-6)
        votes[ytr[i]] = votes.get(ytr[i], 0) + w
    return max(votes.items(), key=lambda kv: kv[1])[0]


def loo_eval(X, y, k=3):
    """留一交叉验证；标准化参数只用训练折计算，避免信息泄漏"""
    n = len(y)
    pred = []
    for i in range(n):
        m = np.ones(n, dtype=bool)
        m[i] = False
        mu, sd = zscore_fit(X[m])
        Ztr = (X[m] - mu) / sd
        pred.append(knn_predict(Ztr, y[m], (X[i] - mu) / sd, k))
    pred = np.array(pred)
    acc = float((pred == y).mean())
    return pred, acc


def forward_select(X, y, cols, max_feats=6, verbose=True):
    """贪心前向选择：每次加入让 LOO 准确率提升最多的特征，无提升则停。
    75 个样本配 21 维距离会严重稀释，选特征比堆特征更有效。"""
    chosen, best, rest = [], 0.0, list(cols)
    while rest and len(chosen) < max_feats:
        cand_scores = []
        for j in rest:
            _, acc = loo_eval(X[:, chosen + [j]], y)
            cand_scores.append((acc, j))
        acc, j = max(cand_scores)
        if acc <= best + 1e-9:
            break
        chosen.append(j)
        rest.remove(j)
        best = acc
        if verbose:
            print(f"    + {FEATURES[j]:<12} → LOO {acc:.1%}")
    return chosen, best


def confusion(y, pred):
    cm = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    for t, p in zip(y, pred):
        cm[CLASSES.index(t), CLASSES.index(p)] += 1
    return cm


# ---------------------------------------------------------------- 抗噪
def add_noise(x, noise, snr_db):
    """按目标信噪比叠加噪声；噪声从真实 C5 录音中截取"""
    if len(noise) < len(x):
        noise = np.tile(noise, int(np.ceil(len(x) / len(noise))))
    n = noise[: len(x)]
    ps = np.mean(x ** 2)
    pn = np.mean(n ** 2) + 1e-12
    target = ps / (10 ** (snr_db / 10))
    return x + n * np.sqrt(target / pn)


def main():
    X, y, files = load_features()
    print(f"样本 {len(y)} 段，原始特征 {X.shape[1]} 个\n")

    # ---------- 特征筛选 ----------
    print("逐特征贪心筛选（全部特征）")
    cols_all, accA = forward_select(X, y, range(X.shape[1]))
    print(f"→ 选中 {len(cols_all)} 个：{[FEATURES[i] for i in cols_all]}\n")

    predA, accA = loo_eval(X[:, cols_all], y)
    cmA = confusion(y, predA)
    print(f"实验 A　筛选后全部特征　　　LOO 准确率 = {accA:.1%}")

    # ---------- 实验 B：禁止使用音量类特征 ----------
    cols_nl_pool = [i for i, k in enumerate(FEATURES) if k not in LEVEL_FEATS]
    print("\n逐特征贪心筛选（禁用 rms_db / peak）")
    cols_nl, _ = forward_select(X, y, cols_nl_pool)
    print(f"→ 选中 {len(cols_nl)} 个：{[FEATURES[i] for i in cols_nl]}\n")
    predB, accB = loo_eval(X[:, cols_nl], y)
    cmB = confusion(y, predB)
    print(f"实验 B　去音量后筛选　　　　LOO 准确率 = {accB:.1%}")

    # 基线：只用音量
    _, accRMS = loo_eval(X[:, [FEATURES.index("rms_db")]], y)
    # 基线：全特征不做筛选
    _, accAll21 = loo_eval(X, y)
    print(f"对照　　不筛选 21 维全用　　 LOO 准确率 = {accAll21:.1%}")
    print(f"基线　　只用 rms_db　　　　 LOO 准确率 = {accRMS:.1%}\n")

    print(f"{'类别':<8}{'精确率':>8}{'召回率':>8}{'样本':>6}")
    print("-" * 32)
    for i, c in enumerate(CLASSES):
        tp = cmA[i, i]
        prec = tp / max(cmA[:, i].sum(), 1)
        rec = tp / max(cmA[i, :].sum(), 1)
        print(f"{LABEL[c]:<8}{prec:>8.0%}{rec:>8.0%}{cmA[i, :].sum():>6}")

    # ---------- 实验 C：抗噪 ----------
    wavs = sorted(glob.glob(os.path.join(DATA, "wav", "*.wav")))
    c5 = [p for p in wavs if "-c5-" in os.path.basename(p).lower()]
    rng = random.Random(0)
    noise_pool = []
    for p in c5:
        xx, _ = load_wav(p)
        noise_pool.append(xx)
    snrs = [20, 10, 5, 0, -5]
    accs = []
    print(f"\n抗噪：训练用干净样本，测试时叠加真实环境噪声")
    print(f"{'SNR(dB)':>8}{'准确率':>9}")
    print("-" * 18)
    Xsel = X[:, cols_all]
    for snr in snrs:
        preds = []
        for i, p in enumerate(wavs):
            x, fs = load_wav(p)
            n = rng.choice(noise_pool)
            xn = add_noise(x, n, snr)
            feats = frame_features(xn, fs)
            xv = np.array([feats[FEATURES[j]] for j in cols_all])
            m = np.ones(len(y), dtype=bool)
            m[i] = False                      # 留一：不拿自己当邻居
            mu, sd = zscore_fit(Xsel[m])
            preds.append(knn_predict((Xsel[m] - mu) / sd, y[m], (xv - mu) / sd, 3))
        acc = float(np.mean(np.array(preds) == y))
        accs.append(acc)
        print(f"{snr:>8}{acc:>8.1%}")

    # ---------- 出图 ----------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, cm, title, acc in [(axes[0], cmA, "筛选后特征", accA),
                               (axes[1], cmB, "禁用音量特征", accB)]:
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(CLASSES)))
        ax.set_yticks(range(len(CLASSES)))
        ax.set_xticklabels([LABEL[c] for c in CLASSES], fontsize=8, rotation=30)
        ax.set_yticklabels([LABEL[c] for c in CLASSES], fontsize=8)
        ax.set_xlabel("预测")
        ax.set_ylabel("真实")
        ax.set_title(f"{title}　准确率 {acc:.1%}", fontsize=10, loc="left")
        for i in range(len(CLASSES)):
            for j in range(len(CLASSES)):
                if cm[i, j]:
                    ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=9,
                            color="white" if cm[i, j] > cm.max() / 2 else "#202124")
    fig.suptitle("项目三　五工况混淆矩阵（留一交叉验证）", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p1 = os.path.join(FIG, "chart-vib-confusion.png")
    fig.savefig(p1)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(snrs, [a * 100 for a in accs], "o-", color="#1a73e8", lw=2)
    ax.axhline(20, color="#9aa0a6", ls="--", lw=1)
    ax.text(snrs[0], 22, "随机猜测 20%", color="#5f6368", fontsize=8)
    ax.set_xlabel("信噪比 (dB)")
    ax.set_ylabel("准确率 (%)")
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("抗噪性能：叠加 C5 真实环境噪声", fontsize=10, loc="left")
    p2 = os.path.join(FIG, "chart-vib-snr.png")
    fig.tight_layout()
    fig.savefig(p2)
    plt.close(fig)

    print(f"\n图 → {os.path.relpath(p1, PROJ)}")
    print(f"图 → {os.path.relpath(p2, PROJ)}")

    with open(os.path.join(DATA, "eval_summary.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["实验", "准确率"])
        w.writerow(["A_全部特征_LOO", f"{accA:.4f}"])
        w.writerow(["B_去音量特征_LOO", f"{accB:.4f}"])
        w.writerow(["对照_21维不筛选_LOO", f"{accAll21:.4f}"])
        w.writerow(["基线_仅rms_db_LOO", f"{accRMS:.4f}"])
        for snr, a in zip(snrs, accs):
            w.writerow([f"C_抗噪_SNR{snr}dB", f"{a:.4f}"])


if __name__ == "__main__":
    main()
