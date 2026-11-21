"""
项目三 · 声振信号特征提取
==========================
输入：data/wav/vib-cX-NN.wav（10 s，44.1 kHz，16 bit，单声道）
输出：data/features.csv（每段一行，20 个特征）

特征分两组，各自的作用不同：
  时域   —— 反映能量与冲击性，对"有无运行""有无遮挡"敏感
  频域   —— 反映转速与谐波结构，对"档位""失衡"敏感

其中「包络调制」是识别扇叶失衡的关键：叶片失衡会让振动幅值以转速频率被调制，
表现为包络谱上出现一根独立的低频峰，与基频无关。

用法：
    python features.py                       # 处理 data/wav
    python features.py --dir data/wav --out data/features.csv
"""

import argparse
import csv
import glob
import os
import wave

import numpy as np

FS_EXPECT = 44100


# ---------------------------------------------------------------- 基础读取
def load_wav(path):
    with wave.open(path, "rb") as w:
        fs = w.getframerate()
        ch = w.getnchannels()
        raw = w.readframes(w.getnframes())
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32768.0
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    return x, fs


# ---------------------------------------------------------------- 特征
def _psd(x, fs):
    n = 1 << int(np.ceil(np.log2(len(x))))
    w = np.hanning(len(x))
    X = np.abs(np.fft.rfft(x * w, n=n))
    fr = np.fft.rfftfreq(n, 1.0 / fs)
    return fr, X ** 2


def _f0_peak(x, fs, lo=20.0, hi=200.0):
    """取 20~200 Hz 内 PSD 最强的谱线并做抛物线插值，即叶片通过频率。

    不用谐波积谱：实测发现风扇的 2 倍频常强于基频，积谱会误锁到次谐波上。
    """
    fr, P = _psd(x, fs)
    m = (fr >= lo) & (fr <= hi)
    idx = np.where(m)[0]
    if len(idx) == 0:
        return 0.0
    k = idx[np.argmax(P[idx])]
    if 0 < k < len(P) - 1:
        a, b, c = P[k - 1], P[k], P[k + 1]
        den = a - 2 * b + c
        delta = 0.5 * (a - c) / den if abs(den) > 1e-20 else 0.0
    else:
        delta = 0.0
    return float(fr[k] + delta * (fr[1] - fr[0]))


def _envelope(x, fs, cutoff=40.0):
    """整流 + 滑动平均得到幅值包络（等价于低通），再返回包络谱"""
    win = max(3, int(fs / cutoff))
    env = np.convolve(np.abs(x), np.ones(win) / win, mode="same")
    env = env - env.mean()
    n = 1 << int(np.ceil(np.log2(len(env))))
    E = np.abs(np.fft.rfft(env * np.hanning(len(env)), n=n))
    fr = np.fft.rfftfreq(n, 1.0 / fs)
    return fr, E


def frame_features(x, fs):
    f = {}
    eps = 1e-12

    # ---------------- 时域 ----------------
    rms = float(np.sqrt(np.mean(x ** 2)))
    peak = float(np.max(np.abs(x)))
    f["rms_db"] = 20 * np.log10(rms + eps)
    f["peak"] = peak
    f["crest"] = peak / (rms + eps)
    f["zcr"] = float(np.mean(np.abs(np.diff(np.sign(x))) > 0)) / 2
    mu, sd = float(np.mean(x)), float(np.std(x))
    f["kurtosis"] = float(np.mean((x - mu) ** 4) / (sd ** 4 + eps))
    f["skew"] = float(np.mean((x - mu) ** 3) / (sd ** 3 + eps))

    # ---------------- 包络调制（失衡特征）----------------
    efr, E = _envelope(x, fs)
    m = (efr >= 1.0) & (efr <= 30.0)
    if m.any():
        pk = E[m].max()
        f["mod_depth"] = float(pk / (E.mean() + eps))     # 调制深度
        f["mod_freq"] = float(efr[m][np.argmax(E[m])])    # 调制频率
    else:
        f["mod_depth"], f["mod_freq"] = 0.0, 0.0

    # ---------------- 频域 ----------------
    fr, P = _psd(x, fs)
    total = float(P[(fr > 10)].sum() + eps)

    f0 = _f0_peak(x, fs)
    f["f0"] = f0
    if f0 > 0:
        b = (fr >= f0 * 0.92) & (fr <= f0 * 1.08)
        f["f0_share"] = float(P[b].sum() / total)
        harm = 0.0
        for k in range(2, 6):
            hb = (fr >= f0 * k * 0.94) & (fr <= f0 * k * 1.06)
            harm += float(P[hb].sum())
        f["harm_ratio"] = float(harm / total)
        # 边带：f0 ± (2~12 Hz) 内能量 / f0 能量，失衡会在这里抬升
        sb = 0.0
        for d in (2, 4, 6, 8, 10, 12):
            for s in (-1, 1):
                c = f0 + s * d
                if 10 < c < fr[-1]:
                    w_ = (fr >= c - 1.0) & (fr <= c + 1.0)
                    sb += float(P[w_].sum())
        f["sideband"] = float(sb / (P[(fr >= f0 * 0.92) & (fr <= f0 * 1.08)].sum() + eps))
    else:
        f["f0_share"] = f["harm_ratio"] = f["sideband"] = 0.0

    f["centroid"] = float((fr * P).sum() / (P.sum() + eps))
    f["bandwidth"] = float(np.sqrt((((fr - f["centroid"]) ** 2) * P).sum() / (P.sum() + eps)))
    cs = np.cumsum(P)
    f["rolloff85"] = float(fr[np.searchsorted(cs, 0.85 * cs[-1])]) if cs[-1] > 0 else 0.0
    pn = P / (P.sum() + eps)
    f["flatness"] = float(np.exp(np.mean(np.log(pn + 1e-20))) / (np.mean(pn) + 1e-20))
    pnn = pn[pn > 0]
    f["entropy"] = float(-(pnn * np.log(pnn)).sum() / np.log(len(pnn) + 1))

    for name, lo, hi in [("band_20_100", 20, 100), ("band_100_500", 100, 500),
                         ("band_500_2k", 500, 2000), ("band_2k_up", 2000, 20000)]:
        b = (fr >= lo) & (fr < hi)
        f[name] = float(P[b].sum() / total)

    return f


FEATURES = ["rms_db", "peak", "crest", "zcr", "kurtosis", "skew",
            "mod_depth", "mod_freq", "f0", "f0_share", "harm_ratio", "sideband",
            "centroid", "bandwidth", "rolloff85", "flatness", "entropy",
            "band_20_100", "band_100_500", "band_500_2k", "band_2k_up"]


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    proj = os.path.dirname(here)
    ap.add_argument("--dir", default=os.path.join(proj, "data", "wav"))
    ap.add_argument("--out", default=os.path.join(proj, "data", "features.csv"))
    ap.add_argument("--normalize", action="store_true",
                    help="按每段自身 RMS 归一化后再提特征，消除录音距离/增益差异")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.dir, "*.wav")))
    if not files:
        raise SystemExit(f"没有找到 wav: {args.dir}")

    rows = []
    for p in files:
        name = os.path.basename(p)
        cond = name.split("-")[1].upper()
        x, fs = load_wav(p)
        if args.normalize:
            r = np.sqrt(np.mean(x ** 2)) + 1e-9
            x = x / r * 0.05
        if fs != FS_EXPECT:
            print(f"⚠️ {name} 采样率 {fs}，与预期 {FS_EXPECT} 不符")
        feats = frame_features(x, fs)
        rows.append({"file": name, "cond": cond, **{k: feats[k] for k in FEATURES}})

    with open(args.out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["file", "cond"] + FEATURES)
        w.writeheader()
        w.writerows(rows)

    print(f"✅ 已提取 {len(rows)} 段 × {len(FEATURES)} 个特征 → {args.out}\n")

    # 按工况打印均值，便于快速检查可分性
    import collections
    grp = collections.defaultdict(list)
    for r in rows:
        grp[r["cond"]].append(r)
    show = ["rms_db", "f0", "harm_ratio", "sideband", "mod_depth", "mod_freq",
            "centroid", "band_20_100", "entropy"]
    print(f"{'工况':<6}" + "".join(f"{k:>12}" for k in show))
    print("-" * (6 + 12 * len(show)))
    for c in sorted(grp):
        vals = [np.mean([r[k] for r in grp[c]]) for k in show]
        print(f"{c:<6}" + "".join(f"{v:>12.3f}" for v in vals))


if __name__ == "__main__":
    main()
