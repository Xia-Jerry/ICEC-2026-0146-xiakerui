"""
课题 08 数据处理与特征提取脚本
=================================
输入：Phyphox「加速度（含 g）」导出的 4 个原始包（.zip，内含 Raw Data.csv，位于桌面）
输出：raw_data/gait-0X-*.csv  +  photos/chart-08-*.png

切段规则（由本人现场确认，非猜测）：
  1) 17-19-27 静止15秒            → 0 ~ 15 s   全部
  2) 17-22-28 慢走30秒            → 0 ~ 30 s   全部
  3) 17-26-55 慢走30秒+跑步15秒   → 30 ~ 45 s  仅取后半段（前 30 s 与 2) 重复）
  4) 17-30-42 上楼梯+下楼梯        → 0 ~ 15 s（上）；35 ~ 50 s（下）
说明：现场实际采集时长按原计划的 1/2 执行，故各段取原计划时长的一半。

用法：python analyze_gait.py
"""

import csv
import io
import os
import zipfile

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150

DESKTOP = os.path.join(os.path.expanduser("~"), "Desktop")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "raw_data")
FIG = os.path.join(ROOT, "photos")
os.makedirs(RAW, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

# (源名称关键字, 输出文件名, 显示名, [(t0, t1, 段落标签), ...])
SEGMENTS = [
    ("17-19-27", "gait-01-static.csv", "静止站立", [(0.0, 15.0, "static")]),
    ("17-22-28", "gait-02-walk.csv", "平地慢走", [(0.0, 30.0, "walk")]),
    ("17-26-55", "gait-03-run.csv", "快跑", [(30.0, 45.0, "run")]),
    ("17-30-42", "gait-04-stairs.csv", "上下楼梯",
     [(0.0, 15.0, "up"), (35.0, 50.0, "down")]),
]


def load_phyphox(keyword):
    """从桌面找到对应 zip，解析 Raw Data.csv → (path, t, x, y, z, A)"""
    cands = [f for f in os.listdir(DESKTOP)
             if f.endswith(".zip") and keyword in f and "加速度" in f]
    if not cands:
        raise FileNotFoundError(f"桌面上找不到含 {keyword} 的加速度压缩包")
    path = os.path.join(DESKTOP, sorted(cands)[0])
    with zipfile.ZipFile(path) as zf:
        text = zf.read("Raw Data.csv").decode("utf-8-sig")
    rows = list(csv.reader(io.StringIO(text)))[1:]
    arr = np.array([[float(v) for v in r[:5]] for r in rows if len(r) >= 5])
    return path, arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4]


def step_freq(t, a, lo=0.8, hi=4.5):
    """在 lo~hi Hz 内用 FFT 找主频（去均值 + 汉宁窗）"""
    x = a - a.mean()
    n = len(x)
    if n < 64:
        return float("nan")
    fs = 1.0 / np.mean(np.diff(t))
    spec = np.abs(np.fft.rfft(x * np.hanning(n)))
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    m = (freqs >= lo) & (freqs <= hi)
    return float(freqs[m][np.argmax(spec[m])])


def window_std(t, a, win=2.0, step=0.5):
    """滑动窗口标准差分布——检验阈值规则在真实判定时的稳定性"""
    fs = 1.0 / np.mean(np.diff(t))
    n = int(win * fs)
    hop = max(1, int(step * fs))
    if n >= len(a):
        return np.array([a.std(ddof=1)])
    return np.array([a[i:i + n].std(ddof=1) for i in range(0, len(a) - n + 1, hop)])


def main():
    results, curves = [], []

    for keyword, out_name, label, ranges in SEGMENTS:
        src, t, x, y, z, a = load_phyphox(keyword)
        print(f"\n[{label}] 源: {os.path.basename(src)}　0~{t[-1]:.2f}s，{len(t)} 帧")

        dt = 1.0 / (len(t) / t[-1])          # 平均采样间隔
        seg_t, seg_x, seg_y, seg_z, seg_a, seg_tag = [], [], [], [], [], []
        cursor = 0.0
        for t0, t1, tag in ranges:
            m = (t >= t0) & (t <= t1)
            print(f"  取 {tag:5s}: {t0:>4.0f}~{t1:<4.0f}s → {int(m.sum()):>5d} 帧 "
                  f"({t[m][0]:.2f}~{t[m][-1]:.2f}s)")
            seg_t.append(t[m] - t[m][0] + cursor)
            cursor = seg_t[-1][-1] + dt
            seg_x.append(x[m]); seg_y.append(y[m]); seg_z.append(z[m]); seg_a.append(a[m])
            seg_tag += [tag] * int(m.sum())

        st = np.concatenate(seg_t)
        sa = np.concatenate(seg_a)
        sx, sy, sz = np.concatenate(seg_x), np.concatenate(seg_y), np.concatenate(seg_z)

        with open(os.path.join(RAW, out_name), "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["Time (s)", "Acceleration x (m/s^2)", "Acceleration y (m/s^2)",
                        "Acceleration z (m/s^2)", "Absolute acceleration (m/s^2)", "segment"])
            for i in range(len(st)):
                w.writerow([f"{st[i]:.6f}", f"{sx[i]:.6E}", f"{sy[i]:.6E}",
                            f"{sz[i]:.6E}", f"{sa[i]:.6E}", seg_tag[i]])
        print(f"  → 写出 {out_name}：{len(st)} 帧，{st[-1]:.2f}s")

        r = dict(label=label, file=out_name, n=len(st), dur=float(st[-1]),
                 fs=len(st) / st[-1], mean=float(sa.mean()), std=float(sa.std(ddof=1)),
                 peak=float(sa.max()), trough=float(sa.min()),
                 p95=float(np.percentile(sa, 95)), freq=step_freq(st, sa),
                 wstd=window_std(st, sa),
                 src=os.path.basename(src),
                 ranges="；".join(f"{a_}~{b_}s({c})" for a_, b_, c in ranges))
        results.append(r)
        curves.append((label, st, sa))

    print("\n" + "=" * 92)
    print(f"{'状态':<10}{'帧数':>8}{'时长s':>8}{'采样Hz':>8}{'均值':>8}{'标准差':>8}"
          f"{'峰值':>8}{'谷值':>8}{'P95':>8}{'主频Hz':>8}")
    for r in results:
        print(f"{r['label']:<10}{r['n']:>8}{r['dur']:>8.2f}{r['fs']:>8.0f}"
              f"{r['mean']:>8.2f}{r['std']:>8.2f}{r['peak']:>8.2f}{r['trough']:>8.2f}"
              f"{r['p95']:>8.2f}{r['freq']:>8.2f}")

    # 图 1：四态波形
    colors = ["#1a73e8", "#188038", "#d93025", "#e37400"]
    fig, axes = plt.subplots(4, 1, figsize=(11, 9))
    for ax, r, (label, st, sa), c in zip(axes, results, curves, colors):
        ax.plot(st, sa, color=c, lw=0.6)
        ax.axhline(9.81, color="#9aa0a6", ls="--", lw=0.9)
        ax.text(st[-1], 9.81, " 9.81", va="bottom", fontsize=7, color="#5f6368")
        ax.set_ylabel("A (m/s²)", fontsize=9)
        ax.set_title(f"{label}　std={r['std']:.2f}　峰值={r['peak']:.1f}　"
                     f"主频={r['freq']:.2f} Hz", fontsize=10, loc="left")
        ax.grid(alpha=0.25, lw=0.5)
        ax.set_xlim(0, st[-1])
    axes[-1].set_xlabel("时间 (s)")
    fig.suptitle("课题 08　四种运动状态的合加速度 A=√(x²+y²+z²)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    p1 = os.path.join(FIG, "chart-08-gait-waveform.png")
    fig.savefig(p1); plt.close(fig)

    # 图 2：特征对比（静止状态无周期运动，主频为噪声，标灰并在图内注明）
    labels = [r["label"] for r in results]
    xpos = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].bar(xpos - 0.2, [r["std"] for r in results], 0.4, label="标准差", color="#1a73e8")
    axes[0].bar(xpos + 0.2, [r["peak"] - r["trough"] for r in results], 0.4,
                label="动态范围(峰-谷)", color="#e37400")
    axes[0].set_xticks(xpos); axes[0].set_xticklabels(labels, fontsize=9)
    axes[0].set_ylabel("m/s²"); axes[0].set_title("波动强度对比", fontsize=10, loc="left")
    axes[0].legend(fontsize=8); axes[0].grid(axis="y", alpha=0.25)

    fcolors = ["#9aa0a6", "#188038", "#d93025", "#e37400"]
    axes[1].bar(xpos, [r["freq"] for r in results], color=fcolors)
    axes[1].set_xticks(xpos); axes[1].set_xticklabels(labels, fontsize=9)
    axes[1].set_ylabel("Hz")
    axes[1].set_title("主频对比（0.8–4.5 Hz 频段）", fontsize=10, loc="left")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].annotate("静止状态无周期性运动\n该值为噪声峰，不具物理意义",
                     xy=(0, results[0]["freq"]), xytext=(0.55, results[0]["freq"] * 0.92),
                     fontsize=7.5, color="#5f6368",
                     arrowprops=dict(arrowstyle="->", color="#9aa0a6", lw=0.8))
    fig.tight_layout()
    p2 = os.path.join(FIG, "chart-08-features.png")
    fig.savefig(p2); plt.close(fig)

    print(f"\n图1 → {os.path.relpath(p1, ROOT)}")
    print(f"图2 → {os.path.relpath(p2, ROOT)}")

    print("\n| 状态 | 均值 (m/s²) | 标准差 | 峰值 | 谷值 | P95 | 主频 (Hz) | 采样率 (Hz) |")
    print("|---|---|---|---|---|---|---|---|")
    for r in results:
        print(f"| {r['label']} | {r['mean']:.2f} | {r['std']:.2f} | {r['peak']:.2f} | "
              f"{r['trough']:.2f} | {r['p95']:.2f} | {r['freq']:.2f} | {r['fs']:.0f} |")

    # 2 秒滑窗 std 分布：阈值规则的稳定性依据
    print("\n2 秒滑窗标准差分布（窗口重叠 0.5 s）——用于确定阈值区间")
    print(f"{'状态':<10}{'窗数':>6}{'P5':>8}{'中位数':>8}{'P95':>8}{'最小':>8}{'最大':>8}")
    for r in results:
        w = r["wstd"]
        print(f"{r['label']:<10}{len(w):>6}{np.percentile(w, 5):>8.2f}{np.median(w):>8.2f}"
              f"{np.percentile(w, 95):>8.2f}{w.min():>8.2f}{w.max():>8.2f}")

    # 供后续填写索引表
    with open(os.path.join(HERE, "_extract_summary.csv"), "w", newline="",
              encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["序号", "状态", "原始数据文件名", "采样时长_s", "采样率_Hz",
                    "手机型号", "采集地点", "同框照片文件名", "备注"])
        for i, r in enumerate(results, 1):
            w.writerow([i, r["label"], r["file"], f"{r['dur']:.1f}", f"{r['fs']:.0f}",
                        "", "", "", f"源 {r['src']}；取 {r['ranges']}"])


if __name__ == "__main__":
    main()
