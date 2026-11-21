"""
项目三 · 手机长录音导入与自动切段
==================================
手机录音机每次录一条长音频（建议 3 分钟），本脚本负责：
  1) 用 ffmpeg 统一转成 44.1kHz / 16bit / 单声道 WAV
  2) 跳过开头 N 秒（默认 5s，留给开机与按键稳定）
  3) 按 10 秒切成 15 段，命名为 vib-cX-NN.wav
  4) 追加写入 data/录音索引.csv

这样采集端只需按 5 次录音键，不必手点 75 次。

用法
----
    python import_phone_audio.py                      # 默认读 桌面/风扇录音
    python import_phone_audio.py --src "D:/录音"       # 指定目录
    python import_phone_audio.py --skip 5 --seg 10 --count 15

文件名要求：手机录音的文件名需含 C1~C5（或 1~5 档位标识），否则按文件顺序映射。
"""

import argparse
import csv
import datetime as dt
import os
import re
import subprocess
import sys
import wave

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
OUT_DIR = os.path.join(PROJ, "data", "wav")
INDEX = os.path.join(PROJ, "data", "录音索引.csv")

COND = {
    "C1": ("正常·低档", "1 档"),
    "C2": ("正常·高档", "3 档"),
    "C3": ("扇叶失衡", "3 档"),
    "C4": ("进风遮挡", "3 档"),
    "C5": ("停机静默", "停机"),
}
AUDIO_EXT = (".m4a", ".mp3", ".wav", ".aac", ".amr", ".flac", ".ogg", ".opus", ".3gp")


def ffmpeg_exe():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def to_wav(src, dst):
    """转码为 44.1kHz/16bit/mono，返回 dst"""
    cmd = [ffmpeg_exe(), "-y", "-loglevel", "error", "-i", src,
           "-ac", "1", "-ar", "44100", "-acodec", "pcm_s16le", dst]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"转码失败 {os.path.basename(src)}: {r.stderr.strip()[:200]}")
    return dst


def read_wav(path):
    with wave.open(path, "rb") as w:
        fs = w.getframerate()
        n = w.getnframes()
        ch = w.getnchannels()
        raw = w.readframes(n)
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32768.0
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    return x, fs


def write_wav(path, x, fs):
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(fs)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes())


def guess_cond(fname):
    m = re.search(r"c([1-5])", fname, re.I)
    if m:
        return f"C{m.group(1)}"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(os.path.expanduser("~"), "Desktop", "风扇录音"))
    ap.add_argument("--skip", type=float, default=5.0, help="跳过开头秒数")
    ap.add_argument("--seg", type=float, default=10.0, help="每段秒数")
    ap.add_argument("--count", type=int, default=15, help="每个工况切几段")
    ap.add_argument("--out-dir", default=None, help="输出目录（默认 data/wav）")
    ap.add_argument("--index", default=None, help="索引 CSV 路径（默认 data/录音索引.csv）")
    ap.add_argument("--prefix", default="vib", help="输出文件名前缀（默认 vib）")
    ap.add_argument("--map", default=None,
                    help='显式映射，如 "C1=一档,C2=三档,C3=三档+贴纸,C4=三档+贴纸+遮挡,C5=空"')
    args = ap.parse_args()

    if not os.path.isdir(args.src):
        sys.exit(f"源目录不存在: {args.src}")

    files = [f for f in sorted(os.listdir(args.src))
             if f.lower().endswith(AUDIO_EXT)]
    if not files:
        sys.exit(f"目录里没有音频文件: {args.src}")

    # 按 C1~C5 归类：优先 --map 显式映射，其次文件名含 Cx，最后按顺序补位
    groups = {c: None for c in COND}
    unmapped = []
    if args.map:
        stem2path = {os.path.splitext(f)[0]: os.path.join(args.src, f) for f in files}
        for pair in args.map.split(","):
            k, _, v = pair.partition("=")
            k, v = k.strip().upper(), v.strip()
            if k in groups and v in stem2path:
                groups[k] = stem2path[v]
            else:
                print(f"⚠️ 映射未命中: {k}={v}")
        files = []          # 已全部通过映射处理
    for f in files:
        c = guess_cond(f)
        if c and groups.get(c) is None:
            groups[c] = os.path.join(args.src, f)
        else:
            unmapped.append(os.path.join(args.src, f))
    if unmapped:
        print(f"⚠️ {len(unmapped)} 个文件无法从文件名识别工况，将按顺序补位：")
        for p in unmapped:
            print("   ", os.path.basename(p))
        for c in COND:
            if groups[c] is None and unmapped:
                groups[c] = unmapped.pop(0)

    missing = [c for c in COND if not groups[c]]
    if missing:
        sys.exit(f"缺少工况文件: {'/'.join(missing)}")

    out_dir = args.out_dir or OUT_DIR
    index_path = args.index or INDEX
    os.makedirs(out_dir, exist_ok=True)
    tmp_dir = os.path.join(PROJ, "data", "_tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    new_file = not os.path.exists(index_path)
    fi = open(index_path, "a", newline="", encoding="utf-8-sig")
    w = csv.writer(fi)
    if new_file:
        w.writerow(["序号", "工况", "工况名称", "文件名", "时长_s", "采样率_Hz", "位深",
                    "声道", "档位", "距离_cm", "录制时间", "备注"])

    print(f"{'工况':<6}{'源文件':<28}{'总长':>8}{'切出':>6}{'有效RMS':>10}{'峰值':>8}")
    print("-" * 74)
    for c, src in groups.items():
        label, gear = COND[c]
        wav = os.path.join(tmp_dir, f"{c}.wav")
        to_wav(src, wav)
        x, fs = read_wav(wav)
        dur = len(x) / fs
        start = int(args.skip * fs)
        need = int(args.count * args.seg * fs)
        if len(x) - start < need:
            print(f"⚠️ {c} 时长不足：可用 {len(x)/fs:.0f}s，需要 {args.skip + args.count * args.seg:.0f}s")

        made = 0
        for i in range(args.count):
            a = start + int(i * args.seg * fs)
            b = a + int(args.seg * fs)
            if b > len(x):
                break
            seg = x[a:b]
            name = f"{args.prefix}-{c.lower()}-{i + 1:02d}.wav"
            write_wav(os.path.join(out_dir, name), seg, fs)
            rms = float(np.sqrt(np.mean(seg ** 2)))
            w.writerow([f"{args.prefix}-{c}-{i + 1:02d}", c, label, name, args.seg, fs, "16bit",
                        "mono", gear, "25", dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        f"源 {os.path.basename(src)}；第 {i + 1} 段 {i * args.seg:.0f}~{(i + 1) * args.seg:.0f}s"])
            made += 1

        seg_all = x[start:start + need]
        rms = float(np.sqrt(np.mean(seg_all ** 2))) if len(seg_all) else 0
        peak = float(np.max(np.abs(seg_all))) if len(seg_all) else 0
        db = 20 * np.log10(rms) if rms > 0 else -120
        print(f"{c:<6}{os.path.basename(src)[:27]:<28}{dur:>7.0f}s{made:>6}{db:>9.1f}dB{peak:>8.2f}")
        if peak > 0.98:
            print(f"   ⚠️ {c} 有削顶，建议下次拉远或降低录音增益")

    fi.close()
    for f in os.listdir(tmp_dir):
        os.remove(os.path.join(tmp_dir, f))
    os.rmdir(tmp_dir)

    def _show(p_):
        try:
            return os.path.relpath(p_, PROJ)
        except ValueError:
            return p_
    print(f"\n✅ 共切出 {args.count * len(COND)} 段 → {_show(out_dir)}")
    print(f"✅ 索引已写入 {_show(index_path)}")


if __name__ == "__main__":
    main()
