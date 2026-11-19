"""
项目三 · 声振信号录制脚本
==========================
把采集规范做成程序：固定 44.1kHz / 16bit / 单声道 / 每段 10 秒，自动命名、自动记索引，
避免手录时改名错乱、时长不一、参数漂移。

用法
----
1) 先测电平（必做，确认麦克风有信号）：
     python record_vibration.py --probe

2) 录训练集（每个工况 15 段，共 5 个工况）：
     python record_vibration.py --cond C1 --count 15
     python record_vibration.py --cond C2 --count 15
     python record_vibration.py --cond C3 --count 15
     python record_vibration.py --cond C4 --count 15
     python record_vibration.py --cond C5 --count 15

3) 另找时间录盲测集（每工况 5 段，共 25 段）：
     python record_vibration.py --cond C1 --count 5 --blind
     ...（C2 ~ C5 同理）

参数
----
--cond   工况编号 C1~C5
--count  本工况录制段数（默认 15）
--dur    单段时长秒（默认 10）
--out    输出目录（默认 ../data/wav，--blind 时为 ../data/wav-blind）
--device 录音设备编号（默认 1 = 麦克风阵列，可用 --list 查看）
"""

import argparse
import csv
import datetime as dt
import os
import sys
import time
import wave

import numpy as np
import sounddevice as sd

FS = 44100
CHANNELS = 1
DTYPE = "int16"

CONDITIONS = {
    "C1": ("正常·低档", "电风扇 1 档，无干扰", "1 档"),
    "C2": ("正常·高档", "电风扇 3 档（或最大档）", "3 档"),
    "C3": ("扇叶失衡", "一片扇叶贴约 2g 胶带，保持 3 档", "3 档"),
    "C4": ("进风遮挡", "书本挡住进风口约 1/2，保持 3 档", "3 档"),
    "C5": ("停机静默", "关机，仅录环境本底噪声", "停机"),
}

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)


def probe(device):
    print("=== 电平测试（录 3 秒）===")
    print("请让设备正常运转，手机/电脑麦克风距设备约 20cm...")
    time.sleep(1)
    rec = sd.rec(int(3 * FS), samplerate=FS, channels=CHANNELS, dtype=DTYPE, device=device)
    sd.wait()
    x = rec[:, 0].astype(np.float64) / 32768.0
    rms = float(np.sqrt(np.mean(x ** 2)))
    peak = float(np.max(np.abs(x)))
    dbfs = 20 * np.log10(rms) if rms > 0 else -120
    print(f"  RMS : {rms:.4f}  ({dbfs:.1f} dBFS)")
    print(f"  峰值: {peak:.4f}")
    if peak >= 0.99:
        print("  ⚠️ 峰值已削顶！请把麦克风音量调低或距离拉远，否则数据失真。")
    elif dbfs < -45:
        print("  ⚠️ 信号偏弱，检查麦克风是否选错设备 / 是否被静音。")
    else:
        print("  ✅ 电平正常，可以开始正式录制。")
    return dbfs


def record(cond, count, dur, out_dir, device, blind):
    if cond not in CONDITIONS:
        sys.exit(f"工况必须是 {'/'.join(CONDITIONS)}")
    label, desc, gear = CONDITIONS[cond]
    os.makedirs(out_dir, exist_ok=True)
    index_path = os.path.join(PROJ, "data", "录音索引.csv")
    os.makedirs(os.path.dirname(index_path), exist_ok=True)

    prefix = "blind" if blind else "vib"
    existing = [f for f in os.listdir(out_dir) if f.startswith(f"{prefix}-{cond.lower()}-")]
    start_no = len(existing) + 1

    new_file = not os.path.exists(index_path)
    f_idx = open(index_path, "a", newline="", encoding="utf-8-sig")
    w = csv.writer(f_idx)
    if new_file:
        w.writerow(["序号", "工况", "工况名称", "文件名", "时长_s", "采样率_Hz", "位深",
                    "声道", "档位", "距离_cm", "录制时间", "备注"])

    print(f"\n=== {cond} {label} ===\n    {desc}")
    print(f"    文件名前缀: {prefix}-{cond.lower()}-XX.wav　共 {count} 段，每段 {dur}s\n")
    input("    按回车开始（首次会倒数 3 秒，然后自动连续录完）...")

    for i in range(count):
        no = start_no + i
        name = f"{prefix}-{cond.lower()}-{no:02d}.wav"
        path = os.path.join(out_dir, name)
        print(f"\n  [{i + 1}/{count}] {name}")
        for s in (3, 2, 1):
            print(f"     {s}...", end="", flush=True)
            time.sleep(1)
        print(" ● 录制中", end="", flush=True)
        rec = sd.rec(int(dur * FS), samplerate=FS, channels=CHANNELS, dtype=DTYPE, device=device)
        sd.wait()
        print(" ✓")

        with wave.open(path, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(FS)
            wf.writeframes(rec.tobytes())

        x = rec[:, 0].astype(np.float64) / 32768.0
        rms = float(np.sqrt(np.mean(x ** 2)))
        print(f"     已保存　RMS={rms:.4f}（{20 * np.log10(rms) if rms else -120:.1f} dBFS）")

        w.writerow([f"{prefix}-{cond}-{no:02d}", cond, label, name, dur, FS, "16bit",
                    "mono", gear, "20", dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), ""])
        f_idx.flush()
        if i < count - 1:
            time.sleep(2)

    f_idx.close()
    print(f"\n✅ {cond} 完成 {count} 段 → {out_dir}")
    print("   别忘了：检查一遍本工况的附加工况条件是否与定义一致（档位/配重/遮挡）。")


def main():
    ap = argparse.ArgumentParser(description="项目三 声振信号录制")
    ap.add_argument("--cond", help="工况 C1~C5")
    ap.add_argument("--count", type=int, default=15, help="录制段数")
    ap.add_argument("--dur", type=float, default=10.0, help="单段时长（秒）")
    ap.add_argument("--out", help="输出目录")
    ap.add_argument("--device", type=int, default=1, help="录音设备编号")
    ap.add_argument("--blind", action="store_true", help="录盲测集（输出到 wav-blind）")
    ap.add_argument("--probe", action="store_true", help="只做电平测试")
    ap.add_argument("--list", action="store_true", help="列出音频设备")
    args = ap.parse_args()

    if args.list:
        print(sd.query_devices())
        return
    if args.probe:
        probe(args.device)
        return
    if not args.cond:
        ap.print_help()
        return

    out_dir = args.out or os.path.join(PROJ, "data", "wav-blind" if args.blind else "wav")
    record(args.cond, args.count, args.dur, out_dir, args.device, args.blind)


if __name__ == "__main__":
    main()
