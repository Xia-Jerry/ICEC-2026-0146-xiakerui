"""
课题 09 照片合规校验
=====================
交图前跑一遍，避免"拍完了才发现没有 EXIF / 主体太小 / 太糊"。

检查项
------
1. EXIF：是否有拍摄时间与机型 —— 缺 EXIF 基本等于无法证明现场实拍
2. 分辨率：长边是否达到原图水平（低于 2000px 多半是被压缩过）
3. 文件名：哈希式文件名（32 位十六进制）通常是微信/QQ 传图的产物
4. 清晰度：拉普拉斯方差，过低提示模糊（模糊可以是难度，但不能糊到无信息）
5. 亮度：平均灰度，提示是否严重欠曝
6. 尺寸一致性：15 张应来自同一部手机

用法
----
    python check_photos.py                    # 检查 ../photos/visual-test/
    python check_photos.py <目录或图片...>     # 检查指定路径（可混合传多张）
"""

import os
import re
import sys

import numpy as np
from PIL import Image, ExifTags

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DIR = os.path.join(os.path.dirname(HERE), "photos", "visual-test")

HASH_NAME = re.compile(r"^[0-9a-f]{16,}\.(jpe?g|png)$", re.I)
IMG_EXTS = (".jpg", ".jpeg", ".png", ".heic", ".webp")


def laplacian_var(gray):
    """拉普拉斯响应的方差，越大越清晰"""
    k = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
    h, w = gray.shape
    if h < 3 or w < 3:
        return 0.0
    out = np.zeros((h - 2, w - 2))
    for i in range(3):
        for j in range(3):
            if k[i, j]:
                out += k[i, j] * gray[i:h - 2 + i, j:w - 2 + j]
    return float(out.var())


def check(path):
    name = os.path.basename(path)
    r = dict(name=name, ok=True, notes=[])

    try:
        im = Image.open(path)
    except Exception as e:
        return dict(name=name, ok=False, notes=[f"无法打开: {e}"], w=0, h=0,
                    kb=0, exif_time="", model="", sharp=0, bright=0)

    w, h = im.size
    r["w"], r["h"] = w, h
    r["kb"] = os.path.getsize(path) / 1024

    ex = im.getexif()
    tags = {ExifTags.TAGS.get(k, k): v for k, v in ex.items()}
    dt = tags.get("DateTimeOriginal") or tags.get("DateTime") or ""
    model = f"{tags.get('Make', '')} {tags.get('Model', '')}".strip()
    r["exif_time"], r["model"] = str(dt), model

    if not ex or not dt:
        r["ok"] = False
        r["notes"].append("无 EXIF 拍摄时间（疑似压缩/截图/传图丢失）")
    if not model:
        r["notes"].append("无机型信息")

    if max(w, h) < 2000:
        r["ok"] = False
        r["notes"].append(f"长边仅 {max(w, h)}px，低于原图水平（≥2000）")

    if HASH_NAME.match(name):
        r["ok"] = False
        r["notes"].append("文件名为哈希式，疑似微信/QQ 传图（非原图）")

    g = np.asarray(im.convert("L").resize((min(w, 800), int(min(w, 800) * h / w))),
                   dtype=np.float64)
    r["sharp"] = laplacian_var(g)
    r["bright"] = float(g.mean())
    if r["sharp"] < 15:
        r["notes"].append(f"清晰度偏低（{r['sharp']:.0f}），可能严重模糊")
    if r["bright"] < 45:
        r["notes"].append(f"画面偏暗（均值 {r['bright']:.0f}）")
    if r["bright"] > 215:
        r["notes"].append(f"画面过曝（均值 {r['bright']:.0f}）")

    return r


def main():
    args = sys.argv[1:]
    paths = []
    if not args:
        if not os.path.isdir(DEFAULT_DIR):
            sys.exit(f"目录不存在: {DEFAULT_DIR}")
        paths = [os.path.join(DEFAULT_DIR, f) for f in sorted(os.listdir(DEFAULT_DIR))
                 if f.lower().endswith(IMG_EXTS) and "selfie" not in f.lower()]
    else:
        for a in args:
            if os.path.isdir(a):
                paths += [os.path.join(a, f) for f in sorted(os.listdir(a))
                          if f.lower().endswith(IMG_EXTS) and "selfie" not in f.lower()]
            else:
                paths.append(a)

    if not paths:
        sys.exit("没有找到待检查的图片")

    results = [check(p) for p in paths]

    print(f"{'文件':<22}{'尺寸':>12}{'KB':>7}{'清晰':>7}{'亮度':>6}  判定")
    print("-" * 92)
    for r in results:
        flag = "✅" if r["ok"] else "❌"
        size = f"{r['w']}x{r['h']}"
        print(f"{r['name'][:21]:<22}{size:>12}{r['kb']:>7.0f}{r['sharp']:>7.0f}"
              f"{r['bright']:>6.0f}  {flag}")
        for n in r["notes"]:
            print(f"    └─ {n}")

    # 一致性
    models = {r["model"] for r in results if r["model"]}
    sizes = {(r["w"], r["h"]) for r in results if r["w"]}
    print("-" * 92)
    print(f"共 {len(results)} 张，通过 {sum(r['ok'] for r in results)} 张，"
          f"不通过 {sum(not r['ok'] for r in results)} 张")
    if len(models) > 1:
        print(f"⚠️ 检出 {len(models)} 种机型，应为同一部手机: {models}")
    if len(sizes) > 1:
        print(f"⚠️ 检出 {len(sizes)} 种分辨率，疑似混入了压缩图或不同设备")
    if len(results) != 15:
        print(f"⚠️ 当前 {len(results)} 张，官方要求正好 15 张")

    times = sorted(r["exif_time"] for r in results if r["exif_time"])
    if times:
        print(f"拍摄时间跨度: {times[0]}  →  {times[-1]}")


if __name__ == "__main__":
    main()
