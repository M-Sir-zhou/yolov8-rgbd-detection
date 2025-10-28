# python D:\ProjectCode\PyCharm\ultralytics-main\scripts\fuse_rgb_depth.py --rgb_dir "D:/ProjectCode/PyCharm/ultralytics-main/datasets/tennis-rgbd/val/rgb" --depth_dir "D:/ProjectCode/PyCharm/ultralytics-main/datasets/tennis-rgbd/val/depth" --out_dir "D:/ProjectCode/PyCharm/ultralytics-main/datasets/tennis-yolo/images/val" --depth_type uint8 --mode sorted  批量融合 rgb + depth 为 4 通道 png
# python.exe d:/ProjectCode/PyCharm/ultralytics-main/scripts/fuse_rgb_depth.py --rgb_dir "D:\ProjectCode\PyCharm\ultralytics-main\datasets\tennis_path\Color" --depth_dir "D:\ProjectCode\PyCharm\ultralytics-main\datasets\tennis_path\Depth" --out_dir "D:\ProjectCode\PyCharm\ultralytics-main\datasets\path_out" --depth_type uint16 --mode name  单张融合 rgb + depth 为 4 通道 png
import argparse
from pathlib import Path

import cv2
import numpy as np


def read_gray(path):
    im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if im is None:
        raise FileNotFoundError(path)
    if im.ndim == 3:
        # 转为单通道灰度（取第0通道或转换）
        if im.shape[2] == 4:
            im = im[:, :, 3]  # 若第四通道为深度/alpha，优先使用
        else:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    return im


def fuse_pair(rgb_path, depth_path, out_path, depth_type="uint16"):
    rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)  # BGR, uint8
    if rgb is None:
        raise FileNotFoundError(rgb_path)
    depth = read_gray(depth_path)

    # 规范深度为单通道
    if depth.dtype == np.uint16:
        d = depth.astype(np.uint16)
    elif depth.dtype == np.uint8:
        d = depth.astype(np.uint8)
    else:
        # float -> 0..1 -> scale to uint16
        d = depth.astype(np.float32)
        d = np.clip(d, 0.0, 1.0)
        d = (d * 65535.0).astype(np.uint16)

    H, W = rgb.shape[:2]
    if (d.shape[0], d.shape[1]) != (H, W):
        d = cv2.resize(d, (W, H), interpolation=cv2.INTER_NEAREST)

    # 根据目标 depth_type 构造 4 通道数组并保存
    out_dir = Path(out_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    if depth_type == "uint8":
        if d.dtype != np.uint8:
            # 将深度归一化到 0..255
            d8 = cv2.normalize(d, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        else:
            d8 = d
        bgra = np.dstack([rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2], d8])
        cv2.imwrite(str(out_path), bgra)
    else:  # uint16
        # 将 rgb 扩展为 uint16（0..255 -> 0..65535）以保证整张图为同一 dtype
        rgb16 = rgb.astype(np.uint16) * 257  # 255->65535
        if d.dtype == np.uint8:
            d16 = d.astype(np.uint16) * 257
        else:
            d16 = d.astype(np.uint16)
        out4 = np.dstack([rgb16[:, :, 0], rgb16[:, :, 1], rgb16[:, :, 2], d16])
        cv2.imwrite(str(out_path), out4)


def pair_by_name(rgb_dir, depth_dir):
    rgb_files = sorted([p for p in Path(rgb_dir).iterdir() if p.is_file()])
    depth_files = sorted([p for p in Path(depth_dir).iterdir() if p.is_file()])

    # 优先按同名文件（stem）匹配
    rgb_map = {p.stem: p for p in rgb_files}
    depth_map = {p.stem: p for p in depth_files}
    pairs = []
    # 找同名
    for stem, rpath in rgb_map.items():
        if stem in depth_map:
            pairs.append((rpath, depth_map[stem]))
    # 若没有同名或不完整，按顺序补齐剩余
    if not pairs:
        minlen = min(len(rgb_files), len(depth_files))
        pairs = list(zip(rgb_files[:minlen], depth_files[:minlen]))
    return pairs


def main():
    parser = argparse.ArgumentParser(description="Fuse RGB + depth mask into 4-channel RGBD PNGs")
    parser.add_argument("--rgb_dir", required=True, help="RGB 图像目录（3 通道）")
    parser.add_argument("--depth_dir", required=True, help="深度/掩码 图像目录（单通道或包含 alpha）")
    parser.add_argument("--out_dir", required=True, help="输出目录（保存 4 通道 PNG）")
    parser.add_argument("--depth_type", choices=["uint8", "uint16"], default="uint16", help="深度通道保存类型")
    parser.add_argument("--mode", choices=["name", "sorted"], default="name", help="匹配模式：按文件名或排序配对")
    args = parser.parse_args()

    rgb_dir = Path(args.rgb_dir)
    depth_dir = Path(args.depth_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "name":
        pairs = pair_by_name(rgb_dir, depth_dir)
    else:
        rgb_files = sorted([p for p in rgb_dir.iterdir() if p.is_file()])
        depth_files = sorted([p for p in depth_dir.iterdir() if p.is_file()])
        minlen = min(len(rgb_files), len(depth_files))
        pairs = list(zip(rgb_files[:minlen], depth_files[:minlen]))

    if not pairs:
        print("未找到可配对的文件。请检查目录和匹配模式。")
        return

    for rpath, dpath in pairs:
        out_path = out_dir / (rpath.stem + "_rgbd.png")
        try:
            fuse_pair(rpath, dpath, out_path, depth_type=args.depth_type)
            print("saved:", out_path)
        except Exception as e:
            print("failed pair:", rpath.name, dpath.name, "->", e)


if __name__ == "__main__":
    main()
