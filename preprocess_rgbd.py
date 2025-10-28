# preprocess_rgbd.py
# 融合RGB-D
import os
import re
import shutil

import cv2
import numpy as np

# 在文件开头定义全局深度范围（需要预先计算）
GLOBAL_DEPTH_MIN = 0  # 替换为实际最小值
GLOBAL_DEPTH_MAX = 10000  # 替换为实际最大值


def combine_rgb_depth(rgb_path, depth_path, output_path):
    """将 RGB 和深度图像合并为 4 通道图像 (RGB + Depth)，使用全局归一化."""
    try:
        # 读取 RGB 图像
        rgb_image = cv2.imread(rgb_path)
        if rgb_image is None:
            print(f"无法读取 RGB 图像: {rgb_path}")
            return False
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)

        # 读取深度图像
        depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth_image is None:
            print(f"无法读取深度图像: {depth_path}")
            return False

        # 确保深度图像是单通道
        if len(depth_image.shape) == 3:
            depth_image = cv2.cvtColor(depth_image, cv2.COLOR_BGR2GRAY)

        # 调整图像尺寸使其匹配
        if rgb_image.shape[:2] != depth_image.shape[:2]:
            print(f"调整图像尺寸: RGB {rgb_image.shape[:2]} -> 深度 {depth_image.shape[:2]}")
            depth_image = cv2.resize(depth_image, (rgb_image.shape[1], rgb_image.shape[0]))

        # 使用全局深度范围归一化
        if GLOBAL_DEPTH_MAX > GLOBAL_DEPTH_MIN:
            # 将深度值裁剪到全局范围后再归一化
            depth_image = np.clip(depth_image, GLOBAL_DEPTH_MIN, GLOBAL_DEPTH_MAX)
            depth_image = ((depth_image - GLOBAL_DEPTH_MIN) / (GLOBAL_DEPTH_MAX - GLOBAL_DEPTH_MIN) * 255).astype(
                np.uint8
            )
        else:
            depth_image = np.zeros_like(depth_image, dtype=np.uint8)

        # 合并 RGB 和深度图像为4通道 (R, G, B, Depth)
        rgba_image = np.concatenate([rgb_image, depth_image[:, :, np.newaxis]], axis=-1)

        # 保存为PNG格式
        cv2.imwrite(output_path, rgba_image)
        print(f"成功合并并保存: {output_path}")
        return True

    except Exception as e:
        print(f"处理图像时出错: {e}")
        return False


def find_matching_depth_file(rgb_file, depth_dir):
    """查找与 RGB 图像匹配的深度图像文件."""
    # 提取 RGB 图像的基本名称（不带扩展名）
    base_name = os.path.splitext(rgb_file)[0]

    # 尝试直接匹配（相同的文件名）
    depth_path = os.path.join(depth_dir, f"{base_name}_depth.png")
    if os.path.exists(depth_path):
        return depth_path

    # 尝试其他可能的命名模式
    patterns = [
        f"{base_name}_depth.png",
        f"{base_name}_d.png",
        f"{base_name}_depth.jpg",
        f"{base_name}_d.jpg",
        f"depth_{base_name}.png",
        f"d_{base_name}.png",
    ]

    for pattern in patterns:
        depth_path = os.path.join(depth_dir, pattern)
        if os.path.exists(depth_path):
            return depth_path

    # 如果以上模式都不匹配，尝试查找包含相同数字的任何文件
    number_match = re.search(r"\d+", base_name)
    if number_match:
        number = number_match.group()
        for depth_file in os.listdir(depth_dir):
            if number in depth_file and depth_file.endswith((".png", ".jpg", ".jpeg")):
                return os.path.join(depth_dir, depth_file)

    print(f"找不到与 {rgb_file} 匹配的深度图像")
    return None


def preprocess_dataset():
    """预处理整个数据集."""
    # 原始数据集路径
    base_path = "D:/ProjectCode/PyCharm/ultralytics-main/datasets/tennis-rgbd"

    # 新数据集路径
    output_base = "D:/ProjectCode/PyCharm/ultralytics-main/datasets/tennis-yolo"

    # 创建输出目录
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(output_base, "images", split), exist_ok=True)
        os.makedirs(os.path.join(output_base, "labels", split), exist_ok=True)

    # 处理每个分割（训练集、验证集、测试集）
    for split in ["train", "val", "test"]:
        rgb_dir = os.path.join(base_path, split, "rgb")
        depth_dir = os.path.join(base_path, split, "depth")
        labels_dir = os.path.join(base_path, split, "labels")

        output_image_dir = os.path.join(output_base, "images", split)
        output_label_dir = os.path.join(output_base, "labels", split)

        print(f"\n处理 {split} 数据集...")
        print(f"RGB 目录: {rgb_dir}")
        print(f"深度目录: {depth_dir}")
        print(f"标签目录: {labels_dir}")

        # 检查目录是否存在
        if not os.path.exists(rgb_dir):
            print(f"RGB 目录不存在: {rgb_dir}")
            continue

        if not os.path.exists(depth_dir):
            print(f"深度目录不存在: {depth_dir}")
            continue

        if not os.path.exists(labels_dir):
            print(f"标签目录不存在: {labels_dir}")
            continue

        # 处理每个图像
        processed_count = 0
        for rgb_file in os.listdir(rgb_dir):
            if rgb_file.endswith((".png", ".jpg", ".jpeg")):
                # 构建 RGB 图像路径
                rgb_path = os.path.join(rgb_dir, rgb_file)

                # 查找匹配的深度图像
                depth_path = find_matching_depth_file(rgb_file, depth_dir)

                if depth_path is None:
                    print(f"找不到对应的深度图像: {rgb_file}")
                    continue

                # 构建标签文件路径
                label_file = os.path.splitext(rgb_file)[0] + ".txt"
                label_path = os.path.join(labels_dir, label_file)

                # 合并 RGB 和深度图像
                output_image_path = os.path.join(output_image_dir, os.path.splitext(rgb_file)[0] + ".png")
                success = combine_rgb_depth(rgb_path, depth_path, output_image_path)

                if success:
                    # 复制标签文件
                    if os.path.exists(label_path):
                        output_label_path = os.path.join(output_label_dir, label_file)
                        shutil.copy2(label_path, output_label_path)
                        print(f"复制标签: {label_path} -> {output_label_path}")
                    else:
                        print(f"警告: 找不到标签文件 {label_path}")

                    processed_count += 1

        print(f"{split} 数据集处理完成: {processed_count} 个图像")


if __name__ == "__main__":
    preprocess_dataset()
    print("数据集预处理完成!")
