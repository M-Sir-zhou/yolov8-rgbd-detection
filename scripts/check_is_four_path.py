# check.py
import os

import cv2
import numpy as np


def verify_image(image_path):
    """验证图像格式."""
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"无法读取图像: {image_path}")
        return False

    print(f"图像路径: {image_path}")
    print(f"图像形状: {img.shape}")
    print(f"图像数据类型: {img.dtype}")
    print(f"最小值: {np.min(img)}, 最大值: {np.max(img)}")
    print("-" * 50)

    # 检查是否为4通道
    if len(img.shape) == 3 and img.shape[2] == 4:
        print("✓ 图像是4通道的")
        return True
    else:
        print("✗ 图像不是4通道的")
        return False


# 验证几张图像
image_dir = "D:/ProjectCode/PyCharm/ultralytics-main/datasets/tennis-yolo/images/train"
image_files = [f for f in os.listdir(image_dir) if f.endswith(".png")][:5]  # 检查前5张图像

for image_file in image_files:
    image_path = os.path.join(image_dir, image_file)
    verify_image(image_path)
