# check_training_model.py
import torch
from ultralytics import YOLO

def check_training_model():
    # 检查预训练权重
    print("=== 检查预训练权重 ===")
    try:
        pretrained = YOLO('D:\\ProjectCode\\PyCharm\\ultralytics-main\\yolov8_4ch_direct.pt')
        first_conv_pretrained = pretrained.model.model[0].conv
        print(f"预训练权重 - 第一个卷积层形状: {first_conv_pretrained.weight.shape}")
    except Exception as e:
        print(f"预训练权重加载失败: {e}")
    
    # 检查训练后的模型
    print("\n=== 检查训练后模型 ===")
    try:
        # 假设训练后的模型在 runs/detect/train/weights/best.pt
        trained = YOLO('D:\\ProjectCode\\PyCharm\\ultralytics-main\\runs\\detect\\train10\\weights\\best.pt')
        first_conv_trained = trained.model.model[0].conv
        print(f"训练后模型 - 第一个卷积层形状: {first_conv_trained.weight.shape}")
    except Exception as e:
        print(f"训练后模型加载失败: {e}")
    
    # 检查配置文件
    print("\n=== 检查配置文件 ===")
    try:
        from_config = YOLO('D:\\ProjectCode\\PyCharm\\ultralytics-main\\ultralytics\\cfg\\models\\v8\\yolov8-rgbd.yaml')
        first_conv_config = from_config.model.model[0].conv
        print(f"配置文件 - 第一个卷积层形状: {first_conv_config.weight.shape}")
    except Exception as e:
        print(f"配置文件加载失败: {e}")

if __name__ == "__main__":
    check_training_model()