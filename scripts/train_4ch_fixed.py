# train_4ch_fixed.py
import torch
from ultralytics import YOLO
import os

def train_4ch_fixed():
    print("=== 强制4通道训练解决方案 ===")
    
    # 方法1：直接使用4通道模型文件
    print("方法1: 直接使用4通道模型文件训练")
    model = YOLO('yolov8_4ch_direct.pt')
    
    # 验证模型是4通道
    first_conv = model.model.model[0].conv
    print(f"训练前模型通道数: {first_conv.weight.shape[1]}")
    
    if first_conv.weight.shape[1] != 4:
        # 手动扩展输入通道数
        first_conv.weight = torch.nn.Parameter(torch.cat([first_conv.weight, torch.zeros_like(first_conv.weight[:, :1])], dim=1))
        print(f"手动扩展后的模型通道数: {first_conv.weight.shape[1]}")
    
    # 开始训练
    results = model.train(
        data='D:/ProjectCode/PyCharm/ultralytics-main/datasets/tennis-yolo/tennis-yolo.yaml',
        epochs=100,
        batch=4,
        imgsz=640,
        device=0,
        save=True,
        exist_ok=True
    )
    
    print("✅ 训练完成!")
    
    # 验证训练后的模型
    trained_model = YOLO('runs/detect/train/weights/best.pt')
    first_conv_trained = trained_model.model.model[0].conv
    trained_channels = first_conv_trained.weight.shape[1]
    print(f"训练后模型通道数: {trained_channels}")
    
    if trained_channels != 4:
        # 手动扩展输入通道数
        first_conv_trained.weight = torch.nn.Parameter(torch.cat([first_conv_trained.weight, torch.zeros_like(first_conv_trained.weight[:, :1])], dim=1))
        print(f"手动扩展后的模型通道数: {first_conv_trained.weight.shape[1]}")
        # 保存扩展后的模型
        trained_model.model.float()  # 确保模型权重为浮点类型
        ckpt = {
            "model": trained_model.model,  # 直接保存模型对象
            "ema": trained_model.model,   # 直接保存模型对象
            "optimizer": None,
            "epoch": -1,
        }
        torch.save(ckpt, 'runs/detect/train/weights/best_4ch.pt')
        print("🎉 成功! 训练后模型已手动扩展为4通道并保存为 best_4ch.pt")
    else:
        print("🎉 成功! 训练后模型保持4通道")

if __name__ == "__main__":
    train_4ch_fixed()