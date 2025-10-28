#检测预训练模型通道数
import torch
model = torch.load('yolov8_4ch_direct.pt', weights_only=False)
print(model['model'].model[0].conv.weight.shape[1])  # 应该输出4
