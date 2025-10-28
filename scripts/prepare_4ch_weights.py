import torch

from ultralytics import YOLO


def create_4ch_with_yolo_class():
    print("使用YOLO类创建4通道模型...")

    # 使用YOLO类创建模型，指定输入通道数
    model = YOLO("ultralytics/cfg/models/v8/yolov8-rgbd.yaml")

    # 检查第一个卷积层
    first_conv = model.model.model[0].conv
    print(f"当前第一个卷积层形状: {first_conv.weight.shape}")

    # 如果输入通道数已经是4，直接保存
    if first_conv.weight.shape[1] == 4:
        print("检测到4通道输入，无需修改，直接保存...")
    # 如果仍然是3通道，手动修改
    elif first_conv.weight.shape[1] == 3:
        print("检测到3通道输入，正在修改为4通道...")

        # 获取原始权重
        original_weight = first_conv.weight.data
        out_channels, _, kh, kw = original_weight.shape

        # 创建新的4通道权重
        new_weight = torch.zeros(out_channels, 4, kh, kw)
        new_weight[:, :3, :, :] = original_weight
        torch.nn.init.normal_(new_weight[:, 3:, :, :], mean=0, std=0.01)

        # 创建新的卷积层
        new_conv = torch.nn.Conv2d(
            4, out_channels, kernel_size=3, stride=2, padding=1, bias=first_conv.bias is not None
        )
        new_conv.weight.data = new_weight

        if first_conv.bias is not None:
            new_conv.bias.data = first_conv.bias.data

        # 替换模型中的卷积层
        model.model.model[0].conv = new_conv

        print(f"修改后第一个卷积层形状: {model.model.model[0].conv.weight.shape}")
    else:
        print(f"⚠️ 未知输入通道数: {first_conv.weight.shape[1]}")

    # 保存模型
    model.save("yolov8_4ch_direct.pt")
    print("✅ 成功保存4通道模型: yolov8_4ch_direct.pt")
    return model


if __name__ == "__main__":
    create_4ch_with_yolo_class()
