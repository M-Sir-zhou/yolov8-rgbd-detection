"""RGBD 4通道模型验证脚本 - 支持命令行参数"""
import argparse
import torch
from pathlib import Path
from ultralytics import YOLO


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='验证YOLOv8 RGBD 4通道模型')
    
    parser.add_argument('--model', type=str, required=True,
                        help='训练好的模型路径 (必需)')
    parser.add_argument('--data', type=str, 
                        default='datasets/tennis-yolo/tennis-yolo.yaml',
                        help='数据集配置文件路径')
    parser.add_argument('--split', type=str, default='val',
                        choices=['val', 'test', 'train'],
                        help='验证数据集分割 (默认: val)')
    parser.add_argument('--batch', type=int, default=1,
                        help='批次大小 (默认: 1)')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='输入图像尺寸 (默认: 640)')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='设备 (默认: cuda:0)')
    parser.add_argument('--verbose', action='store_true', default=True,
                        help='详细输出')
    parser.add_argument('--save-json', action='store_true',
                        help='保存COCO格式的结果JSON文件')
    parser.add_argument('--save-hybrid', action='store_true',
                        help='保存标签+预测的混合结果')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 70)
    print("YOLOv8 RGBD 4通道模型验证")
    print("=" * 70)
    
    # 验证模型文件
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"✗ 模型文件不存在: {model_path}")
        return
    
    print(f"\n模型: {model_path}")
    
    # 验证模型通道数
    try:
        model_data = torch.load(str(model_path), weights_only=False)
        channels = model_data['model'].model[0].conv.weight.shape[1]
        print(f"模型输入通道: {channels}")
        
        if channels != 4:
            print(f"⚠️  警告: 模型是{channels}通道，不是4通道RGBD模型")
    except Exception as e:
        print(f"⚠️  无法验证模型通道数: {e}")
    
    # 加载模型
    print(f"\n加载模型...")
    model = YOLO(str(model_path))
    
    # 自动设备选择
    device = args.device
    if device.startswith('cuda') and not torch.cuda.is_available():
        print(f"⚠️  CUDA不可用，切换到CPU")
        device = 'cpu'
    
    print(f"\n验证配置:")
    print(f"  数据集:     {args.data}")
    print(f"  分割:       {args.split}")
    print(f"  批次大小:   {args.batch}")
    print(f"  图像尺寸:   {args.imgsz}")
    print(f"  设备:       {device}")
    print("-" * 70)
    
    # 开始验证
    try:
        results = model.val(
            data=args.data,
            split=args.split,
            batch=args.batch,
            imgsz=args.imgsz,
            device=device,
            verbose=args.verbose,
            save_json=args.save_json,
            save_hybrid=args.save_hybrid,
        )
        
        print("\n" + "=" * 70)
        print("✅ 验证完成！")
        print("=" * 70)
        
        # 显示结果
        print(f"\n性能指标:")
        print(f"  mAP50:      {results.box.map50:.4f}")
        print(f"  mAP50-95:   {results.box.map:.4f}")
        print(f"  Precision:  {results.box.mp:.4f}")
        print(f"  Recall:     {results.box.mr:.4f}")
        
    except Exception as e:
        print(f"\n❌ 验证失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
