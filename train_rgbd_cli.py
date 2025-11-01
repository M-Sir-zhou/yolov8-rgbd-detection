"""RGBD 4通道模型训练脚本 - 支持命令行参数"""
import argparse
import torch
from pathlib import Path
from ultralytics import YOLO


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='训练YOLOv8 RGBD 4通道模型')
    
    # 必需参数
    parser.add_argument('--data', type=str, 
                        default='datasets/tennis-yolo/tennis-yolo.yaml',
                        help='数据集配置文件路径')
    parser.add_argument('--model', type=str, 
                        default='yolov8_4ch_direct.pt',
                        help='4通道预训练模型路径')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=100,
                        help='训练轮数 (默认: 100)')
    parser.add_argument('--batch', type=int, default=4,
                        help='批次大小 (默认: 4)')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='输入图像尺寸 (默认: 640)')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='训练设备 (默认: cuda:0, 可选: cpu)')
    
    # 输出参数
    parser.add_argument('--name', type=str, default='train_rgbd',
                        help='实验名称 (默认: train_rgbd)')
    parser.add_argument('--project', type=str, default='runs/detect',
                        help='项目保存目录 (默认: runs/detect)')
    
    # 训练配置
    parser.add_argument('--patience', type=int, default=50,
                        help='早停耐心值 (默认: 50)')
    parser.add_argument('--workers', type=int, default=0,
                        help='数据加载进程数 (Windows建议为0)')
    parser.add_argument('--cache', action='store_true',
                        help='是否缓存数据 (默认: False)')
    parser.add_argument('--amp', action='store_true',
                        help='是否启用AMP混合精度 (默认: False)')
    
    # 数据增强参数（RGBD建议禁用）
    parser.add_argument('--mosaic', type=float, default=0.0,
                        help='Mosaic增强概率 (RGBD建议0.0)')
    parser.add_argument('--mixup', type=float, default=0.0,
                        help='Mixup增强概率 (RGBD建议0.0)')
    parser.add_argument('--copy-paste', type=float, default=0.0,
                        help='Copy-Paste增强概率 (RGBD建议0.0)')
    
    # 其他参数
    parser.add_argument('--no-plots', action='store_true',
                        help='不生成训练可视化图表')
    parser.add_argument('--verbose', action='store_true', default=True,
                        help='详细输出模式')
    
    return parser.parse_args()


def verify_model_channels(model_path, expected_channels=4):
    """验证模型输入通道数"""
    try:
        model = YOLO(model_path)
        first_layer = model.model.model[0].conv
        actual_channels = first_layer.weight.shape[1]
        
        print(f"模型: {model_path}")
        print(f"输入通道数: {actual_channels}")
        
        if actual_channels == expected_channels:
            print(f"✓ 模型确认为{expected_channels}通道")
            return True
        else:
            print(f"✗ 错误: 模型是{actual_channels}通道，期望{expected_channels}通道")
            return False
    except Exception as e:
        print(f"✗ 模型验证失败: {e}")
        return False


def main():
    # 解析参数
    args = parse_args()
    
    print("=" * 70)
    print("YOLOv8 RGBD 4通道模型训练")
    print("=" * 70)
    
    # 1. 验证模型
    print("\n[1/3] 验证模型...")
    if not verify_model_channels(args.model, expected_channels=4):
        print("\n❌ 模型验证失败，请检查模型是否为4通道")
        return
    
    # 2. 验证数据集
    print("\n[2/3] 验证数据集...")
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"✗ 数据集配置文件不存在: {data_path}")
        return
    print(f"✓ 数据集: {data_path}")
    
    # 3. 开始训练
    print("\n[3/3] 开始训练...")
    print("\n训练配置:")
    print(f"  模型:       {args.model}")
    print(f"  数据集:     {args.data}")
    print(f"  轮数:       {args.epochs}")
    print(f"  批次大小:   {args.batch}")
    print(f"  图像尺寸:   {args.imgsz}")
    print(f"  设备:       {args.device}")
    print(f"  实验名称:   {args.name}")
    print(f"  进程数:     {args.workers}")
    print(f"  AMP:        {args.amp}")
    print(f"  Mosaic:     {args.mosaic}")
    print(f"  Mixup:      {args.mixup}")
    print(f"  Copy-Paste: {args.copy_paste}")
    print("-" * 70)
    
    try:
        model = YOLO(args.model)
        
        # 自动设备选择
        device = args.device
        if device.startswith('cuda') and not torch.cuda.is_available():
            print(f"⚠️  CUDA不可用，切换到CPU")
            device = 'cpu'
        
        results = model.train(
            data=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=device,
            name=args.name,
            project=args.project,
            patience=args.patience,
            save=True,
            plots=not args.no_plots,
            verbose=args.verbose,
            workers=args.workers,
            cache=args.cache,
            amp=args.amp,
            mosaic=args.mosaic,
            mixup=args.mixup,
            copy_paste=args.copy_paste,
        )
        
        print("\n" + "=" * 70)
        print("✅ 训练完成！")
        print("=" * 70)
        
        # 获取保存路径
        save_dir = Path(model.trainer.save_dir)
        best_pt = save_dir / 'weights' / 'best.pt'
        last_pt = save_dir / 'weights' / 'last.pt'
        
        print(f"\n保存目录: {save_dir}")
        print(f"最佳模型: {best_pt}")
        print(f"最后模型: {last_pt}")
        
        # 验证训练后的模型
        model_to_check = best_pt if best_pt.exists() else last_pt
        
        if model_to_check.exists():
            print(f"\n验证训练后的模型...")
            best_model = torch.load(str(model_to_check), weights_only=False)
            channels = best_model['model'].model[0].conv.weight.shape[1]
            print(f"训练后模型通道数: {channels}")
            
            if channels == 4:
                print("✓ 训练后模型仍然是4通道 ✓")
            else:
                print(f"✗ 警告: 训练后模型变成了{channels}通道")
        
        print("\n使用以下命令进行验证:")
        print(f"python val_rgbd.py --model {best_pt}")
        
    except Exception as e:
        print(f"\n❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
