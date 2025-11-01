"""直接使用Python API训练RGBD模型"""
from ultralytics import YOLO
import torch

def main():
    print("=" * 60)
    print("开始训练RGBD 4通道模型")
    print("=" * 60)

    # 1. 加载4通道模型
    print("\n1. 加载模型...")
    model = YOLO('yolov8_4ch_direct.pt')

    # 验证模型是4通道
    first_layer = model.model.model[0].conv
    print(f"第一层卷积输入通道: {first_layer.weight.shape[1]}")
    assert first_layer.weight.shape[1] == 4, "模型不是4通道！"
    print("✓ 模型确认为4通道")

    # 2. 开始训练
    print("\n2. 开始训练...")
    print("配置:")
    print("  - 数据集: datasets/tennis-yolo/tennis-yolo.yaml")
    print("  - Epochs: 100")
    print("  - Batch size: 4")
    print("  - Image size: 640")
    print("  - Device: cuda:0")

    try:
        results = model.train(
            data='datasets/tennis-yolo/tennis-yolo.yaml',
            epochs=100,
            imgsz=640,
            batch=4,
            device='cuda:0' if torch.cuda.is_available() else 'cpu',  # 自动检测CUDA
            name='train_rgbd_python_api',
            project='runs/detect',
            patience=50,
            save=True,
            plots=True,
            verbose=True,
            workers=0,  # Windows需要设置为0避免多进程问题
            cache=False,  # 不使用缓存，确保重新加载数据
            amp=False,  # 禁用AMP以避免检查3通道模型的问题
            # 禁用所有可能导致buffer问题的数据增强
            mosaic=0.0,  # 禁用 Mosaic
            copy_paste=0.0,  # 禁用 Copy-Paste
            mixup=0.0,  # 禁用 Mixup
        )
        
        print(f"\n训练设备: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
        
        print("\n" + "=" * 60)
        print("✅ 训练完成！")
        print("=" * 60)
        
        # 使用训练器返回的实际保存路径
        from pathlib import Path
        save_dir = Path(model.trainer.save_dir)
        best_pt = save_dir / 'weights' / 'best.pt'
        last_pt = save_dir / 'weights' / 'last.pt'
        
        print(f"保存目录: {save_dir}")
        print(f"最佳模型: {best_pt}")
        print(f"最后模型: {last_pt}")
        
        # 验证训练后的模型（使用存在的模型文件）
        model_to_check = best_pt if best_pt.exists() else last_pt
        
        if model_to_check.exists():
            best_model = torch.load(str(model_to_check), weights_only=False)
            channels = best_model['model'].model[0].conv.weight.shape[1]
            print(f"\n训练后模型通道数: {channels}")
            
            if channels == 4:
                print("✓ 训练后模型仍然是4通道 ✓")
            else:
                print(f"✗ 警告: 训练后模型变成了{channels}通道")
        else:
            print(f"\n⚠️ 警告: 找不到模型文件 {model_to_check}")
            
    except Exception as e:
        print(f"\n❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
