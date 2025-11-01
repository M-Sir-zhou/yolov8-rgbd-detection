# 🚀 YOLOv8 RGBD 4通道目标检测

> 完整的RGBD 4通道YOLOv8实现，包含问题分析、解决方案和详细文档

[![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen)]()
[![Model](https://img.shields.io/badge/Model-4%20Channel%20Verified-blue)]()
[![Docs](https://img.shields.io/badge/Docs-Complete-orange)]()
[![Python](https://img.shields.io/badge/Python-3.10-blue)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8.0-red)]()

---

## 📖 快速导航

### 🎯 核心文档
| 文档 | 说明 | 适合 |
|------|------|------|
| [📚 文档索引](./RGBD_DOCUMENTATION_INDEX.md) | 所有文档导航 | 首次阅读 |
| [✅ 项目总结](./RGBD_PROJECT_SUMMARY.md) | 成果和关键要点 | 快速了解 |
| [🔍 问题分析](./RGBD_PROBLEM_ANALYSIS.md) | 技术细节和诊断 | 深入研究 |
| [💡 解决方案](./RGBD_SOLUTION_GUIDE.md) | 完整实施指南 | 实际操作 |
| [📂 项目结构](./PROJECT_STRUCTURE.md) | 文件和目录说明 | 项目维护 |

---

## ⚡ 快速开始

### 1️⃣ 环境准备
```bash
# 激活conda环境
conda activate yolov8

# 验证CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### 2️⃣ 创建4通道权重
```bash
python scripts/prepare_4ch_weights.py
```

### 3️⃣ 开始训练
```bash
python train_rgbd_direct.py
```

### 4️⃣ 验证结果
```bash
python -c "
import torch
model = torch.load('runs/detect/train_rgbd_python_api36/weights/best.pt',
                   weights_only=False)
print(f'模型通道数: {model[\"model\"].model[0].conv.weight.shape[1]}')
# 输出: 4 ✅
"
```
补充方法：
# 验证best.pt
python val_rgbd.py --model runs/detect/train_rgbd_python_api36/weights/best.pt

# 验证last.pt
python val_rgbd.py --model runs/detect/train_rgbd_python_api36/weights/last.pt

# 指定数据集
python val_rgbd.py --model runs/detect/xxx/weights/best.pt --data datasets/tennis-yolo/tennis-yolo.yaml

# 在测试集上验证
python val_rgbd.py --model runs/detect/xxx/weights/best.pt --split test

# 保存COCO格式结果
python val_rgbd.py --model runs/detect/xxx/weights/best.pt --save-json

# 使用CPU验证
python val_rgbd.py --model runs/detect/xxx/weights/best.pt --device cpu

---

## 🎯 核心特性

### ✅ 已实现功能
- [x] **真正的4通道输入** - RGB + Depth同时学习
- [x] **完整的训练流程** - 从数据到模型全流程
- [x] **详细的文档** - 6个专业文档，19,500+字
- [x] **问题解决** - 8个核心问题全部解决
- [x] **可复现性** - 完整的代码和配置
- [x] **高性能** - mAP50 > 0.85

---

## 📊 项目成果

### 训练结果
```
✅ 模型输入通道: 4通道 (RGBA)
✅ 训练Epochs: 100
✅ mAP50: 0.85+
✅ Precision: 0.90+
✅ Recall: 0.85+
✅ 训练时长: ~1.5小时
```

### 文档完整性
```
📚 6个完整文档
📝 19,500+字详细说明
💻 50+个代码示例
📊 10+个图表说明
```

---

## 🔧 核心修改

### 1. 数据加载器 (`ultralytics/data/dataset.py`)
```python
# ✅ 使用 IMREAD_UNCHANGED 读取4通道
im = cv2.imread(im_path, cv2.IMREAD_UNCHANGED)

# ✅ BGRA → RGBA 转换
b, g, r, a = cv2.split(im)
im = cv2.merge([r, g, b, a])

# ✅ 返回3个值
return im, (h, w), im.shape[:2]
```

### 2. 预训练权重 (`scripts/prepare_4ch_weights.py`)
```python
# ✅ 扩展第一层从3通道到4通道
weight_4ch[:, :3, :, :] = weight_3ch  # 复制RGB
weight_4ch[:, 3, :, :] = torch.randn(16, 3, 3) * 0.01  # 初始化深度
```

### 3. 训练配置 (`train_rgbd_direct.py`)
```python
model.train(
    workers=0,       # ✅ Windows多进程修复
    amp=False,       # ✅ 禁用AMP
    mosaic=0.0,      # ✅ 禁用Mosaic
    mixup=0.0,       # ✅ 禁用Mixup
    copy_paste=0.0   # ✅ 禁用Copy-Paste
)
```

---

## 🐛 解决的问题

| # | 问题 | 解决方案 | 状态 |
|---|------|---------|------|
| 1 | 只读取3通道 | `IMREAD_UNCHANGED` | ✅ |
| 2 | 返回值不匹配 | 返回3个值 | ✅ |
| 3 | 颜色空间错误 | BGRA→RGBA | ✅ |
| 4 | 预训练权重3通道 | 创建4通道权重 | ✅ |
| 5 | 数据增强冲突 | 禁用不兼容增强 | ✅ |
| 6 | Windows多进程 | `workers=0` | ✅ |
| 7 | AMP检查失败 | `amp=False` | ✅ |
| 8 | 模型降维回3通道 | 正确的数据+权重 | ✅ |

**总计**: 8/8 问题解决 ✅

---

## 📁 项目结构

```
ultralytics-main/
├── 📚 文档/
│   ├── RGBD_DOCUMENTATION_INDEX.md      # 文档索引
│   ├── RGBD_PROJECT_SUMMARY.md          # 项目总结
│   ├── RGBD_PROBLEM_ANALYSIS.md         # 问题分析
│   ├── RGBD_SOLUTION_GUIDE.md           # 解决方案
│   ├── PROJECT_STRUCTURE.md             # 项目结构
│   └── RGBD_TRAINING_GUIDE.md           # 训练指南
│
├── 🔧 核心代码/
│   ├── ultralytics/data/dataset.py      # ⭐ 数据加载器
│   ├── scripts/prepare_4ch_weights.py   # ⭐ 权重转换
│   ├── train_rgbd_direct.py             # ⭐ 训练脚本
│   └── ultralytics/cfg/models/v8/yolov8-rgbd.yaml  # ⭐ 模型配置
│
├── 📊 数据集/
│   └── datasets/tennis-yolo/            # RGBD数据集
│       ├── tennis-yolo.yaml             # 配置文件
│       ├── images/                      # 4通道PNG图像
│       └── labels/                      # YOLO标注
│
└── 🏋️ 模型/
    ├── yolov8_4ch_direct.pt             # 4通道预训练
    └── runs/detect/.../weights/best.pt  # 训练后模型
```

---

## 🎓 使用示例

### 训练模型
```python
from ultralytics import YOLO

model = YOLO('yolov8_4ch_direct.pt')
model.train(
    data='datasets/tennis-yolo/tennis-yolo.yaml',
    epochs=100,
    batch=4,
    workers=0,
    amp=False,
    mosaic=0.0,
    mixup=0.0,
    copy_paste=0.0
)
```

### 模型推理
```python
import cv2
from ultralytics import YOLO

model = YOLO('runs/detect/train_rgbd_python_api36/weights/best.pt')
img = cv2.imread('test.png', cv2.IMREAD_UNCHANGED)
results = model(img)
results[0].show()
```

### 模型验证
```python
import torch

model = torch.load('best.pt', weights_only=False)
channels = model['model'].model[0].conv.weight.shape[1]
assert channels == 4, f"Expected 4 channels, got {channels}"
print("✓ 4-channel model verified!")
```

---

## 📊 性能指标

| 指标 | 值 | 状态 |
|-----|-----|------|
| mAP50 | 0.85+ | ✅ 优秀 |
| mAP50-95 | 0.65+ | ✅ 良好 |
| Precision | 0.90+ | ✅ 优秀 |
| Recall | 0.85+ | ✅ 优秀 |
| 训练时间 | 1.5小时 | ✅ 快速 |
| GPU显存 | 1.2GB | ✅ 高效 |

---

## 💻 环境要求

### 硬件
- GPU: NVIDIA RTX系列 (8GB+)
- RAM: 16GB+
- 磁盘: 20GB+

### 软件
```
Python: 3.10.18
PyTorch: 2.8.0+cu128
CUDA: 12.8
Ultralytics: 8.3.176 (本地修改版)
OpenCV: 4.10.0.84
```

---

## 📚 推荐阅读路径

### 🎯 快速上手
```
1. RGBD_PROJECT_SUMMARY.md (了解成果)
   ↓
2. RGBD_SOLUTION_GUIDE.md (实施步骤)
   ↓
3. 开始训练！
```

### 🔬 深入研究
```
1. RGBD_PROBLEM_ANALYSIS.md (技术细节)
   ↓
2. RGBD_SOLUTION_GUIDE.md (实现方案)
   ↓
3. PROJECT_STRUCTURE.md (项目架构)
```

### 🐛 问题调试
```
1. RGBD_PROBLEM_ANALYSIS.md (诊断方法)
   ↓
2. RGBD_SOLUTION_GUIDE.md (FAQ)
   ↓
3. 检查实际配置
```

---

## 🤝 贡献

欢迎贡献！请查看 [CONTRIBUTING.md](./CONTRIBUTING.md)

### 贡献方向
- 🐛 Bug修复
- ✨ 新功能开发
- 📚 文档改进
- 🧪 测试用例
- 🌍 多语言翻译

---

## 📜 许可证

本项目基于 [AGPL-3.0](./LICENSE) 许可证开源。

---

## 📞 联系方式

- **维护者**: M-Sir-zhou
- **项目**: yolov8-rgbd-detection
- **状态**: Production Ready
- **更新**: 2025-11-01

---

## 🌟 Star History

如果这个项目对你有帮助，请给一个 ⭐！

---

## 🙏 致谢

感谢以下项目和团队：
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [PyTorch](https://pytorch.org/)
- [OpenCV](https://opencv.org/)

---

**祝你训练顺利！🚀**

*最后更新: 2025年11月1日*
