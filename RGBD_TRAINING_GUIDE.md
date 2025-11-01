# RGBD 4通道模型训练完整指南

## 📋 目录
1. [预训练权重转换为4通道](#1-预训练权重转换为4通道)
2. [数据集准备和融合](#2-数据集准备和融合)
3. [完整训练流程](#3-完整训练流程)
4. [验证模型](#4-验证模型)

---

## 1. 预训练权重转换为4通道

### 🎯 为什么需要转换？

标准的YOLOv8预训练权重（如 `yolov8n.pt`）的第一层卷积只接受 **3通道输入（RGB）**：
```python
Conv2d(3, 16, kernel_size=3, stride=2, padding=1)  # 输入: 3通道
```

而RGBD模型需要 **4通道输入（RGB + Depth）**：
```python
Conv2d(4, 16, kernel_size=3, stride=2, padding=1)  # 输入: 4通道
```

### 📝 转换方法

#### 方法1：使用 `prepare_4ch_weights.py`（推荐）

```bash
# 1. 进入脚本目录
cd D:\ProjectCode\PyCharm\ultralytics-main\scripts

# 2. 运行转换脚本
python prepare_4ch_weights.py
```

**脚本工作原理：**
1. 加载原始3通道模型（基于 `yolov8-rgbd.yaml` 配置）
2. 提取第一层卷积权重：`[16, 3, 3, 3]`
3. 创建新的4通道权重：`[16, 4, 3, 3]`
4. **复制RGB权重**：前3个通道保持不变
5. **初始化深度权重**：第4通道使用小随机值（均值0，标准差0.01）
6. 保存为 `yolov8_4ch_direct.pt`

**代码解析：**
```python
# 获取原始3通道权重
original_weight = first_conv.weight.data  # shape: [16, 3, 3, 3]
out_channels, _, kh, kw = original_weight.shape

# 创建4通道权重矩阵
new_weight = torch.zeros(out_channels, 4, kh, kw)

# 复制RGB通道（保留预训练特征）
new_weight[:, :3, :, :] = original_weight

# 初始化深度通道（小随机值，让模型从零学习深度特征）
torch.nn.init.normal_(new_weight[:, 3:, :, :], mean=0, std=0.01)

# 创建新的卷积层并替换
new_conv = torch.nn.Conv2d(4, out_channels, kernel_size=3, stride=2, padding=1)
new_conv.weight.data = new_weight
```

#### 方法2：手动转换（理解原理）

```python
import torch
from ultralytics import YOLO

# 1. 加载配置文件创建模型
model = YOLO('ultralytics/cfg/models/v8/yolov8-rgbd.yaml')

# 2. 检查第一层
first_conv = model.model.model[0].conv
print(f"原始形状: {first_conv.weight.shape}")  # 可能是 [16, 3, 3, 3]

# 3. 创建4通道权重
original = first_conv.weight.data
new_weight = torch.zeros(16, 4, 3, 3)
new_weight[:, :3, :, :] = original  # RGB通道
torch.nn.init.normal_(new_weight[:, 3:, :, :], std=0.01)  # Depth通道

# 4. 替换卷积层
new_conv = torch.nn.Conv2d(4, 16, kernel_size=3, stride=2, padding=1, bias=False)
new_conv.weight.data = new_weight
model.model.model[0].conv = new_conv

# 5. 保存
model.save('yolov8_4ch_custom.pt')
```

### ✅ 验证转换结果

```python
import torch

# 加载转换后的模型
model = torch.load('yolov8_4ch_direct.pt', weights_only=False)

# 检查输入通道数
channels = model['model'].model[0].conv.weight.shape[1]
print(f"输入通道数: {channels}")  # 应该输出: 4

# 检查权重形状
weight_shape = model['model'].model[0].conv.weight.shape
print(f"权重形状: {weight_shape}")  # 应该是: torch.Size([16, 4, 3, 3])

if channels == 4:
    print("✅ 转换成功！这是一个4通道RGBD模型")
else:
    print("❌ 转换失败！仍然是3通道模型")
```

---

## 2. 数据集准备和融合

### 🎯 目标
将分离的RGB图像和Depth图像融合为单个4通道PNG文件。

### 📁 数据集结构

**原始结构（分离的RGB和Depth）：**
```
datasets/tennis-rgbd/
├── train/
│   ├── rgb/          # RGB图像（3通道）
│   │   ├── img_001.png
│   │   ├── img_002.png
│   │   └── ...
│   ├── depth/        # 深度图像（单通道）
│   │   ├── img_001.png
│   │   ├── img_002.png
│   │   └── ...
│   └── labels/       # YOLO标签
│       ├── img_001.txt
│       └── ...
├── val/
│   ├── rgb/
│   ├── depth/
│   └── labels/
└── test/
    ├── rgb/
    ├── depth/
    └── labels/
```

**目标结构（融合后的4通道）：**
```
datasets/tennis-yolo/
├── images/
│   ├── train/
│   │   ├── img_001_rgbd.png  # 4通道PNG (RGBA，A=Depth)
│   │   ├── img_002_rgbd.png
│   │   └── ...
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   │   ├── img_001.txt
│   │   └── ...
│   ├── val/
│   └── test/
└── tennis-yolo.yaml  # 数据集配置文件
```

### 🔧 融合方法

#### 方法1：使用 `fuse_rgb_depth.py`（批量处理）

```bash
# 融合训练集
python scripts/fuse_rgb_depth.py \
    --rgb_dir "D:/ProjectCode/PyCharm/ultralytics-main/datasets/tennis-rgbd/train/rgb" \
    --depth_dir "D:/ProjectCode/PyCharm/ultralytics-main/datasets/tennis-rgbd/train/depth" \
    --out_dir "D:/ProjectCode/PyCharm/ultralytics-main/datasets/tennis-yolo/images/train" \
    --depth_type uint8 \
    --mode sorted

# 融合验证集
python scripts/fuse_rgb_depth.py \
    --rgb_dir "D:/ProjectCode/PyCharm/ultralytics-main/datasets/tennis-rgbd/val/rgb" \
    --depth_dir "D:/ProjectCode/PyCharm/ultralytics-main/datasets/tennis-rgbd/val/depth" \
    --out_dir "D:/ProjectCode/PyCharm/ultralytics-main/datasets/tennis-yolo/images/val" \
    --depth_type uint8 \
    --mode sorted

# 融合测试集
python scripts/fuse_rgb_depth.py \
    --rgb_dir "D:/ProjectCode/PyCharm/ultralytics-main/datasets/tennis-rgbd/test/rgb" \
    --depth_dir "D:/ProjectCode/PyCharm/ultralytics-main/datasets/tennis-rgbd/test/depth" \
    --out_dir "D:/ProjectCode/PyCharm/ultralytics-main/datasets/tennis-yolo/images/test" \
    --depth_type uint8 \
    --mode sorted
```

**参数说明：**
- `--rgb_dir`: RGB图像目录
- `--depth_dir`: 深度图像目录
- `--out_dir`: 输出目录（保存4通道PNG）
- `--depth_type`: 深度通道类型
  - `uint8`: 0-255范围（推荐，文件小）
  - `uint16`: 0-65535范围（精度高，文件大）
- `--mode`: 文件匹配模式
  - `sorted`: 按文件名排序配对
  - `name`: 按文件名（stem）匹配

#### 方法2：使用 `preprocess_rgbd.py`（自动化处理）

```bash
python preprocess_rgbd.py
```

**此脚本会自动：**
1. 读取 `tennis-rgbd` 数据集
2. 匹配RGB和Depth图像
3. 融合为4通道PNG
4. 复制标签文件到 `tennis-yolo` 目录

### 📊 融合原理详解

```python
def fuse_pair(rgb_path, depth_path, out_path, depth_type='uint8'):
    # 1. 读取RGB图像（3通道，uint8）
    rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)  # BGR格式
    
    # 2. 读取深度图像（单通道，可能是uint8或uint16）
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    
    # 3. 调整深度图尺寸与RGB匹配
    if (depth.shape[0], depth.shape[1]) != (rgb.shape[0], rgb.shape[1]):
        depth = cv2.resize(depth, (rgb.shape[1], rgb.shape[0]), 
                          interpolation=cv2.INTER_NEAREST)
    
    # 4. 深度归一化到0-255（如果选择uint8模式）
    if depth_type == 'uint8':
        if depth.dtype != np.uint8:
            # 线性归一化
            depth8 = cv2.normalize(depth, None, 0, 255, 
                                  cv2.NORM_MINMAX).astype(np.uint8)
        else:
            depth8 = depth
        
        # 5. 合并为4通道 BGRA（OpenCV格式）
        bgra = np.dstack([rgb[:,:,0],  # B通道
                         rgb[:,:,1],   # G通道
                         rgb[:,:,2],   # R通道
                         depth8])      # A通道（深度）
        
        # 6. 保存为PNG（支持alpha通道）
        cv2.imwrite(str(out_path), bgra)
```

### 🖼️ 验证融合结果

```python
import cv2
import numpy as np

# 读取4通道图像
img_path = "datasets/tennis-yolo/images/train/img_001_rgbd.png"
img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

print(f"图像形状: {img.shape}")  # 应该是 (H, W, 4)
print(f"数据类型: {img.dtype}")  # 应该是 uint8

# 分离通道
b, g, r, depth = cv2.split(img)

# 可视化
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
axes[0].imshow(cv2.cvtColor(img[:,:,:3], cv2.COLOR_BGR2RGB))
axes[0].set_title('RGB')
axes[1].imshow(r, cmap='gray')
axes[1].set_title('Red Channel')
axes[2].imshow(g, cmap='gray')
axes[2].set_title('Green Channel')
axes[3].imshow(depth, cmap='jet')
axes[3].set_title('Depth Channel')
plt.show()

print("✅ 4通道图像验证完成！")
```

### ⚠️ 重要注意事项

1. **标签文件同步**
   - 确保每个融合后的图像都有对应的标签文件
   - 标签文件名要匹配（除了扩展名）

2. **深度图质量**
   - 确保深度图没有缺失值或NaN
   - 深度范围要合理（不全是0或全是最大值）

3. **文件命名一致性**
   - RGB和Depth文件名要能正确匹配
   - 推荐使用 `mode=sorted` 确保顺序匹配

---

## 3. 完整训练流程

### 步骤1：准备4通道预训练权重

```bash
python scripts/prepare_4ch_weights.py
```

验证生成的 `yolov8_4ch_direct.pt` 文件。

### 步骤2：融合数据集

```bash
# 方式1：使用自动化脚本
python preprocess_rgbd.py

# 方式2：手动批量处理
python scripts/fuse_rgb_depth.py \
    --rgb_dir "datasets/tennis-rgbd/train/rgb" \
    --depth_dir "datasets/tennis-rgbd/train/depth" \
    --out_dir "datasets/tennis-yolo/images/train" \
    --depth_type uint8 \
    --mode sorted
```

### 步骤3：配置数据集YAML

创建 `datasets/tennis-yolo/tennis-yolo.yaml`:
```yaml
# 数据集根目录
path: D:/ProjectCode/PyCharm/ultralytics-main/datasets/tennis-yolo

# 训练/验证/测试集
train: images/train
val: images/val
test: images/test

# 类别数
nc: 1

# 类别名称
names:
  0: tennis_ball

# RGBD标志
rgbd: true
channels: 4  # RGB + Depth
```

### 步骤4：开始训练

```python
from ultralytics import YOLO

# 加载4通道预训练模型
model = YOLO('yolov8_4ch_direct.pt')

# 开始训练
results = model.train(
    data='datasets/tennis-yolo/tennis-yolo.yaml',  # 数据集配置
    epochs=100,                                     # 训练轮数
    imgsz=640,                                      # 图像尺寸
    batch=4,                                        # 批次大小
    device=0,                                       # GPU设备（0=第一块GPU）
    project='runs/detect',                          # 项目目录
    name='train_rgbd',                              # 实验名称
    pretrained=True,                                # 使用预训练权重
    patience=50,                                    # 早停耐心值
    save=True,                                      # 保存模型
    plots=True,                                     # 生成可视化图表
    verbose=True                                    # 详细输出
)

print("训练完成！")
print(f"最佳模型: {results.save_dir}/weights/best.pt")
```

或使用命令行：
```bash
yolo detect train \
    model=yolov8_4ch_direct.pt \
    data=datasets/tennis-yolo/tennis-yolo.yaml \
    epochs=100 \
    imgsz=640 \
    batch=4 \
    device=0 \
    name=train_rgbd
```

### 步骤5：监控训练

训练过程中会生成：
- `runs/detect/train_rgbd/weights/best.pt` - 最佳模型
- `runs/detect/train_rgbd/weights/last.pt` - 最后一轮模型
- `runs/detect/train_rgbd/results.png` - 训练曲线
- `runs/detect/train_rgbd/confusion_matrix.png` - 混淆矩阵

---

## 4. 验证模型

### 验证是否为4通道模型

```python
import torch

# 加载训练好的模型
model_path = "runs/detect/train_rgbd/weights/best.pt"
model = torch.load(model_path, weights_only=False)

# 检查第一层卷积
first_conv = model['model'].model[0].conv
channels = first_conv.weight.shape[1]

print("=" * 60)
print(f"模型路径: {model_path}")
print(f"第一层卷积输入通道数: {channels}")
print(f"权重形状: {first_conv.weight.shape}")

if channels == 4:
    print("✅ 这是一个RGBD 4通道模型")
else:
    print("❌ 这不是4通道模型")
print("=" * 60)
```

### 测试推理

```python
from ultralytics import YOLO
import cv2

# 加载模型
model = YOLO('runs/detect/train_rgbd/weights/best.pt')

# 读取4通道测试图像
test_img = cv2.imread('datasets/tennis-yolo/images/test/test_001_rgbd.png', 
                      cv2.IMREAD_UNCHANGED)

# 推理
results = model.predict(test_img, save=True, conf=0.5)

# 查看结果
for r in results:
    print(f"检测到 {len(r.boxes)} 个对象")
    print(f"置信度: {r.boxes.conf}")
    print(f"边界框: {r.boxes.xyxy}")
```

### 性能评估

```python
# 在验证集上评估
metrics = model.val(data='datasets/tennis-yolo/tennis-yolo.yaml')

print(f"mAP50: {metrics.box.map50:.4f}")
print(f"mAP50-95: {metrics.box.map:.4f}")
print(f"Precision: {metrics.box.mp:.4f}")
print(f"Recall: {metrics.box.mr:.4f}")
```

---

## 🎯 关键要点总结

### 第一条建议：预训练权重转换
✅ **必须步骤：**
1. 使用 `prepare_4ch_weights.py` 创建4通道预训练权重
2. 验证第一层卷积输入通道为4
3. 训练时使用转换后的 `yolov8_4ch_direct.pt`

❌ **常见错误：**
- 直接使用 `yolov8n.pt`（3通道）训练RGBD模型
- 忘记验证转换是否成功
- 配置文件设置 `ch: 4` 但未转换权重

### 第三条建议：数据加载器
✅ **必须步骤：**
1. 将RGB和Depth融合为单个4通道PNG文件
2. 使用 `cv2.IMREAD_UNCHANGED` 读取完整4通道
3. 在数据集YAML中设置 `rgbd: true` 和 `channels: 4`

❌ **常见错误：**
- RGB和Depth分开存放但未融合
- 使用 `cv2.IMREAD_COLOR` 只读取3通道
- 深度通道未正确归一化

---

## 🔍 故障排查

### 问题1：训练时报错 "shape mismatch"
**原因：** 模型期望4通道输入，但数据只有3通道
**解决：** 确保数据融合为4通道PNG，使用 `cv2.IMREAD_UNCHANGED` 读取

### 问题2：模型仍然是3通道
**原因：** 预训练权重未正确转换
**解决：** 重新运行 `prepare_4ch_weights.py`，验证输出文件

### 问题3：深度信息没有被使用
**原因：** 第4通道全是0或未正确加载
**解决：** 检查融合脚本，确保深度图正确读取和归一化

### 问题4：性能不如RGB模型
**原因：** 深度信息质量差或未经过足够训练
**解决：** 
- 检查深度图质量
- 增加训练轮数
- 调整学习率和数据增强

---

## 📚 参考资料

- YOLOv8官方文档: https://docs.ultralytics.com
- 项目仓库: https://github.com/M-Sir-zhou/yolov8-rgbd-detection
- 相关脚本:
  - `scripts/prepare_4ch_weights.py` - 权重转换
  - `scripts/fuse_rgb_depth.py` - 数据融合
  - `preprocess_rgbd.py` - 自动化预处理
  - `scripts/train_4ch_fixed.py` - 训练脚本示例

---

**最后更新：** 2025年10月31日
