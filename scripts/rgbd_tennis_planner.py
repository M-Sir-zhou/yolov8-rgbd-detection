#   python "D:\ProjectCode\PyCharm\ultralytics-main\scripts\rgbd_tennis_planner.py" --input "D:\ProjectCode\PyCharm\ultralytics-main\datasets\path_out\rgb(1)_rgbd.png" --out "D:\ProjectCode\PyCharm\ultralytics-main\scripts\path_results\path(1).png" --weights "D:\ProjectCode\PyCharm\ultralytics-main\runs\detect\train16\weights\best.pt" --device 0
import cv2
import numpy as np
import argparse
import os
from math import hypot

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

# ...existing code...
def detect_with_yolo(rgb_image, model=None, conf=0.25, imgsz=640, device='cpu'):
    """更稳健的 YOLO 推理：尝试 RGB/BGR 输入并打印调试信息"""
    if model is None:
        return []

    img = rgb_image
    if img.dtype != np.uint8:
        img = (cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)).astype(np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    for mode in ('rgb', 'bgr'):
        inp = img if mode == 'rgb' else cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        try:
            results = model(inp, device=device, imgsz=imgsz, conf=conf)
        except Exception as e:
            print(f"YOLO 推理失败 ({mode}):", e)
            continue

        if not results or len(results) == 0:
            continue

        r = results[0]
        if not (hasattr(r, "boxes") and r.boxes is not None and len(r.boxes) > 0):
            continue

        try:
            xyxy = r.boxes.xyxy.cpu().numpy()
            scores = r.boxes.conf.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy()
        except Exception:
            # 兼容不同版本
            try:
                xyxy = np.array([b.xyxy for b in r.boxes])
                scores = np.zeros(len(xyxy))
                classes = np.zeros(len(xyxy), dtype=int)
            except Exception:
                continue

        dets = []
        for (x1, y1, x2, y2), sc, cl in zip(xyxy, scores, classes):
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            dets.append({'bbox': (x1, y1, x2 - x1, y2 - y1), 'score': float(sc), 'class': int(cl)})
        print(f"YOLO({mode}) detected {len(dets)} boxes; model.names:", getattr(model.model, 'names', {}))
        return dets

    return []
# ...existing code...

# ...existing code...
def load_rgbd(rgb_path, depth_path=None):
    """
    加载 RGB 和 深度图，返回 (rgb(H,W,3) uint8, depth(H,W) float32 in 0..1 or None)
    兼容 uint8/uint16/float 的输入，若 RGB 为 uint16 会转换为 uint8 以供 cvtColor 使用。
    """
    if not os.path.exists(rgb_path):
        raise FileNotFoundError(f"rgb not found: {rgb_path}")
    im = cv2.imread(rgb_path, cv2.IMREAD_UNCHANGED)
    if im is None:
        raise FileNotFoundError(f"cannot read rgb image: {rgb_path}")

    depth = None

    def to_uint8(img):
        # 将任意整型/浮点图转换为 uint8，保留亮度关系（uint16 -> 0..255）
        if img.dtype == np.uint8:
            return img
        if np.issubdtype(img.dtype, np.integer):
            # 整型（如 uint16）：按 0..65535 -> 0..255 映射
            return (img.astype(np.uint32) // 257).astype(np.uint8)
        else:
            # 浮点：归一化后映射到 0..255
            tmp = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
            return tmp.astype(np.uint8)

    # 情况1：4通道图像（BGR + depth），且未另外给 depth_path
    if im.ndim == 3 and im.shape[2] == 4 and depth_path is None:
        bgr = im[:, :, :3]
        depth_ch = im[:, :, 3]
        bgr8 = to_uint8(bgr)
        rgb = cv2.cvtColor(bgr8, cv2.COLOR_BGR2RGB)
        depth = _process_depth_channel(depth_ch)
        return rgb, depth

    # 情况2：3通道 RGB/BGR (+ 可选单独 depth 文件)
    if im.ndim == 3 and im.shape[2] >= 3:
        bgr = im[:, :, :3]
        bgr8 = to_uint8(bgr)
        rgb = cv2.cvtColor(bgr8, cv2.COLOR_BGR2RGB)
    elif im.ndim == 2:
        # 单通道图像视为灰度 RGB
        im8 = to_uint8(im)
        rgb = cv2.cvtColor(im8, cv2.COLOR_GRAY2RGB)
    else:
        raise ValueError(f"unsupported rgb image shape: {im.shape}")

    if depth_path:
        if not os.path.exists(depth_path):
            raise FileNotFoundError(f"depth not found: {depth_path}")
        d = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if d is None:
            raise FileNotFoundError(f"cannot read depth image: {depth_path}")
        if d.ndim == 3:
            d = d[:, :, 0]
        if (d.shape[0], d.shape[1]) != (rgb.shape[0], rgb.shape[1]):
            d = cv2.resize(d, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
        depth = _process_depth_channel(d)

    return rgb, depth
# ...existing code...

def _process_depth_channel(depth_ch):
    """将各种深度通道类型归一化为 float32 0..1"""
    if depth_ch is None:
        return None
    if depth_ch.dtype == np.uint16:
        depth = depth_ch.astype(np.float32)
        mn, mx = float(depth.min()), float(depth.max())
        if mx > mn:
            depth = (depth - mn) / (mx - mn)
        else:
            depth = depth * 0.0
    elif depth_ch.dtype == np.uint8:
        depth = depth_ch.astype(np.float32) / 255.0
    else:
        # float32/float64 假定已在 0..1 或米单位（尝试归一化）
        depth = depth_ch.astype(np.float32)
        mn, mx = float(depth.min()), float(depth.max())
        if mx > mn:
            depth = (depth - mn) / (mx - mn)
    return depth

def _ensure_uint8_rgb(rgb):
    """确保 rgb 为 uint8 三通道图像（返回 new array）"""
    if rgb is None:
        return None
    if rgb.dtype == np.uint8:
        out = rgb
    elif np.issubdtype(rgb.dtype, np.integer):
        out = (rgb.astype(np.uint32) // 257).astype(np.uint8)  # uint16 -> uint8 映射
    else:
        tmp = cv2.normalize(rgb, None, 0, 255, cv2.NORM_MINMAX)
        out = tmp.astype(np.uint8)
    # 若有 4 通道（意外），去掉第4通道
    if out.ndim == 3 and out.shape[2] == 4:
        out = out[:, :, :3]
    # 若单通道，转为三通道
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2RGB)
    return out

def color_detect_tennis(rgb, debug_out_dir=None):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    candidates = [
        (np.array([20, 80, 80]), np.array([45, 255, 255])),
        (np.array([18, 60, 60]), np.array([50, 255, 255])),
        (np.array([15, 40, 40]), np.array([60, 255, 255])),
        (np.array([10, 30, 30]), np.array([75, 255, 255])),
    ]
    best_mask = None
    best_count = -1
    for idx, (low, high) in enumerate(candidates):
        mask = cv2.inRange(hsv, low, high)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
        cnt = int((mask > 0).sum())
        if cnt > best_count:
            best_count = cnt
            best_mask = mask
        if debug_out_dir:
            os.makedirs(debug_out_dir, exist_ok=True)
            cv2.imwrite(os.path.join(debug_out_dir, f"mask_try_{idx}.png"), mask)
    if best_count <= 5:
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        b_chan = lab[:, :, 2]
        th = max(150, int(np.percentile(b_chan, 95)))
        mask_lab = (b_chan >= th).astype(np.uint8) * 255
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_lab = cv2.morphologyEx(mask_lab, cv2.MORPH_OPEN, k, iterations=1)
        mask_lab = cv2.morphologyEx(mask_lab, cv2.MORPH_CLOSE, k, iterations=1)
        if int((mask_lab > 0).sum()) > best_count:
            best_mask = mask_lab
            best_count = int((mask_lab > 0).sum())
            if debug_out_dir:
                cv2.imwrite(os.path.join(debug_out_dir, "mask_lab.png"), mask_lab)
    mask = best_mask if best_mask is not None else np.zeros(hsv.shape[:2], dtype=np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 20:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        M = cv2.moments(cnt)
        if M.get("m00", 0) != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = x + w // 2, y + h // 2
        detections.append({'bbox': (x, y, w, h), 'centroid': (cx, cy), 'area': area})
    return detections, mask

def estimate_positions(detections, depth):
    pts = []
    for d in detections:
        x, y, w, h = d['bbox']
        cx, cy = d['centroid']
        z = None
        if depth is not None:
            dx1, dy1 = max(0, x), max(0, y)
            dx2, dy2 = min(depth.shape[1], x + w), min(depth.shape[0], y + h)
            crop = depth[dy1:dy2, dx1:dx2]
            if crop.size > 0:
                z = float(np.median(crop))
        pts.append({'centroid': (cx, cy), 'z': z, 'bbox': d['bbox'], 'area': d['area']})
    return pts

def plan_path_nn(points, start_idx=0):
    if not points:
        return []
    unused = list(range(len(points)))
    order = []
    cur = unused.pop(start_idx if start_idx < len(unused) else 0)
    order.append(cur)
    while unused:
        best = None
        bestd = float('inf')
        x0, y0 = points[cur]['centroid']
        z0 = points[cur]['z'] if points[cur]['z'] is not None else 0.0
        for j in unused:
            x1, y1 = points[j]['centroid']
            z1 = points[j]['z'] if points[j]['z'] is not None else 0.0
            d = hypot(x1 - x0, y1 - y0) + 0.5 * abs(z1 - z0)
            if d < bestd:
                bestd = d
                best = j
        unused.remove(best)
        order.append(best)
        cur = best
    return order

def visualize(rgb, detections_pts, order, out_path, mask=None):
    vis = rgb.copy()
    vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
    for i, p in enumerate(detections_pts):
        cx, cy = p['centroid']
        x, y, w, h = p['bbox']
        z = p['z']
        cv2.rectangle(vis_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label = f"{i}"
        if z is not None:
            label += f" z={z:.2f}"
        cv2.putText(vis_bgr, label, (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    if order:
        for i in range(len(order) - 1):
            a = detections_pts[order[i]]['centroid']
            b = detections_pts[order[i + 1]]['centroid']
            cv2.line(vis_bgr, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), (0, 0, 255), 2)
        s = detections_pts[order[0]]['centroid']
        cv2.circle(vis_bgr, (int(s[0]), int(s[1])), 6, (255, 0, 0), -1)
    # 保存 RGB 可视化
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or '.', exist_ok=True)
    cv2.imwrite(out_path, vis_bgr)
    # 若需要，返回保存路径
    return out_path

def main():
    parser = argparse.ArgumentParser(description="基于 RGBD 的网球聚类与路径规划（测试版）")
    parser.add_argument('--input', '-i', required=True, help="输入 RGBD 图像路径 (四通道 PNG 或 3通道 + 单通道深度可另传)")
    parser.add_argument('--out', '-o', default='plan_vis.png', help="输出可视化路径图片")
    parser.add_argument('--weights', '-w', default=None, help="YOLO 权重文件 (.pt)，不提供则使用颜色检测")
    parser.add_argument('--conf', type=float, default=0.25, help="YOLO 置信度阈值")
    parser.add_argument('--imgsz', type=int, default=640, help="YOLO 推理输入尺寸")
    parser.add_argument('--device', default='cpu', help="设备: 'cpu' 或 GPU 索引如 '0' 或 'cuda:0'")
    args = parser.parse_args()

    # 加载 RGBD 图像（保留原有 load_rgbd）
    rgb, depth = load_rgbd(args.input)

    # 若提供权重且 ultralytics 可用，则加载模型
    yolo_model = None
    if args.weights:
        if YOLO is None:
            print("ultralytics 未安装或无法导入，无法使用 YOLO；将回退到颜色检测。")
        else:
            try:
                yolo_model = YOLO(args.weights)
                print("已加载 YOLO 权重:", args.weights)
            except Exception as e:
                print("加载 YOLO 权重失败，回退到颜色检测。错误：", e)
                yolo_model = None

    # 使用 YOLO 检测（如果模型可用），否则使用颜色分割
    detections = []
    mask = None
    if yolo_model is not None:
        yolo_dets = detect_with_yolo(rgb, model=yolo_model, conf=args.conf, imgsz=args.imgsz, device=args.device)
        if yolo_dets:
            for d in yolo_dets:
                x, y, w, h = d['bbox']
                cx, cy = x + w // 2, y + h // 2
                detections.append({'bbox': (x, y, w, h), 'centroid': (cx, cy), 'area': w * h})
        else:
            print("YOLO 未检测到目标，回退到颜色分割。")
            detections, mask = color_detect_tennis(rgb)
    else:
        detections, mask = color_detect_tennis(rgb)

    pts = estimate_positions(detections, depth)
    order = plan_path_nn(pts)

    # 输出路径顺序
    print("检测到对象数:", len(pts))
    for idx, i in enumerate(order):
        c = pts[i]['centroid']
        z = pts[i]['z']
        print(f"{idx+1}. id={i}, centroid={c}, depth={z}")

    # 可视化并保存
    mask_val = mask if 'mask' in locals() else None
    visualize(rgb, pts, order, args.out, mask=mask_val)
    print("可视化已保存到:", os.path.abspath(args.out))

if __name__ == '__main__':
    main()