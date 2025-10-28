import cv2
import time
import argparse
import os
import numpy as np
from ultralytics import YOLO

def draw_boxes(frame, boxes, scores=None, classes=None, names=None, colors=None):
    """兼容处理：即使 scores/classes 为空也绘制 boxes（labels 使用类 id）"""
    if boxes is None or len(boxes) == 0:
        return
    # 保证数组
    boxes = np.asarray(boxes)
    n = boxes.shape[0]
    if scores is None or len(scores) != n:
        scores = np.zeros(n)
    if classes is None or len(classes) != n:
        classes = np.zeros(n, dtype=int)
    names = {} if names is None else names
    colors = np.zeros((max(1, len(names)), 3), dtype=np.uint8) if colors is None else colors

    for i in range(n):
        x1, y1, x2, y2 = boxes[i].astype(int)
        conf = float(scores[i])
        cls = int(classes[i])
        color = tuple(int(c) for c in colors[cls % len(colors)])
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{names.get(cls, str(cls))} {conf:.2f}"
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.rectangle(frame, (x1, y1 - t_size[1] - 6), (x1 + t_size[0] + 6, y1), color, -1)
        cv2.putText(frame, label, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

def main():
    parser = argparse.ArgumentParser(description="实时 YOLO 视频检测")
    parser.add_argument('--weights', '-w', required=True, help="YOLO 权重文件 (.pt)")
    parser.add_argument('--source', '-s', default=0, help="视频源：文件路径 或 摄像头索引（0）")
    parser.add_argument('--out', '-o', default=None, help="保存输出视频路径（可选）")
    parser.add_argument('--imgsz', type=int, default=640, help="网络输入尺寸")
    parser.add_argument('--conf', type=float, default=0.25, help="置信度阈值")
    parser.add_argument('--device', default='0', help="设备: 'cpu' 或 GPU 索引如 '0' 或 'cuda:0'")
    args = parser.parse_args()

    # 解析 source（数字 => 摄像头）
    try:
        source = int(args.source)
    except Exception:
        source = args.source

    # 加载模型
    model = YOLO(args.weights)

    # 准备视频流
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("无法打开视频源:", args.source)
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0

    # 输出视频 writer
    writer = None
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.out, fourcc, fps_in, (width, height))

    # 颜色和类名
    names = model.model.names if hasattr(model, "model") and hasattr(model.model, "names") else {}
    rng = np.random.default_rng(12345)
    colors = (rng.integers(0, 255, size=(max(1, len(names)), 3))).astype(np.uint8)

    prev_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO 推理（传入 BGR 图像）
        results = model(frame, device=args.device, imgsz=args.imgsz, conf=args.conf)
        r = results[0]

        boxes = np.empty((0,4))
        scores = np.array([])
        classes = np.array([])
        if hasattr(r, "boxes") and r.boxes is not None and len(r.boxes) > 0:
            boxes = r.boxes.xyxy.cpu().numpy()
            # 有些 ultralytics 版本上面属性名可能不同，做容错
            try:
                scores = r.boxes.conf.cpu().numpy()
                classes = r.boxes.cls.cpu().numpy()
            except Exception:
                # 如果没有 conf/cls 属性，构造占位
                scores = np.zeros(len(boxes))
                classes = np.zeros(len(boxes), dtype=int)

        # 打印检测数量用于调试
        print(f"detected: {len(boxes)} boxes")

        # 绘制框
        draw_boxes(frame, boxes, scores, classes, names, colors)

        # FPS 显示
        now = time.time()
        fps = 1.0 / (now - prev_time) if now != prev_time else 0.0
        prev_time = now
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        # 显示
        cv2.imshow("YOLO RealTime", frame)
        if writer:
            # 确保 writer 可写并写入 uint8 BGR 帧
            if writer.isOpened():
                if frame.dtype != np.uint8:
                    frame_to_write = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)
                else:
                    frame_to_write = frame
                writer.write(frame_to_write)
            else:
                print("Warning: video writer is not opened, output will not be saved.")

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()




#  mkdir "D:\ProjectCode\PyCharm\ultralytics-main\scripts\path_results" -ErrorAction SilentlyContinue 
# python d:\ProjectCode\PyCharm\ultralytics-main\scripts\realtime_yolo_video.py --weights "D:\ProjectCode\PyCharm\ultralytics-main\runs\detect\train19\weights\best.pt" --source "D:\123\Videos\Screenshot\apple.mp4" --out "D:\ProjectCode\PyCharm\ultralytics-main\scripts\path_results\out.mp4" --device 0

