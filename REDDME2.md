yolov8模型配置修改：
D:\ProjectCode\PyCharm\ultralytics-main\ultralytics\cfg\models\v8\yolov8-rgbd.yaml
自训练数据集配置：
D:\ProjectCode\PyCharm\ultralytics-main\datasets\tennis-yolo\tennis-yolo.yaml
拼接4通道：
D:\ProjectCode\PyCharm\ultralytics-main\preprocess_rgbd.py

训练：
yolo detect train data=D:/ProjectCode/PyCharm/ultralytics-main/datasets/tennis-yolo/tennis-yolo.yaml model=D:/ProjectCode/PyCharm/ultralytics-main/ultralytics/cfg/models/v8/yolov8-rgbd.yaml epochs=100 batch=4 lr0=0.001 imgsz=640

rgbd
yolo detect train data=D:/ProjectCode/PyCharm/ultralytics-main/datasets/tennis-yolo/tennis-yolo.yaml model=d:/ProjectCode/PyCharm/ultralytics-main/ultralytics/cfg/models/v8/yolov8-rgbd.yaml epochs=100 batch=4 imgsz=640 pretrained=False device=0

yolo detect train data=D:\ProjectCode\PyCharm\ultralytics-main\datasets\QRcode\data.yaml model=yolov8n.yaml epochs=100 batch=4 lr0=0.001 imgsz=640

验证：
yolo val model=D:\ProjectCode\PyCharm\ultralytics-main\runs\detect\train20\weights\best.pt data=D:/ProjectCode/PyCharm/ultralytics-main/datasets/tennis_dark/tennis_dark.yaml plots=True

检测：
yolo predict model="D:\ProjectCode\PyCharm\ultralytics-main\runs\detect\train16\weights\best.pt" source="D:\ProjectCode\PyCharm\ultralytics-main\datasets\tennis-yolo\images\test\tennis (1).png"

