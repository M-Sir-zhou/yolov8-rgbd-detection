import cv2
p = r"D:\ProjectCode\PyCharm\ultralytics-main\datasets\tennis-yolo\images\train\tennis (1).png"
im = cv2.imread(p, cv2.IMREAD_UNCHANGED)
print(p, im.shape)  # 期望 (H, W, 4)