from ultralytics import YOLO
from ultralytics import RTDETR

import torch
import os

if __name__ == '__main__':
    device = torch.device("cuda:0")

    # 训练train
    Dir = os.getcwd()
    # model=YOLO(r"yolov8n.pt")
    # model=YOLO(r"best.pt")
    # model = YOLO(r'yolov8n-multi-p7-1.yaml',task='multi-task')
    # model.train(data=Dir + r"/multi/multi.yaml", epochs=1, imgsz=1280, device=device, batch=3)

    model= YOLO(r"yolov8n-seg-p7-1-C2f.yaml")
    model.train(data=Dir + r"/segment/segment.yaml", epochs=1, imgsz=1280, device=device, batch=3)

    # 验证val
    # model=YOLO(r"./runs/pose/train/weights/best.pt")
    # metrics = model.val(iou=0.4)

    # 预测predict
    # model = YOLO(r"./runs/pose/train/weights/best.pt")
    # model.predict('./pose/images/test/0_59-0.png', imgsz=640, conf=0.5) # predict on an image
    # results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image

    # 导出模型
    # path = model.export(format="onnx")  # export the model to ONNX format