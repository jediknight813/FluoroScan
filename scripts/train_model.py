from ultralytics import YOLO

model = YOLO('yolov8n.yaml')
model.train(data='./config.yaml', epochs=200, device=0, workers=1, imgsz=512, batch=4)
