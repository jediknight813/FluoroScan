from ultralytics import YOLO

# model = YOLO("yolov8n.yaml")
# model = YOLO("yolov8n.pt")

# model.train(data="./config.yaml", epochs=1)
# metrics = model.val()
# results = model("https://ultralytics.com/images/bus.jpg")
# path = model.export(format="onnx")


# from ultralytics import YOLO

# Load a model
model = YOLO('./best.pt')
results = model(['./training_data/images/1_Alzheimers_Disease.jpeg', './training_data/images/3_Lung_Cancer.jpeg'])

# Process results list
for result in results:
    print(result)
