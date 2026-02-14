from ultralytics import YOLO
import os 

cwd = os.getcwd()
filename = "data/test.jpg" # in real word case this image will be taken from somewhere like S3 

filepath = os.path.join(cwd, filename)

model = YOLO("yolo11n.pt")

results = model(filepath)
print(len(results))

for result in results:
    names = [result.names[cls.item()] for cls in result.boxes.cls.int()]

print(names)

