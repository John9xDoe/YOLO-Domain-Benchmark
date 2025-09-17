from ultralytics import  YOLO

# ---
import os

# Проверь точные пути
print("Текущая рабочая папка:", os.getcwd())
print("Существует ли data/images/train:", os.path.exists("data/images/train"))
print("Существует ли data/images/val:", os.path.exists("data/images/val"))

if os.path.exists("data/images/train"):
    print("Файлы в train:", os.listdir("data/images/train"))
if os.path.exists("data/images/val"):
    print("Файлы в val:", os.listdir("data/images/val"))
# ---

model = YOLO('yolov8n.yaml')

model = model.train(
    data='synth_data.yml',
    epochs=100,
    imgsz=640,
    pretrained=False,
    project='my_project',
    name='exp1'
)

metrics = model.val()
y_pred = model.predict(source='datasample.png', save=True)

model.save('models/synth-100epchs-80train-20val')