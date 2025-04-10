from ultralytics import YOLO

# Load a model
model = YOLO("/home/tml/VSProjects/EnpoNet/EnpoNet/ultralytics/runs/detect/train5/weights/best.pt")

# Customize validation settings
validation_results = model.val(data="/home/tml/VSProjects/EnpoNet/EnpoNet/configs/data_cfgs/enpo_80.yaml", batch=1,  device="1")