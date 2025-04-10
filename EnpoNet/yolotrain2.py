from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)

# Train the model with 2 GPUs
results = model.train(data="/home/tml/VSProjects/EnpoNet/EnpoNet/configs/data_cfgs/enpo_80.yaml", epochs=20, imgsz=640, device="1")