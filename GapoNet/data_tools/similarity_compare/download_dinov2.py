from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import requests

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

# proxy = 'http://localhost:8118'
proxy = 'http://192.168.32.148:8118'
proxies = {
    'http': proxy,
    'https': proxy,
}

processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base', proxies=proxies)
model = AutoModel.from_pretrained('facebook/dinov2-base', proxies=proxies)

inputs = processor(images=image, return_tensors="pt")  # [1, 3, 224, 224]
outputs = model(**inputs)
last_hidden_states = outputs.last_hidden_state  # [1, 257, 768]
pooler_output = outputs.pooler_output  # [1, 768]
pass