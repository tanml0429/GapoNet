import tensorflow as tf
import numpy as np
from PIL import Image
import os
from tqdm import tqdm

# 计算信息熵
def entropy(data):
    unique, counts = np.unique(data, return_counts=True)
    probabilities = counts / len(data)
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy

# 读取图片数据集并计算信息熵
def process_images_in_batches(image_paths, batch_size=1000):
    entropy_values = []
    
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Processing images in batches"):
        batch_data = []
        
        for img_path in image_paths[i:i+batch_size]:
            img = Image.open(img_path)
            img_data = np.array(img).flatten()
            batch_data.extend(img_data)
        
        batch_data = np.array(batch_data)
        
        with tf.device('/GPU:0'):
            batch_entropy = entropy(batch_data)
        
        entropy_values.append(batch_entropy)
    
    total_entropy = np.sum(entropy_values)
    return total_entropy

# 读取图片数据集
image_folder = "/home/tml/datasets/enpo_dataset/mixed_69959/images/train"
image_paths = [os.path.join(image_folder, img) for img in os.listdir(image_folder)]

# 按批次处理图片数据并计算信息熵
total_entropy = process_images_in_batches(image_paths, batch_size=1000)

print(f"信息熵: {total_entropy}")
