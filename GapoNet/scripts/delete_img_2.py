import cv2
import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# 设置光流法参数
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# 读取图像文件夹中的图像
image_folder = 'path_to_image_folder'
image_files = sorted(os.listdir(image_folder))

# 定义处理函数
def process_image(prev_gray, current_gray, filename1, filename2):
    flow = cv2.calcOpticalFlowFarneback(prev_gray, current_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    mean_magnitude = np.mean(magnitude)
    if mean_magnitude < 10.0:
        os.remove(os.path.join(image_folder, filename2))
        print(f"Deleted {filename2} due to low optical flow magnitude")

# 创建线程池
with ThreadPoolExecutor() as executor:
    for i in range(len(image_files)):
        prev_gray = cv2.cvtColor(cv2.imread(os.path.join(image_folder, image_files[i])), cv2.COLOR_BGR2GRAY)
        for j in range(i+1, len(image_files)):
            current_gray = cv2.cvtColor(cv2.imread(os.path.join(image_folder, image_files[j])), cv2.COLOR_BGR2GRAY)
            executor.submit(process_image, prev_gray, current_gray, image_files[i], image_files[j])

