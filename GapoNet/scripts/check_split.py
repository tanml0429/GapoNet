import os
import shutil
import matplotlib.pyplot as plt
import cv2
from pathlib import Path



# 聚类结果文件夹路径
tar_path = f'/data/tml/split_polyp_v5_format'
target_path = Path(tar_path)
normal_path = f'{target_path}/normal_endoscopy/images'
n_labels_path = f'{target_path}/normal_endoscopy/labels'
dyed_path = f'{target_path}/dyed_endoscopy/images'
d_labels_path = f'{target_path}/dyed_endoscopy/labels'

result_path = f'{target_path.parent}/check_split_polyp_v5_format'
# 染色内镜图片文件夹路径
check_normal_path = f'{result_path}/normal_endoscopy/images'
check_n_labels_path = f'{result_path}/normal_endoscopy/labels'
check_dyed_path = f'{result_path}/dyed_endoscopy/images'
check_d_labels_path = f'{result_path}/dyed_endoscopy/labels'


os.makedirs(f'{check_normal_path}', exist_ok=True)
os.makedirs(f'{check_n_labels_path}', exist_ok=True)
os.makedirs(f'{check_dyed_path}', exist_ok=True)
os.makedirs(f'{check_d_labels_path}', exist_ok=True)

result_file = f'{result_path}/result.txt'
if not os.path.exists(result_file):
    with open(result_file, 'w') as f:
        f.write('')

done_files = set()
if os.path.exists(result_file):
    with open(result_file, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            done_files.add(line)


#清空文件夹

# for path in [check_dyed_path, check_dyed_path, check_n_labels_path, check_d_labels_path]:
#     for root, dirs, files in os.walk(path):
#         for file in files:
#             os.remove(os.path.join(root, file))

# 遍历聚类结果文件夹
for path in [normal_path, dyed_path]:
    for filename in os.listdir(path):
        print (filename)
        print (path)
        filepath = os.path.join(path, filename)
        labelpath = f'{filepath}'.replace('/images/', '/labels/').replace('.jpg', '.txt')
        if filename in done_files:
            continue
        if os.path.isfile(filepath):
            # 显示图片
            print("当前图片：", filename)
            img = cv2.imread(filepath)
            cv2.imshow("Image", img)
            key = cv2.waitKey(0)
            with open(result_file, 'a') as f:
                if key == ord('d'):
                    f.write(filename + ': 染色内镜\n')
                elif key == ord('n'):
                    f.write(filename + ': 普通内镜\n')
            if key == ord('d'):
                print("User pressed 'd'")
                shutil.copy(filepath, os.path.join(check_dyed_path, filename))
                shutil.copy(labelpath, os.path.join(check_d_labels_path, filename.replace('jpg', 'txt')))
            elif key == ord('n'):
                print("User pressed 'n'")
                shutil.copy(filepath, os.path.join(check_normal_path, filename))
                shutil.copy(labelpath, os.path.join(check_n_labels_path, filename.replace('jpg', 'txt')))
            cv2.destroyAllWindows()
            # os.system("display " + filepath)
            
            # 提示用户输入标签
            # label = input("请输入标签（d代表染色内镜，n代表普通内镜）：")
            
            # 复制文件到相应文件夹
            # if label == 'd':
            #     shutil.copy(filepath, os.path.join(check_dyed_path, filename))
            #     shutil.copy(labelpath, os.path.join(check_d_labels_path, filename.replace('jpg', 'txt')))

            # elif label == 'n':
            #     shutil.copy(filepath, os.path.join(check_normal_path, filename))
            #     shutil.copy(labelpath, os.path.join(check_n_labels_path, filename.replace('jpg', 'txt')))
