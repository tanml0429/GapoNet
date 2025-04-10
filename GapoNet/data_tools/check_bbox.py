import os

def check_yolo_bbox_format(folder_path):
    error_files = []
    error_txt = open("/home/tml/VSProjects/GapoNet/GapoNet/data_tools/error_log.txt", "w")
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            with open(os.path.join(folder_path, filename), 'r') as file:
                lines = file.readlines()
                for line_num, line in enumerate(lines, start=1):
                    values = line.strip().split()
                    x, y, width, height = map(float, values[1:])
                    x_min = x - width / 2
                    y_min = y - height / 2
                    x_max = x_min + width
                    y_max = y_min + height
                    
                    if x_min < 0 or y_min < 0 or x_max > 1 or y_max > 1:
                        error_msg = f"Error in file: {filename}, line: {line_num} - x_min: {x_min}, y_min: {y_min}, x_max: {x_max}, y_max: {y_max}\n"
                        error_files.append(error_msg)
                        error_txt.write(error_msg)
    error_txt.close()  # 关闭错误信息文本文件
    if error_files:
        print("Errors found in the following files:")
        for error in error_files:
            print(error)
    
    else:
        print("All files have correct YOLO bbox format.")
    print(len(error_files))

folder_path = "/home/tml/datasets/enpo_dataset/enpo_12910_origin/labels"
check_yolo_bbox_format(folder_path)
