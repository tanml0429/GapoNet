import os


bbox = [0, 152, 367, 530]
w = 611
h = 530
y = [0, 0, 0, 0]
y[0] = (bbox[0] + bbox[2]) / 2 / w
y[1] = (bbox[1] + bbox[3]) / 2 / h
y[2] = (bbox[2] - bbox[0]) / w
y[3] = (bbox[3] - bbox[1]) / h
# 取小数点后六位并四舍五入
y = [round(i, 6) for i in y]
print(y)

x_min = y[0] - y[2] / 2
y_min = y[1] - y[3] / 2
x_max = x_min + y[2]
y_max = y_min + y[3]

print(x_min, y_min, x_max, y_max)