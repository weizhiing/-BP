from tqdm import tqdm

import BP_net as bp
import numpy as np
from PIL import Image


# todo: use softmax
# 28*28=784   12
# [784,30,14]
def get_max_index(arr):
    max = -10
    maxIdx = 0
    for i in range(len(arr)):
        if arr[i] > max:
            max = arr[i]
            maxIdx = i
    return maxIdx


def do_file(file_name):
    img = Image.open(file_name)
    pixels = list(img.getdata())
    for i in range(len(pixels)):
        pixels[i] /= 255
    img.close()
    return pixels


network = bp.BP_Network([784, 50, 12], 0.05, 'sigmoidal')
# 读取
train_number = 621 * 12
train_set = np.ndarray(train_number, np.object)
for imgIdx in range(621):
    for wordIdx in range(1, 13):
        fileName = 'train/' + str(wordIdx) + '/' + str(imgIdx) + '.bmp'
        try:
            data = np.array(do_file(fileName))
            train_set[(wordIdx - 1) * 621 + imgIdx] = data
        except:
            continue

# 关联
expect_set = np.ndarray(train_number, np.object)
tmp_set = []
for wordIdx in range(1, 13):
    expectation = np.zeros(12)
    expectation[wordIdx - 1] = 1
    for imgIdx in range(621):
        # expect_set[(wordIdx - 1) * 621 + imgIdx] = expectation
        tmp = [train_set[((wordIdx - 1) * 621 + imgIdx)], expectation]
        tmp_set.append(tmp)

train_time = 0
np.random.shuffle(tmp_set)
tmp_set = np.array(tmp_set)
train_set = tmp_set[:, 0]
expect_set = tmp_set[:, 1]

# 训练开始
for i in tqdm(range(50)):
    precision = 0
    for idx in range(train_number):
        input_data = train_set[idx]
        if input_data is not None:
            expectation = expect_set[idx]
            network.train_start(input_data, expectation, 5)

# #测试
# print("测试开始***************")
# test_len=0
# test_success_len=0
# for imgIdx in range(621):
#     for wordIdx in range(1, 13):
#         fileName = 'my_test/' + str(wordIdx) + '/' + str(imgIdx) + '.bmp'
#         try:
#             data = np.array(do_file(fileName))
#             test_len=test_len+1
#             output_data = network.forward_count(data)
#             ss= get_max_index(output_data)
#             print(str(ss)+" "+str(wordIdx-1))
#             if ss == wordIdx-1:
#                 test_success_len = test_success_len + 1
#         except:
#             continue
#
# print("成功率："+str(test_success_len/test_len))
file = "pred.txt"
f = open(file, 'w')
for img in range(1,1801):
    fileName = 'test/' + str(img) + '.bmp'
    try:
        data = np.array(do_file(fileName))
    except Exception:
        continue
    output_data = network.forward_count(data)
    f.write(str(get_max_index(output_data) + 1) + '\n')
f.close()
