import os
import json
import matplotlib.pyplot as plt
folder_path = "output/experiment/1adult12/ep3"
import plotly.graph_objects as go
import kaleido
import numpy as np
# 遍历文件夹中的文件




ial_accuracy_list = []
ial_epoch_list = []
ial_similarity_list = []
ial_time_list = []

for filename in os.listdir(folder_path):
    # 检查文件扩展名是否为 .json
    if filename.endswith(".json"):
        # 构建文件的完整路径
        file_path = os.path.join(folder_path, filename)
        # 打开 JSON 文件并读取内容
        with open(file_path, "r") as json_file:
            print(file_path)
            metadata = json.load(json_file)

            if 'ial_time' in metadata:
                ial_time_list.append(metadata['ial_time'])

            if 'best_ial_acc' in metadata:
                ial_accuracy_list.append(metadata['best_ial_acc'])

            if 'ial_val_scores' in metadata:
                ial_epoch_list.append(len(metadata['ial_val_scores']))

            for i in range(len(metadata['similarity_list_ial'])):
                ial_similarity_list.append(metadata['similarity_list_ial'][i])






        # 处理读取的 JSON 数据
        # 在这里进行你想要的操作，比如打印或其他处
print('acc',sum(ial_accuracy_list)/len(ial_accuracy_list))
print('epoch',sum(ial_epoch_list)/len(ial_epoch_list))
print('similarity',"{:.4f}".format(sum(ial_similarity_list)/len(ial_similarity_list)))
print('time',sum(ial_time_list)/len(ial_time_list))
print('over')





