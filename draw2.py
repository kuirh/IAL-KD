import os
import json
import matplotlib.pyplot as plt
folder_path = "output/experiment/8california_housing12/ep2"
import plotly.graph_objects as go
import kaleido
# 遍历文件夹中的文件
teacher_avg_acc_list = {0:[],1:[],2:[],3:[],4:[]}
teacher_0 = []





for filename in os.listdir(folder_path):
    # 检查文件扩展名是否为 .json
    if filename.endswith(".json"):
        # 构建文件的完整路径
        file_path = os.path.join(folder_path, filename)

        # 打开 JSON 文件并读取内容
        with open(file_path, "r") as json_file:
            metadata = json.load(json_file)
            teacher_0.append(metadata['teacher_best_acc_list'][0])

            # x 轴的 epoch 数
            data = [
                go.Scatter(x=list(range(1, len(metadata['add_val_scores_list']) + 1),), y=metadata['add_val_scores_list'], mode='lines',
                           name='Old format input in Model M^t', marker=dict(color='#7E2F8E')),
                go.Scatter(x=list(range(1, len(metadata['student_best_acc_list']) + 1), ),
                           y=metadata['student_best_acc_list'], mode='lines',
                           name='New format input in Model M^t', marker=dict(color='#0072BD')),
                go.Scatter(x=list(range(1, len(metadata['direct_best_acc_list']) + 1)), y=metadata['direct_best_acc_list'], mode='lines',
                           name='Model DirectRetraining', marker=dict(color='#EDB120'))
            ]
            layout = go.Layout(autosize=True, width=800, height=500,
                               xaxis=dict(title='Stage t', dtick=1), yaxis=dict(title='Accuracy(%)',range=[-1,-0.5]),
                               legend=dict(yanchor="bottom", y=0.01, xanchor="right", x=0.99),
                               margin=dict(l=10, r=10, t=10, b=10),
                               )

            # 创建图表布局

            # 创建图表对象
            fig = go.Figure(data=data, layout=layout)

            # 显示图表
            fig.write_image(file_path+".png", format="png")


        # 处理读取的 JSON 数据
        # 在这里进行你想要的操作，比如打印或其他处理
        print('over')

        for i in range (len(metadata['student_best_acc_list'])):
            fr=(metadata['teacher_best_acc_list'][i]-metadata['add_val_scores_list'][i])/metadata['teacher_best_acc_list'][i]
            print(i,fr)
            teacher_avg_acc_list[i]
            teacher_avg_acc_list[i].append(fr)
            print('over')

for i in range(5):
    print("{:.4f}".format(sum(teacher_avg_acc_list[i]) / len(teacher_avg_acc_list[i])))


print("{:.4f}".format(sum(teacher_0) / len(teacher_0)))
