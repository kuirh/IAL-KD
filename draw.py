import os
import json
import matplotlib.pyplot as plt
folder_path = "output/experiment/9year12/ab_loss"
import plotly.graph_objects as go
import kaleido
import numpy as np
# 遍历文件夹中的文件
def calculate_column_averages(model_accuracy_array):
    max_length_1 = max(len(sublist) for sublist in model_accuracy_array)

    # Pad the shorter sublists with None values
    padded_array = [sublist + [None] * (max_length_1 - len(sublist)) for sublist in model_accuracy_array]

    # Convert the padded array to a NumPy ndarray


    num_rows = len(padded_array)
    num_cols = len(padded_array[0])
    column_sums = [0] * num_cols
    column_counts = [0] * num_cols

    for row in padded_array:
        for col_idx, value in enumerate(row):
            if value is not None:
                column_sums[col_idx] += value
                column_counts[col_idx] += 1

    column_averages = [column_sums[col_idx] / column_counts[col_idx] if column_counts[col_idx] > 0 else None for col_idx in range(num_cols)]
    return column_averages
# Define the function to calculate the average length of lists in a list of lists
def calculate_column_time(lists):
    # Calculate the total length of all lists
    total_length = sum(len(inner_list) for inner_list in lists)
    # Calculate the number of lists
    number_of_lists = len(lists)
    # Calculate the average length
    average = total_length / number_of_lists if number_of_lists > 0 else 0
    return average

# Test the function with a placeholder list of lists
# Uncomment the line below to test the function
# average_length([[1, 2, 3], [4, 5], [6, 7, 8, 9]])

def draw_3_list(model1_accuracy_list,model2_accuracy_list,model3_accuracy_list,file_path):
    data = [
        go.Scatter(x=list(range(len(model1_accuracy_list))), y=model1_accuracy_list, mode='lines',
                   name='Model M^t', marker=dict(color='#0072BD')),  # 设置为plotly模板中的第一个颜色
        go.Scatter(x=list(range(len(model2_accuracy_list))), y=model2_accuracy_list, mode='lines',
                   name='Model M^{t-1}', marker=dict(color='#D95319')),  # 设置为plotly模板中的第二个颜色
        go.Scatter(x=list(range(len(model3_accuracy_list))), y=model3_accuracy_list, mode='lines',
                   name='Model DirectRetraining', marker=dict(color='#EDB120')),  # 设置为plotly模板中的第三个颜色
    ]
    # 0072BD
    # D95319
    # EDB120
    # 7E2F8E
    # 77AC30
    # 4DBEEE
    # A2142F
    # 创建图表布局
    #title='Model Accuracy'#paper_bgcolor="LightSteelBlue",
    layout = go.Layout(autosize=True,width=800,height=500,
                       xaxis=dict(title='Epoch'), yaxis=dict(title='Accuracy(%)'),
                       legend=dict(yanchor="bottom",y=0.01,xanchor="right",x=0.99),
                       margin=dict(l=10, r=10, t=10, b=10),
                       )

    # 创建图表对象
    fig = go.Figure(data=data, layout=layout)

    # 显示图表
    fig.write_image(file_path + ".png", format="png")


def draw_best_list(model1_accuracy_best, model2_accuracy_best, model3_accuracy_best, file_path):
    data = [model1_accuracy_best, model2_accuracy_best, model3_accuracy_best]
    labels = ['Model M^t', 'Model M^{t-1}', 'Model DirectRetraining']

    fig = go.Figure()

    for i in range(len(data)):
        fig.add_trace(go.Box(y=data[i], name=labels[i], marker=dict(color=['#0072BD', '#D95319', '#EDB120'][i])))

    # 使用与 draw_3_list 相同的布局设置
    layout = go.Layout(autosize=True, width=800, height=500,
                        yaxis=dict(title='Accuracy(%)'),
                       legend=dict(yanchor="bottom", y=0.01, xanchor="right", x=0.99),
                       margin=dict(l=10, r=10, t=10, b=10),
                       )

    # 应用新的布局
    fig.update_layout(layout)

    # 显示图表
    fig.write_image(file_path + ".png", format="png")



model1_accuracy_array = []
model2_accuracy_array = []
model3_accuracy_array = []

model1_accuracy_best = []
model2_accuracy_best = []
model3_accuracy_best = []

model1_accuracy_old = []
model3_accuracy_old=[]

model1_time_list = []
model2_time_list = []
model3_time_list = []
model_accuracy_best=[]

model_s_similarity_list = []
model_d_similarity_list = []

for filename in os.listdir(folder_path):
    # 检查文件扩展名是否为 .json
    if filename.endswith(".json"):
        # 构建文件的完整路径
        file_path = os.path.join(folder_path, filename)
        # 打开 JSON 文件并读取内容
        with open(file_path, "r") as json_file:
            print(file_path)
            metadata = json.load(json_file)
            print(metadata['seed'])
            model1_accuracy_list = []
            model2_accuracy_list = []
            model3_accuracy_list = []

            if 'f_best_acc' in metadata:
                model_accuracy_best.append(metadata['f_best_acc'])

            for i in range(len(metadata['student_val_scores'])):
                model1_accuracy_list.append(metadata['student_val_scores'][i]['val']['score'])
            for i in range(len(metadata['teacher_val_scores'])):
                model2_accuracy_list.append(metadata['teacher_val_scores'][i]['val']['score'])
            for i in range(len(metadata['direct_val_scores'])):
                model3_accuracy_list.append(metadata['direct_val_scores'][i]['val']['score'])

            if 'student_similarity' in metadata:
                for i in range(len(metadata['student_similarity'])):
                    if 0 <= metadata['student_similarity'][i] <= 1:
                        model_s_similarity_list.append(metadata['student_similarity'][i])

            if 'direct_similarity' in metadata:
                for i in range(len(metadata['direct_similarity'])):
                    if 0 <= metadata['direct_similarity'][i] <= 1:
                        model_d_similarity_list.append(metadata['direct_similarity'][i])

            model1_time_list.append(metadata['distill_execution_time'])
            model2_time_list.append(metadata['teacher_executiontime'])
            model3_time_list.append(metadata['direct_execution_time'])
            model1_accuracy_array.append(model1_accuracy_list)
            model1_accuracy_best.append(metadata['distill_best_acc'])
            model2_accuracy_array.append(model2_accuracy_list)
            model2_accuracy_best.append(metadata['teacher_best_acc'])
            model3_accuracy_array.append(model3_accuracy_list)
            model3_accuracy_best.append(metadata['direct_best_acc'])
            # model1_accuracy_old.append(max(metadata['add_val_scores']))
            # model3_accuracy_old.append(max(metadata['old_val_scores']))

            # x 轴的 epoch 数
            #draw_3_list(model1_accuracy_list,model2_accuracy_list,model3_accuracy_list,file_path)


        # 处理读取的 JSON 数据
        # 在这里进行你想要的操作，比如打印或其他处理



column_averages_1 = calculate_column_averages(model1_accuracy_array)
column_averages_2 = calculate_column_averages(model2_accuracy_array)
column_averages_3 = calculate_column_averages(model3_accuracy_array)


print(calculate_column_time(model1_accuracy_array),calculate_column_time(model2_accuracy_array),calculate_column_time(model3_accuracy_array))

draw_3_list(model1_accuracy_list, model2_accuracy_list, model3_accuracy_list, os.path.join(folder_path, "average"))
draw_best_list(model1_accuracy_best, model2_accuracy_best, model3_accuracy_best, os.path.join(folder_path, "best"))
print(sum(model1_accuracy_best)/len(model1_accuracy_best),sum(model2_accuracy_best)/len(model2_accuracy_best),sum(model3_accuracy_best)/len(model3_accuracy_best))
print(sum(model1_time_list)/len(model1_time_list),sum(model2_time_list)/len(model2_time_list),sum(model3_time_list)/len(model3_time_list))
#draw_best_list(model1_accuracy_old,model2_accuracy_best ,model3_accuracy_old, os.path.join(folder_path, "old"))
if len(model_accuracy_best)!=0:
    print('early stop',sum(model_accuracy_best)/len(model_accuracy_best))

if len(model_d_similarity_list)!=0:
    print("{:.4f}".format(sum(model_s_similarity_list)/len(model_s_similarity_list)),"{:.4f}".format(sum(model_d_similarity_list)/len(model_d_similarity_list)))
print('over')

