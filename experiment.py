import json
import os
import pickle
import random

import numpy
import torch

import distillresnet
from Tabular import lib
# def setup_seed(seed):
#     random.seed(seed)  # Python的随机性
#     os.environ['PYTHONHASHSEED'] = str(seed)  # 设置Python哈希种子，为了禁止hash随机化，使得实验可复现
#     numpy.random.seed(seed)  # numpy的随机性
#     torch.manual_seed(seed)  # torch的CPU随机性，为CPU设置随机种子
#     torch.cuda.manual_seed(seed)  # torch的GPU随机性，为当前GPU设置随机种子
#     torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.   torch的GPU随机性，为所有GPU设置随机种子
#     torch.backends.cudnn.deterministic = True  # 选择确定性算法
#     torch.backends.cudnn.benchmark = False  # if benchmark=True, deterministic will be False
#     torch.backends.cudnn.enabled = False

def delete_pickle_files(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".pickle"):
                file_path = os.path.join(root, file)
                os.remove(file_path)
                print(f"Deleted file: {file_path}")


def metatrain(teacherdataargs,teacherargs,studentdataargs,studentargs,directargs,ranseed):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    teacher_executiontime,distill_execution_time,teacher_val_scores,student_val_scores,teacher_best_acc,distill_best_acc,add_val_scores,add_acc,student_similarity,direct_execution_time,direct_val_scores,direct_best_acc,old_val_scores,old_acc,direct_similarity,f_best_acc=distillresnet.distillresnet(teacherdataargs, studentdataargs,directdataargs, teacherargs, studentargs,directargs,device,temperature=3, alpha=0.9,
                                alpha_decay=0.85, teacher_need_train=1)
    print("蒸馏模型运算耗时: {:.2f} 秒".format(distill_execution_time))



    # direct_execution_time,direct_val_scores,direct_best_acc,old_val_scores,old_acc,direct_similarity=distillresnet.directresnet(directdataargs, teacherdataargs,directargs, teacherargs,device)
    print("直接模型运算耗时: {:.2f} 秒".format(direct_execution_time))





    output_directory=os.path.join('output',teacherdataargs['data']['path'])
    output_filename=f"file_{distill_best_acc}_{distill_execution_time}_{direct_best_acc}_{direct_execution_time}_{add_acc/teacher_best_acc}.json"
    os.makedirs(output_directory, exist_ok=True)
    output_path = os.path.join(output_directory,output_filename)
    output_meta={'seed':ranseed,'student_val_scores':student_val_scores,'teacher_val_scores':teacher_val_scores,'direct_val_scores':direct_val_scores,'teacherdataargs':teacherdataargs, 'studentdataargs':studentdataargs, 'teacherargs':teacherargs, 'studentargs':studentargs, 'directargs':directargs
                 ,'teacher_best_acc':teacher_best_acc,'distill_best_acc':distill_best_acc,'direct_best_acc':direct_best_acc,'teacher_executiontime':teacher_executiontime,'distill_execution_time':distill_execution_time,'direct_execution_time':direct_execution_time,'add_val_scores':add_val_scores,'add_acc':add_acc,'old_val_scores':old_val_scores,'old_acc':old_acc,'student_similarity':student_similarity,'direct_similarity':direct_similarity,'f_best_acc':f_best_acc}
    with open(output_path, 'w') as file:
        json.dump(output_meta, file)
    delete_pickle_files(teacherdataargs['data']['path'])

# 定义温度参数
for _ in range(200):
    ranseed = random.randint(1, 2**32 - 1)


    teacherdataargs = {
        'seed': ranseed,  # 随机种子
        'data': {
            'path': 'data/helena',  # 数据集路径
            'normalization': "standard",  # 数据归一化方法
            'cat_policy': 'indices',  # 类别特征编码策略
            'cat_min_frequency': 0.0,  # 类别特征最小频率
            'y_policy': 'mean_std',  # 目标变量处理策略
            'attribute_mask': 'random'
        }
    }

    studentdataargs = {
        'seed': ranseed,  # 随机种子
        'data': {
            'path': 'data/helena',  # 数据集路径
            'normalization': "standard",  # 数据归一化方法
            'cat_policy': 'indices',  # 类别特征编码策略
            'cat_min_frequency': 0.0,  # 类别特征最小频率
            'y_policy': 'mean_std',  # 目标变量处理策略
            'attribute_mask': 'none'
        }
    }

    directdataargs = {
        'seed':ranseed ,  # 随机种子
        'data': {
            'path': 'data/helena',  # 数据集路径
            'normalization': "standard",  # 数据归一化方法
            'cat_policy': 'indices',  # 类别特征编码策略
            'cat_min_frequency': 0.0,  # 类别特征最小频率
            'y_policy': 'mean_std',  # 目标变量处理策略
            'attribute_mask': 'none'
        }
    }

    teacherargs = {
        'model': {
            "activation": "relu",
            "d": 339,
            "d_embedding": 348,
            "d_hidden_factor": 4,
            "hidden_dropout": 0.2209100377552283,
            "n_layers": 8,
            "normalization": "batchnorm",
            "residual_dropout": 0
        },
        'training': {
            "batch_size": 512,
            "eval_batch_size": 128,
            "lr": 0.001,
            "n_epochs": 50,
            "optimizer": "adamw",
            "patience": 12,
            "weight_decay": 0.0
        },
        'name': 'teacher'
    }

    studentargs = {
        'model': {
            "activation": "relu",
            "d": 339,
            "d_embedding": 348,
            "d_hidden_factor": 4,
            "hidden_dropout": 0.2209100377552283,
            "n_layers": 8,
            "normalization": "batchnorm",
            "residual_dropout": 0
        },
        'training': {
            "batch_size": 512,
            "eval_batch_size": 128,
            "lr": 0.001,
            "n_epochs": 50,
            "optimizer": "adamw",
            "patience": 12,
            "weight_decay": 0.0
        },
        'name': 'student'
    }

    directargs = {
        'model': {
            "activation": "relu",
            "d": 339,
            "d_embedding": 348,
            "d_hidden_factor": 4,
            "hidden_dropout": 0.2209100377552283,
            "n_layers": 8,
            "normalization": "batchnorm",
            "residual_dropout": 0
        },
        'training': {
            "batch_size": 512,
            "eval_batch_size": 128,
            "lr": 0.001,
            "n_epochs": 50,
            "optimizer": "adamw",
            "patience": 12,
            "weight_decay": 0.0
        },
        'name': 'direct'
    }
    metatrain(teacherdataargs, teacherargs, studentdataargs, studentargs, directargs, ranseed=ranseed)


#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"



