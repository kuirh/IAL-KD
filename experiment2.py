import json
import os
import pickle
import random

import numpy
import torch
# %%
import copy
import math
import typing as ty
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import TensorDataset
import sys
import time
sys.path.append('/home/zettakit/wjr/new')

from Tabular import lib
import matplotlib.pyplot as plt
import Tabular.distillresnet as distillresnet
from Tabular import lib
ranseed = 123

def setup_seed(seed):
    random.seed(seed)  # Python的随机性
    os.environ['PYTHONHASHSEED'] = str(seed)  # 设置Python哈希种子，为了禁止hash随机化，使得实验可复现
    numpy.random.seed(seed)  # numpy的随机性
    torch.manual_seed(seed)  # torch的CPU随机性，为CPU设置随机种子
    torch.cuda.manual_seed(seed)  # torch的GPU随机性，为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.   torch的GPU随机性，为所有GPU设置随机种子
    torch.backends.cudnn.deterministic = True  # 选择确定性算法
    torch.backends.cudnn.benchmark = False  # if benchmark=True, deterministic will be False
    torch.backends.cudnn.enabled = False

dataargs = {
        'seed': ranseed,  # 随机种子
        'data': {
            'path': '../data/covtype',  # 数据集路径
            'normalization': "standard",  # 数据归一化方法
            'cat_policy': 'indices',  # 类别特征编码策略
            'cat_min_frequency': 0.0,  # 类别特征最小频率
            'y_policy': 'mean_std',  # 目标变量处理策略
            'attribute_mask': 'random'
        }
    }

modelargs = {
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
        "n_epochs": 128,
        "optimizer": "adamw",
        "patience": 16,
        "weight_decay": 0.0
    },
    'name': 'teacher'
}



from Tabular import lib
class ResNet(nn.Module):
    def __init__(
            self,
            *,
            d_numerical: int,
            categories: ty.Optional[ty.List[int]],
            d_embedding: int,
            d: int,
            d_hidden_factor: float,
            n_layers: int,
            activation: str,
            normalization: str,
            hidden_dropout: float,
            residual_dropout: float,
            d_out: int,
    ) -> None:
        super().__init__()

        # 根据normalization的值，字典中的键会被映射到相应的归一化层类(nn.BatchNorm1d或nn.LayerNorm)。然后，该类会被调用并传入参数d来创建一个归一化层对象
        def make_normalization():
            return {'batchnorm': nn.BatchNorm1d, 'layernorm': nn.LayerNorm}[
                normalization
            ](d)

        self.main_activation = lib.get_activation_fn(activation)
        self.last_activation = lib.get_nonglu_activation_fn(activation)
        self.residual_dropout = residual_dropout
        self.hidden_dropout = hidden_dropout

        d_in = d_numerical
        d_hidden = int(d * d_hidden_factor)

        if categories is not None:
            d_in += len(categories) * d_embedding
            category_offsets = torch.tensor([0] + categories[:-1]).cumsum(0)
            self.register_buffer('category_offsets', category_offsets)
            self.category_embeddings = nn.Embedding(sum(categories), d_embedding)
            nn.init.kaiming_uniform_(self.category_embeddings.weight, a=math.sqrt(5))
            print(f'{self.category_embeddings.weight.shape=}')

        self.first_layer = nn.Linear(d_in, d)
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        'norm': make_normalization(),
                        'linear0': nn.Linear(
                            d, d_hidden * (2 if activation.endswith('glu') else 1)
                        ),
                        'linear1': nn.Linear(d_hidden, d),
                    }
                )
                for _ in range(n_layers)
            ]
        )
        self.last_normalization = make_normalization()
        self.head = nn.Linear(d, d_out)

    def forward(self, x_num: Tensor, x_cat: ty.Optional[Tensor]) -> Tensor:
        x = []
        if x_num is not None:
            x.append(x_num)
        if x_cat is not None:
            x.append(
                self.category_embeddings(x_cat + self.category_offsets[None]).view(
                    x_cat.size(0), -1
                )
            )
        x = torch.cat(x, dim=-1)

        x = self.first_layer(x)
        for layer in self.layers:
            layer = ty.cast(ty.Dict[str, nn.Module], layer)
            z = x
            z = layer['norm'](z)
            z = layer['linear0'](z)
            z = self.main_activation(z)
            if self.hidden_dropout:
                z = F.dropout(z, self.hidden_dropout, self.training)
            z = layer['linear1'](z)
            if self.residual_dropout:
                z = F.dropout(z, self.residual_dropout, self.training)
            x = x + z
        x = self.last_normalization(x)
        x = self.last_activation(x)
        last_hint_layers=x
        x = self.head(x)
        if x.size(-1) != 1:
            last_hint_layers=x
        x = x.squeeze(-1)
        return x,last_hint_layers

class TabularDataset(Dataset):
    tensors: ty.Tuple[Tensor, ...]

    def __init__(self, args, dataset_dir):
        self.dataset_dir = lib.get_path(args['data']['path'])
        self.D = lib.Dataset.from_dir(dataset_dir)
        self.Xnumpy = self.D.build_X(
            normalization=args['data'].get('normalization'),
            num_nan_policy='mean',
            cat_nan_policy='new',
            cat_policy=args['data'].get('cat_policy', 'indices'),
            cat_min_frequency=args['data'].get('cat_min_frequency', 0.0),
            seed=args['seed'],
        )
        if not isinstance(self.Xnumpy, tuple):
            self.Xnumpy = (self.Xnumpy, None)

        self.Ynumpy, self.y_info = self.D.build_y(args['data'].get('y_policy'))
        self.X = tuple(None if x is None else lib.to_tensors(x) for x in self.Xnumpy)
        self.Y = lib.to_tensors(self.Ynumpy)
        # self.D=torch.from_numpy(self.D)
        # self.X=torch.from_numpy(self.X)
        # self.Y=torch.from_numpy(self.Y)
        l = []
        for x in self.X:
            l.append(None if x is None else x['train'])
        l.append(self.Y['train'])
        self.tensors = tuple(l)
        # self.tensors = self.X+ tuple(self.Y)

    def __getitem__(self, index):
        # return tuple(tensor[index] for tensor in self.tensors)
        return tuple(tensor[index] if tensor is not None else None for tensor in self.tensors)

    def __len__(self):
        return self.tensors[0].size(0)

class CombinedDataset(Dataset):
    tensors: ty.Tuple[Tensor, ...]

    def __init__(self, dataset1, dataset2):
        self.tensors = dataset1.tensors + dataset2.tensors

    def __getitem__(self, index):
        return tuple(tensor[index] if tensor is not None else None for tensor in self.tensors)

    def __len__(self):
        return self.tensors[0].size(0)

def distillation_loss(student_outputs, teacher_outputs, temperature):
    soft_teacher_outputs = F.softmax(teacher_outputs / temperature, dim=1)
    loss = F.kl_div(F.log_softmax(student_outputs, dim=1), soft_teacher_outputs, reduction='batchmean')
    return loss

def model_train(model, dataset, args, device):
    starttime=time.time()
    val_scores = []
    trainloader = DataLoader(dataset, batch_size=args['training']['batch_size'], shuffle=True,
                             collate_fn=distillresnet.collate_none_fn)
    teacheroptimizer = lib.make_optimizer(
        args['training']['optimizer'],
        model.parameters(),
        args['training']['lr'],
        args['training']['weight_decay'],
    )

    model.train()
    best_acc = float('-inf')
    patience = args['training']['patience']
    early_stopping_counter = 0

    for epoch in range(args['training']['n_epochs']):
        for x_num, x_cat, y in trainloader:
            # 省略前向传播和损失计算的代码
            x_num = x_num.to(device)
            x_cat = None if x_cat is None else x_cat.to(device)
            y = y.view(-1).to(device)


            # 清除梯度
            teacheroptimizer.zero_grad()

            # 前向传播
            output,last_output = model(x_num, x_cat)



            # 计算损失
            loss = distillresnet.loss_fn(dataset, output, y)
            # 反向传播和优化
            loss.backward()
            teacheroptimizer.step()

        # 打印每个epoch的损失
        # accuracy = compute_accuracy(output, y)
        print(f"Epoch {epoch + 1}/{args['training']['n_epochs']}, Loss: {loss.item()}")
        metrics, predictions = distillresnet.evaluate(model, dataset, args, [lib.VAL, lib.TEST], device)
        model.train()
        # 保存val准确率
        #val_score = metrics['val']['score']
        val_scores.append(metrics)
        # 绘制准确率曲线

        # 早停判断
        if metrics['val']['score'] > best_acc:
            best_acc = metrics['test']['score']
            early_stopping_counter = 0
            # 保存最优模型
            print('Saving model')
            endtime=time.time()
            torch.save(model.state_dict(), f"best_{args['name']}_model.pth")
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= patience:
                print("Early stopping triggered.")
                executiontime=endtime-starttime
                return  executiontime,val_scores,best_acc
    return endtime-starttime, val_scores, best_acc


def muti_epoch_train(teacherdataset,studentdataset,device):
    alpha = 0.9
    alpha_decay = 0.95
    teacher_val_scores = []
    student_val_scores = []

    teachermodel = ResNet(
        d_numerical=0 if teacherdataset.X[0]['train'] is None else teacherdataset.X[0]['train'].shape[1],
        categories=lib.get_categories(teacherdataset.X[1]),
        d_out=teacherdataset.D.info['n_classes'] if teacherdataset.D.is_multiclass else 1,
        **modelargs['model'],
    ).to(device)

    studentmodel = ResNet(
        d_numerical=0 if studentdataset.X[0]['train'] is None else studentdataset.X[0]['train'].shape[1],
        categories=lib.get_categories(studentdataset.X[1]),
        d_out=studentdataset.D.info['n_classes'] if studentdataset.D.is_multiclass else 1,
        **modelargs['model'],
    ).to(device)

    combined_dataset = CombinedDataset(teacherdataset, studentdataset)
    print('训练老师')

    #####训练老师
    teacher_executiontime, teacher_val_scores, teacher_best_acc = model_train(teachermodel, teacherdataset, modelargs,device)
    teachermodel.load_state_dict(torch.load('best_teacher_model.pth'))
    #####训练学生
    #################初始化student
    print('训练学生')

    pretrained_dict = torch.load('best_teacher_model.pth')

    model_params = studentmodel.named_parameters()

    # 加载除第一层以外的参数，第一层做特殊处理
    for name, param in model_params:
        if name.startswith('first_layer_weight'):
            tensort = pretrained_dict[name]
            # 计算每行的平均值
            mean_values = torch.mean(tensort, dim=1, keepdim=True)
            # 将平均值列与原始张量连接起来
            first_tensor = torch.cat((tensort, mean_values), dim=1)
            normt = tensort.size(1) / (tensort.size(1) + 1)
            first_tensor = torch.mul(first_tensor, normt)
            param.data.copy_(first_tensor)
        if name.startswith('first_layer_bias'):
            tensort = pretrained_dict[name]
            normt = tensort.size(1) / (tensort.size(1) + 1)
            first_tensor = torch.mul(tensort, normt)

        if not name.startswith('first_layer'):
            param.data.copy_(pretrained_dict[name])
    #############

    studentmodel.train()
    best_acc = float('-inf')
    patience = modelargs['training']['patience']
    early_stopping_counter = 0

    studentoptimizer = lib.make_optimizer(
        modelargs['training']['optimizer'],
        studentmodel.parameters(),
        modelargs['training']['lr'],
        modelargs['training']['weight_decay'],
    )
    # 训练循环

    studentmodel.train()
    trainloader = DataLoader(combined_dataset, batch_size=modelargs['training']['batch_size'], shuffle=True,
                             collate_fn=distillresnet.collate_none_fn)
    starttime = time.time()
    alpha = 0.9
    alpha_decay = 0.95
    for epoch in range(modelargs['training']['n_epochs']):
        alpha = alpha * alpha_decay
        for x_num_t, x_cat_t, _, x_num, x_cat, y in trainloader:
            x_num = x_num.to(device)
            x_cat = None if x_cat is None else x_cat.to(device)
            x_num_t = x_num_t.to(device)
            x_cat_t = None if x_cat_t is None else x_cat_t.to(device)
            y = y.view(-1).to(device)

            # 清除梯度
            studentoptimizer.zero_grad()

            # 前向传播
            student_output, student_lasthidden_output = studentmodel(x_num, x_cat)
            teacher_output, teacher_lasthidden_output = teachermodel(x_num_t, x_cat_t)

            # 计算蒸馏损失
            distillation_loss_value = distillation_loss(student_lasthidden_output, teacher_lasthidden_output,
                                                        temperature)

            # distillation_loss_value = nn.CrossEntropyLoss(student_output, torch.argmax(teacher_output, dim=1))
            # 计算总损失
            loss = alpha * distillation_loss_value + (1 - alpha) * distillresnet.loss_fn(studentdataset, student_output, y)

            # loss = loss_fn(student_output, y)
            # 反向传播和优化
            loss.backward()
            studentoptimizer.step()

        # 打印每个epoch的损失
        print(f"Epoch {epoch + 1}/{modelargs['training']['n_epochs']}, Loss: {loss.item()}")
        metrics, predictions = distillresnet.evaluate(studentmodel, studentdataset, modelargs, [lib.VAL, lib.TEST], device)
        studentmodel.train()
        # 保存val准确率
        # student_val_score = metrics['val']['score']
        student_val_scores.append(metrics)


        # 早停判断
        if metrics['val']['score'] > best_acc:
            best_acc = metrics['test']['score']
            early_stopping_counter = 0
            # 保存最优模型
            print('Saving model')
            endtime = time.time()
            torch.save(studentmodel.state_dict(), 'best_student_model.pth')
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= patience:
                print("Early stopping triggered.")
                # 绘制准确率曲线
                execution_time = endtime - starttime
                return teacher_executiontime, execution_time, teacher_val_scores, student_val_scores, teacher_best_acc, best_acc

    return teacher_executiontime, endtime - starttime, teacher_val_scores, student_val_scores, teacher_best_acc, best_acc


def datasetmask(input_Dataset,remove_idx):
    Dataset=copy.deepcopy(input_Dataset)

    for k in Dataset.Xnumpy[0]:
        Dataset.Xnumpy[0][k] = np.delete(Dataset.Xnumpy[0][k], remove_idx, axis=1)
    for k in Dataset.X[0]:
        Dataset.X[0][k] = np.delete(Dataset.X[0][k], remove_idx, axis=1)
    l = []
    for x in Dataset.X:
        l.append(None if x is None else x['train'])
    l.append(Dataset.Y['train'])
    Dataset.tensors = tuple(l)

    return Dataset





#####超参数
temperature = 3
for _ in range(128):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    increment_epoch = 5
    ranseed=random.randint(1, 1000)
    setup_seed(ranseed)
    dataset_dir = lib.get_path(dataargs['data']['path'])

    dataset = TabularDataset(dataargs, dataset_dir)


    print(dataset.Xnumpy[0]['train'].shape[1])
    remove_idx = random.sample(range(0, dataset.Xnumpy[0]['train'].shape[1]), increment_epoch)
    remove_idx_list=[]
    for i in range(len(remove_idx),0, -1):
        sublist = remove_idx[:i]
        remove_idx_list.append(sublist)




    teacher_val_scores_list=[]
    student_val_scores_list=[]
    direct_val_scores_list=[]
    teacher_best_acc_list=[]
    student_best_acc_list=[]
    direct_best_acc_list=[]
    add_val_scores_list=[]
    ######
    for i in range(len(remove_idx_list)):


        teacherdataset=datasetmask(dataset,remove_idx_list[i])
        if i+1<len(remove_idx_list):
            studentdataset=datasetmask(dataset,remove_idx_list[i+1])
        else:
            studentdataset=copy.deepcopy(dataset)
        adddataset=copy.deepcopy(studentdataset)







        for k in adddataset.X[0]:

            diff_col = None
            for col in range(teacherdataset.X[0][k].size(1)):
                if teacherdataset.X[0][k][0, col] != adddataset.X[0][k][0, col]:
                    diff_col = col
                    break

            if diff_col is  None:
                diff_col = teacherdataset.X[0][k].size(1)

            column = adddataset.X[0][k][:, diff_col]  # 获取要替换的列
            mean_value = torch.mean(column)  # 计算列的平均值
            adddataset.X[0][k][:, diff_col] = mean_value  # 将列的所有值替换为平均值

        l = []
        for x in adddataset.X:
            l.append(None if x is None else x['train'])
        l.append(adddataset.Y['train'])
        adddataset.tensors = tuple(l)

        teacher_executiontime, student_executiontime, teacher_val_scores, student_val_scores, teacher_best_acc, best_acc=muti_epoch_train(teacherdataset,studentdataset,device)
        teacher_val_scores_list.append(teacher_val_scores)
        student_val_scores_list.append(student_val_scores)
        teacher_best_acc_list.append(teacher_best_acc)
        student_best_acc_list.append(best_acc)



        studentmodel = ResNet(
            d_numerical=0 if studentdataset.X[0]['train'] is None else studentdataset.X[0]['train'].shape[1],
            categories=lib.get_categories(studentdataset.X[1]),
            d_out=studentdataset.D.info['n_classes'] if studentdataset.D.is_multiclass else 1,
            **modelargs['model'],
        ).to(device)
        studentmodel.load_state_dict(torch.load('best_student_model.pth'))
        # 设置模型为评估模式
        studentmodel.eval()
        print('验证增量效果')
        metrics_add, predictions_add = distillresnet.evaluate(studentmodel, adddataset, modelargs, [lib.VAL, lib.TEST], device)
        add_val_scores_list.append(metrics_add['test']['score'])




        ###
        print('不使用蒸馏训练')
        directmodel = ResNet(
            d_numerical=0 if dataset.X[0]['train'] is None else dataset.X[0]['train'].shape[1],
            categories=lib.get_categories(dataset.X[1]),
            d_out=dataset.D.info['n_classes'] if dataset.D.is_multiclass else 1,
            **modelargs['model'],
        ).to(device)
        direct_executiontime, direct_val_scores, direct_best_acc = model_train(directmodel, dataset, modelargs,device)
        direct_val_scores_list.append(direct_val_scores)
        direct_best_acc_list.append(direct_best_acc)


    output_directory=os.path.join('output2',dataargs['data']['path'])
    output_filename=f"file{add_val_scores_list}.json"
    os.makedirs(output_directory, exist_ok=True)
    output_path = os.path.join(output_directory,output_filename)
    output_meta={'add_val_scores_list':add_val_scores_list,'teacher_best_acc_list':teacher_best_acc_list,'student_best_acc_list':student_best_acc_list,'direct_best_acc_list':direct_best_acc_list,'seed':ranseed,'student_val_scores_list':student_val_scores_list,'teacher_val_scores_list':teacher_val_scores_list}
    with open(output_path, 'w') as file:
        json.dump(output_meta, file)






