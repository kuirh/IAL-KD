# %%
import copy
import math
import typing as ty
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
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


# 1. d_numerical：输入的数值特征的维度。它表示数值特征的数量。

#
# 2. categories：一个可选的整数列表，表示类别特征的数量。如果数据集中没有类别特征，则为None。
#
# 3. d_embedding：类别特征的嵌入维度。它决定了类别特征在嵌入层中的表示维度。
#
# 4. d：残差块的隐藏层维度。它决定了残差块内部全连接层的维度。
#
# 5. d_hidden_factor：隐藏层维度的因子。它是一个浮点数，用于计算隐藏层维度。隐藏层维度等于输入维度乘以d_hidden_factor的整数部分。
#
# 6. n_layers：残差块的数量。它决定了模型中残差块的层数。
#
# 7. activation：激活函数的名称。它决定了模型中使用的激活函数类型。
#
# 8. normalization：归一化层的类型。它决定了模型中使用的归一化层类型，可以是batchnorm（批归一化）或layernorm（层归一化）。
#
# 9. hidden_dropout：隐藏层的随机失活率。它决定了在训练过程中应用于隐藏层输出的随机失活的比例。
#
# 10. residual_dropout：残差连接的随机失活率。它决定了在训练过程中应用于残差连接输出的随机失活的比例。
#
# 11. d_out：模型的输出维度。它决定了模型输出的维度大小。
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

class IALNet(nn.Module):
    def __init__(self, teacheroutputsize, input_size2, output_size):
        super(IALNet, self).__init__()
        self.normal1=nn.BatchNorm1d(teacheroutputsize + input_size2)
        self.fc1 = nn.Linear(teacheroutputsize + input_size2, teacheroutputsize + input_size2)
        self.fc2 = nn.Linear(teacheroutputsize + input_size2, output_size)
        self.fc3 = nn.Linear(output_size, output_size)
        self.normal2=nn.BatchNorm1d(output_size)

        self.sigmoid = nn.Sigmoid()


    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)
        x=self.normal1(x)
        x = self.fc1(x)
        x = self.fc2(x)+x1
        x = self.fc3(x)
        x=self.normal2(x)
        x = x.squeeze(-1)
        return x

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
        if args['data'].get('attribute_mask') == 'random':
            remove_col = np.random.randint(0, self.Xnumpy[0]['train'].shape[1])
            self.re_col = remove_col
            for k in self.Xnumpy[0]:
                # arr = np.delete(arr, remove_col, axis=1)
                self.Xnumpy[0][k] = np.delete(self.Xnumpy[0][k], remove_col, axis=1)
        if isinstance(args['data'].get('attribute_mask'), int):
            remove_col = args['data'].get('attribute_mask')
            for k in self.Xnumpy[0]:
                # arr = np.delete(arr, remove_col, axis=1)
                self.Xnumpy[0][k] = np.delete(self.Xnumpy[0][k], remove_col, axis=1)
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




def evaluate(model, dataset, args, parts, de):
    device = de
    metrics = {}
    predictions = {}
    model.eval()
    with torch.no_grad():
        for part in parts:
            inputs = [
                (
                    None if dataset.X[0] is None else dataset.X[0][part][idx].to(device),
                    None if dataset.X[1] is None else dataset.X[1][part][idx].to(device)
                )
                for idx in lib.IndexLoader(dataset.D.size(part), args['training']['eval_batch_size'], False, device)
            ]
            outputs = [model(*input)[0] for input in inputs]
            predictions[part] = torch.cat(outputs).to('cpu').detach().numpy()





            # predictions[part] = (
            #     torch.cat(
            #         [
            #             model(
            #                 None if dataset.X[0] is None else dataset.X[0][part][idx].to(device),
            #                 None if dataset.X[1] is None else dataset.X[1][part][idx].to(device),
            #             )
            #             for idx in lib.IndexLoader(
            #             dataset.D.size(part),
            #             args['training']['eval_batch_size'],
            #             False,
            #             device,
            #         )
            #         ]
            #     )
            #     .device()
            #     .detach()
            #     .numpy()
            # )
            metrics[part] = lib.calculate_metrics(
                dataset.D.info['task_type'],
                dataset.Y[part].numpy(),  # type: ignore[code]
                predictions[part],  # type: ignore[code]
                'logits',
                dataset.y_info,
            )
        for part, part_metrics in metrics.items():
            print(f'[{part:<5}]', lib.make_summary(part_metrics))
    return metrics, predictions

def loss_fn(dataset,output,y):
    if dataset.D.is_binclass:
        return F.binary_cross_entropy_with_logits(output, y.to(torch.float32))
    if dataset.D.is_multiclass:
        return F.cross_entropy(output, y)
    else:
        return F.mse_loss(output, y.float())

def distillation_loss(dataset,student_outputs, teacher_outputs, temperature):
    if dataset.D.is_multiclass:
        soft_teacher_outputs = F.softmax(teacher_outputs / temperature, dim=1)
        return  F.kl_div(F.log_softmax(student_outputs, dim=1), soft_teacher_outputs, reduction='batchmean')
    if dataset.D.is_binclass:
        return F.binary_cross_entropy_with_logits(teacher_outputs, student_outputs)
    else:
        return F.mse_loss(student_outputs,teacher_outputs)

def compute_similarity(tensor1,tensor2):
    combined_tensor = torch.cat((tensor1, tensor2))

    # 归一化合并后的tensor


    normalized_combined_tensor = combined_tensor / torch.norm(combined_tensor)

    # 将归一化后的tensor拆分为两个tensor
    normalized_tensor1 = normalized_combined_tensor[:tensor1.size()[0]]
    normalized_tensor2 = normalized_combined_tensor[tensor1.size()[0]:]



    # 计算标准化的欧几里得距离
    normalized_distance = torch.dist(normalized_tensor1, normalized_tensor2)



    # 将标准化的欧几里得距离转换为相似度度量
    similarity = 1 - normalized_distance
    return similarity.item()

# 定义准

def model_train_ialteacher(model, dataset,valdataset, args, device):
    model = model.to(device)
    starttime=time.time()
    val_scores = []
    trainloader = DataLoader(dataset, batch_size=args['training']['batch_size'], shuffle=True,
                             collate_fn=collate_none_fn)
    testloader=DataLoader(valdataset, batch_size=args['training']['batch_size'], shuffle=True,
                           collate_fn=collate_none_fn)
    teacheroptimizer = lib.make_optimizer(
        args['training']['optimizer'],
        model.parameters(),
        args['training']['lr'],
        args['training']['weight_decay'],
    )

    model.train()
    best_acc = float('-inf')
    best_test_acc=float('-inf')
    patience = args['training']['patience']
    early_stopping_counter = 0

    for epoch in range(args['training']['n_epochs']):
        for data in trainloader:
            x_num, x_cat, y=data[:3]
            # 省略前向传播和损失计算的代码
            x_num = x_num.to(device)
            x_cat = None if x_cat is None else x_cat.to(device)
            y = y.view(-1).to(device)


            # 清除梯度
            teacheroptimizer.zero_grad()

            # 前向传播
            output,last_output = model(x_num, x_cat)


            # 计算损失
            loss = F.cross_entropy(output, y)
            # 反向传播和优化
            loss.backward()
            teacheroptimizer.step()
    return 1


# 老师模型训练循环

def model_train(model, dataset, args, device):
    starttime=time.time()
    val_scores = []
    trainloader = DataLoader(dataset, batch_size=args['training']['batch_size'], shuffle=True,
                             collate_fn=collate_none_fn)
    teacheroptimizer = lib.make_optimizer(
        args['training']['optimizer'],
        model.parameters(),
        args['training']['lr'],
        args['training']['weight_decay'],
    )

    model.train()
    best_acc = float('-inf')
    best_test_acc=float('-inf')
    patience = args['training']['patience']
    early_stopping_counter = 0

    for epoch in range(args['training']['n_epochs']):
        for data in trainloader:
            x_num, x_cat, y=data[:3]
            # 省略前向传播和损失计算的代码
            x_num = x_num.to(device)
            x_cat = None if x_cat is None else x_cat.to(device)
            y = y.view(-1).to(device)


            # 清除梯度
            teacheroptimizer.zero_grad()

            # 前向传播
            output,last_output = model(x_num, x_cat)



            # 计算损失
            loss = loss_fn(dataset,output, y)
            # 反向传播和优化
            loss.backward()
            teacheroptimizer.step()

        # 打印每个epoch的损失
        # accuracy = compute_accuracy(output, y)
        print(f"Epoch {epoch + 1}/{args['training']['n_epochs']}, Loss: {loss.item()}")
        metrics, predictions = evaluate(model, dataset, args, [lib.VAL, lib.TEST], device)
        model.train()
        # 保存val准确率
        #val_score = metrics['val']['score']
        val_scores.append(metrics)
        # 绘制准确率曲线

        # 早停判断
        if metrics['val']['score'] > best_acc:
            best_acc = metrics['val']['score']
            best_test_acc=metrics['test']['score']
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
                return  executiontime,val_scores,best_test_acc
    return endtime-starttime, val_scores, best_test_acc

def model_load(model):
    # 加载保存的模型参数
    model.load_state_dict(torch.load('best_teacher_model.pth'))
    # 设置模型为评估模式
    model.eval()


def collate_none_fn(batch_list):
    batch_size = len(batch_list)
    list_size = len(batch_list[0])
    Xnum, Xcat, Y, Xnum2, Xcat2, Y = None, None, None, None, None, None
    if list_size == 3:
        Xnum = torch.cat([item[0] for item in batch_list]).reshape(batch_size, -1)
        if batch_list[0][1] is not None:
            Xcat = torch.cat([item[1] for item in batch_list]).reshape(batch_size, -1)
        else:
            Xcat = None
        Y = torch.stack([item[2] for item in batch_list]).reshape(batch_size, -1)
        return Xnum, Xcat, Y
    else:
        Xnum = torch.cat([item[0] for item in batch_list]).reshape(batch_size, -1)
        if batch_list[0][1] is not None:
            Xcat = torch.cat([item[1] for item in batch_list]).reshape(batch_size, -1)
        else:
            Xcat = None
        Y = torch.stack([item[2] for item in batch_list]).reshape(batch_size, -1)
        if torch.is_tensor(batch_list[0][4]):
            Xnum2 = torch.cat([item[3] for item in batch_list]).reshape(batch_size, -1)
        else:
            Xnum2 = torch.stack([item[3] for item in batch_list]).reshape(batch_size, -1)
        if batch_list[0][4] is not None:
            Xcat2 = torch.cat([item[4] for item in batch_list]).reshape(batch_size, -1)
        else:
            Xcat2 = None
        return Xnum, Xcat, Y, Xnum2, Xcat2, Y





def distillresnet(teacherdataargs, studentdataargs, directdataargs,teacherargs, studentargs, directargs,de,temperature, alpha, alpha_decay,
                  teacher_need_train):



    temperature = temperature
    alpha = alpha
    alpha_decay = alpha_decay
    device = de
    teacher_val_scores = []
    student_val_scores = []
    add_val_scores=[]
    similarity_list_distill=[]
    direct_val_scores = []
    old_val_scores=[]
    similarity_list_direct=[]
    dataset_dir = lib.get_path(teacherdataargs['data']['path'])
    teacherdataset = TabularDataset(teacherdataargs, dataset_dir)
    studentdataset = TabularDataset(studentdataargs, dataset_dir)
    directdataset = TabularDataset(directdataargs, dataset_dir)
    #################################ADDdataset
#################################ADDdataset
    adddataset = copy.deepcopy(studentdataset)

    remove_col = teacherdataset.re_col
    for k in adddataset.X[0]:
        column = adddataset.X[0][k][:, remove_col]  # 获取要替换的列
        mean_value = torch.mean(column)  # 计算列的平均值
        adddataset.X[0][k][:, remove_col] = mean_value  # 将列的所有值替换为平均值
   ###################################



    combined_dataset = CombinedDataset(teacherdataset, studentdataset)
    teachermodel = ResNet(
        d_numerical=0 if teacherdataset.X[0]['train'] is None else teacherdataset.X[0]['train'].shape[1],
        categories=lib.get_categories(teacherdataset.X[1]),
        d_out=teacherdataset.D.info['n_classes'] if teacherdataset.D.is_multiclass else 1,
        **teacherargs['model'],
    ).to(device)

    studentmodel = ResNet(
        d_numerical=0 if studentdataset.X[0]['train'] is None else studentdataset.X[0]['train'].shape[1],
        categories=lib.get_categories(studentdataset.X[1]),
        d_out=studentdataset.D.info['n_classes'] if studentdataset.D.is_multiclass else 1,
        **studentargs['model'],
    ).to(device)

    if teacher_need_train == 1:
        teacher_executiontime,teacher_val_scores,teacher_best_acc=model_train(teachermodel, teacherdataset, teacherargs, device)
        model_load(teachermodel)
    else:
        model_load(teachermodel)

#################初始化student
    # pretrained_dict = torch.load('best_teacher_model.pth')
    #
    # model_params = studentmodel.named_parameters()
    #
    # # 加载除第一层以外的参数，第一层做特殊处理
    # for name, param in model_params:
    #     if name.startswith('first_layer_weight'):
    #         tensort = pretrained_dict[name]
    #         # 计算每行的平均值
    #         mean_values = torch.mean(tensort, dim=1, keepdim=True)
    #         # 将平均值列与原始张量连接起来
    #         first_tensor = torch.cat((tensort, mean_values), dim=1)
    #         normt = tensort.size(1) / (tensort.size(1) + 1)
    #         first_tensor = torch.mul(first_tensor, normt)
    #         param.data.copy_(first_tensor)
    #     if name.startswith('first_layer_bias'):
    #         tensort = pretrained_dict[name]
    #         normt = tensort.size(1) / (tensort.size(1) + 1)
    #         first_tensor = torch.mul(tensort, normt)
    #
    #     if not name.startswith('first_layer'):
    #         param.data.copy_(pretrained_dict[name])


    #############



    trainloader = DataLoader(combined_dataset, batch_size=teacherargs['training']['batch_size'], shuffle=True,
                             collate_fn=collate_none_fn)
    studentoptimizer = lib.make_optimizer(
        studentargs['training']['optimizer'],
        studentmodel.parameters(),
        studentargs['training']['lr'],
        studentargs['training']['weight_decay'],
    )
    # 训练循环





    studentmodel.train()
    best_acc = float('-inf')
    best_test_acc=float('-inf')
    add_acc=float('-inf')
    patience = studentargs['training']['patience']
    early_stopping_counter = 0

    starttime = time.time()

    for epoch in range(studentargs['training']['n_epochs']):
        alpha = alpha * alpha_decay
        for x_num_t, x_cat_t, _, x_num, x_cat, y in trainloader:
            x_num = x_num.to(device)
            x_cat = None if x_cat is None else x_cat.to(device)
            x_num_t = x_num_t.to(device)
            x_cat_t = None if x_cat_t is None else x_cat_t.to(device)
            y = y.view(-1).to(device)
            #print(x_num_t.dtype, x_cat_t, _, x_num.dtype, x_cat, y.dtype)

            # 清除梯度
            studentoptimizer.zero_grad()

            # 前向传播
            teacher_output,teacher_lasthidden_output = teachermodel(x_num_t, x_cat_t)
            student_output,student_lasthidden_output = studentmodel(x_num, x_cat)


            # 计算蒸馏损失
            distillation_loss_value = distillation_loss(studentdataset,student_lasthidden_output, teacher_lasthidden_output, temperature)
            #print(distillation_loss_value,(1 - alpha) * loss_fn(studentdataset,student_output, y))
            #计算相似度
            # distillation_loss_value = nn.CrossEntropyLoss(student_output, torch.argmax(teacher_output, dim=1))
            # 计算总损失
            loss = alpha * distillation_loss_value + (1 - alpha) * loss_fn(studentdataset,student_output, y)

            # loss = loss_fn(student_output, y)
            # 反向传播和优化
            loss.backward()
            studentoptimizer.step()

        # 打印每个epoch的损失
        print(f"Epoch {epoch + 1}/{studentargs['training']['n_epochs']}, Loss: {loss.item()}")
        similarity= compute_similarity(student_lasthidden_output, teacher_lasthidden_output)
        similarity_list_distill.append(similarity)

        metrics, predictions = evaluate(studentmodel, studentdataset, studentargs, [lib.VAL, lib.TEST], device)
        if epoch+1==5:
            f_best_acc=metrics['test']['score']
        studentmodel.train()
        # 保存val准确率
        #student_val_score = metrics['val']['score']
        student_val_scores.append(metrics)

#############add
        metrics_add, predictions_add = evaluate(studentmodel, adddataset, studentargs, [lib.VAL, lib.TEST], device)
###################
        studentmodel.train()
        add_val_scores.append(metrics_add)


        # 早停判断
        if metrics['val']['score'] > best_acc:
            best_acc = metrics['val']['score']
            best_test_acc=metrics['test']['score']
            add_acc=metrics_add['test']['score']
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
                execution_time_d = endtime - starttime
                break


########直接训练


    combined_dataset_d = CombinedDataset(teacherdataset, directdataset)
    directmodel = ResNet(
        d_numerical=0 if directdataset.X[0]['train'] is None else directdataset.X[0]['train'].shape[1],
        categories=lib.get_categories(directdataset.X[1]),
        d_out=directdataset.D.info['n_classes'] if directdataset.D.is_multiclass else 1,
        **directargs['model'],
    ).to(device)
    model_load(teachermodel)
    teachermodel.eval()
    trainloader = DataLoader(combined_dataset_d, batch_size=teacherargs['training']['batch_size'], shuffle=True,
                             collate_fn=collate_none_fn)

    directoptimizer = lib.make_optimizer(
        directargs['training']['optimizer'],
        directmodel.parameters(),
        directargs['training']['lr'],
        directargs['training']['weight_decay'],
    )

    #############
    directmodel.train()
    distill_best_acc = float('-inf')
    distill_test_best_acc = float('-inf')
    old_acc = float('-inf')
    patience = directargs['training']['patience']
    early_stopping_counter = 0

    # 训练循环

    starttime = time.time()

    for epoch in range(directargs['training']['n_epochs']):
        for x_num_t, x_cat_t, _, x_num, x_cat, y in trainloader:
            x_num = x_num.to(device)
            x_cat = None if x_cat is None else x_cat.to(device)
            x_num_t = x_num_t.to(device)
            x_cat_t = None if x_cat_t is None else x_cat_t.to(device)
            y = y.view(-1).to(device)

            # 清除梯度
            directoptimizer.zero_grad()

            # 前向传播
            teacher_output, teacher_lasthidden_output = teachermodel(x_num_t, x_cat_t)
            student_output, student_lasthidden_output = directmodel(x_num, x_cat)

            # 计算蒸馏损失

            # 计算相似度

            # distillation_loss_value = nn.CrossEntropyLoss(student_output, torch.argmax(teacher_output, dim=1))
            # 计算总损失
            loss = loss_fn(directdataset, student_output, y)

            # loss = loss_fn(student_output, y)
            # 反向传播和优化
            loss.backward()
            directoptimizer.step()

        # 打印每个epoch的损失
        print(f"Epoch {epoch + 1}/{directargs['training']['n_epochs']}, Loss: {loss.item()}")
        similarity = compute_similarity(student_lasthidden_output, teacher_lasthidden_output)
        similarity_list_direct.append(similarity)
        metrics, predictions = evaluate(directmodel, directdataset, directargs, [lib.VAL, lib.TEST], device)
        directmodel.train()
        # 保存val准确率
        # student_val_score = metrics['val']['score']
        direct_val_scores.append(metrics)

        #############old
        metrics_old, predictions_old = evaluate(directmodel, adddataset, directargs, [lib.VAL, lib.TEST], device)
        ###################
        directmodel.train()
        old_val_scores.append(metrics_old)

        # 早停判断
        if metrics['val']['score'] > distill_best_acc:
            distill_best_acc = metrics['val']['score']
            distill_test_best_acc = metrics['test']['score']
            old_acc = metrics_old['test']['score']
            early_stopping_counter = 0
            # 保存最优模型
            print('Saving model')
            endtime = time.time()
            torch.save(directmodel.state_dict(), 'best_direct_model.pth')
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= patience:
                print("Early stopping triggered.")
                # 绘制准确率曲线

                break
    execution_time_d = time.time() - starttime

    return teacher_executiontime, execution_time_d, teacher_val_scores, student_val_scores, teacher_best_acc, best_test_acc,add_val_scores,add_acc,similarity_list_distill,endtime - starttime, direct_val_scores, distill_test_best_acc, old_val_scores, old_acc, similarity_list_direct,f_best_acc










# def directresnet(teacherdataargs, directdataargs, teacherargs, directargs, de):
#     device = de
#     direct_val_scores = []
#     old_val_scores=[]
#     similarity_list=[]
#     dataset_dir = lib.get_path(teacherdataargs['data']['path'])
#     teacherdataset = TabularDataset(teacherdataargs, dataset_dir)
#     directdataset = TabularDataset(directdataargs, dataset_dir)
# #################################ADDdataset
#     olddataset = copy.deepcopy(directdataset)
#
#     remove_col = teacherdataset.re_col
#     for k in olddataset.X[0]:
#         column = olddataset.X[0][k][:, remove_col]  # 获取要替换的列
#         mean_value = torch.mean(column)  # 计算列的平均值
#         olddataset.X[0][k][:, remove_col] = mean_value  # 将列的所有值替换为平均值
#    ###################################
#
#
#
#     combined_dataset = CombinedDataset(teacherdataset, directdataset)
#     teachermodel = ResNet(
#         d_numerical=0 if teacherdataset.X[0]['train'] is None else teacherdataset.X[0]['train'].shape[1],
#         categories=lib.get_categories(teacherdataset.X[1]),
#         d_out=teacherdataset.D.info['n_classes'] if teacherdataset.D.is_multiclass else 1,
#         **teacherargs['model'],
#     ).to(device)
#
#     directmodel = ResNet(
#         d_numerical=0 if directdataset.X[0]['train'] is None else directdataset.X[0]['train'].shape[1],
#         categories=lib.get_categories(directdataset.X[1]),
#         d_out=directdataset.D.info['n_classes'] if directdataset.D.is_multiclass else 1,
#         **directargs['model'],
#     ).to(device)
#
#     model_load(teachermodel)
#
#     #############
#
#     directmodel.train()
#     best_acc = float('-inf')
#     old_acc=float('-inf')
#     patience = directargs['training']['patience']
#     early_stopping_counter = 0
#
#     directoptimizer = lib.make_optimizer(
#         directargs['training']['optimizer'],
#         directmodel.parameters(),
#         directargs['training']['lr'],
#         directargs['training']['weight_decay'],
#     )
#     # 训练循环
#
#
#
#
#
#
#
#
#
#
#     directmodel.train()
#     trainloader = DataLoader(combined_dataset, batch_size=teacherargs['training']['batch_size'], shuffle=True,
#                              collate_fn=collate_none_fn)
#     starttime = time.time()
#
#     for epoch in range(directargs['training']['n_epochs']):
#         for x_num_t, x_cat_t, _, x_num, x_cat, y in trainloader:
#             x_num = x_num.to(device)
#             x_cat = None if x_cat is None else x_cat.to(device)
#             x_num_t = x_num_t.to(device)
#             x_cat_t = None if x_cat_t is None else x_cat_t.to(device)
#             y = y.view(-1).to(device)
#
#             # 清除梯度
#             directoptimizer.zero_grad()
#
#             # 前向传播
#             student_output,student_lasthidden_output = directmodel(x_num, x_cat)
#             teacher_output,teacher_lasthidden_output = teachermodel(x_num_t, x_cat_t)
#
#             # 计算蒸馏损失
#
#             #计算相似度
#
#             # distillation_loss_value = nn.CrossEntropyLoss(student_output, torch.argmax(teacher_output, dim=1))
#             # 计算总损失
#             loss = loss_fn(directdataset,student_output, y)
#
#             # loss = loss_fn(student_output, y)
#             # 反向传播和优化
#             loss.backward()
#             directoptimizer.step()
#
#         # 打印每个epoch的损失
#         print(f"Epoch {epoch + 1}/{directargs['training']['n_epochs']}, Loss: {loss.item()}")
#         similarity = compute_similarity(student_lasthidden_output, teacher_lasthidden_output)
#         similarity_list.append(similarity)
#         metrics, predictions = evaluate(directmodel, directdataset, directargs, [lib.VAL, lib.TEST], device)
#         directmodel.train()
#         # 保存val准确率
#         #student_val_score = metrics['val']['score']
#         direct_val_scores.append(metrics)
#
# #############old
#         metrics_old, predictions_old = evaluate(directmodel, olddataset, directargs, [lib.VAL, lib.TEST], device)
# ###################
#         old_val_scores.append(metrics_old)
#
#
#         # 早停判断
#         if metrics['val']['score'] > best_acc:
#             best_acc = metrics['val']['score']
#             old_acc=metrics_old['val']['score']
#             early_stopping_counter = 0
#             # 保存最优模型
#             print('Saving model')
#             endtime = time.time()
#             torch.save(directmodel.state_dict(), 'best_direct_model.pth')
#         else:
#             early_stopping_counter += 1
#             if early_stopping_counter >= patience:
#                 print("Early stopping triggered.")
#                 # 绘制准确率曲线
#
#                 return endtime - starttime,direct_val_scores,best_acc,old_val_scores,old_acc,similarity_list
#
#     return  endtime - starttime, direct_val_scores, best_acc,old_val_scores,old_acc,similarity_list


# def directresnet( directdataargs, directargs,de):
#     device=de
#     dataset_dir = lib.get_path(directdataargs['data']['path'])
#     directdataset = TabularDataset(directdataargs, dataset_dir)
#     directmodel = ResNet(
#         d_numerical=0 if directdataset.X[0]['train'] is None else directdataset.X[0]['train'].shape[1],
#         categories=lib.get_categories(directdataset.X[1]),
#         d_out=directdataset.D.info['n_classes'] if directdataset.D.is_multiclass else 1,
#         **directargs['model'],
#     ).to(device)
#     direct_execution_time,direct_val_scores,direct_best_acc=model_train(directmodel, directdataset, directargs,device)
#
#
#     return direct_execution_time,direct_val_scores,direct_best_acc

def ialresnet(teacherdataargs, teacherargs,ialdataargs,ialargs, de):
    device = de
    teacher_val_scores = []
    ial_val_scores=[]
    ial_old_val_scores=[]
    similarity_list_ial=[]
    dataset_dir = lib.get_path(teacherdataargs['data']['path'])
    teacherdataset = TabularDataset(teacherdataargs, dataset_dir)

    ialdataset = TabularDataset(ialdataargs, dataset_dir)
#################################ADDdataset
    olddataset = copy.deepcopy(ialdataset)

    remove_col = teacherdataset.re_col
    for k in olddataset.X[0]:
        column = olddataset.X[0][k][:, remove_col]  # 获取要替换的列
        mean_value = torch.mean(column)  # 计算列的平均值
        olddataset.X[0][k][:, remove_col] = mean_value  # 将列的所有值替换为平均值
   ###################################

    remove_col = teacherdataset.re_col
    for k in ialdataset.X[0]:
        column = ialdataset.X[0][k][:, remove_col]  # 获取要替换的列
          # 计算列的平均值
        ialdataset.X[0][k] = column  # 将列的所有值替换为平均值
        l = []
        for x in ialdataset.X:
            l.append(None if x is None else x['train'])
        l.append(ialdataset.Y['train'])
        ialdataset.tensors = tuple(l)


    combined_ial_dataset = CombinedDataset(teacherdataset, ialdataset)
    # 计算划分比例
    split_point = int(0.8 * len(combined_ial_dataset))  # 80%用于训练集，20%用于验证集
    trainset = torch.utils.data.dataset.Subset(combined_ial_dataset, range(split_point))
    testset = torch.utils.data.dataset.Subset(combined_ial_dataset, range(split_point, len(combined_ial_dataset)))
    # 划分数据集

    teachermodel = ResNet(
        d_numerical=0 if teacherdataset.X[0]['train'] is None else teacherdataset.X[0]['train'].shape[1],
        categories=lib.get_categories(teacherdataset.X[1]),
        d_out=teacherdataset.D.info['n_classes'] if teacherdataset.D.is_multiclass else 1,
        **teacherargs['model'],
    ).to(device)

    model_train_ialteacher(teachermodel, trainset,testset, teacherargs,device)




    ialmodel= IALNet( teacheroutputsize=teacherdataset.D.info['n_classes'] if teacherdataset.D.is_multiclass else 339,input_size2=1,output_size=teacherdataset.D.info['n_classes'] if teacherdataset.D.is_multiclass else 1).to(device)

    ialmodel.train()
    best_ial_acc = float('-inf')
    patience = teacherargs['training']['patience']
    early_stopping_counter = 0

    ialoptimizer = lib.make_optimizer(
        ialargs['training']['optimizer'],
        ialmodel.parameters(),
        ialargs['training']['lr'],
        ialargs['training']['weight_decay'],
    )
#     # 训练循环
    ialmodel.train()
    trainloader = DataLoader(trainset, batch_size=ialargs['training']['batch_size'], shuffle=True,
                             collate_fn=collate_none_fn)
    testloader= DataLoader(testset, batch_size=ialargs['training']['batch_size'], shuffle=True,
                           collate_fn=collate_none_fn)
    starttime = time.time()

    for epoch in range(ialargs['training']['n_epochs']):
        test_loss = 0.0
        total_samples = 0
        correct_predictions = 0
        for param in teachermodel.parameters():
            param.requires_grad = False

        for x_num_t, x_cat_t, _, x_num, x_cat, y in trainloader:
            x_num = x_num.to(device)
            x_cat = None if x_cat is None else x_cat.to(device)
            x_num_t = x_num_t.to(device)
            x_cat_t = None if x_cat_t is None else x_cat_t.to(device)
            y = y.view(-1).to(device)
            # # 清除梯度
            ialoptimizer.zero_grad()
            ialmodel.train()
            # # 前向传播
            with torch.no_grad():
                teacher_output, teacher_lasthidden_output = teachermodel(x_num_t, x_cat_t)
            output = ialmodel(teacher_lasthidden_output, x_num)
            predicted_labels_1 = torch.argmax(output, dim=1)
            similarity_list_ial.append(compute_similarity(teacher_lasthidden_output, output))

            loss = loss_fn(ialdataset, output, y)
            y_1=y
            # loss = loss_fn(student_output, y)
            # 反向传播和优化
            loss.backward()
            ialoptimizer.step()


        # 打印每个epoch的损失
            ialmodel.eval()  # 设置模型为评估模式


            with torch.no_grad():
                for x_num_t, x_cat_t, _, x_num, x_cat, y in testloader:
                    x_num = x_num.to(device)
                    x_cat = None if x_cat is None else x_cat.to(device)
                    x_num_t = x_num_t.to(device)
                    x_cat_t = None if x_cat_t is None else x_cat_t.to(device)
                    y = y.view(-1).to(device)

                    # 前向传播
                    teacher_output, teacher_lasthidden_output = teachermodel(x_num_t, x_cat_t)
                    output = ialmodel(teacher_lasthidden_output, x_num)

                    # 计算损失
                    loss = loss_fn(ialdataset, output, y)
                    test_loss += loss.item()

                    # 计算准确率
                    predicted_labels = torch.argmax(output, dim=1)
                    correct_predictions += torch.sum(predicted_labels == y).item()
                    total_samples += y.size(0)
        average_loss = test_loss / len(testloader)
        accuracy = correct_predictions / total_samples
        ial_val_scores.append(accuracy)
        print("Test Loss: {:.4f}".format(average_loss))
        print("Test Accuracy: {:.2%}".format(accuracy))

#补充代码

            # 计算推理时间
        endtime = time.time()
        inference_time = endtime - starttime
        print("Inference Time: {:.2f} seconds".format(inference_time))
            #
        if accuracy > best_ial_acc:
            best_ial_acc = accuracy
            early_stopping_counter = 0
            # 保存最优模型
            print('Saving model')
            endtime = time.time()
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= patience:
                print("Early stopping triggered.")
                # 绘制准确率曲线

                break
    return inference_time, ial_val_scores, best_ial_acc, similarity_list_ial
            # student_output, student_lasthidden_output = directmodel(x_num, x_cat)
            # teacher_output, teacher_lasthidden_output = teachermodel(x_num_t, x_cat_t)
#
#         # 打印每个epoch的损失
#         print(f"Epoch {epoch + 1}/{directargs['training']['n_epochs']}, Loss: {loss.item()}")
#         similarity = compute_similarity(student_lasthidden_output, teacher_lasthidden_output)
#         similarity_list.append(similarity)
#         metrics, predictions = evaluate(directmodel, directdataset, directargs, [lib.VAL, lib.TEST], device)
#         directmodel.train()
#         # 保存val准确率
#         #student_val_score = metrics['val']['score']
#         direct_val_scores.append(metrics)
#
# #############old
#         metrics_old, predictions_old = evaluate(directmodel, olddataset, directargs, [lib.VAL, lib.TEST], device)
# ###################
#         old_val_scores.append(metrics_old)
#
#
#         # 早停判断
#         if metrics['val']['score'] > best_acc:
#             best_acc = metrics['val']['score']
#             old_acc=metrics_old['val']['score']
#             early_stopping_counter = 0
#             # 保存最优模型
#             print('Saving model')
#             endtime = time.time()
#             torch.save(directmodel.state_dict(), 'best_direct_model.pth')
#         else:
#             early_stopping_counter += 1
#             if early_stopping_counter >= patience:
#                 print("Early stopping triggered.")
#                 # 绘制准确率曲线
#
#                 return endtime - starttime,direct_val_scores,best_acc,old_val_scores,old_acc,similarity_list
#
#     return  endtime - starttime, direct_val_scores, best_acc,old_val_scores,old_acc,similarity_list