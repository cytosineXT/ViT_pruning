"""### Importing Libraries"""
# import timm
import torch
from torch import nn, einsum
# from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
# from torchvision.datasets import CIFAR10, CIFAR100, ImageNet
from torch.optim import Adam, lr_scheduler
# from torchvision import transforms

#import matplotlib.pyplot as plt
import numpy as np
#import pandas as pd

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# from einops import rearrange, repeat
# from einops.layers.torch import Rearrange
# import pickle
import os
from thop import profile
import time
# import re

from timm.models.layers import DropPath, PatchEmbed, trunc_normal_, lecun_normal_
from deit_modified import  DynaLinear
from deit_modified_ghost import VisionTransformer
# from functools import partial

from tqdm import tqdm
import logging
from pathlib import Path

def increment_path(path, exist_ok=False, sep="", mkdir=True):
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(""), path.suffix) if path.is_file() else (path, "")
        for n in range(2, 9999):
            p = f"{path}{sep}{n}{suffix}"  # increment path
            if not os.path.exists(p):  #
                break
        path = Path(p)
    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  # make directory
    return path

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])
 
    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
 
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger

#try:
    #from google.colab import drive
    #drive.mount("/content/gdrive")
    #colab = True
#except:
    #colab = False

#helpers
#def show_torch_img(image):
    #plt.imshow(transforms.ToPILImage()(image))

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def soft_cross_entropy(predicts, targets):
    student_likelihood = nn.functional.log_softmax(predicts, dim=-1)
    targets_prob = nn.functional.softmax(targets, dim=-1)
    return -torch.sum(targets_prob * student_likelihood, dim=-1).mean()

import matplotlib.pyplot as plt
import seaborn as sns
def visualize_head_importance(head_importance,save_dir):
    """
    可视化 head_importance 的热力图
    Args:
        head_importance (torch.Tensor): 尺寸为 (n_layers, n_heads) 的 tensor，表示每一层每个注意力头的重要性
    """
    # 确保 head_importance 是在 CPU 上的 numpy 数组
    if isinstance(head_importance, torch.Tensor):
        head_importance = head_importance.detach().cpu().numpy()

    plt.figure(figsize=(12, 6))
    sns.heatmap(head_importance, annot=True, cmap="YlGnBu", cbar=True)
    
    plt.title("每Transformer层注意力头重要性图")
    plt.xlabel("注意力头")
    plt.ylabel("Transformer层")

    plt.savefig(os.path.join(save_dir,'headimportance.png')) 

def visualize_neuron_importance(neuron_importance,save_dir):
    """
    Visualizes the neuron importance as a heatmap.
    
    Args:
    - neuron_importance (list of Tensors): List of tensors where each tensor contains the importance of neurons in a specific layer.
    """
    
    # Convert neuron_importance to a numpy array for easy plotting
    importance_matrix = np.array([imp.cpu().numpy() for imp in neuron_importance])
    # Plot the heatmap
    plt.figure(figsize=(20, 10))  # Adjust the size based on your needs
    sns.heatmap(importance_matrix, cmap="YlGnBu", cbar=True)
    plt.title("每MLP层神经元重要性图")
    plt.xlabel("神经元")
    plt.ylabel("MLP层")
    plt.savefig(os.path.join(save_dir,'neuronimportance.png'))

# def visualize_neuron_importance2(neuron_importance, layer_idx=None):
#     """
#     Visualizes the neuron importance using a bar chart.
    
#     Args:
#     - neuron_importance (list of Tensors): List of tensors where each tensor contains the importance of neurons in a specific layer.
#     - layer_idx (int, optional): If specified, only visualizes the importance for the given layer index. Otherwise, visualizes all layers.
#     """

#     # Convert neuron_importance to a numpy array for easy plotting
#     importance_values = [imp.cpu().numpy() for imp in neuron_importance]

#     if layer_idx is not None:
#         # Plot for a specific layer
#         plt.figure(figsize=(10, 6))
#         plt.bar(range(len(importance_values[layer_idx])), importance_values[layer_idx])
#         plt.title(f'第{layer_idx}MLP层神经元重要性图 ')
#         plt.xlabel('神经元')
#         plt.ylabel('重要性得分')
#         plt.savefig('./code/modelshessian2/neuronimportance.png')
#     else:
#         # Plot all layers
#         for i, importance in enumerate(importance_values):
#             plt.figure(figsize=(10, 6))
#             plt.bar(range(len(importance)), importance)
#             plt.title(f'Neuron Importance for Layer {i}')
#             plt.xlabel('Neuron Index')
#             plt.ylabel('Importance Score')
#             plt.savefig(f'./code/modelshessian2/neuronimportance{i}.png')


"""### Importance reordering"""
def compute_neuron_head_importance(
    eval_dataloader, model, n_layers, n_heads,device, loss_fn=nn.CrossEntropyLoss()
    ):
    """ This method shows how to compute:
        - neuron importance scores based on loss according to http://arxiv.org/abs/1905.10650
    """
    head_importance = torch.zeros(n_layers, n_heads).to(device)
    head_mask = torch.ones(n_layers, n_heads).to(device)
    head_mask.requires_grad_(requires_grad=True)

    intermediate_weight = []
    intermediate_bias = []
    output_weight = []
    for name, w in model.named_parameters(): #这里好像是筛选层 但是怎么最后三个list还是空的呢 这里是筛选神经元的
        if 'intermediate' in name:
            if w.dim() > 1:
                intermediate_weight.append(w)
            else:
                intermediate_bias.append(w)

        if 'output' in name and 'attention' not in name:
            if w.dim() > 1:
                output_weight.append(w)
    
    neuron_importance = []
    for w in intermediate_weight:
        neuron_importance.append(torch.zeros(w.shape[0]).to(device))
    
    model.to(device)

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch = tuple(t.to(device) for t in batch)
        input_ids, label_ids = batch

        #-----------一 loss重要性----------------
        # calculate head importance 
        outputs = model(input_ids, head_mask=head_mask)
        loss = loss_fn(outputs, label_ids)
        loss.backward()
        head_importance += head_mask.grad.abs().detach()
        # calculate  neuron importance
        for w1, b1, w2, current_importance in zip(intermediate_weight, intermediate_bias, output_weight, neuron_importance):
            current_importance += ((w1 * w1.grad).sum(dim=1) + b1 * b1.grad).abs().detach()
            current_importance += ((w2 * w2.grad).sum(dim=0)).abs().detach()

        # # #-----------二 entropy重要性----------------
        # # outputs , hidden, attn_weights = model(input_ids, return_states=True, return_attn_weight = True)
        # # # attn_weights = attn  # assuming this is a list of attention scores per layer
        # # for layer in range(n_layers):
        # #     for head in range(n_heads):
        # #         # Computing entropy for each head
        # #         probs = attn_weights[layer][:, head, :, :].mean(dim=0)
        # #         # max_entropy = torch.log(torch.tensor(probs.shape[0], device=probs.device))  # 最大熵值
        # #         # entropy = -(probs * torch.log(probs + 1e-10)).sum() / max_entropy #做个归一化
        # #         entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=[1]).mean(dim=0)
        # #         # entropy = -(probs * torch.log(probs + 1e-10)).sum()
        # #         head_importance[layer, head] += entropy.detach()
        # # head_importance[layer, head] /= n_layers * n_heads
        # # # for w1, b1, w2, current_importance in zip(intermediate_weight, intermediate_bias, output_weight, neuron_importance):
        # #     neuron_output = torch.relu(torch.matmul(hidden[-2], w1.T) + b1)  # using ReLU as the activation function
        # #     neuron_entropy = -(neuron_output * torch.log(neuron_output + 1e-10)).sum(dim=0)
        # #     current_importance += neuron_entropy.abs().detach()
        # outputs, hidden, attn_weights = model(input_ids, return_states=True, return_attn_weight=True)
        # for layer in range(n_layers):
        #     for head in range(n_heads):
        #         attn_score = attn_weights[layer][:, head, :, :]  # shape: [batch_size, seq_len, seq_len]
        #         probs = attn_score.mean(dim=0)  # mean over batch
        #         probs = probs / probs.sum(dim=-1, keepdim=True)  # normalize over seq_len
        #         max_entropy = torch.log(torch.tensor(probs.shape[0], device=probs.device))  # 最大熵值
        #         entropy = max_entropy+(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean()  # entropy per head 负号哪来的
        #         head_importance[layer, head] += entropy.detach()
        # head_importance /= (n_layers * n_heads)        # Normalize head importance
        # for w1, b1, w2, current_importance in zip(intermediate_weight, intermediate_bias, output_weight, neuron_importance):
        #     neuron_output = torch.relu(torch.matmul(hidden[-2], w1.T) + b1)  # ReLU激活
        #     # 归一化每个神经元的输出
        #     neuron_output = neuron_output / (neuron_output.sum(dim=-1, keepdim=True) + 1e-10)
        #     # 计算每个神经元的熵
        #     neuron_entropy = -(neuron_output * torch.log(neuron_output + 1e-10)).sum(dim=0)
        #     # 更新神经元重要性
        #     current_importance += neuron_entropy.abs().detach()
        
        # #-----------三 hessian重要性----------------
        # outputs = model(input_ids, head_mask=head_mask)
        # loss = loss_fn(outputs, label_ids)
        # grads = torch.autograd.grad(loss, head_mask, create_graph=True)[0]
        # for layer in range(n_layers):
        #     for head in range(n_heads):
        #         hessian = torch.autograd.grad(grads[layer, head], head_mask, retain_graph=True)[0]
        #         head_importance[layer, head] += hessian[layer, head].abs().detach()
        #         # head_importance[layer, head] = head_importance[layer, head] + hessian.abs().detach()

        # for w1, b1, w2, current_importance in zip(intermediate_weight, intermediate_bias, output_weight, neuron_importance):
        #     grads_w1 = torch.autograd.grad(loss, w1, create_graph=True)[0]
        #     grads_b1 = torch.autograd.grad(loss, b1, create_graph=True)[0]
        #     grads_w2 = torch.autograd.grad(loss, w2, create_graph=True)[0]

        #     hessian_w1 = torch.autograd.grad(grads_w1.sum(), w1, retain_graph=True)[0]
        #     hessian_b1 = torch.autograd.grad(grads_b1.sum(), b1, retain_graph=True)[0]
        #     hessian_w2 = torch.autograd.grad(grads_w2.sum(), w2, retain_graph=True)[0]

        #     current_importance += (hessian_w1 * w1).sum(dim=1).abs().detach()
        #     current_importance += (hessian_b1 * b1).abs().detach()
        #     current_importance += (hessian_w2 * w2).sum(dim=0).abs().detach()

        break
    return head_importance, neuron_importance #好吧 实际上neuron_importance根本就没做，都是空的，在reorder_head_neuron.py里也把neuron相关的删掉了，太逗了

def reorder_neuron_head(model, head_importance, neuron_importance):

    model = model.module if hasattr(model, 'module') else model

    # reorder heads and ffn neurons
    for layer, current_importance in enumerate(neuron_importance): #草 之前没报错是因为之前没跑 neuron_importance是空的。。
        # reorder heads --> [layer][0]: attention module
        idx = torch.sort(head_importance[layer], descending=True)[-1]
        model.transformer.layers[layer][0].fn.reorder_heads(idx)
        # reorder neurons --> [layer][1]: feed-forward module
        idx = torch.sort(current_importance, descending=True)[-1]
        model.transformer.layers[layer][1].fn.reorder_intermediate_neurons(idx)
        model.transformer.layers[layer][1].fn.reorder_output_neurons(idx)

    for layer, current_importance in enumerate(neuron_importance):
        # reorder heads --> [layer]: block module
        idx = torch.sort(head_importance[layer], descending=True)[-1]
        model.blocks[layer].attn.reorder_heads(idx)

    #     # reorder neurons --> MLP module
    #     idx = torch.sort(current_importance, descending=True)[-1]
    #     model.blocks[layer].mlp.reorder_intermediate_neurons(idx)
    #     model.blocks[layer].mlp.reorder_output_neurons(idx)
    '''
    发生异常: AttributeError
    'Mlp' object has no attribute 'reorder_intermediate_neurons'
    File "/home/ljm/workspace/jxt/ViT_pruning/code/utils.py", line 187, in reorder_neuron_head
        model.blocks[layer].mlp.reorder_intermediate_neurons(idx)
    File "/home/ljm/workspace/jxt/ViT_pruning/code/utils.py", line 257, in train
        reorder_neuron_head(model, head_importance, neuron_importance) #这里是将其按重要性排序的
    File "/home/ljm/workspace/jxt/ViT_pruning/code/main.py", line 215, in <module>
        train(model,
    AttributeError: 'Mlp' object has no attribute 'reorder_intermediate_neurons'
    '''

"""### Training"""

def train(
    model, train_data, eval_data, device,
    mode = "finetuning", width_list = None,
    weights_file = None, model_path = "./",
    loss_fn=nn.CrossEntropyLoss(), epochs=10,depth_list=None,logger=None,savedir=None,test_loader=None,
        **args
    ):
    assert mode in ["finetuning", "width", "depth"], "Wrong mode input"

    model.to(device)

    if weights_file is not None:
        model.load_state_dict(torch.load(weights_file))

    optimizer = Adam(model.parameters(), lr=1e-4)#优化器和调度器
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=50)

    if mode == "finetuning":
        train_model(model, path=model_path,
                    train_data=train_data, eval_data=eval_data,
                    epochs=epochs, loss_fn=loss_fn,
                    optimizer=optimizer, scheduler=scheduler, device=device,logger=logger,test_loader=test_loader
                    )
    
    if mode == "width":
            print("Training Distillation")
            # teacher_model = timm.create_model(args['model_architecture'], pretrained=False)
            # torch.save(model.state_dict(), args.model_path)
            teacher_model = VisionTransformer(
                img_size=args["img_size"],
                patch_size=args["patch_size"],
                num_classes=args["num_classes"],
                embed_dim=args["embed_dim"],
                depth=args["depth"],
                num_heads=args["num_heads"],
                mlp_ratio=args["mlp_ratio"],
                qkv_bias = args["qkv_bias"],
                in_chans=args["in_chans"],
                mha_width=args["mha_width"],
                mlp_width=args["mlp_width"],
                no_ghost=args["no_ghost"],
                ghost_mode=args["ghost_mode"],
            )
            # 加载预训练权重
            pretrained_weights = torch.load(model_path,weights_only=True)

            # 创建一个新的字典，只包含模型中存在的权重
            new_weights = {}
            for key in teacher_model.state_dict().keys():
                if key in pretrained_weights:
                    new_weights[key] = pretrained_weights[key]

            # 将新权重加载到模型中
            teacher_model.load_state_dict(new_weights, strict=False)
            teacher_model.to(device)
            model.load_state_dict(new_weights, strict=False)

            # print(teacher_model.state_dict().keys())
            # teacher_model.load_state_dict(torch.load(model_path))
            head_importance, neuron_importance = compute_neuron_head_importance( #这里是计算神经元和头重要性的
                eval_data, model, args["depth"], args["heads"],
                loss_fn=loss_fn, device=device
                )
            visualize_head_importance(head_importance,savedir)
            visualize_neuron_importance(neuron_importance,savedir)
            # reorder_neuron_head(model, head_importance, neuron_importance) #这里是将其按重要性排序的 草 跑不了就别跑 nnd
            #width_list = sorted(width_list, reverse=True)
            for i, width in enumerate(tqdm(width_list, desc="Width", leave=False)):
                logger.info(f"\nWidth: {width}")
                model.apply(lambda m: setattr(m, 'width_mult', width))
                path = os.path.join(savedir, f"Width{width}_model_width_distillation.pt")
                train_distillation(
                    model, teacher_model = teacher_model,path=path,
                    train_data = train_data, eval_data = eval_data,
                    epochs = 1, device=device,
                    optimizer=optimizer, scheduler=scheduler,loss_fn=loss_fn,logger=logger
                )
                model.load_state_dict(new_weights, strict=False)
            logger.info("Fine tuning after distillation")
            for i, width in enumerate(tqdm(width_list, desc="Width", leave=False)):
                path = os.path.join(savedir, f"Width{width}_model_width_distillation.pt")
                logger.info(f"\nWidth: {width}")
                model.apply(lambda m: setattr(m, 'width_mult', width))
                train_model(
                    model, path = path,
                    train_data=train_data, eval_data = eval_data,
                    epochs=epochs, loss_fn =loss_fn,
                    optimizer=optimizer, scheduler=scheduler, device=device,logger=logger
                )
                model.load_state_dict(torch.load(path), strict=False)

    if mode == "depth":
        width_list = sorted(width_list, reverse=True)
        for i, width in enumerate(tqdm(width_list, desc="Width", leave=False)):
            teacher_model = VisionTransformer(
                img_size=args["img_size"],
                patch_size=args["patch_size"],
                num_classes=args["num_classes"],
                embed_dim=args["embed_dim"],
                depth=args["depth"],
                num_heads=args["num_heads"],
                mlp_ratio=args["mlp_ratio"],
                qkv_bias=args["qkv_bias"],
                in_chans=args["in_chans"],
                mha_width=args["mha_width"],
                mlp_width=args["mlp_width"],
                no_ghost=args["no_ghost"],
                ghost_mode=args["ghost_mode"],
            )
            teacher_model.apply(lambda m: setattr(m, 'width_mult', width))
            teacher_model.to(device)
            path = os.path.join(model_path, f"Width{width}_model_width_distillation.pt")
            teacher_model.load_state_dict(torch.load(path), strict=False)
            for i, depth in enumerate(tqdm(depth_list, desc="Depth", leave=False)):
                model.apply(lambda m: setattr(m, 'width_mult', width))
                model.apply(lambda m: setattr(m, 'depth', depth))
                path = os.path.join(savedir, f"Width{width}_Depth{depth}_model_width_distillation.pt")
                logger.info(path)
                logger.info("Training Distillation")
                train_distillation(
                    model, teacher_model=teacher_model, path=path,
                    train_data=train_data, eval_data=eval_data,
                    epochs=epochs, device=device,
                    optimizer=optimizer, scheduler=scheduler, loss_fn=loss_fn,logger=logger
                )
                model.load_state_dict(torch.load(path), strict=False)#为啥会报错找不到文件夹。。


def train_model(model, train_data, eval_data, path, epochs, loss_fn, optimizer, scheduler, device='cuda', logger=None,test_loader=None,**args):
    model.train()
    best_eval_loss = 1e8
    model.to(device)
    
    for epoch in tqdm(range(epochs), desc="Epochs", leave=False):
        total_loss = 0.0
        logger.info(f"Epoch: {epoch}")

        for i, data in enumerate(tqdm(train_data, desc="Training", leave=False)):
            inputs, labels = tuple(t.to(device) for t in data)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()*inputs.size(0)

        logger.info(f"Train loss = {total_loss/len(train_data.sampler):.4f}")

        model.eval()
        eval_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for i, data in enumerate(tqdm(eval_data, desc="Evaluating", leave=False)):
                inputs, labels = tuple(t.to(device) for t in data)
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                eval_loss += loss.item()*inputs.size(0)

        eval_loss = eval_loss/len(eval_data.sampler)
        scheduler.step(metrics=eval_loss)

        if eval_loss < best_eval_loss:
            torch.save(model.state_dict(), path)
            best_eval_loss = eval_loss

        logger.info(f"Validation loss = {eval_loss:.4f}")

        logger.info(path)
        # model.load_state_dict(torch.load(path,weights_only=True), strict=False)

        model.to(device)
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for i, data in enumerate(tqdm(test_loader, desc="Evaluating", leave=False)):
                inputs, labels = tuple(t.to(device) for t in data)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        logger.info(f'Accuracy of the  network on the test images: %0.2f %%' % accuracy)
        num_params = sum(p.numel() for p in model.parameters())
        logger.info('Number of parameters: %d' % num_params)
        inputs, _ = next(iter(test_loader))
        inputs = inputs.to(device)
        with torch.no_grad():
            start_time = time.time()
            outputs = model(inputs)
            end_time = time.time()
        inference_time = end_time - start_time
        logger.info('Inference time: %.2fms' % (inference_time * 1000))
        flops, params = profile(model, inputs=(inputs,))
        logger.info('Number of parameters: %d' % params)
        logger.info('FLOPS: %.2fG' % (flops / 1e9))

def train_naive(model, train_data, eval_data, path, epochs, loss_fn, 
                        width_list, optimizer, scheduler, layernorm=False, device='cuda', **args):
    model.train()
    best_eval_loss = 1e8

    model.to(device)
    
    for epoch in tqdm(range(epochs), desc="Epochs", leave=False):
        total_loss = 0.0
        print(f"\nEpoch: {epoch}")

        for i, data in enumerate(tqdm(train_data, desc="Training", leave=False)):

            inputs, labels = tuple(t.to(device) for t in data)
            optimizer.zero_grad()
            width_list_loss = 0.0
            for j, width in enumerate(width_list):
                model.apply(lambda m: setattr(m, 'width_mult', width))
                if layernorm:
                    outputs = model(inputs, width_n=j)
                else:
                    outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                width_list_loss += loss.item()
                loss.backward()
            optimizer.step()

            total_loss += (width_list_loss/len(width_list))*inputs.size(0)
        
        print(f"Train loss = {total_loss/len(train_data.sampler)}")

        model.eval()
        eval_loss = 0.0

        with torch.no_grad():
            for i, data in enumerate(tqdm(eval_data, desc="Evaluating", leave=False)):
                inputs, labels = tuple(t.to(device) for t in data)
                width_list_loss = 0.0
                for j, width in enumerate(width_list):
                    model.apply(lambda m: setattr(m, 'width_mult', width))
                    if layernorm:
                        outputs = model(inputs, width_n=j)
                    else:
                        outputs = model(inputs)
                    loss = loss_fn(outputs, labels)
                    width_list_loss += loss.item()

                eval_loss += (width_list_loss/len(width_list))*inputs.size(0)
        
        eval_loss = eval_loss/len(eval_data.sampler)
        scheduler.step(metrics=eval_loss)

        if eval_loss < best_eval_loss:
            torch.save(model.state_dict(), path)
            best_eval_loss = eval_loss

        print(f"Validation loss = {eval_loss}")

def train_sandwich(model, train_data, eval_data, path, epochs, loss_fn,  
                        optimizer, scheduler, layernorm=False, width_min = 0.25, width_max = 1, n_widths=5 , device='cuda',
                   **args
                   ):
    model.train()
    best_eval_loss = 1e8
    
    for epoch in tqdm(range(epochs), desc="Epochs", leave=False):
        total_loss = 0.0
        print(f"\nEpoch: {epoch}")

        for i, data in enumerate(tqdm(train_data, desc="Training", leave=False)):

            inputs, labels = tuple(t.to(device) for t in data)
            optimizer.zero_grad()
            width_list_loss = 0.0
            width_list = list(np.random.choice(np.arange(256*width_min, 256*width_max), n_widths-2))
            width_list = [width_min] + width_list + [width_max]
            for j, width in enumerate(width_list):
                model.apply(lambda m: setattr(m, 'width_mult', width))
                if layernorm:
                    outputs = model(inputs, width_n=j)
                else:
                    outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                width_list_loss += loss.item()
                loss.backward()
            optimizer.step()

            total_loss += (width_list_loss/len(width_list))*inputs.size(0)
        
        print(f"Train loss = {total_loss/len(train_data.sampler)}")

        model.eval()
        eval_loss = 0.0

        with torch.no_grad():
            for i, data in enumerate(tqdm(eval_data, desc="Evaluating", leave=False)):
                inputs, labels = tuple(t.to(device) for t in data)
                width_list_loss = 0.0
                for j, width in enumerate(width_list):
                    model.apply(lambda m: setattr(m, 'width_mult', width))
                    if layernorm:
                        outputs = model(inputs, width_n=j)
                    else:
                        outputs = model(inputs)
                    loss = loss_fn(outputs, labels)
                    width_list_loss += loss.item()

                eval_loss += (width_list_loss/len(width_list))*inputs.size(0)
        
        eval_loss = eval_loss/len(eval_data.sampler)
        scheduler.step(metrics=eval_loss)

        if eval_loss < best_eval_loss:
            torch.save(model.state_dict(), path)
            best_eval_loss = eval_loss

        print(f"Validation loss = {eval_loss}")

def train_incremental(model, train_data, eval_data, path, 
                      epochs, loss_fn, optimizer, scheduler, 
                      freeze_width=None, device='cuda', **args):
    model.train()
    best_eval_loss = 1e8

    def zero_grad_dyna_linear(x, width):
        if isinstance(x, DynaLinear):
            x.set_grad_to_zero(width)
    
    for epoch in tqdm(range(epochs), desc="Epochs", leave=False):
        total_loss = 0.0
        print(f"\nEpoch: {epoch}")

        for i, data in enumerate(tqdm(train_data, desc="Training", leave=False)):

            inputs, labels = tuple(t.to(device) for t in data)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            
            # important step to not destroy previously learned parameters
            if freeze_width:
                model.apply(lambda x: zero_grad_dyna_linear(x, freeze_width))

            optimizer.step()

            total_loss += loss.item()*inputs.size(0)

        print(f"Train loss = {total_loss/len(train_data.sampler)}")

        model.eval()
        eval_loss = 0.0

        with torch.no_grad():
            for i, data in enumerate(tqdm(eval_data, desc="Evaluating", leave=False)):
                inputs, labels = tuple(t.to(device) for t in data)
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                eval_loss += loss.item()*inputs.size(0)

        eval_loss = eval_loss/len(eval_data.sampler)
        scheduler.step(metrics=eval_loss)

        if eval_loss < best_eval_loss:
            torch.save(model.state_dict(), path)
            best_eval_loss = eval_loss

        print(f"Validation loss = {eval_loss}")

def train_distillation(model, teacher_model, train_data, eval_data, path,
                      epochs, optimizer, scheduler, loss_fn, device='cuda',lambda1=0.5, lambda2=0.5,logger=None,
                      **args):
    model.train()
    best_eval_loss = 1e8

    loss_mse = nn.MSELoss()

    def zero_grad_dyna_linear(x, width):
        if isinstance(x, DynaLinear):
            x.set_grad_to_zero(width)

    for epoch in tqdm(range(epochs), desc="Epochs", leave=False):
        total_loss = 0.0
        logger.info(f"\nEpoch: {epoch}")

        for i, data in enumerate(tqdm(train_data, desc="Training", leave=False)):
            inputs, labels = tuple(t.to(device) for t in data)
            optimizer.zero_grad()
            with torch.no_grad():
                teacher_out, teacher_hidden = teacher_model(inputs, return_states=True)
            student_out, student_hidden = model(inputs, return_states=True)
            loss1 = soft_cross_entropy(student_out, teacher_out.detach())
            loss2 = loss_mse(student_hidden, teacher_hidden.detach())

            loss = loss1*lambda1 + loss2*lambda2
            loss.backward()

            optimizer.step()

            total_loss += loss.item()*inputs.size(0)

        logger.info(f"Train loss = {total_loss/len(train_data.sampler)}")

        # model.eval()
        # eval_loss = 0.0
        # correct = 0
        # total = 0

        # with torch.no_grad():
        #     for i, data in enumerate(tqdm(eval_data, desc="Evaluating", leave=False)):
        #         inputs, labels = tuple(t.to(device) for t in data)
        #         start_time = time.time()
        #         outputs = model(inputs)
        #         end_time = time.time()
        #         loss = loss_fn(outputs, labels)
        #         eval_loss += loss.item()*inputs.size(0)
        #         _, predicted = torch.max(outputs.data, 1)
        #         total += labels.size(0)
        #         correct += (predicted == labels).sum().item()

        # eval_loss = eval_loss/len(eval_data.sampler)
        # scheduler.step(metrics=eval_loss)
        
        # print(f"Validation loss = {eval_loss}")
        # accuracy = 100 * correct / total
        # print(f'Accuracy of the  network on the test images: %d %%' % accuracy)
        # num_params = sum(p.numel() for p in model.parameters())
        # print('Number of parameters: %d' % num_params)
            
        # inference_time = end_time - start_time
        # print('Inference time: %.2fms' % (inference_time * 1000))
        # flops, params = profile(model, inputs=(inputs,))
        # print('Number of parameters: %d' % params)
        # print('FLOPS: %.2fG' % (flops / 1e9))

        # if eval_loss < best_eval_loss:
        #     torch.save(model.state_dict(), path)
        #     best_eval_loss = eval_loss
        
        model.eval()
        eval_loss = 0.0

        with torch.no_grad():
            for i, data in enumerate(tqdm(eval_data, desc="Evaluating", leave=False)):
                inputs, labels = tuple(t.to(device) for t in data)
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                eval_loss += loss.item()*inputs.size(0)

        eval_loss = eval_loss/len(eval_data.sampler)
        scheduler.step(metrics=eval_loss)

        if eval_loss < best_eval_loss:
            torch.save(model.state_dict(), path)#草 就这行 老是存不了
            best_eval_loss = eval_loss

        logger.info(f"Validation loss = {eval_loss}")

        model.to(device)
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for i, data in enumerate(tqdm(eval_data, desc="Evaluating", leave=False)):
                inputs, labels = tuple(t.to(device) for t in data)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        logger.info(f'Accuracy of the  network on the test images: %0.2f %%' % accuracy)
        num_params = sum(p.numel() for p in model.parameters())
        logger.info('Number of parameters: %d' % num_params)
        inputs, _ = next(iter(eval_data))
        inputs = inputs.to(device)
        with torch.no_grad():
            start_time = time.time()
            outputs = model(inputs)
            end_time = time.time()
        inference_time = end_time - start_time
        logger.info('Inference time: %.2fms' % (inference_time * 1000))
        flops, params = profile(model, inputs=(inputs,))
        logger.info('Number of parameters: %d' % params)
        logger.info('FLOPS: %.2fG' % (flops / 1e9))

        

def print_metrics(model, test_data, metric_funcs, device, loss_fn=None, width_list=None, width_switch=False):
    model.eval()
    model.to(device)

    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()

    if width_list:
        print(f"Width | {'Loss':^20}", end = "")
        for metric, args in metric_funcs:
            print(f" | {metric.__name__:^20}", end = "")
        print()
        for k, width in enumerate(width_list):
            print(f"{width:^5}", end = "")
            model.apply(lambda m: setattr(m, 'width_mult', width))
            preds = []
            truths = []
            
            total_loss = 0
            with torch.no_grad():
                for i, data in enumerate(test_data):
                    inputs, labels = tuple(t.to(device) for t in data)
                    if width_switch:
                        outputs = model(inputs, width_n=k)
                    else:
                        outputs = model(inputs)
                    loss = loss_fn(outputs, labels)
                    total_loss += loss.item()*inputs.size(0)
                    preds = preds + list(
                        torch.argmax(
                            nn.functional.softmax(outputs.cpu(), dim=1), 
                            dim=1
                            )
                        )
                    truths = truths + list(labels.cpu())
            test_loss = total_loss/len(test_data.sampler)
            print(f" | {test_loss:^20.4f}", end = "")
            for metric, args in metric_funcs:
                perf = metric(truths, preds, **args)
                print(f" | {perf:^20.4f}", end = "")
            print()
    else:
        preds = []
        truths = []
        total_loss = 0
        with torch.no_grad():
            for i, data in enumerate(test_data):
                inputs, labels = tuple(t.to(device) for t in data)
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                total_loss += loss.item()*inputs.size(0)
                preds = preds + list(
                    torch.argmax(
                        nn.functional.softmax(outputs.cpu(), dim=1), 
                        dim=1
                        )
                    )
                truths = truths + list(labels.cpu())
        test_loss = total_loss/len(test_data.sampler)
        print(f"Loss: {test_loss}")
        for metric, args  in metric_funcs:
            perf = metric(truths, preds, **args)
            print(f"{metric.__name__}: {perf:^.4f}")

def print_accuracy(model, test_data, loss_fn=None, width_list=None, width_switch=False, device=None):
    model.eval()
    model.to(device)
    metric_funcs = [(accuracy_score, {})]

    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()

    if width_list:
        print(f"Width | {'Loss':^20}", end = "")
        for metric, args in metric_funcs:
            print(f" | {metric.__name__:^20}", end = "")
        print()
        for k, width in enumerate(width_list):
            print(f"{width:^5}", end = "")
            model.apply(lambda m: setattr(m, 'width_mult', width))
            preds = []
            truths = []
            
            total_loss = 0
            with torch.no_grad():
                for i, data in enumerate(test_data):
                    inputs, labels = tuple(t.to(device) for t in data)
                    if width_switch:
                        outputs = model(inputs, width_n=k)
                    else:
                        outputs = model(inputs)
                    loss = loss_fn(outputs, labels)
                    total_loss += loss.item()*inputs.size(0)
                    preds = preds + list(
                        torch.argmax(
                            nn.functional.softmax(outputs.cpu(), dim=1), 
                            dim=1
                            )
                        )
                    truths = truths + list(labels.cpu())
            test_loss = total_loss/len(test_data.sampler)
            print(f" | {test_loss:^20.4f}", end = "")
            for metric, args in metric_funcs:
                perf = metric(truths, preds, **args)
                print(f" | {perf:^20.4f}", end = "")
            print()
    else:
        preds = []
        truths = []
        total_loss = 0
        with torch.no_grad():
            for i, data in enumerate(test_data):
                inputs, labels = tuple(t.to(device) for t in data)
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                total_loss += loss.item()*inputs.size(0)
                preds = preds + list(
                    torch.argmax(
                        nn.functional.softmax(outputs.cpu(), dim=1), 
                        dim=1
                        )
                    )
                truths = truths + list(labels.cpu())
        test_loss = total_loss/len(test_data.sampler)
        print(f"Loss: {test_loss}")
        for metric, args  in metric_funcs:
            perf = metric(truths, preds, **args)
            print(f"{metric.__name__}: {perf:^.4f}")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
